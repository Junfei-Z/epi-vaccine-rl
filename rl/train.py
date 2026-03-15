# -*- coding: utf-8 -*-
"""
rl/train.py — PPO training loop, evaluation, and early-stopping utilities.

Classes
-------
StoppingCriteria  — configuration dataclass for early-stopping rules
TrainingMonitor   — tracks best model and decides when to stop

Functions
---------
quick_eval_det    — deterministic episode evaluation (no exploration)
run_training      — full warm/cold-start PPO training loop
"""

import os
import numpy as np
import torch
from collections import deque

from config import D
from env import make_env_from_graph
from rl.ppo import PPO, PPOBuffer


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class StoppingCriteria:
    """
    Configuration for early-stopping rules.

    Stopping is triggered when either:
    (a) no improvement for `patience` consecutive evaluation rounds, or
    (b) the rolling window mean falls below best_score - min_delta.

    Parameters
    ----------
    patience  : int   — max evaluation rounds without improvement before stop
    min_delta : float — minimum improvement to count as 'better'
    window    : int   — rolling window size for plateau detection
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, window: int = 5):
        self.patience  = patience
        self.min_delta = min_delta
        self.window    = window


class TrainingMonitor:
    """
    Tracks training progress and saves the best model checkpoint.

    Note: scores are negated deaths (higher = fewer deaths = better),
    so improvement means increasing score.

    Parameters
    ----------
    criteria : StoppingCriteria
    """

    def __init__(self, criteria: StoppingCriteria):
        self.criteria      = criteria
        self.best_score    = -float('inf')
        self.best_state    = None
        self.no_improve    = 0
        self.window_scores = deque(maxlen=criteria.window)

    def update(self, score: float, agent) -> bool:
        """
        Register a new evaluation score and decide whether to stop.

        Parameters
        ----------
        score : float — evaluation metric (higher = better)
        agent : object with .state_dict() — model to checkpoint

        Returns
        -------
        stop : bool — True if training should terminate
        """
        self.window_scores.append(score)

        if score > self.best_score + self.criteria.min_delta:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in agent.state_dict().items()}
            self.no_improve = 0
        else:
            self.no_improve += 1

        stop = False
        if self.no_improve >= self.criteria.patience:
            stop = True
        if len(self.window_scores) == self.criteria.window:
            mean_score = sum(self.window_scores) / float(self.criteria.window)
            if mean_score < self.best_score - self.criteria.min_delta:
                stop = True

        return stop

    def load_best(self, agent):
        """Restore the best checkpoint into `agent`. Returns agent."""
        if self.best_state is not None:
            agent.load_state_dict(self.best_state)
        return agent


# ---------------------------------------------------------------------------
# Deterministic evaluation
# ---------------------------------------------------------------------------

def quick_eval_det(ppo: PPO, env_args: tuple, n_eval: int = 3) -> float:
    """
    Evaluate the current policy deterministically over n_eval episodes.

    Uses the distribution mean (no sampling) so results are reproducible.
    The environment is rebuilt fresh for each episode to avoid state leakage.

    Parameters
    ----------
    ppo      : PPO agent
    env_args : tuple (G, groups, deg_dict, params_global, capacity_daily,
                      reward_scale, seed_counts, substeps, dt)
    n_eval   : number of episodes to average over

    Returns
    -------
    mean final deaths across episodes (float)
    """
    G, groups, deg_dict, params_global, capacity_daily, reward_scale, \
        seed_counts, substeps, dt = env_args

    vals = []
    ppo.policy.eval()

    for _ in range(n_eval):
        env, _, _, _, _ = make_env_from_graph(
            G=G, groups=groups, deg_dict=deg_dict,
            params_global=params_global, capacity_daily=capacity_daily,
            reward_scale=reward_scale, seed_counts=seed_counts,
            substeps=substeps, dt=dt, deterministic=True,
        )
        state = env.reset(seed_counts=seed_counts)
        done  = False

        while not done:
            with torch.no_grad():
                s_t    = torch.from_numpy(state).float()
                dist   = ppo.policy.dist(s_t)
                act    = torch.clamp(dist.mean, min=1e-6)
                shares = (act / act.sum()).detach().cpu().numpy()
            state, _, done, _ = env.step(shares)

        vals.append(int(np.sum(env.status == D)))

    ppo.policy.train()
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(
    G,
    groups: dict,
    deg_dict: dict,
    params_global: dict,
    capacity_daily: int,
    prior_path: str = None,
    max_episodes: int = 200,
    reward_scale: float = 1.0,
    episodes_per_update: int = 15,
    warm_mean_episodes: int = 8,
    window_size: int = 40,
    rel_std_thresh: float = 0.05,
    patience: int = 3,
    min_episodes: int = 40,
    seed_counts: dict = None,
    substeps: int = 1,
    dt: float = 1.0,
    label: str = None,
    out_dir: str = '.',
) -> tuple:
    """
    Run a full warm-start or cold-start PPO training loop.

    Warm start (prior_path given)
    ------------------------------
    • Episode 0: follow the ODE prior exactly (imitation episode)
    • Episodes 1 … warm_mean_episodes-1: sample with high temperature (warm)
    • Episodes warm_mean_episodes+: sample with cold temperature (exploit)
    • KL penalty toward prior decays throughout

    Cold start (prior_path=None)
    ----------------------------
    • All episodes use cold temperature sampling from the start
    • Weaker prior_weight / prior_decay defaults

    Early stopping
    --------------
    Triggered when the relative std of the last `window_size` evaluation
    death counts drops below `rel_std_thresh` for `patience` consecutive
    update rounds, indicating convergence.

    Parameters
    ----------
    G, groups, deg_dict, params_global, capacity_daily : env spec
    prior_path          : path to .npy prior file, or None for cold start
    max_episodes        : hard cap on training episodes
    reward_scale        : death penalty multiplier
    episodes_per_update : rollout episodes accumulated before each PPO update
    warm_mean_episodes  : episodes using high-temperature sampling
    window_size         : rolling window for convergence check
    rel_std_thresh      : relative std threshold for early stopping
    patience            : number of consecutive plateaus before stopping
    min_episodes        : minimum episodes before early stopping is checked
    seed_counts         : initial infection counts per group
    substeps, dt        : env disease-progression sub-stepping
    label               : if given, save best policy to '{out_dir}/best_policy_{label}.pt'
    out_dir             : directory for saved model files

    Returns
    -------
    ppo       : trained PPO agent (best checkpoint loaded)
    hist_eval : list of deterministic eval death counts per update round
    """
    os.makedirs(out_dir, exist_ok=True)

    # build env once to determine state_dim
    env, obs, _, _, _ = make_env_from_graph(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity_daily,
        reward_scale=reward_scale, seed_counts=seed_counts,
        substeps=substeps, dt=dt,
    )
    state_dim = int(len(obs))
    prior     = None if prior_path is None else np.load(prior_path)

    # warm vs cold hyperparameters
    is_warm      = prior is not None
    prior_weight = 0.5   if is_warm else 0.3   # reduced: prior guides but doesn't dominate
    prior_decay  = 0.97  if is_warm else 0.99  # faster decay: RL takes over sooner
    prior_alpha  = 20.0  if is_warm else 30.0  # softer prior Dirichlet

    ppo = PPO(
        state_dim=state_dim, action_dim=3,
        lr_actor=2e-4, lr_critic=6e-4,
        gamma=0.99, K_epochs=10, eps_clip=0.2,
        prior_policy=prior,
        T_horizon=params_global['T_HORIZON'],
        prior_weight=prior_weight,
        prior_decay=prior_decay,
        prior_alpha=prior_alpha,
        entropy_coef=0.001, entropy_decay=0.99,
        sample_temp_warm=3.0, sample_temp_cold=1.0,  # fixed: warm=explore, cold=exploit
    )
    buffer = PPOBuffer()

    best_death       = float('inf')
    best_state       = None
    hist_eval        = []
    patience_counter = 0

    env_args = (G, groups, deg_dict, params_global, capacity_daily,
                reward_scale, seed_counts, substeps, dt)

    for ep in range(max_episodes):
        state = env.reset(seed_counts=seed_counts)
        done  = False

        while not done:
            s_t = torch.from_numpy(state).float()

            # action selection strategy per episode phase
            if prior is not None and ep == 0:
                # episode 0: imitate prior exactly (no exploration)
                shares        = prior[env.day]
                action_tensor = torch.tensor(shares, dtype=torch.float32)
                action_tensor = torch.clamp(action_tensor, min=1e-6)
                action_tensor = action_tensor / action_tensor.sum()
                log_prob      = ppo.policy_old.dist(s_t).log_prob(action_tensor).detach()
            elif ep < warm_mean_episodes:
                # warm-up: high temperature → more exploration
                action_tensor, log_prob = ppo.policy.sample_with_temp(
                    s_t, ppo.policy_old, sample_temp=ppo.sample_temp_warm,
                )
            else:
                # exploitation: lower temperature → sharper actions
                action_tensor, log_prob = ppo.policy.sample_with_temp(
                    s_t, ppo.policy_old, sample_temp=ppo.sample_temp_cold,
                )

            next_state, reward, done, _ = env.step(action_tensor.numpy())
            buffer.states.append(s_t)
            buffer.actions.append(action_tensor)
            buffer.log_probs.append(log_prob)
            buffer.rewards.append(reward)
            buffer.is_terminals.append(done)
            state = next_state

        # PPO update every `episodes_per_update` episodes
        if (ep + 1) % episodes_per_update == 0:
            ppo.update(buffer)
            eval_deaths = quick_eval_det(ppo, env_args, n_eval=3)
            hist_eval.append(eval_deaths)
            if eval_deaths < best_death:
                best_death = eval_deaths
                best_state = {k: v.cpu().clone()
                              for k, v in ppo.policy.state_dict().items()}

        # early stopping: check convergence after min_episodes
        if window_size and len(hist_eval) >= window_size and ep >= min_episodes:
            recent  = np.array(hist_eval[-window_size:])
            rel_std = recent.std() / max(1.0, recent.mean())
            if rel_std < rel_std_thresh:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[run_training] Early stop at episode {ep} "
                          f"(rel_std={rel_std:.4f})")
                    break
            else:
                patience_counter = 0

    # flush any remaining buffer transitions
    if len(buffer.states) > 0:
        ppo.update(buffer)
        eval_deaths = quick_eval_det(ppo, env_args, n_eval=3)
        hist_eval.append(eval_deaths)
        if eval_deaths < best_death:
            best_death = eval_deaths
            best_state = {k: v.cpu().clone()
                          for k, v in ppo.policy.state_dict().items()}

    # restore best checkpoint and optionally save to disk
    if best_state is not None:
        ppo.policy.load_state_dict(best_state)
        ppo.policy_old.load_state_dict(ppo.policy.state_dict())
        if label is not None:
            save_path = os.path.join(out_dir, f'best_policy_{label}.pt')
            torch.save(ppo.policy.state_dict(), save_path)
            print(f"[run_training] Best policy saved → {save_path}  "
                  f"(deaths={best_death:.1f})")

    return ppo, hist_eval
