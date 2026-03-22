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
from rl.model import NodeScoringPolicy


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

    # warm vs cold hyperparameters.
    # Warm start uses behavioural-cloning initialisation: the first
    # `warm_mean_episodes` episodes purely imitate the ODE prior (no KL
    # penalty during PPO updates).  After that, free PPO takes over.
    # Setting prior_weight=0 removes the KL constraint entirely so the
    # policy can improve beyond the ODE solution once initialised.
    is_warm      = prior is not None
    prior_weight = 0.0                          # no KL penalty: BC init + free PPO
    prior_decay  = 0.99  if is_warm else 0.99
    prior_alpha  = 20.0  if is_warm else 30.0

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
        sample_temp_warm=3.0, sample_temp_cold=1.0,
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
            if prior is not None and ep < warm_mean_episodes:
                # Behavioural-cloning phase: imitate ODE prior exactly.
                # Repeating over warm_mean_episodes episodes gives the policy
                # a strong initialisation near the ODE solution before free
                # PPO exploration begins.  No temperature scaling needed here
                # since we are following the prior deterministically.
                shares        = prior[env.day]
                action_tensor = torch.tensor(shares, dtype=torch.float32)
                action_tensor = torch.clamp(action_tensor, min=1e-6)
                action_tensor = action_tensor / action_tensor.sum()
                log_prob      = ppo.policy_old.dist(s_t).log_prob(action_tensor).detach()
            elif prior is not None and ep < warm_mean_episodes * 2:
                # exploration phase: high temperature around the now-initialised policy
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


# ---------------------------------------------------------------------------
# Node-level RL training
# ---------------------------------------------------------------------------

def run_training_node_rl(
    G,
    groups: dict,
    deg_dict: dict,
    params_global: dict,
    capacity_daily: int,
    max_episodes: int = 300,
    episodes_per_update: int = 10,
    lr: float = 3e-4,
    gamma: float = 0.99,
    K_epochs: int = 8,
    eps_clip: float = 0.2,
    window_size: int = 30,
    rel_std_thresh: float = 0.05,
    patience: int = 4,
    min_episodes: int = 40,
    seed_counts: dict = None,
    label: str = None,
    out_dir: str = '.',
) -> tuple:
    """
    Train a NodeScoringPolicy via PPO.

    At each step the policy:
      1. Receives 34-dim global state (group aggregates + 3 pressure features)
      2. Computes per-node features for all susceptible nodes (6-dim each)
      3. Scores every susceptible node via a shared MLP
      4. Selects top-V_MAX nodes via Gumbel-top-k (stochastic training)
      5. Passes selected node IDs directly to env.step_node_ids()

    This gives RL genuine information ODE cannot access:
      - Which specific nodes are surrounded by infectious neighbours RIGHT NOW
      - Enabling pre-emptive vaccination before the epidemic wave hits

    Parameters
    ----------
    G, groups, deg_dict, params_global, capacity_daily : env spec
    max_episodes        : hard cap on training episodes
    episodes_per_update : rollout episodes before each PPO update
    lr                  : learning rate (shared for scorer + critic)
    gamma               : discount factor
    K_epochs            : PPO gradient steps per update
    eps_clip            : PPO clip parameter
    window_size         : rolling window for early stopping
    rel_std_thresh      : relative std threshold for early stopping
    patience            : consecutive plateau rounds before stopping
    min_episodes        : minimum episodes before early stopping is active
    seed_counts         : initial infection counts per group
    label               : if given, save best policy to out_dir
    out_dir             : directory for saved model files

    Returns
    -------
    policy    : NodeScoringPolicy (best checkpoint)
    hist_eval : list of deterministic eval death counts per update round
    """
    os.makedirs(out_dir, exist_ok=True)

    env, obs, _, _, _ = make_env_from_graph(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity_daily,
        seed_counts=seed_counts, deterministic=False,
    )

    policy   = NodeScoringPolicy(hidden=64)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    MSE       = torch.nn.MSELoss()

    # buffer stores per-step (log_prob, reward, value, done)
    buf_logp    = []
    buf_rewards = []
    buf_values  = []
    buf_dones   = []

    best_death       = float('inf')
    best_state       = None
    hist_eval        = []
    patience_counter = 0

    def eval_det(n_eval=3):
        """Deterministic evaluation: top-k selection, no Gumbel noise."""
        policy.eval()
        deaths = []
        for _ in range(n_eval):
            env_e, _, _, _, _ = make_env_from_graph(
                G=G, groups=groups, deg_dict=deg_dict,
                params_global=params_global, capacity_daily=capacity_daily,
                seed_counts=seed_counts, deterministic=True,
            )
            env_e.reset(seed_counts=seed_counts)
            done = False
            while not done:
                g_state = torch.from_numpy(env_e.obs_with_pressure()).float()
                s_ids, feats = env_e.node_features()
                if len(s_ids) == 0:
                    _, _, done, _ = env_e.step_node_ids([])
                    continue
                f_t = torch.from_numpy(feats).float()
                with torch.no_grad():
                    idxs, _ = policy.select(g_state, f_t, capacity_daily,
                                            deterministic=True)
                selected = [s_ids[i] for i in idxs.tolist()]
                _, _, done, _ = env_e.step_node_ids(selected)
            deaths.append(int(np.sum(env_e.status == D)))
        policy.train()
        return float(np.mean(deaths))

    for ep in range(max_episodes):
        env.reset(seed_counts=seed_counts)
        done = False

        ep_logp    = []
        ep_rewards = []
        ep_values  = []
        ep_dones   = []

        while not done:
            g_state = torch.from_numpy(env.obs_with_pressure()).float()
            s_ids, feats = env.node_features()

            if len(s_ids) == 0:
                _, reward, done, _ = env.step_node_ids([])
                ep_logp.append(torch.tensor(0.0))
                ep_rewards.append(reward)
                ep_values.append(policy.value(g_state).squeeze())
                ep_dones.append(float(done))
                continue

            f_t = torch.from_numpy(feats).float()
            idxs, log_prob = policy.select(g_state, f_t, capacity_daily,
                                           deterministic=False)
            selected = [s_ids[i] for i in idxs.tolist()]

            _, reward, done, _ = env.step_node_ids(selected)

            ep_logp.append(log_prob)
            ep_rewards.append(reward)
            ep_values.append(policy.value(g_state).squeeze())
            ep_dones.append(float(done))

        buf_logp.extend(ep_logp)
        buf_rewards.extend(ep_rewards)
        buf_values.extend(ep_values)
        buf_dones.extend(ep_dones)

        # PPO update every episodes_per_update episodes
        if (ep + 1) % episodes_per_update == 0:
            rewards  = torch.tensor(buf_rewards, dtype=torch.float32)
            dones    = torch.tensor(buf_dones,   dtype=torch.float32)
            values   = torch.stack(buf_values).detach()
            old_logp = torch.stack(buf_logp).detach()

            # GAE advantage
            adv  = torch.zeros_like(rewards)
            gae  = 0.0
            next_v = torch.cat([values[1:], torch.tensor([0.0])])
            for t in reversed(range(len(rewards))):
                delta  = rewards[t] + gamma * next_v[t] * (1 - dones[t]) - values[t]
                gae    = delta + gamma * 0.95 * (1 - dones[t]) * gae
                adv[t] = gae
            returns = adv + values
            if adv.std() > 1e-6:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # K gradient steps
            for _ in range(K_epochs):
                # recompute log_probs requires re-running policy — use stored
                # log_probs as proxy (single-step PPO without re-scoring batch)
                ratios = torch.exp(
                    torch.stack(buf_logp) - old_logp
                ).detach()   # treat as IS weights (approx PPO)

                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * adv
                v_pred = torch.stack(buf_values).squeeze()

                loss = (-torch.min(surr1, surr2)
                        + 0.5 * MSE(v_pred, returns)).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

            buf_logp.clear(); buf_rewards.clear()
            buf_values.clear(); buf_dones.clear()

            eval_deaths = eval_det(n_eval=3)
            hist_eval.append(eval_deaths)
            if eval_deaths < best_death:
                best_death = eval_deaths
                best_state = {k: v.cpu().clone()
                              for k, v in policy.state_dict().items()}

            print(f"[node_rl] ep={ep+1:3d}  eval_deaths={eval_deaths:.1f}")

        # early stopping
        if window_size and len(hist_eval) >= window_size and ep >= min_episodes:
            recent  = np.array(hist_eval[-window_size:])
            rel_std = recent.std() / max(1.0, recent.mean())
            if rel_std < rel_std_thresh:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[node_rl] Early stop at episode {ep}")
                    break
            else:
                patience_counter = 0

    if best_state is not None:
        policy.load_state_dict(best_state)
        if label is not None:
            save_path = os.path.join(out_dir, f'best_node_policy_{label}.pt')
            torch.save(policy.state_dict(), save_path)
            print(f"[node_rl] Best policy saved → {save_path} (deaths={best_death:.1f})")

    return policy, hist_eval
