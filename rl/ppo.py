# -*- coding: utf-8 -*-
"""
rl/ppo.py — PPO algorithm with optional ODE-prior regularisation.

Classes
-------
PPOBuffer  — experience replay buffer (one episode at a time)
PPO        — Proximal Policy Optimisation with:
               • GAE advantage estimation
               • KL-divergence penalty toward an ODE prior policy
               • Decaying entropy bonus
               • Temperature-scaled exploration warm-up
"""

import torch
import torch.nn as nn
import torch.optim as optim

from config import MIN_CONC
from rl.model import ActorCritic


# ---------------------------------------------------------------------------
# Experience buffer
# ---------------------------------------------------------------------------

class PPOBuffer:
    """
    Flat experience buffer for one or more rollout episodes.

    Stores (state, action, log_prob, reward, done) tuples as Python lists
    of tensors/scalars so they can be stacked into batches during the update.
    """

    def __init__(self):
        self.states       = []
        self.actions      = []
        self.log_probs    = []
        self.rewards      = []
        self.is_terminals = []

    def clear(self):
        """Empty all stored transitions."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.is_terminals.clear()


# ---------------------------------------------------------------------------
# PPO agent
# ---------------------------------------------------------------------------

class PPO:
    """
    Proximal Policy Optimisation for vaccine allocation.

    The standard PPO clipped objective is augmented with a KL-divergence
    penalty that encourages the policy to stay close to an ODE-derived prior,
    which is gradually relaxed via `prior_decay` over training.

    Parameters
    ----------
    state_dim, action_dim : network input/output dimensions
    lr_actor, lr_critic   : separate Adam learning rates
    gamma                 : discount factor
    K_epochs              : number of gradient steps per PPO update
    eps_clip              : PPO clipping parameter ε
    prior_policy          : np.ndarray of shape (T, 3) or None
                            per-day simplex prior from ODE solution
    T_horizon             : episode length (used to decode day from obs)
    prior_weight          : initial KL penalty coefficient
    prior_decay           : multiplicative decay of prior_weight per K_epoch step
    prior_alpha           : Dirichlet concentration scale for prior distribution
    entropy_coef          : initial entropy bonus coefficient
    entropy_decay         : multiplicative decay of entropy_coef per K_epoch step
    sample_temp_warm      : Dirichlet temperature during early warm-up episodes
    sample_temp_cold      : Dirichlet temperature during later episodes
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float,
        lr_critic: float,
        gamma: float,
        K_epochs: int,
        eps_clip: float,
        prior_policy=None,
        T_horizon: int = 60,
        prior_weight: float = 1.6,
        prior_decay: float = 0.995,
        prior_alpha: float = 60.0,
        entropy_coef: float = 0.001,
        entropy_decay: float = 0.99,
        sample_temp_warm: float = 2.0,
        sample_temp_cold: float = 2.5,
    ):
        self.gamma    = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip

        self.policy     = ActorCritic(state_dim, action_dim)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(),  'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
        ])
        self.MSE = nn.MSELoss()

        # prior: store as tensor on CPU; moved to device lazily in update()
        self.prior      = (None if prior_policy is None
                           else torch.tensor(prior_policy, dtype=torch.float32))
        self.T_horizon  = T_horizon

        self.prior_weight = prior_weight
        self.prior_decay  = prior_decay
        self.prior_alpha  = prior_alpha

        self.entropy_coef  = entropy_coef
        self.entropy_decay = entropy_decay

        self.sample_temp_warm = sample_temp_warm
        self.sample_temp_cold = sample_temp_cold

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state_tensor: torch.Tensor) -> tuple:
        """
        Sample an action using the frozen old policy.

        Parameters
        ----------
        state_tensor : Tensor of shape (state_dim,)

        Returns
        -------
        action : Tensor of shape (action_dim,)
        logp   : scalar Tensor
        """
        return self.policy.act_from_old(state_tensor, self.policy_old)

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, buffer: PPOBuffer, use_gae: bool = True, lam: float = 0.95) -> None:
        """
        Run K_epochs of PPO updates on the stored buffer transitions.

        Advantage estimation
        --------------------
        use_gae=True  → Generalised Advantage Estimation (λ-returns)
        use_gae=False → simple Monte-Carlo returns minus baseline

        Prior KL regularisation
        -----------------------
        When a prior policy is available, adds a KL(current ‖ prior) penalty
        per transition. The prior distribution at day t is a Dirichlet with
        concentration = prior_vec[t] * prior_alpha + MIN_CONC.
        The KL is computed analytically via lgamma / digamma identities.
        The penalty weight `pw` decays by prior_decay each gradient step so
        the prior influence fades as the agent gains experience.

        Parameters
        ----------
        buffer  : PPOBuffer populated by the calling training loop
        use_gae : whether to use GAE (True) or MC returns (False)
        lam     : GAE lambda parameter
        """
        rewards      = torch.tensor(buffer.rewards,      dtype=torch.float32)
        dones        = torch.tensor(buffer.is_terminals, dtype=torch.float32)
        states       = torch.stack(buffer.states).detach()
        actions      = torch.stack(buffer.actions).detach()
        old_logprobs = torch.stack(buffer.log_probs).detach()

        # --- advantage estimation ---
        with torch.no_grad():
            values      = self.policy.critic(states).squeeze()
            next_values = torch.cat([values[1:], torch.tensor([0.0])])

            if use_gae:
                adv = torch.zeros_like(rewards)
                gae = 0.0
                for t in reversed(range(len(rewards))):
                    delta  = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
                    gae    = delta + self.gamma * lam * (1 - dones[t]) * gae
                    adv[t] = gae
                returns = adv + values
            else:
                ret     = torch.zeros_like(rewards)
                running = 0.0
                for t in reversed(range(len(rewards))):
                    if dones[t]:
                        running = 0.0
                    running = rewards[t] + self.gamma * running
                    ret[t]  = running
                returns = ret
                adv     = returns - values

        # normalise advantages
        if adv.std() > 1e-6:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # --- K gradient steps ---
        pw = self.prior_weight
        for _ in range(self.K_epochs):
            logprobs, v_pred, entropy = self.policy.evaluate(states, actions)
            v_pred = v_pred.squeeze()

            ratios = torch.exp(logprobs - old_logprobs)
            surr1  = ratios * adv
            surr2  = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MSE(v_pred, returns)
                - self.entropy_coef * entropy
            )

            # optional KL penalty toward ODE prior
            if self.prior is not None:
                # decode which day each transition belongs to
                # obs[-1] stores day / T_horizon (see env._obs)
                days_idx  = (states[:, -1] * self.T_horizon).round().long().clamp(0, self.T_horizon - 1)
                prior_vec = self.prior.to(states.device)[days_idx]

                # normalise prior shares to valid simplex
                eps_clip  = 1e-6
                prior_vec = torch.clamp(prior_vec, min=eps_clip)
                prior_vec = prior_vec / (prior_vec.sum(dim=-1, keepdim=True) + 1e-12)
                prior_vec = torch.clamp(prior_vec, max=1.0 - eps_clip)
                prior_vec = prior_vec / (prior_vec.sum(dim=-1, keepdim=True) + 1e-12)

                # Dirichlet KL(current ‖ prior) via analytic formula
                dist_cur   = self.policy.dist(states)
                conc_cur   = dist_cur.concentration
                conc_prior = prior_vec * self.prior_alpha + MIN_CONC

                sa    = conc_cur.sum(-1, keepdim=True)
                sb    = conc_prior.sum(-1, keepdim=True)
                term1 = torch.lgamma(sa) - torch.sum(torch.lgamma(conc_cur),   dim=-1, keepdim=True)
                term2 = torch.lgamma(sb) - torch.sum(torch.lgamma(conc_prior), dim=-1, keepdim=True)
                term3 = torch.sum(
                    (conc_cur - conc_prior) * (torch.digamma(conc_cur) - torch.digamma(sa)),
                    dim=-1, keepdim=True,
                )
                kl   = (term1 - term2 + term3).squeeze(-1)
                loss = loss + pw * kl

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

            # decay entropy bonus and prior weight each gradient step
            self.entropy_coef = max(0.0003, self.entropy_coef * self.entropy_decay)
            pw *= self.prior_decay

        # persist decayed prior_weight for next update call
        self.prior_weight = pw
        self.policy_old.load_state_dict(self.policy.state_dict())
        buffer.clear()
