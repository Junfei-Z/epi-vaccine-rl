# -*- coding: utf-8 -*-
"""
rl/model.py — Actor-Critic neural network for vaccine allocation PPO.

Architecture
------------
Actor  : Linear(state_dim→128) → Tanh → Linear(128→64) → Tanh → Linear(64→3)
         Output passed through Softplus * CONC_SCALE + MIN_CONC to get
         Dirichlet concentration parameters (always positive).

Critic : Linear(state_dim→128) → Tanh → Linear(128→64) → Tanh → Linear(64→1)
         Outputs a scalar state-value estimate.

The policy outputs a Dirichlet distribution over the 3-group simplex,
so actions are naturally constrained to [0,1]^3 with sum = 1.
"""

import torch
import torch.nn as nn
from torch.distributions import Dirichlet

from config import MIN_CONC, CONC_SCALE


class ActorCritic(nn.Module):
    """
    Combined actor-critic network with a Dirichlet policy head.

    Parameters
    ----------
    state_dim  : int — dimension of the observation vector (31 for default env)
    action_dim : int — number of groups to allocate to (default 3: X, Y, Z)
    """

    def __init__(self, state_dim: int, action_dim: int = 3):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 64),        nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        self.softplus = nn.Softplus()

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 64),        nn.Tanh(),
            nn.Linear(64, 1),
        )

    def dist(self, state: torch.Tensor) -> Dirichlet:
        """
        Compute the Dirichlet distribution for the given state.

        Concentration parameters are guaranteed positive via:
            conc = Softplus(actor(state)) * CONC_SCALE + MIN_CONC

        NaN/Inf are replaced with MIN_CONC for numerical safety.

        Parameters
        ----------
        state : Tensor of shape (..., state_dim)

        Returns
        -------
        Dirichlet distribution object
        """
        conc = self.softplus(self.actor(state)) * CONC_SCALE + MIN_CONC
        conc = torch.nan_to_num(conc, nan=MIN_CONC, posinf=10.0, neginf=MIN_CONC)
        return Dirichlet(conc)

    def act_from_old(
        self,
        state: torch.Tensor,
        policy_old: 'ActorCritic',
    ) -> tuple:
        """
        Sample an action using the old (frozen) policy distribution.

        Used during rollout collection so the stored log-probs are consistent
        with the old policy that will be used in the PPO importance ratio.

        Parameters
        ----------
        state      : Tensor of shape (state_dim,)
        policy_old : ActorCritic — the frozen snapshot policy

        Returns
        -------
        action : Tensor of shape (action_dim,), sums to 1
        logp   : scalar Tensor — log-probability under old policy
        """
        with torch.no_grad():
            d      = policy_old.dist(state)
            action = d.sample()
            action = torch.clamp(action, min=1e-6)
            action = action / action.sum()
            logp   = d.log_prob(action)
        return action, logp

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple:
        """
        Evaluate log-prob, value, and entropy under the current policy.

        Called during the PPO update step.

        Parameters
        ----------
        state  : Tensor of shape (batch, state_dim)
        action : Tensor of shape (batch, action_dim)

        Returns
        -------
        logp    : Tensor (batch,)
        value   : Tensor (batch, 1)
        entropy : Tensor (batch,)
        """
        d    = self.dist(state)
        logp = d.log_prob(action)
        v    = self.critic(state)
        ent  = d.entropy()
        return logp, v, ent

    def sample_with_temp(
        self,
        state: torch.Tensor,
        policy_old: 'ActorCritic',
        sample_temp: float = 2.0,
    ) -> tuple:
        """
        Sample with temperature scaling applied to the old policy's concentration.

        Dividing concentration by sample_temp > 1 flattens the Dirichlet,
        producing more exploratory (uniform-like) samples during warm-up.
        The log-prob is still computed under the un-tempered old distribution
        so the PPO importance ratio remains correct.

        Parameters
        ----------
        state       : Tensor of shape (state_dim,)
        policy_old  : ActorCritic — the frozen snapshot policy
        sample_temp : float > 1 → more exploration; = 1 → standard sampling

        Returns
        -------
        action : Tensor of shape (action_dim,), sums to 1
        logp   : scalar Tensor — log-prob under old (un-tempered) policy
        """
        with torch.no_grad():
            d_old        = policy_old.dist(state)
            conc_tempered = torch.clamp(d_old.concentration / sample_temp, min=MIN_CONC)
            d_temp        = Dirichlet(conc_tempered)
            action        = d_temp.sample()
            action        = torch.clamp(action, min=1e-6)
            action        = action / action.sum()
            logp          = d_old.log_prob(action)   # under original dist
        return action, logp
