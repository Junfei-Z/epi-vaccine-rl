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

# ---------------------------------------------------------------------------
# Node Scoring Policy
# ---------------------------------------------------------------------------

class NodeScoringPolicy(nn.Module):
    """
    Node-level vaccine allocation policy.

    Instead of outputting group-level Dirichlet shares, this policy scores
    every susceptible node individually and selects the top V_MAX to vaccinate.

    Each node's input is a concatenation of:
      - Local node features  (6-dim): degree, infectious-neighbour fraction,
                                      group one-hot (3), normalised day
      - Global epidemic state (34-dim): 30 compartment fractions + day/T
                                        + 3 pressure features
    Total input dim: 40

    The critic takes only the 34-dim global state (shared across all nodes).

    Parameters
    ----------
    global_dim   : dimension of global state vector (default 34)
    node_feat_dim: dimension of per-node feature vector (default 6)
    hidden       : hidden layer width
    """

    NODE_FEAT_DIM  = 6    # degree_norm, inf_nbr_frac, gX, gY, gZ, day_norm
    GLOBAL_DIM     = 34   # 30 compartment fracs + day/T + 3 pressure features

    def __init__(self, hidden: int = 64):
        super().__init__()
        in_dim = self.NODE_FEAT_DIM + self.GLOBAL_DIM

        # Shared MLP scorer — same weights applied to every node
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        # Critic takes global state only
        self.critic = nn.Sequential(
            nn.Linear(self.GLOBAL_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),          nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def score(
        self,
        global_state: torch.Tensor,
        node_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a scalar score for each susceptible node.

        Parameters
        ----------
        global_state : Tensor (global_dim,)
        node_feats   : Tensor (n_susceptible, node_feat_dim)

        Returns
        -------
        scores : Tensor (n_susceptible,)
        """
        n = node_feats.shape[0]
        g = global_state.unsqueeze(0).expand(n, -1)   # (n, global_dim)
        x = torch.cat([node_feats, g], dim=-1)         # (n, 40)
        return self.scorer(x).squeeze(-1)              # (n,)

    def select(
        self,
        global_state: torch.Tensor,
        node_feats: torch.Tensor,
        k: int,
        deterministic: bool = False,
        score_bias: torch.Tensor = None,
    ) -> tuple:
        """
        Select k nodes to vaccinate and return their indices + log-prob.

        Training  (deterministic=False): sample k nodes via Gumbel-top-k so
          the selection is stochastic and differentiable.
        Evaluation (deterministic=True): take the top-k by raw score.

        Parameters
        ----------
        global_state  : Tensor (global_dim,)
        node_feats    : Tensor (n_susceptible, node_feat_dim)
        k             : number of nodes to select (V_MAX_DAILY)
        deterministic : if True, greedy top-k; else Gumbel-top-k sampling
        score_bias    : Tensor (n_susceptible,) or None — additive bias from
                        OC warm-start, applied to scores before selection but
                        NOT included in log-prob computation (so PPO ratios
                        stay correct as the bias decays)

        Returns
        -------
        indices  : LongTensor (min(k, n),) — selected positions in node_feats
        log_prob : scalar Tensor — sum of log-probs (None if deterministic)
        """
        n = node_feats.shape[0]
        k = min(k, n)
        if k == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor(0.0)

        scores = self.score(global_state, node_feats)   # (n,)

        # biased scores for selection; unbiased scores for log-prob
        biased = scores + score_bias if score_bias is not None else scores

        if deterministic:
            indices = torch.topk(biased, k).indices
            return indices, None

        # Gumbel-top-k: add Gumbel noise then take top-k
        gumbel = -torch.log(-torch.log(torch.rand_like(biased) + 1e-10) + 1e-10)
        perturbed = biased + gumbel
        indices   = torch.topk(perturbed, k).indices

        # log-prob uses unbiased scores so PPO ratios remain correct
        log_probs = torch.log_softmax(scores, dim=0)
        log_prob  = log_probs[indices].sum()

        return indices, log_prob

    def value(self, global_state: torch.Tensor) -> torch.Tensor:
        """Critic value estimate from global state."""
        return self.critic(global_state)


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
