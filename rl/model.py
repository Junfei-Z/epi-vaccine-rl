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
      - Global epidemic state (31-dim): 30 compartment fractions + day/T
    Total input dim: 37

    The critic takes only the 31-dim global state (shared across all nodes).

    Parameters
    ----------
    global_dim   : dimension of global state vector (default 31)
    node_feat_dim: dimension of per-node feature vector (default 6)
    hidden       : hidden layer width
    """

    NODE_FEAT_DIM  = 6    # degree_norm, inf_nbr_frac, gX, gY, gZ, day_norm
    GLOBAL_DIM     = 31   # 30 compartment fracs + day/T

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

        Uses Gumbel-Top-K during training (deterministic=False) for stochastic
        exploration, and greedy Top-K during evaluation (deterministic=True).

        Parameters
        ----------
        global_state  : Tensor (global_dim,)
        node_feats    : Tensor (n_susceptible, node_feat_dim)
        k             : number of nodes to select (V_MAX_DAILY)
        deterministic : if False, use Gumbel noise for exploration (training);
                        if True, use greedy top-k (evaluation)
        score_bias    : Tensor (n_susceptible,) or None — additive bias from
                        OC warm-start, applied to scores before selection but
                        NOT included in log-prob computation (so PPO ratios
                        stay correct as the bias decays)

        Returns
        -------
        indices  : LongTensor (min(k, n),) — selected positions in node_feats
        log_prob : scalar Tensor or None — sum of log-probs for PPO
                   (None when deterministic=True)
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

        # Gumbel-Top-K: add Gumbel noise for stochastic exploration
        gumbel = -torch.log(-torch.log(torch.rand_like(biased) + 1e-10) + 1e-10)
        perturbed = biased + gumbel
        indices = torch.topk(perturbed, k).indices

        # log-prob from unbiased scores for PPO importance ratio
        log_probs = torch.log_softmax(scores, dim=0)
        log_prob  = log_probs[indices].sum()

        return indices, log_prob

    def value(self, global_state: torch.Tensor) -> torch.Tensor:
        """Critic value estimate from global state."""
        return self.critic(global_state)


# ---------------------------------------------------------------------------
# Naive Node Policy (N independent Bernoulli + projection)
# ---------------------------------------------------------------------------

class NaiveNodePolicy(nn.Module):
    """
    Truly naive individual RL policy for vaccine allocation.

    Uses the SAME MLP architecture as NodeScoringPolicy, but treats each
    node's vaccination decision as an INDEPENDENT Bernoulli:

        p_i = sigmoid(score_i)
        a_i ~ Bernoulli(p_i)   for each node independently
        then project to exactly K nodes

    Action space = {0,1}^N with constraint sum=K → effectively 2^N choices.

    log pi(a|s) = sum_i [ a_i * log(p_i) + (1-a_i) * log(1-p_i) ]

    This sum has N terms, so gradient variance grows with N — the key
    trainability disadvantage vs Node RL's O(1)-variance top-K approach.

    Projection (non-differentiable):
    - Training: sample Bernoulli, then if count != K, sort by p_i and
      take top-K of the sampled set (or add highest-p unsampled nodes).
    - Evaluation: directly take top-K by p_i (deterministic mode).
    """

    NODE_FEAT_DIM = 6
    GLOBAL_DIM = 31

    def __init__(self, hidden: int = 64):
        super().__init__()
        in_dim = self.NODE_FEAT_DIM + self.GLOBAL_DIM

        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        self.critic = nn.Sequential(
            nn.Linear(self.GLOBAL_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),          nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def score(self, global_state, node_feats):
        n = node_feats.shape[0]
        g = global_state.unsqueeze(0).expand(n, -1)
        x = torch.cat([node_feats, g], dim=-1)
        return self.scorer(x).squeeze(-1)

    def select(self, global_state, node_feats, k,
               deterministic=False):
        """
        Select k nodes via independent Bernoulli sampling + projection.

        Training (deterministic=False):
            1. Compute p_i = sigmoid(score_i) for all N nodes
            2. Sample a_i ~ Bernoulli(p_i) independently
            3. Project to exactly K: if too many selected, keep K with
               highest p_i; if too few, add nodes with highest p_i
            4. log_prob = sum of Bernoulli log-probs for ALL N nodes
               (selected nodes contribute log(p_i), others log(1-p_i))

        Evaluation (deterministic=True):
            Take K nodes with highest sigmoid probability.
            log_prob = None.

        Returns
        -------
        indices  : LongTensor (k,)
        log_prob : scalar Tensor or None
        """
        n = node_feats.shape[0]
        k = min(k, n)
        if k == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor(0.0)

        scores = self.score(global_state, node_feats)  # (n,)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=10.0, neginf=-10.0)
        probs = torch.sigmoid(scores).clamp(1e-6, 1 - 1e-6)  # (n,)

        if deterministic:
            indices = torch.topk(probs, k).indices
            return indices, None

        # Independent Bernoulli sampling
        samples = torch.bernoulli(probs)  # (n,) of 0.0 / 1.0

        # Projection to exactly K
        selected_mask = samples.bool()
        n_selected = selected_mask.sum().item()

        if n_selected == k:
            indices = torch.where(selected_mask)[0]
        elif n_selected > k:
            # too many: keep K with highest probability among selected
            candidates = torch.where(selected_mask)[0]
            top_k_in_candidates = torch.topk(probs[candidates], k).indices
            indices = candidates[top_k_in_candidates]
        else:
            # too few: keep all selected + add highest-p unselected
            selected_ids = torch.where(selected_mask)[0].tolist()
            unselected = torch.where(~selected_mask)[0]
            need = k - n_selected
            top_unselected = torch.topk(probs[unselected], min(need, len(unselected))).indices
            extra_ids = unselected[top_unselected].tolist()
            indices = torch.tensor(selected_ids + extra_ids, dtype=torch.long)

        # Compute log-prob: sum over ALL N nodes (Bernoulli log-likelihood)
        # This is what makes gradient variance O(N)
        action_vec = torch.zeros(n)
        action_vec[indices] = 1.0
        log_prob = (action_vec * torch.log(probs + 1e-10)
                    + (1 - action_vec) * torch.log(1 - probs + 1e-10)).sum()

        return indices, log_prob

    def value(self, global_state):
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
