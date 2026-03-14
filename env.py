# -*- coding: utf-8 -*-
"""
env.py — Node-level epidemic simulation environment.

EpidemicNodeEnv simulates disease spread and vaccination on a contact
network at the individual-node level. Each node carries one of 10
compartmental states (S, E, P, A, I, L, H, R, V, D).

The RL agent interacts via step(shares) where `shares` is a length-3
simplex vector giving the fraction of the daily vaccine budget to
allocate to groups X, Y, Z.
"""

import math
import numpy as np

from config import S, E, P, A, I, L, H, R, V, D
from graph import get_contact_matrix


# ---------------------------------------------------------------------------
# Core environment
# ---------------------------------------------------------------------------

class EpidemicNodeEnv:
    """
    Node-level stochastic/deterministic epidemic environment.

    Parameters
    ----------
    G               : networkx Graph
    node_to_group   : dict {node_id -> group_index (1,2,3)}
    group_nodes     : dict {group_index -> sorted list of node ids}
    deg_dict        : dict {node_id -> degree}
    contact_matrix  : np.ndarray (3,3) inter-group contact rates
    params_by_group : nested param dict (see make_env_from_graph)
    capacity_daily  : int, max vaccine doses per day (V_MAX)
    reward_scale    : float, multiplier on death penalty
    substeps        : int, disease-progression sub-steps per day
    dt              : float, time increment per substep
    rng_seed        : int, numpy RNG seed
    deterministic   : bool, use expected-value transitions instead of sampling
    """

    def __init__(
        self,
        G,
        node_to_group: dict,
        group_nodes: dict,
        deg_dict: dict,
        contact_matrix,
        params_by_group: dict,
        capacity_daily: int,
        reward_scale: float = 1.0,
        substeps: int = 1,
        dt: float = 1.0,
        rng_seed: int = 0,
        deterministic: bool = False,
    ):
        self.G            = G
        self.node_to_group = node_to_group
        self.group_nodes  = group_nodes
        self.deg          = deg_dict
        self.C            = np.array(contact_matrix, dtype=float)
        self.params       = params_by_group
        self.V_MAX        = capacity_daily
        self.V_MAX_DAILY  = capacity_daily      # alias used by external helpers
        self.reward_scale = float(reward_scale)
        self.substeps     = max(1, int(substeps))
        self.dt           = dt / self.substeps
        self.rng          = np.random.default_rng(rng_seed)
        self.N            = len(self.G.nodes())
        self.groups       = [1, 2, 3]
        self.Ng           = np.array(
            [len(group_nodes[1]), len(group_nodes[2]), len(group_nodes[3])],
            dtype=float,
        )
        self.day    = 0
        self.status = np.full(self.N, S, dtype=np.int32)

        self.deterministic = bool(deterministic)
        # fixed vaccination order within each group: high-degree nodes first
        self.fixed_order = {
            g: sorted(
                list(self.group_nodes[g]),
                key=lambda x: (self.deg[x], x),
                reverse=True,
            )
            for g in self.groups
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_k_in_state(self, cur: np.ndarray, group: int, state: int, k: int) -> list:
        """Return up to k node ids from `group` that are in `state`, highest degree first."""
        if k <= 0:
            return []
        picked = []
        for node in self.fixed_order[group]:
            if cur[node] == state:
                picked.append(node)
                if len(picked) >= k:
                    break
        return picked

    def _move_k(self, cur: np.ndarray, group: int, src: int, dst: int, k: int) -> int:
        """Move up to k nodes in `group` from state `src` to `dst`. Returns count moved."""
        picked = self._pick_k_in_state(cur, group, src, k)
        for node in picked:
            cur[node] = dst
        return len(picked)

    def _group_comp_counts(self) -> np.ndarray:
        """Return (3, 10) int array: compartment counts per group."""
        res = np.zeros((3, 10), dtype=np.int32)
        for g in self.groups:
            node_ids = self.group_nodes[g]
            vals     = self.status[node_ids]
            for state in range(10):
                res[g - 1, state] = np.sum(vals == state)
        return res

    def _obs(self) -> np.ndarray:
        """
        Observation: normalised compartment counts (30 values) + day (1 value).
        Shape: (31,)
        """
        comp = self._group_comp_counts().astype(np.float32)
        x    = comp.flatten() / float(self.N)
        return np.concatenate([x, np.array([self.day], dtype=np.float32)])

    def _project_doses(self, shares) -> np.ndarray:
        """
        Convert a simplex share vector to feasible integer dose counts.

        Respects (a) the daily capacity V_MAX and (b) available susceptibles
        per group. Uses Largest Remainder Method for the integer rounding.

        Parameters
        ----------
        shares : array-like of length 3, non-negative, need not sum to 1

        Returns
        -------
        base : np.ndarray of shape (3,), integer doses per group
        """
        shares = np.clip(np.array(shares, dtype=float), 0.0, 1.0)
        ssum   = shares.sum()
        if ssum <= 0:
            shares = np.array([1.0, 0.0, 0.0])
        shares = shares / shares.sum()

        avail = np.array(
            [np.sum(self.status[self.group_nodes[g]] == S) for g in self.groups],
            dtype=int,
        )

        raw  = shares * self.V_MAX
        raw  = np.minimum(raw, avail.astype(float))
        base = np.floor(raw).astype(int)

        rem = int(self.V_MAX - base.sum())
        if rem > 0:
            frac  = raw - base
            order = np.argsort(-frac)
            for gi in order:
                if rem == 0:
                    break
                room = avail[gi] - base[gi]
                if room > 0:
                    add      = min(rem, room)
                    base[gi] += add
                    rem      -= add
            # fallback: fill by share size if fractional pass didn't exhaust rem
            if rem > 0:
                for gi in np.argsort(-shares):
                    if rem == 0:
                        break
                    room = avail[gi] - base[gi]
                    if room > 0:
                        add      = min(rem, room)
                        base[gi] += add
                        rem      -= add

        return base.astype(int)

    def _choose_to_vaccinate(self, group: int, k: int) -> list:
        """Return up to k susceptible nodes from `group`, highest degree first."""
        if k <= 0:
            return []
        cand = [n for n in self.group_nodes[group] if self.status[n] == S]
        return sorted(cand, key=lambda x: self.deg[x], reverse=True)[:k]

    def _compute_lambdas(self) -> np.ndarray:
        """Compute force-of-infection λ for each group based on current state."""
        comp = self._group_comp_counts().astype(float)
        wA   = self.params['wA']; wP = self.params['wP']; wI = self.params['wI']
        Ic   = np.array(
            [wA*comp[i, A] + wP*comp[i, P] + wI*comp[i, I] for i in range(3)],
            dtype=float,
        )
        beta = np.array(
            [self.params[1]['beta'], self.params[2]['beta'], self.params[3]['beta']],
            dtype=float,
        )
        frac = Ic / np.maximum(self.Ng, 1.0)
        return np.array([beta[i] * np.dot(self.C[i], frac) for i in range(3)], dtype=float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed_counts: dict = None) -> np.ndarray:
        """
        Reset environment to all-susceptible state then seed infections.

        Parameters
        ----------
        seed_counts : dict {group_index -> int} or None
                      number of initially infected nodes per group

        Returns
        -------
        obs : initial observation array
        """
        self.status[:] = S
        self.day = 0
        if seed_counts is not None:
            for g in self.groups:
                k = int(seed_counts.get(g, 0))
                if k > 0:
                    cand = list(self.group_nodes[g])
                    sel  = self.rng.choice(cand, size=min(k, len(cand)), replace=False)
                    for node in sel:
                        self.status[node] = I
        return self._obs()

    def step(self, shares) -> tuple:
        """
        Execute one environment day: vaccinate then advance disease.

        Parameters
        ----------
        shares : array-like of length 3 — allocation fractions for X, Y, Z

        Returns
        -------
        obs     : np.ndarray, next observation
        reward  : float, -deaths_today * reward_scale - 0.01 * unused_doses
        done    : bool
        info    : dict with keys 'doses', 'details', 'mask', 'unused'
        """
        doses = self._project_doses(shares)

        # availability mask (which groups still have susceptibles)
        avail_counts = np.array(
            [np.sum(self.status[self.group_nodes[g]] == S) for g in self.groups],
            dtype=int,
        )
        mask = (avail_counts > 0).astype(int)

        day_start = self.status.copy()

        # --- vaccination ---
        details = []
        for gi, group in enumerate(self.groups):
            sel = self._choose_to_vaccinate(group, int(doses[gi]))
            for node in sel:
                nbrs          = list(self.G.neighbors(node))
                inf_nbr_count = sum(1 for u in nbrs if day_start[u] in (P, A, I))
                details.append({
                    'day':           int(self.day),
                    'id':            int(node),
                    'group':         int(group),
                    'degree':        int(self.deg[node]),
                    'inf_nbr_count': int(inf_nbr_count),
                })
                self.status[node] = V

        # --- disease progression (substeps) ---
        cur = self.status.copy()
        for _ in range(self.substeps):
            # recompute compartment counts & lambda at each substep
            comp = np.zeros((3, 10), dtype=float)
            for g in self.groups:
                node_ids = self.group_nodes[g]
                vals     = cur[node_ids]
                for state in range(10):
                    comp[g - 1, state] = np.sum(vals == state)

            wA = self.params['wA']; wP = self.params['wP']; wI = self.params['wI']
            Ic  = np.array(
                [wA*comp[i, A] + wP*comp[i, P] + wI*comp[i, I] for i in range(3)],
                dtype=float,
            )
            beta = np.array(
                [self.params[1]['beta'], self.params[2]['beta'], self.params[3]['beta']],
                dtype=float,
            )
            frac = Ic / np.maximum(self.Ng, 1.0)
            lam  = np.array(
                [beta[i] * np.dot(self.C[i], frac) for i in range(3)],
                dtype=float,
            )

            next_state = cur.copy()

            if not self.deterministic:
                # stochastic: individual Bernoulli transitions
                for g in self.groups:
                    lam_g = lam[g - 1]
                    eps   = self.params[g].get('epsilon', 0.5)
                    tauE  = self.params[g]['tauE']; tauP = self.params[g]['tauP']
                    tauA  = self.params[g]['tauA']; tauI = self.params[g]['tauI']
                    tauL  = self.params[g]['tauL']; tauH = self.params[g]['tauH']
                    sprob = self.params[g]['s']
                    pprob = self.params[g]['p']
                    dprob = self.params[g]['d']

                    for node in self.group_nodes[g]:
                        state = cur[node]
                        if state == S:
                            if self.rng.random() < 1.0 - math.exp(-lam_g * self.dt):
                                next_state[node] = E
                        elif state == V:
                            if self.rng.random() < 1.0 - math.exp(-lam_g * (1.0 - eps) * self.dt):
                                next_state[node] = E
                        elif state == E:
                            if self.rng.random() < 1.0 - math.exp(-tauE * self.dt):
                                next_state[node] = P
                        elif state == P:
                            if self.rng.random() < 1.0 - math.exp(-tauP * self.dt):
                                next_state[node] = I if self.rng.random() < sprob else A
                        elif state == A:
                            if self.rng.random() < 1.0 - math.exp(-tauA * self.dt):
                                next_state[node] = R
                        elif state == I:
                            if self.rng.random() < 1.0 - math.exp(-tauI * self.dt):
                                next_state[node] = L
                        elif state == L:
                            if self.rng.random() < 1.0 - math.exp(-tauL * self.dt):
                                next_state[node] = H if self.rng.random() < pprob else R
                        elif state == H:
                            if self.rng.random() < 1.0 - math.exp(-tauH * self.dt):
                                next_state[node] = D if self.rng.random() < dprob else R
            else:
                # deterministic: move expected number of nodes per transition
                for g in self.groups:
                    lam_g = lam[g - 1]
                    eps   = self.params[g].get('epsilon', 0.5)
                    tauE  = self.params[g]['tauE']; tauP = self.params[g]['tauP']
                    tauA  = self.params[g]['tauA']; tauI = self.params[g]['tauI']
                    tauL  = self.params[g]['tauL']; tauH = self.params[g]['tauH']
                    sprob = self.params[g]['s']
                    pprob = self.params[g]['p']
                    dprob = self.params[g]['d']

                    node_ids = self.group_nodes[g]
                    cnt = {st: int(np.sum(cur[node_ids] == st)) for st in range(10)}

                    k_SE = int(round(cnt[S] * (1.0 - math.exp(-lam_g * self.dt))))
                    self._move_k(next_state, g, S, E, k_SE)

                    k_VE = int(round(cnt[V] * (1.0 - math.exp(-lam_g * (1.0 - eps) * self.dt))))
                    self._move_k(next_state, g, V, E, k_VE)

                    k_EP = int(round(cnt[E] * (1.0 - math.exp(-tauE * self.dt))))
                    self._move_k(next_state, g, E, P, k_EP)

                    k_Pexit = int(round(cnt[P] * (1.0 - math.exp(-tauP * self.dt))))
                    k_PI    = int(round(k_Pexit * sprob))
                    self._move_k(next_state, g, P, I, k_PI)
                    self._move_k(next_state, g, P, A, k_Pexit - k_PI)

                    k_AR = int(round(cnt[A] * (1.0 - math.exp(-tauA * self.dt))))
                    self._move_k(next_state, g, A, R, k_AR)

                    k_IL = int(round(cnt[I] * (1.0 - math.exp(-tauI * self.dt))))
                    self._move_k(next_state, g, I, L, k_IL)

                    k_Lexit = int(round(cnt[L] * (1.0 - math.exp(-tauL * self.dt))))
                    k_LH    = int(round(k_Lexit * pprob))
                    self._move_k(next_state, g, L, H, k_LH)
                    self._move_k(next_state, g, L, R, k_Lexit - k_LH)

                    k_Hexit = int(round(cnt[H] * (1.0 - math.exp(-tauH * self.dt))))
                    k_HD    = int(round(k_Hexit * dprob))
                    self._move_k(next_state, g, H, D, k_HD)
                    self._move_k(next_state, g, H, R, k_Hexit - k_HD)

            cur = next_state

        self.status = cur

        # --- reward ---
        prev_deaths  = int(np.sum(day_start == D))
        new_deaths   = int(np.sum(self.status == D))
        deaths_today = new_deaths - prev_deaths
        unused       = max(0, int(self.V_MAX_DAILY) - int(doses.sum()))
        reward       = -float(deaths_today) * self.reward_scale - 0.01 * float(unused)

        self.day += 1
        done = self.day >= int(self.params.get('T_HORIZON', self.day))

        return self._obs(), reward, done, {
            'doses':   doses,
            'details': details,
            'mask':    mask,
            'unused':  unused,
        }


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_env_from_graph(
    G,
    groups: dict,
    deg_dict: dict,
    params_global: dict,
    capacity_daily: int,
    reward_scale: float = 1.0,
    seed_counts: dict = None,
    substeps: int = 1,
    dt: float = 1.0,
    deterministic: bool = False,
) -> tuple:
    """
    Build an EpidemicNodeEnv from a graph and a flat params_global dict.

    Constructs the group-structured params_by_group dict expected by
    EpidemicNodeEnv, calls reset(), and returns the initial observation.

    Returns
    -------
    env, obs, C, node_to_group, group_nodes
    """
    C, N_g, node_to_group = get_contact_matrix(G, groups)

    group_nodes = {
        1: sorted(list(groups['X'])),
        2: sorted(list(groups['Y'])),
        3: sorted(list(groups['Z'])),
    }

    params_by_group = {
        'wA': params_global['wA'],
        'wP': params_global['wP'],
        'wI': params_global['wI'],
        1: {
            'beta': params_global['beta'],
            'tauE': params_global['tauE'], 'tauP': params_global['tauP'],
            'tauA': params_global['tauA'], 'tauI': params_global['tauI'],
            'tauL': params_global['tauL'], 'tauH': params_global['tauH'],
            's': params_global['sX'], 'p': params_global['pX'], 'd': params_global['dX'],
            'epsilon': params_global.get('epsilonX', 0.5),
        },
        2: {
            'beta': params_global['beta'],
            'tauE': params_global['tauE'], 'tauP': params_global['tauP'],
            'tauA': params_global['tauA'], 'tauI': params_global['tauI'],
            'tauL': params_global['tauL'], 'tauH': params_global['tauH'],
            's': params_global['sY'], 'p': params_global['pY'], 'd': params_global['dY'],
            'epsilon': params_global.get('epsilonY', 0.5),
        },
        3: {
            'beta': params_global['beta'],
            'tauE': params_global['tauE'], 'tauP': params_global['tauP'],
            'tauA': params_global['tauA'], 'tauI': params_global['tauI'],
            'tauL': params_global['tauL'], 'tauH': params_global['tauH'],
            's': params_global['sZ'], 'p': params_global['pZ'], 'd': params_global['dZ'],
            'epsilon': params_global.get('epsilonZ', 0.5),
        },
        'T_HORIZON': params_global['T_HORIZON'],
    }

    env = EpidemicNodeEnv(
        G, node_to_group, group_nodes, deg_dict, C, params_by_group,
        capacity_daily, reward_scale=reward_scale,
        substeps=substeps, dt=dt, deterministic=deterministic,
    )

    default_seeds = {
        1: params_global.get('INIT_INFECTED_X', 0),
        2: params_global.get('INIT_INFECTED_Y', 0),
        3: params_global.get('INIT_INFECTED_Z', 0),
    }
    obs = env.reset(seed_counts=seed_counts or default_seeds)
    return env, obs, C, node_to_group, group_nodes


def build_env(args: tuple, deterministic: bool = True) -> tuple:
    """
    Convenience wrapper for experiments: unpack args tuple and build env.

    Parameters
    ----------
    args : (G, groups, deg_dict, params_global, capacity_daily, seed_counts)

    Returns
    -------
    env, obs, C
    """
    G, groups, deg, params_global, capacity, seed_counts = args
    env, obs, C, _, _ = make_env_from_graph(
        G, groups, deg, params_global, capacity,
        reward_scale=1.0, seed_counts=seed_counts,
        substeps=1, dt=1.0, deterministic=deterministic,
    )
    return env, obs, C
