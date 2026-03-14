# -*- coding: utf-8 -*-
"""
simulate.py — ODE-guided simulation and RL policy evaluation.

Contains three layers of simulation:

1. Low-level primitives
   vaccinate_by_priority  — apply integer doses to env, record node details
   progress_one_day       — deterministic disease-progression for one day

2. Episode-level runners
   day0_report            — print day-0 infection diagnostics
   simulate_episode       — run one episode with a fixed share sequence
   simulate_with_ode_doses — run full ODE-guided episode with priority allocation

3. RL evaluation
   evaluate_and_export    — evaluate a trained PPO agent and export CSVs
"""

import math
import numpy as np
import pandas as pd
import torch

from config import S, E, P, A, I, L, H, R, V, D
from allocation import allocate_by_priority
from env import build_env, make_env_from_graph


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _capacity(env) -> int:
    """Return the environment's daily vaccine capacity."""
    return int(getattr(env, 'V_MAX_DAILY', getattr(env, 'V_MAX', 0)))


# ---------------------------------------------------------------------------
# 1. Low-level primitives
# ---------------------------------------------------------------------------

def vaccinate_by_priority(env, doses: np.ndarray) -> list:
    """
    Apply integer dose counts to the environment, highest-degree nodes first.

    Records per-node vaccination details for later analysis.

    Parameters
    ----------
    env   : EpidemicNodeEnv (status will be mutated)
    doses : np.ndarray of shape (3,) — integer doses for groups X, Y, Z

    Returns
    -------
    details : list of dicts with keys day, id, group, degree, inf_nbr_count
    """
    details   = []
    day_start = env.status.copy()

    for gi, g in enumerate([1, 2, 3]):
        k = int(doses[gi])
        if k <= 0:
            continue
        sel = env._choose_to_vaccinate(g, k)
        for node in sel:
            nbrs          = list(env.G.neighbors(node))
            inf_nbr_count = sum(1 for u in nbrs if day_start[u] in (P, A, I))
            details.append({
                'day':           int(env.day),
                'id':            int(node),
                'group':         int(g),
                'degree':        int(env.deg[node]),
                'inf_nbr_count': int(inf_nbr_count),
            })
            env.status[node] = V

    return details


def progress_one_day(env) -> int:
    """
    Advance disease state by one day using deterministic (expected-value) transitions.

    Operates on env.status in-place. Used by the ODE-guided simulator to keep
    disease dynamics consistent with the ODE model rather than stochastic sampling.

    Parameters
    ----------
    env : EpidemicNodeEnv

    Returns
    -------
    deaths_today : int — new deaths recorded this day
    """
    cur = env.status.copy()

    for _ in range(env.substeps):
        comp = np.zeros((3, 10), dtype=float)
        for g in env.groups:
            node_ids = env.group_nodes[g]
            vals     = cur[node_ids]
            for state in range(10):
                comp[g - 1, state] = np.sum(vals == state)

        wA = env.params['wA']; wP = env.params['wP']; wI = env.params['wI']
        Ic  = np.array(
            [wA*comp[i, A] + wP*comp[i, P] + wI*comp[i, I] for i in range(3)],
            dtype=float,
        )
        beta = np.array(
            [env.params[1]['beta'], env.params[2]['beta'], env.params[3]['beta']],
            dtype=float,
        )
        frac = Ic / np.maximum(env.Ng, 1.0)
        lam  = np.array(
            [beta[i] * np.dot(env.C[i], frac) for i in range(3)],
            dtype=float,
        )

        next_state = cur.copy()

        for g in env.groups:
            lam_g = lam[g - 1]
            eps   = env.params[g].get('epsilon', 0.5)
            tauE  = env.params[g]['tauE']; tauP = env.params[g]['tauP']
            tauA  = env.params[g]['tauA']; tauI = env.params[g]['tauI']
            tauL  = env.params[g]['tauL']; tauH = env.params[g]['tauH']
            sprob = env.params[g]['s']
            pprob = env.params[g]['p']
            dprob = env.params[g]['d']

            node_ids = env.group_nodes[g]
            cnt = {st: int(np.sum(cur[node_ids] == st)) for st in range(10)}

            k_SE = int(round(cnt[S] * (1.0 - math.exp(-lam_g * env.dt))))
            env._move_k(next_state, g, S, E, k_SE)

            k_VE = int(round(cnt[V] * (1.0 - math.exp(-lam_g * (1.0 - eps) * env.dt))))
            env._move_k(next_state, g, V, E, k_VE)

            k_EP = int(round(cnt[E] * (1.0 - math.exp(-tauE * env.dt))))
            env._move_k(next_state, g, E, P, k_EP)

            k_Pexit = int(round(cnt[P] * (1.0 - math.exp(-tauP * env.dt))))
            k_PI    = int(round(k_Pexit * sprob))
            env._move_k(next_state, g, P, I, k_PI)
            env._move_k(next_state, g, P, A, k_Pexit - k_PI)

            k_AR = int(round(cnt[A] * (1.0 - math.exp(-tauA * env.dt))))
            env._move_k(next_state, g, A, R, k_AR)

            k_IL = int(round(cnt[I] * (1.0 - math.exp(-tauI * env.dt))))
            env._move_k(next_state, g, I, L, k_IL)

            k_Lexit = int(round(cnt[L] * (1.0 - math.exp(-tauL * env.dt))))
            k_LH    = int(round(k_Lexit * pprob))
            env._move_k(next_state, g, L, H, k_LH)
            env._move_k(next_state, g, L, R, k_Lexit - k_LH)

            k_Hexit = int(round(cnt[H] * (1.0 - math.exp(-tauH * env.dt))))
            k_HD    = int(round(k_Hexit * dprob))
            env._move_k(next_state, g, H, D, k_HD)
            env._move_k(next_state, g, H, R, k_Hexit - k_HD)

        cur = next_state

    prev_deaths  = int(np.sum(env.status == D))
    env.status   = cur
    new_deaths   = int(np.sum(env.status == D))
    return new_deaths - prev_deaths


# ---------------------------------------------------------------------------
# 2. Episode-level runners
# ---------------------------------------------------------------------------

def day0_report(name: str, args: tuple, shares_day0) -> None:
    """
    Print day-0 infection diagnostics: group sizes, force of infection,
    expected exposures, and projected doses.

    Parameters
    ----------
    name        : scenario label for console output
    args        : env args tuple
    shares_day0 : length-3 array — day-0 allocation shares
    """
    env, _, C = build_env(args, deterministic=True)
    comp0      = env._group_comp_counts()
    pg         = args[3]  # params_global

    wA = pg['wA']; wP = pg['wP']; wI = pg['wI']
    Ic   = np.array(
        [wA*comp0[i, A] + wP*comp0[i, P] + wI*comp0[i, I] for i in range(3)],
        dtype=float,
    )
    beta = np.array([pg['beta']] * 3, dtype=float)
    frac = Ic / np.maximum(env.Ng, 1.0)
    lam  = np.array([beta[i] * np.dot(C[i], frac) for i in range(3)], dtype=float)

    S0  = comp0[:, S].astype(float)
    exp = S0 * (1.0 - np.exp(-lam))   # expected exposures on day 0
    doses = env._project_doses(np.asarray(shares_day0, dtype=float))

    print(f"\n[{name}]")
    print("Group sizes (X,Y,Z):", env.Ng)
    print("lambda (X,Y,Z):", lam)
    print("day0 expected exposures (X,Y,Z):", np.round(exp, 3))
    print("day0 doses (X,Y,Z):", doses, "  sum:", doses.sum(), "  capacity:", env.V_MAX)


def simulate_episode(name: str, args: tuple, shares_seq: np.ndarray) -> int:
    """
    Run one deterministic episode with a fixed per-day share sequence.

    Parameters
    ----------
    name       : label for console output
    args       : env args tuple
    shares_seq : np.ndarray of shape (T, 3) — allocation shares per day

    Returns
    -------
    final_deaths : int
    """
    env, _, _ = build_env(args, deterministic=True)
    done = False
    while not done:
        t      = env.day
        shares = shares_seq[t] if t < len(shares_seq) else shares_seq[-1]
        _, _, done, _ = env.step(shares)

    final_deaths = int(np.sum(env.status == D))
    print(f"[{name}] episode final deaths: {final_deaths}")
    return final_deaths


def simulate_with_ode_doses(
    env,
    doses_seq: np.ndarray,
    priority_order,
    seed_counts: dict,
) -> tuple:
    """
    Run a full ODE-guided episode: vaccinate by priority then advance disease.

    The environment is reset deterministically before starting. Doses may be
    provided as integer counts or as simplex shares (detected automatically).

    Parameters
    ----------
    env            : EpidemicNodeEnv (will be reset inside)
    doses_seq      : np.ndarray of shape (T, 3) — ODE integer doses or shares
    priority_order : list [g1,g2,g3] or callable(t) -> list
    seed_counts    : dict {group_index -> initial infected count}

    Returns
    -------
    df_nodes     : pd.DataFrame — per-node vaccination records
    df_days      : pd.DataFrame — per-day allocation summary
    final_deaths : int
    """
    env.deterministic = True
    env.reset(seed_counts=seed_counts)

    # day-0 compartment diagnostics
    comp0 = env._group_comp_counts().astype(int)
    print('day0 I/P/A:',
          [int(comp0[i, I]) for i in range(3)],
          [int(comp0[i, P]) for i in range(3)],
          [int(comp0[i, A]) for i in range(3)])

    df_nodes_rows, df_days_rows = [], []
    T = len(doses_seq)

    for t in range(T):
        cap   = _capacity(env)
        avail = np.array(
            [np.sum(env.status[env.group_nodes[g]] == S) for g in [1, 2, 3]],
            dtype=int,
        )
        req = doses_seq[t]
        # auto-detect shares input (row sums to ~1)
        if np.isclose(np.sum(req), 1.0, atol=1e-6):
            req = np.floor(np.array(req, dtype=float) * cap).astype(int)

        order       = priority_order(t) if callable(priority_order) else priority_order
        final_doses = allocate_by_priority(req, avail, cap, order)
        details     = vaccinate_by_priority(env, final_doses)
        deaths_today = progress_one_day(env)
        unused      = max(0, cap - int(final_doses.sum()))

        df_days_rows.append({
            'day':          int(env.day),
            'X':            int(final_doses[0]),
            'Y':            int(final_doses[1]),
            'Z':            int(final_doses[2]),
            'unused':       int(unused),
            'V_MAX_DAILY':  int(cap),
            'deaths_today': int(deaths_today),
        })
        df_nodes_rows.extend(details)
        env.day += 1

        if env.day >= int(env.params.get('T_HORIZON', T)):
            break

    df_nodes     = pd.DataFrame(df_nodes_rows)
    df_days      = pd.DataFrame(df_days_rows)
    final_deaths = int(np.sum(env.status == D))
    return df_nodes, df_days, final_deaths


# ---------------------------------------------------------------------------
# 3. RL evaluation and export
# ---------------------------------------------------------------------------

def evaluate_and_export(
    agent,
    G,
    groups: dict,
    deg_dict: dict,
    params_global: dict,
    capacity_daily: int,
    label: str,
    seed_counts: dict = None,
    substeps: int = 1,
    dt: float = 1.0,
    sample_action: bool = False,
    deterministic: bool = True,
    out_dir: str = '.',
) -> tuple:
    """
    Evaluate a trained PPO agent for one episode and export results to CSV.

    Parameters
    ----------
    agent          : PPO instance with a .policy (ActorCritic) attribute
    G, groups, deg_dict, params_global, capacity_daily : env spec
    label          : tag appended to output file names
    seed_counts    : initial infection counts per group
    sample_action  : if True sample from Dirichlet; if False use distribution mean
    deterministic  : use deterministic disease transitions
    out_dir        : directory for output CSV files (default: current dir)

    Returns
    -------
    df_nodes     : pd.DataFrame — per-node vaccination records
    df_days      : pd.DataFrame — per-day allocation summary
    final_deaths : int
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    env, _, _, _, _ = make_env_from_graph(
        G, groups, deg_dict, params_global, capacity_daily,
        reward_scale=1.0, seed_counts=seed_counts,
        substeps=substeps, dt=dt, deterministic=deterministic,
    )
    agent.policy.eval()
    obs  = env.reset(seed_counts=seed_counts)
    done = False
    rows_nodes = []
    rows_days  = []

    while not done:
        s_t = torch.from_numpy(obs).float()
        with torch.no_grad():
            dist = agent.policy.dist(s_t)
            act  = dist.sample() if sample_action else dist.mean
            act  = torch.clamp(act, min=1e-6)
            shares = (act / act.sum()).detach().cpu().numpy()

        obs, _, done, info = env.step(shares)

        for detail in info.get('details', []):
            rows_nodes.append({
                'day':           detail['day'],
                'node_id':       detail['id'],
                'group':         detail['group'],
                'degree':        detail['degree'],
                'inf_nbr_count': detail['inf_nbr_count'],
            })

        doses = info.get('doses', np.zeros(3, dtype=int))
        rows_days.append({
            'day':         env.day - 1,
            'X':           int(doses[0]),
            'Y':           int(doses[1]),
            'Z':           int(doses[2]),
            'unused':      int(info.get('unused', 0)),
            'V_MAX_DAILY': int(capacity_daily),
        })

    agent.policy.train()

    df_nodes = pd.DataFrame(rows_nodes)
    df_days  = pd.DataFrame(rows_days)
    if not df_nodes.empty:
        df_nodes = df_nodes[df_nodes['degree'] >= 0]

    df_nodes.to_csv(os.path.join(out_dir, f'rl_daily_vaccinations_nodes_{label}.csv'), index=False)
    df_days.to_csv(os.path.join(out_dir,  f'rl_daily_vaccinations_days_{label}.csv'),  index=False)

    final_deaths = int(np.sum(env.status == D))
    return df_nodes, df_days, final_deaths
