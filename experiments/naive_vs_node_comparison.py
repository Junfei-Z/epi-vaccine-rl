# -*- coding: utf-8 -*-
"""
experiments/naive_vs_node_comparison.py

Comprehensive comparison: Naive RL vs Node RL (with terminal reward) vs OC-Guided.

Sweeps:
  A. Severity  — pY/dY scaled together (4 levels)
  B. Beta      — transmissibility (6 values)
  C. V_MAX     — daily vaccine budget (5 values)
  D. Network   — BA, ER, WS, Regular
  E. Discount  — gamma = 0.8, 0.9, 0.95, 0.99, 1.0
  F. Horizon   — T = 30, 60

Outputs per run:
  - Death counts (mean ± std over n_eval stochastic episodes)
  - Training runtime (wall-clock seconds)
  - Vaccination plans: per-day node IDs, group, degree (saved as JSON)
"""

import os
import sys
import copy
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch

from config import PARAMS_NODE_RL, to_params_global, D, S
from graph import (build_graph_and_groups, build_graph_er,
                   build_graph_ws, build_graph_regular)
from ode_solver import solve, allocations_from_solution
from allocation import strict_priority_window_fill, cap_to_capacity
from env import make_env_from_graph
from rl.train import run_training_node_rl, run_training_naive_rl


# =====================================================================
# Evaluation helpers with vaccination plan logging
# =====================================================================

def _make_stochastic_env(G, groups, deg_dict, params_global,
                         capacity, seed_counts, rng_seed):
    env, _, _, _, _ = make_env_from_graph(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        seed_counts=seed_counts, deterministic=False,
    )
    env.rng = np.random.default_rng(rng_seed)
    env.reset(seed_counts=seed_counts)
    return env


def _node_group(env, node_id):
    """Return group name ('X', 'Y', or 'Z') for a node."""
    for g in env.groups:
        if node_id in env.group_nodes[g]:
            return g
    return '?'


def eval_with_plan(policy, policy_type, G, groups, deg_dict, params_global,
                   capacity, seed_counts, n_eval=10):
    """
    Evaluate a policy and record vaccination plans.

    Parameters
    ----------
    policy      : NaiveNodePolicy or NodeScoringPolicy
    policy_type : 'naive' or 'node'
    ...

    Returns
    -------
    deaths : list of int
    plans  : list of dicts, one per eval episode. Each dict has:
             {'deaths': int, 'days': [{day, node_ids, groups, degrees}, ...]}
    """
    policy.eval()
    deaths = []
    plans = []

    for i in range(n_eval):
        env = _make_stochastic_env(G, groups, deg_dict, params_global,
                                   capacity, seed_counts, rng_seed=2000 + i)
        done = False
        day_records = []

        while not done:
            g_state = torch.from_numpy(env.obs_with_pressure()).float()
            s_ids, feats = env.node_features()

            if len(s_ids) == 0:
                _, _, done, _ = env.step_node_ids([])
                day_records.append({
                    'day': env.day - 1,
                    'node_ids': [],
                    'groups': [],
                    'degrees': [],
                })
                continue

            f_t = torch.from_numpy(feats).float()
            with torch.no_grad():
                idxs, _ = policy.select(g_state, f_t, capacity,
                                        deterministic=True)
            selected = [s_ids[j] for j in idxs.tolist()]

            # record plan details
            sel_groups = [_node_group(env, nid) for nid in selected]
            sel_degrees = [env.deg.get(nid, 0) for nid in selected]
            current_day = env.day

            _, _, done, _ = env.step_node_ids(selected)

            day_records.append({
                'day': current_day,
                'node_ids': selected,
                'groups': sel_groups,
                'degrees': sel_degrees,
            })

        d = int(np.sum(env.status == D))
        deaths.append(d)
        plans.append({'deaths': d, 'seed': 2000 + i, 'days': day_records})

    policy.train()
    return deaths, plans


def eval_oc_with_plan(G, groups, deg_dict, params_global, capacity,
                      seed_counts, doses_seq, env_class_groups, n_eval=10):
    """
    Evaluate OC-Guided and record which nodes get vaccinated.
    """
    deaths = []
    plans = []
    T = len(doses_seq)

    for i in range(n_eval):
        env = _make_stochastic_env(G, groups, deg_dict, params_global,
                                   capacity, seed_counts, rng_seed=2000 + i)
        day_records = []
        for t in range(T):
            total = max(1, int(doses_seq[t].sum()))
            shares = doses_seq[t].astype(float) / total

            # record which nodes will be vaccinated (before step)
            # env.step internally calls _pick_k_in_state for each group
            # We need to peek at who gets vaccinated
            current_day = env.day

            _, _, done, info = env.step(shares)

            # after step, we can infer vaccinated nodes from status change
            # but simpler: record the group-level allocation
            day_records.append({
                'day': current_day,
                'doses_X': int(round(shares[0] * capacity)),
                'doses_Y': int(round(shares[1] * capacity)),
                'doses_Z': int(round(shares[2] * capacity)),
            })

            if done:
                break

        d = int(np.sum(env.status == D))
        deaths.append(d)
        plans.append({'deaths': d, 'seed': 2000 + i, 'days': day_records})

    return deaths, plans


# =====================================================================
# Run one comparison (3 methods for one param setting)
# =====================================================================

def run_comparison(label, params, G, groups, deg_dict, out_dir,
                   gamma=0.99, terminal_reward_scale=1.0,
                   n_eval=10, max_ep=300):
    """
    Run Naive RL, Node RL, and OC-Guided for one parameter setting.

    Returns list of result dicts with deaths, runtime, and saves plans.
    """
    os.makedirs(out_dir, exist_ok=True)
    capacity    = params['V_MAX_DAILY']
    seed_counts = {
        1: int(params.get('INIT_INFECTED_X', params['INITIAL_INFECTED'])),
        2: int(params.get('INIT_INFECTED_Y', 0)),
        3: int(params.get('INIT_INFECTED_Z', 0)),
    }
    params_global = to_params_global(params)
    results = []

    # --- 1. OC-Guided ---
    print(f"\n  [{label}] Solving ODE ...")
    t0 = time.time()
    states_opt, ctrl_opt, _ = solve(params, init_pattern='hcp')
    ax, ay, az = allocations_from_solution(states_opt, ctrl_opt)
    ax, ay, az = strict_priority_window_fill(ax, ay, az, capacity, priority='Z')
    ax, ay, az = cap_to_capacity(ax, ay, az, capacity)
    doses_seq = np.stack([ax, ay, az], axis=1)
    oc_time = time.time() - t0

    oc_deaths, oc_plans = eval_oc_with_plan(
        G, groups, deg_dict, params_global, capacity,
        seed_counts, doses_seq, groups, n_eval)

    results.append({
        'setting': label, 'method': 'OC-Guided',
        'deaths_mean': round(np.mean(oc_deaths), 1),
        'deaths_std': round(np.std(oc_deaths), 1),
        'runtime_sec': round(oc_time, 1),
        'gamma': '-',
    })
    with open(os.path.join(out_dir, f'plan_oc_{label}.json'), 'w') as f:
        json.dump(oc_plans, f)
    print(f"  [{label}] OC-Guided: {np.mean(oc_deaths):.1f} ± {np.std(oc_deaths):.1f}"
          f"  ({oc_time:.1f}s)")

    # --- 2. Node RL (with terminal reward) ---
    print(f"  [{label}] Training Node RL (gamma={gamma}) ...")
    t0 = time.time()
    node_policy, node_hist = run_training_node_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=max_ep, seed_counts=seed_counts,
        gamma=gamma,
        terminal_reward_scale=terminal_reward_scale,
        label=f'node_{label}', out_dir=out_dir,
    )
    node_time = time.time() - t0

    node_deaths, node_plans = eval_with_plan(
        node_policy, 'node', G, groups, deg_dict, params_global,
        capacity, seed_counts, n_eval)

    results.append({
        'setting': label, 'method': 'Node RL',
        'deaths_mean': round(np.mean(node_deaths), 1),
        'deaths_std': round(np.std(node_deaths), 1),
        'runtime_sec': round(node_time, 1),
        'gamma': gamma,
    })
    with open(os.path.join(out_dir, f'plan_node_{label}.json'), 'w') as f:
        json.dump(node_plans, f)
    print(f"  [{label}] Node RL: {np.mean(node_deaths):.1f} ± {np.std(node_deaths):.1f}"
          f"  ({node_time:.1f}s)")

    # --- 3. Naive RL ---
    print(f"  [{label}] Training Naive RL (gamma={gamma}) ...")
    t0 = time.time()
    naive_policy, naive_hist = run_training_naive_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=max_ep, seed_counts=seed_counts,
        gamma=gamma,
        terminal_reward_scale=terminal_reward_scale,
        label=f'naive_{label}', out_dir=out_dir,
    )
    naive_time = time.time() - t0

    naive_deaths, naive_plans = eval_with_plan(
        naive_policy, 'naive', G, groups, deg_dict, params_global,
        capacity, seed_counts, n_eval)

    results.append({
        'setting': label, 'method': 'Naive RL',
        'deaths_mean': round(np.mean(naive_deaths), 1),
        'deaths_std': round(np.std(naive_deaths), 1),
        'runtime_sec': round(naive_time, 1),
        'gamma': gamma,
    })
    with open(os.path.join(out_dir, f'plan_naive_{label}.json'), 'w') as f:
        json.dump(naive_plans, f)
    print(f"  [{label}] Naive RL: {np.mean(naive_deaths):.1f} ± {np.std(naive_deaths):.1f}"
          f"  ({naive_time:.1f}s)")

    return results


# =====================================================================
# Sweep definitions
# =====================================================================

BASE_OUT = 'results/naive_vs_node'


def _build_default_graph(params):
    return build_graph_and_groups(
        n=params['N'], m=params['BA_M'], seed=params['SEED'],
        high_risk_prob=params['HIGH_RISK_PROB'],
        alpha_std=params['ALPHA_STD'],
    )


def sweep_severity(n_eval=10, max_ep=300):
    """Sweep A: severity (pY, dY scaled together)."""
    levels = [
        {'pY': 0.20, 'dY': 0.27, 'label': 'baseline'},
        {'pY': 0.30, 'dY': 0.40, 'label': 'moderate'},
        {'pY': 0.40, 'dY': 0.50, 'label': 'severe'},
        {'pY': 0.50, 'dY': 0.65, 'label': 'critical'},
    ]
    params_base = dict(PARAMS_NODE_RL)
    G, groups, deg_dict = _build_default_graph(params_base)
    all_results = []

    for sev in levels:
        params = dict(params_base)
        params['pY'] = sev['pY']
        params['dY'] = sev['dY']
        label = f"severity_{sev['label']}"
        out = os.path.join(BASE_OUT, 'severity', sev['label'])
        print(f"\n{'#'*62}")
        print(f"# SEVERITY: {sev['label']} (pY={sev['pY']}, dY={sev['dY']})")
        print(f"{'#'*62}")
        r = run_comparison(label, params, G, groups, deg_dict, out,
                           n_eval=n_eval, max_ep=max_ep)
        all_results.extend(r)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(BASE_OUT, 'severity', 'results.csv'), index=False)
    print(f"\n{'='*62}\nSEVERITY RESULTS\n{'='*62}")
    print(df.to_string(index=False))
    return df


def sweep_beta(n_eval=10, max_ep=300):
    """Sweep B: transmissibility."""
    betas = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    params_base = dict(PARAMS_NODE_RL)
    G, groups, deg_dict = _build_default_graph(params_base)
    all_results = []

    for beta in betas:
        params = dict(params_base)
        params['beta'] = beta
        label = f"beta_{beta:.2f}"
        out = os.path.join(BASE_OUT, 'beta', f'beta_{beta:.2f}')
        print(f"\n{'#'*62}")
        print(f"# BETA: {beta}")
        print(f"{'#'*62}")
        r = run_comparison(label, params, G, groups, deg_dict, out,
                           n_eval=n_eval, max_ep=max_ep)
        all_results.extend(r)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(BASE_OUT, 'beta', 'results.csv'), index=False)
    print(f"\n{'='*62}\nBETA RESULTS\n{'='*62}")
    print(df.to_string(index=False))
    return df


def sweep_vmax(n_eval=10, max_ep=300):
    """Sweep C: daily vaccine budget."""
    vmaxes = [5, 10, 20, 40, 60]
    params_base = dict(PARAMS_NODE_RL)
    G, groups, deg_dict = _build_default_graph(params_base)
    all_results = []

    for vmax in vmaxes:
        params = dict(params_base)
        params['V_MAX_DAILY'] = vmax
        label = f"vmax_{vmax}"
        out = os.path.join(BASE_OUT, 'vmax', f'vmax_{vmax}')
        print(f"\n{'#'*62}")
        print(f"# V_MAX: {vmax}")
        print(f"{'#'*62}")
        r = run_comparison(label, params, G, groups, deg_dict, out,
                           n_eval=n_eval, max_ep=max_ep)
        all_results.extend(r)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(BASE_OUT, 'vmax', 'results.csv'), index=False)
    print(f"\n{'='*62}\nV_MAX RESULTS\n{'='*62}")
    print(df.to_string(index=False))
    return df


def sweep_network(n_eval=10, max_ep=300):
    """Sweep D: network type."""
    params_base = dict(PARAMS_NODE_RL)
    n = params_base['N']
    seed = params_base['SEED']
    hrp = params_base['HIGH_RISK_PROB']
    alpha_std = params_base['ALPHA_STD']
    avg_deg = 6

    networks = [
        ('BA', build_graph_and_groups(n=n, m=params_base['BA_M'], seed=seed,
                                      high_risk_prob=hrp, alpha_std=alpha_std)),
        ('ER', build_graph_er(n=n, avg_degree=avg_deg, seed=seed,
                              high_risk_prob=hrp, alpha_std=alpha_std)),
        ('WS', build_graph_ws(n=n, avg_degree=avg_deg, p_rewire=0.1, seed=seed,
                              high_risk_prob=hrp, alpha_std=alpha_std)),
        ('Regular', build_graph_regular(n=n, degree=avg_deg, seed=seed,
                                        high_risk_prob=hrp, alpha_std=alpha_std)),
    ]
    all_results = []

    for net_name, (G, groups, deg_dict) in networks:
        label = f"net_{net_name}"
        out = os.path.join(BASE_OUT, 'network', net_name)
        print(f"\n{'#'*62}")
        print(f"# NETWORK: {net_name}")
        print(f"{'#'*62}")
        r = run_comparison(label, params_base, G, groups, deg_dict, out,
                           n_eval=n_eval, max_ep=max_ep)
        all_results.extend(r)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(BASE_OUT, 'network', 'results.csv'), index=False)
    print(f"\n{'='*62}\nNETWORK RESULTS\n{'='*62}")
    print(df.to_string(index=False))
    return df


def sweep_discount(n_eval=10, max_ep=300):
    """Sweep E: discount factor gamma."""
    gammas = [0.8, 0.9, 0.95, 0.99, 1.0]
    params_base = dict(PARAMS_NODE_RL)
    G, groups, deg_dict = _build_default_graph(params_base)
    all_results = []

    for gamma in gammas:
        label = f"gamma_{gamma:.2f}"
        out = os.path.join(BASE_OUT, 'discount', f'gamma_{gamma:.2f}')
        print(f"\n{'#'*62}")
        print(f"# GAMMA: {gamma}")
        print(f"{'#'*62}")
        r = run_comparison(label, params_base, G, groups, deg_dict, out,
                           gamma=gamma, n_eval=n_eval, max_ep=max_ep)
        all_results.extend(r)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(BASE_OUT, 'discount', 'results.csv'), index=False)
    print(f"\n{'='*62}\nDISCOUNT FACTOR RESULTS\n{'='*62}")
    print(df.to_string(index=False))
    return df


def sweep_horizon(n_eval=10, max_ep=300):
    """Sweep F: time horizon T."""
    horizons = [30, 60]
    params_base = dict(PARAMS_NODE_RL)
    G, groups, deg_dict = _build_default_graph(params_base)
    all_results = []

    for T in horizons:
        params = dict(params_base)
        params['T_HORIZON'] = T
        label = f"horizon_{T}"
        out = os.path.join(BASE_OUT, 'horizon', f'T_{T}')
        print(f"\n{'#'*62}")
        print(f"# HORIZON: T={T}")
        print(f"{'#'*62}")
        r = run_comparison(label, params, G, groups, deg_dict, out,
                           n_eval=n_eval, max_ep=max_ep)
        all_results.extend(r)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(BASE_OUT, 'horizon', 'results.csv'), index=False)
    print(f"\n{'='*62}\nHORIZON RESULTS\n{'='*62}")
    print(df.to_string(index=False))
    return df


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Naive RL vs Node RL vs OC-Guided comparison')
    parser.add_argument('--sweep', type=str, default='all',
                        choices=['all', 'severity', 'beta', 'vmax',
                                 'network', 'discount', 'horizon'],
                        help='Which sweep to run (default: all)')
    parser.add_argument('--n_eval', type=int, default=10)
    parser.add_argument('--max_ep', type=int, default=300)
    args = parser.parse_args()

    if args.sweep in ('all', 'severity'):
        sweep_severity(n_eval=args.n_eval, max_ep=args.max_ep)
    if args.sweep in ('all', 'beta'):
        sweep_beta(n_eval=args.n_eval, max_ep=args.max_ep)
    if args.sweep in ('all', 'vmax'):
        sweep_vmax(n_eval=args.n_eval, max_ep=args.max_ep)
    if args.sweep in ('all', 'network'):
        sweep_network(n_eval=args.n_eval, max_ep=args.max_ep)
    if args.sweep in ('all', 'discount'):
        sweep_discount(n_eval=args.n_eval, max_ep=args.max_ep)
    if args.sweep in ('all', 'horizon'):
        sweep_horizon(n_eval=args.n_eval, max_ep=args.max_ep)

    print("\n\nAll sweeps complete.")
