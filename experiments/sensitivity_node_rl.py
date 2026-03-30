# -*- coding: utf-8 -*-
"""
experiments/sensitivity_node_rl.py — Sensitivity analysis for Node RL.

Four sweeps, each comparing 5 methods:
  1. OC-Guided
  2. Group RL (cold)
  3. Group RL (warm)
  4. Node RL (cold)
  5. NodeHorizon (α=3, warm)

Sweeps:
  A. Severity  — pY, dY scaled together
  B. Beta      — transmissibility
  C. V_MAX     — daily vaccine budget
  D. Network   — BA, ER, WS, Regular
"""

import os, sys, copy
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
from prior import build_feasible_prior_from_doses
from rl.train import run_training, run_training_node_rl


# =====================================================================
# Shared evaluation helpers
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


def eval_node_policy(policy, G, groups, deg_dict, params_global,
                     capacity, seed_counts, n_eval=10):
    deaths = []
    policy.eval()
    for i in range(n_eval):
        env = _make_stochastic_env(G, groups, deg_dict, params_global,
                                   capacity, seed_counts, rng_seed=2000 + i)
        done = False
        while not done:
            g_state = torch.from_numpy(env.obs_with_pressure()).float()
            s_ids, feats = env.node_features()
            if len(s_ids) == 0:
                _, _, done, _ = env.step_node_ids([])
                continue
            f_t = torch.from_numpy(feats).float()
            with torch.no_grad():
                idxs, _ = policy.select(g_state, f_t, capacity,
                                        deterministic=True)
            selected = [s_ids[i] for i in idxs.tolist()]
            _, _, done, _ = env.step_node_ids(selected)
        deaths.append(int(np.sum(env.status == D)))
    policy.train()
    return deaths


def eval_oc_stochastic(G, groups, deg_dict, params_global, capacity,
                       seed_counts, doses_seq, n_eval=10):
    deaths = []
    T = len(doses_seq)
    for i in range(n_eval):
        env = _make_stochastic_env(G, groups, deg_dict, params_global,
                                   capacity, seed_counts, rng_seed=2000 + i)
        for t in range(T):
            total = max(1, int(doses_seq[t].sum()))
            shares = doses_seq[t].astype(float) / total
            _, _, done, _ = env.step(shares)
            if done:
                break
        deaths.append(int(np.sum(env.status == D)))
    return deaths


def eval_group_policy(ppo, G, groups, deg_dict, params_global,
                      capacity, seed_counts, n_eval=10):
    deaths = []
    ppo.policy.eval()
    for i in range(n_eval):
        env = _make_stochastic_env(G, groups, deg_dict, params_global,
                                   capacity, seed_counts, rng_seed=2000 + i)
        state = env._obs()
        done = False
        while not done:
            with torch.no_grad():
                s_t = torch.from_numpy(state).float()
                dist = ppo.policy.dist(s_t)
                act = torch.clamp(dist.mean, min=1e-6)
                shares = (act / act.sum()).numpy()
            state, _, done, _ = env.step(shares)
        deaths.append(int(np.sum(env.status == D)))
    ppo.policy.train()
    return deaths


def _get_seed_counts(params):
    return {
        1: int(params.get('INIT_INFECTED_X', params['INITIAL_INFECTED'])),
        2: int(params.get('INIT_INFECTED_Y', 0)),
        3: int(params.get('INIT_INFECTED_Z', 0)),
    }


def run_5methods(label, params, G, groups, deg_dict, out_dir,
                 n_eval=10, max_ep_node=300, max_ep_group=200):
    """Run all 5 methods for one parameter setting. Returns list of result dicts."""
    os.makedirs(out_dir, exist_ok=True)
    capacity    = params['V_MAX_DAILY']
    seed_counts = _get_seed_counts(params)
    params_global = to_params_global(params)

    # ODE solution
    print(f"\n  [{label}] Solving ODE ...")
    states_opt, ctrl_opt, _ = solve(params, init_pattern='hcp')
    ax, ay, az = allocations_from_solution(states_opt, ctrl_opt)
    ax, ay, az = strict_priority_window_fill(ax, ay, az, capacity, priority='Z')
    ax, ay, az = cap_to_capacity(ax, ay, az, capacity)
    doses_seq  = np.stack([ax, ay, az], axis=1)

    results = []

    # --- 1. OC-Guided ---
    oc_d = eval_oc_stochastic(G, groups, deg_dict, params_global, capacity,
                              seed_counts, doses_seq, n_eval)
    results.append({'setting': label, 'method': 'OC_Guided',
                    'deaths_mean': round(np.mean(oc_d), 1),
                    'deaths_std': round(np.std(oc_d), 1)})
    print(f"  [{label}] OC-Guided: {np.mean(oc_d):.1f} +/- {np.std(oc_d):.1f}")

    # --- 2. Group RL (cold) ---
    print(f"  [{label}] Training Group RL (cold) ...")
    ppo_cold, _ = run_training(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        prior_path=None, max_episodes=max_ep_group,
        label=f'group_cold_{label}', out_dir=out_dir, seed_counts=seed_counts,
    )
    gc_d = eval_group_policy(ppo_cold, G, groups, deg_dict, params_global,
                             capacity, seed_counts, n_eval)
    results.append({'setting': label, 'method': 'Group_RL (cold)',
                    'deaths_mean': round(np.mean(gc_d), 1),
                    'deaths_std': round(np.std(gc_d), 1)})
    print(f"  [{label}] Group RL (cold): {np.mean(gc_d):.1f} +/- {np.std(gc_d):.1f}")

    # --- 3. Group RL (warm) ---
    print(f"  [{label}] Training Group RL (warm) ...")
    doses_path = os.path.join(out_dir, 'ode_doses.npy')
    prior_path = os.path.join(out_dir, 'ode_prior.npy')
    np.save(doses_path, doses_seq)
    build_feasible_prior_from_doses(
        doses_path=doses_path,
        args=(G, groups, deg_dict, params_global, capacity, seed_counts),
        label=label, bias=[0, 0, 1], save_path=prior_path,
    )
    ppo_warm, _ = run_training(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        prior_path=prior_path, max_episodes=max_ep_group,
        label=f'group_warm_{label}', out_dir=out_dir, seed_counts=seed_counts,
    )
    gw_d = eval_group_policy(ppo_warm, G, groups, deg_dict, params_global,
                             capacity, seed_counts, n_eval)
    results.append({'setting': label, 'method': 'Group_RL (warm)',
                    'deaths_mean': round(np.mean(gw_d), 1),
                    'deaths_std': round(np.std(gw_d), 1)})
    print(f"  [{label}] Group RL (warm): {np.mean(gw_d):.1f} +/- {np.std(gw_d):.1f}")

    # --- 4. Node RL (cold) ---
    print(f"  [{label}] Training Node RL (cold) ...")
    p_node, _ = run_training_node_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=max_ep_node, seed_counts=seed_counts,
        label=f'node_cold_{label}', out_dir=out_dir,
    )
    nc_d = eval_node_policy(p_node, G, groups, deg_dict, params_global,
                            capacity, seed_counts, n_eval)
    results.append({'setting': label, 'method': 'Node_RL (cold)',
                    'deaths_mean': round(np.mean(nc_d), 1),
                    'deaths_std': round(np.std(nc_d), 1)})
    print(f"  [{label}] Node RL (cold): {np.mean(nc_d):.1f} +/- {np.std(nc_d):.1f}")

    # --- 5. NodeHorizon (α=3, warm) ---
    print(f"  [{label}] Training NodeHorizon (a=3, warm) ...")
    p_hz, _ = run_training_node_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=max_ep_node, seed_counts=seed_counts,
        label=f'horizon_{label}', out_dir=out_dir,
        terminal_reward_scale=3.0,
        doses_seq=doses_seq, bias_strength=1.0, bias_decay_episodes=20,
    )
    hz_d = eval_node_policy(p_hz, G, groups, deg_dict, params_global,
                            capacity, seed_counts, n_eval)
    results.append({'setting': label, 'method': 'NodeHorizon (a=3,warm)',
                    'deaths_mean': round(np.mean(hz_d), 1),
                    'deaths_std': round(np.std(hz_d), 1)})
    print(f"  [{label}] NodeHorizon: {np.mean(hz_d):.1f} +/- {np.std(hz_d):.1f}")

    return results


def print_sweep_results(name, df):
    print(f"\n{'='*72}")
    print(f"SENSITIVITY: {name}")
    print(f"{'='*72}")
    for s in df['setting'].unique():
        print(f"\n--- {s} ---")
        sub = df[df['setting'] == s][['method', 'deaths_mean', 'deaths_std']]
        print(sub.to_string(index=False))


# =====================================================================
# Sweep A: Severity (pY, dY)
# =====================================================================

SEVERITY_LEVELS = [
    {'pY': 0.20, 'dY': 0.27, 'label': 'baseline'},
    {'pY': 0.30, 'dY': 0.40, 'label': 'moderate'},
    {'pY': 0.40, 'dY': 0.50, 'label': 'severe'},
    {'pY': 0.50, 'dY': 0.65, 'label': 'critical'},
]


def sweep_severity(base_out='results/sensitivity_node/severity', n_eval=10):
    all_results = []
    params_base = dict(PARAMS_NODE_RL)
    G, groups, deg_dict = build_graph_and_groups(
        n=params_base['N'], m=params_base['BA_M'], seed=params_base['SEED'],
        high_risk_prob=params_base['HIGH_RISK_PROB'],
        alpha_std=params_base['ALPHA_STD'],
    )
    for sev in SEVERITY_LEVELS:
        params = dict(params_base)
        params['pY'] = sev['pY']
        params['dY'] = sev['dY']
        label = f"pY={sev['pY']}_dY={sev['dY']}_{sev['label']}"
        out = os.path.join(base_out, sev['label'])
        print(f"\n{'#'*62}")
        print(f"# SEVERITY: {label}")
        print(f"{'#'*62}")
        r = run_5methods(label, params, G, groups, deg_dict, out, n_eval)
        all_results.extend(r)
    df = pd.DataFrame(all_results)
    print_sweep_results('SEVERITY (pY, dY)', df)
    df.to_csv(os.path.join(base_out, 'results_severity.csv'), index=False)
    return df


# =====================================================================
# Sweep B: Beta (transmissibility)
# =====================================================================

BETA_VALUES = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]


def sweep_beta(base_out='results/sensitivity_node/beta', n_eval=10):
    all_results = []
    params_base = dict(PARAMS_NODE_RL)
    G, groups, deg_dict = build_graph_and_groups(
        n=params_base['N'], m=params_base['BA_M'], seed=params_base['SEED'],
        high_risk_prob=params_base['HIGH_RISK_PROB'],
        alpha_std=params_base['ALPHA_STD'],
    )
    for beta in BETA_VALUES:
        params = dict(params_base)
        params['beta'] = beta
        label = f"beta={beta:.2f}"
        out = os.path.join(base_out, f'beta_{beta:.2f}')
        print(f"\n{'#'*62}")
        print(f"# BETA: {beta}")
        print(f"{'#'*62}")
        r = run_5methods(label, params, G, groups, deg_dict, out, n_eval)
        all_results.extend(r)
    df = pd.DataFrame(all_results)
    print_sweep_results('BETA (transmissibility)', df)
    df.to_csv(os.path.join(base_out, 'results_beta.csv'), index=False)
    return df


# =====================================================================
# Sweep C: V_MAX (vaccine budget)
# =====================================================================

VMAX_VALUES = [5, 10, 20, 40, 60]


def sweep_vmax(base_out='results/sensitivity_node/vmax', n_eval=10):
    all_results = []
    params_base = dict(PARAMS_NODE_RL)
    G, groups, deg_dict = build_graph_and_groups(
        n=params_base['N'], m=params_base['BA_M'], seed=params_base['SEED'],
        high_risk_prob=params_base['HIGH_RISK_PROB'],
        alpha_std=params_base['ALPHA_STD'],
    )
    for vmax in VMAX_VALUES:
        params = dict(params_base)
        params['V_MAX_DAILY'] = vmax
        label = f"V_MAX={vmax}"
        out = os.path.join(base_out, f'vmax_{vmax}')
        print(f"\n{'#'*62}")
        print(f"# V_MAX: {vmax}")
        print(f"{'#'*62}")
        r = run_5methods(label, params, G, groups, deg_dict, out, n_eval)
        all_results.extend(r)
    df = pd.DataFrame(all_results)
    print_sweep_results('V_MAX (vaccine budget)', df)
    df.to_csv(os.path.join(base_out, 'results_vmax.csv'), index=False)
    return df


# =====================================================================
# Sweep D: Network type
# =====================================================================

def sweep_network(base_out='results/sensitivity_node/network', n_eval=10):
    all_results = []
    params_base = dict(PARAMS_NODE_RL)
    n = params_base['N']
    seed = params_base['SEED']
    hrp = params_base['HIGH_RISK_PROB']
    alpha_std = params_base['ALPHA_STD']
    avg_deg = 6  # BA with m=3 gives avg degree ~6

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

    for net_name, (G, groups, deg_dict) in networks:
        label = f"net={net_name}"
        out = os.path.join(base_out, net_name)
        print(f"\n{'#'*62}")
        print(f"# NETWORK: {net_name}")
        print(f"{'#'*62}")
        r = run_5methods(label, params_base, G, groups, deg_dict, out, n_eval)
        all_results.extend(r)

    df = pd.DataFrame(all_results)
    print_sweep_results('NETWORK TYPE', df)
    df.to_csv(os.path.join(base_out, 'results_network.csv'), index=False)
    return df


# =====================================================================
# Main: run all 4 sweeps
# =====================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=str, default='all',
                        choices=['all', 'severity', 'beta', 'vmax', 'network'],
                        help='Which sweep to run (default: all)')
    parser.add_argument('--n_eval', type=int, default=10)
    args = parser.parse_args()

    if args.sweep in ('all', 'severity'):
        sweep_severity(n_eval=args.n_eval)
    if args.sweep in ('all', 'beta'):
        sweep_beta(n_eval=args.n_eval)
    if args.sweep in ('all', 'vmax'):
        sweep_vmax(n_eval=args.n_eval)
    if args.sweep in ('all', 'network'):
        sweep_network(n_eval=args.n_eval)

    print("\n\nAll sensitivity sweeps complete.")
