# -*- coding: utf-8 -*-
"""
experiments/rerun_node_methods.py — Rerun only Node RL (cold) and NodeHorizon
for severity sweep after code changes (31-dim obs, no degree bias).

Reuses OC/Group RL results from previous run.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch

from config import PARAMS_NODE_RL, to_params_global, D, S
from graph import build_graph_and_groups
from ode_solver import solve, allocations_from_solution
from allocation import strict_priority_window_fill, cap_to_capacity
from env import make_env_from_graph
from rl.train import run_training_node_rl


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


SEVERITY_LEVELS = [
    {'pY': 0.20, 'dY': 0.27, 'label': 'baseline'},
    {'pY': 0.30, 'dY': 0.40, 'label': 'moderate'},
    {'pY': 0.40, 'dY': 0.50, 'label': 'severe'},
    {'pY': 0.50, 'dY': 0.65, 'label': 'critical'},
]

# Previous results for methods that didn't change
PREV_RESULTS = {
    'baseline': {'OC_Guided': (25.5, 3.7), 'Group_RL (cold)': (28.5, 4.7), 'Group_RL (warm)': (27.1, 6.3)},
    'moderate': {'OC_Guided': (60.5, 6.5), 'Group_RL (cold)': (58.1, 6.1), 'Group_RL (warm)': (59.6, 9.3)},
    'severe':   {'OC_Guided': (91.2, 7.3), 'Group_RL (cold)': (97.3, 5.3), 'Group_RL (warm)': (95.7, 8.3)},
    'critical': {'OC_Guided': (149.4, 7.0), 'Group_RL (cold)': (142.5, 10.8), 'Group_RL (warm)': (149.5, 7.9)},
}


def run_rerun(out_dir='results/sensitivity_node/severity_v2', n_eval=10,
              max_episodes=300):
    os.makedirs(out_dir, exist_ok=True)
    params_base = dict(PARAMS_NODE_RL)

    # Build graph once
    G, groups, deg_dict = build_graph_and_groups(
        n=params_base['N'], m=params_base['BA_M'], seed=params_base['SEED'],
        high_risk_prob=params_base['HIGH_RISK_PROB'],
        alpha_std=params_base['ALPHA_STD'],
    )

    all_results = []

    for sev in SEVERITY_LEVELS:
        params = dict(params_base)
        params['pY'] = sev['pY']
        params['dY'] = sev['dY']
        label = sev['label']
        capacity = params['V_MAX_DAILY']
        seed_counts = {
            1: int(params.get('INIT_INFECTED_X', params['INITIAL_INFECTED'])),
            2: int(params.get('INIT_INFECTED_Y', 0)),
            3: int(params.get('INIT_INFECTED_Z', 0)),
        }
        params_global = to_params_global(params)
        sub_dir = os.path.join(out_dir, label)
        os.makedirs(sub_dir, exist_ok=True)

        print(f"\n{'#'*62}")
        print(f"# SEVERITY: {label} (pY={sev['pY']}, dY={sev['dY']})")
        print(f"{'#'*62}")

        # Copy previous results
        for method, (mean, std) in PREV_RESULTS[label].items():
            all_results.append({'setting': label, 'method': method,
                                'deaths_mean': mean, 'deaths_std': std})
            print(f"  [{label}] {method}: {mean} +/- {std} (previous)")

        # ODE solution for warm-start
        states_opt, ctrl_opt, _ = solve(params, init_pattern='hcp')
        ax, ay, az = allocations_from_solution(states_opt, ctrl_opt)
        ax, ay, az = strict_priority_window_fill(ax, ay, az, capacity, priority='Z')
        ax, ay, az = cap_to_capacity(ax, ay, az, capacity)
        doses_seq = np.stack([ax, ay, az], axis=1)

        # Node RL (cold) — rerun with 31-dim obs
        print(f"\n  [{label}] Training Node RL (cold) ...")
        p_cold, _ = run_training_node_rl(
            G=G, groups=groups, deg_dict=deg_dict,
            params_global=params_global, capacity_daily=capacity,
            max_episodes=max_episodes, seed_counts=seed_counts,
            label=f'node_cold_{label}_v2', out_dir=sub_dir,
        )
        cold_d = eval_node_policy(p_cold, G, groups, deg_dict, params_global,
                                  capacity, seed_counts, n_eval)
        all_results.append({'setting': label, 'method': 'Node_RL (cold)',
                            'deaths_mean': round(np.mean(cold_d), 1),
                            'deaths_std': round(np.std(cold_d), 1)})
        print(f"  [{label}] Node RL (cold): {np.mean(cold_d):.1f} +/- {np.std(cold_d):.1f}")

        # NodeHorizon (α=3, warm) — rerun with 31-dim obs + no degree bias
        print(f"\n  [{label}] Training NodeHorizon (a=3, warm) ...")
        p_hz, _ = run_training_node_rl(
            G=G, groups=groups, deg_dict=deg_dict,
            params_global=params_global, capacity_daily=capacity,
            max_episodes=max_episodes, seed_counts=seed_counts,
            label=f'horizon_{label}_v2', out_dir=sub_dir,
            terminal_reward_scale=3.0,
            doses_seq=doses_seq, bias_strength=1.0, bias_decay_episodes=20,
        )
        hz_d = eval_node_policy(p_hz, G, groups, deg_dict, params_global,
                                capacity, seed_counts, n_eval)
        all_results.append({'setting': label, 'method': 'NodeHorizon (a=3,warm)',
                            'deaths_mean': round(np.mean(hz_d), 1),
                            'deaths_std': round(np.std(hz_d), 1)})
        print(f"  [{label}] NodeHorizon: {np.mean(hz_d):.1f} +/- {np.std(hz_d):.1f}")

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 72)
    print("SEVERITY SWEEP v2 (31-dim obs, no degree bias)")
    print("=" * 72)
    for s in ['baseline', 'moderate', 'severe', 'critical']:
        print(f"\n--- {s.upper()} ---")
        sub = df[df['setting'] == s][['method', 'deaths_mean', 'deaths_std']]
        print(sub.to_string(index=False))

    csv_path = os.path.join(out_dir, 'results_severity_v2.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved -> {csv_path}")
    return df


if __name__ == '__main__':
    run_rerun()
