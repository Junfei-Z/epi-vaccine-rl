# -*- coding: utf-8 -*-
"""
experiments/ablation_gumbel_topk.py — Ablation: Gumbel-Top-K vs Greedy Top-K

Compares Gumbel-Top-K (stochastic exploration during training) vs
plain greedy Top-K (deterministic during training) for Node RL.

OC-Guided and previous RL results are referenced from existing experiments.

Tests on two parameter settings for generality:
  - baseline (standard params, dY=0.27)
  - moderate severity (dY=0.40)
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
            selected = [s_ids[j] for j in idxs.tolist()]
            _, _, done, _ = env.step_node_ids(selected)
        deaths.append(int(np.sum(env.status == D)))
    policy.train()
    return deaths


def eval_oc_guided(G, groups, deg_dict, params_global, params,
                   capacity, seed_counts, n_eval=10):
    """Evaluate OC-Guided method."""
    states_opt, ctrl_opt, _ = solve(params, init_pattern='hcp')
    ax, ay, az = allocations_from_solution(states_opt, ctrl_opt)
    ax, ay, az = strict_priority_window_fill(ax, ay, az, capacity, priority='Z')
    ax, ay, az = cap_to_capacity(ax, ay, az, capacity)
    doses_seq = np.stack([ax, ay, az], axis=1)

    deaths = []
    for i in range(n_eval):
        env = _make_stochastic_env(G, groups, deg_dict, params_global,
                                   capacity, seed_counts, rng_seed=2000 + i)
        done = False
        day = 0
        while not done:
            if day < len(doses_seq):
                shares = doses_seq[day] / max(doses_seq[day].sum(), 1)
            else:
                shares = np.array([1/3, 1/3, 1/3])
            _, _, done, _ = env.step(shares)
            day += 1
        deaths.append(int(np.sum(env.status == D)))
    return deaths


def run_ablation(out_dir='results/ablation_gumbel', n_eval=10, max_episodes=300):
    os.makedirs(out_dir, exist_ok=True)
    params_base = dict(PARAMS_NODE_RL)

    G, groups, deg_dict = build_graph_and_groups(
        n=params_base['N'], m=params_base['BA_M'], seed=params_base['SEED'],
        high_risk_prob=params_base['HIGH_RISK_PROB'],
        alpha_std=params_base['ALPHA_STD'],
    )

    scenarios = [
        {'label': 'baseline', 'dY': 0.27},
        {'label': 'moderate', 'dY': 0.40},
    ]

    all_results = []

    for sc in scenarios:
        params = dict(params_base)
        params['dY'] = sc['dY']
        capacity = params['V_MAX_DAILY']
        seed_counts = {
            1: int(params.get('INIT_INFECTED_X', params['INITIAL_INFECTED'])),
            2: int(params.get('INIT_INFECTED_Y', 0)),
            3: int(params.get('INIT_INFECTED_Z', 0)),
        }
        params_global = to_params_global(params)
        sub_dir = os.path.join(out_dir, sc['label'])
        os.makedirs(sub_dir, exist_ok=True)

        print(f"\n{'='*62}")
        print(f"SCENARIO: {sc['label']} (dY={sc['dY']})")
        print(f"{'='*62}")

        # --- 1. OC-Guided (reference) ---
        print(f"\n  Evaluating OC-Guided ...")
        oc_deaths = eval_oc_guided(G, groups, deg_dict, params_global, params,
                                   capacity, seed_counts, n_eval)
        all_results.append({
            'scenario': sc['label'], 'method': 'OC-Guided',
            'deaths_mean': round(np.mean(oc_deaths), 1),
            'deaths_std': round(np.std(oc_deaths), 1),
        })
        print(f"  OC-Guided: {np.mean(oc_deaths):.1f} +/- {np.std(oc_deaths):.1f}")

        # --- 2. Node RL with Gumbel-Top-K (default) ---
        print(f"\n  Training Node RL (Gumbel-Top-K) ...")
        policy_gumbel, _ = run_training_node_rl(
            G=G, groups=groups, deg_dict=deg_dict,
            params_global=params_global, capacity_daily=capacity,
            max_episodes=max_episodes, seed_counts=seed_counts,
            label=f'node_gumbel_{sc["label"]}', out_dir=sub_dir,
            explore_during_training=True,
        )
        gumbel_deaths = eval_node_policy(policy_gumbel, G, groups, deg_dict,
                                         params_global, capacity, seed_counts, n_eval)
        all_results.append({
            'scenario': sc['label'], 'method': 'Node RL (Gumbel-Top-K)',
            'deaths_mean': round(np.mean(gumbel_deaths), 1),
            'deaths_std': round(np.std(gumbel_deaths), 1),
        })
        print(f"  Node RL Gumbel: {np.mean(gumbel_deaths):.1f} +/- {np.std(gumbel_deaths):.1f}")

        # --- 3. Node RL with Greedy Top-K ---
        print(f"\n  Training Node RL (Greedy Top-K) ...")
        policy_greedy, _ = run_training_node_rl(
            G=G, groups=groups, deg_dict=deg_dict,
            params_global=params_global, capacity_daily=capacity,
            max_episodes=max_episodes, seed_counts=seed_counts,
            label=f'node_greedy_{sc["label"]}', out_dir=sub_dir,
            explore_during_training=False,
        )
        greedy_deaths = eval_node_policy(policy_greedy, G, groups, deg_dict,
                                         params_global, capacity, seed_counts, n_eval)
        all_results.append({
            'scenario': sc['label'], 'method': 'Node RL (Greedy Top-K)',
            'deaths_mean': round(np.mean(greedy_deaths), 1),
            'deaths_std': round(np.std(greedy_deaths), 1),
        })
        print(f"  Node RL Greedy: {np.mean(greedy_deaths):.1f} +/- {np.std(greedy_deaths):.1f}")

    # Summary
    df = pd.DataFrame(all_results)
    print("\n" + "=" * 62)
    print("ABLATION: Gumbel-Top-K vs Greedy Top-K")
    print("=" * 62)
    for sc_label in df['scenario'].unique():
        print(f"\n--- {sc_label.upper()} ---")
        sub = df[df['scenario'] == sc_label][['method', 'deaths_mean', 'deaths_std']]
        print(sub.to_string(index=False))

    csv_path = os.path.join(out_dir, 'results_gumbel_ablation.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved -> {csv_path}")
    return df


if __name__ == '__main__':
    run_ablation()
