# -*- coding: utf-8 -*-
"""
experiments/n_scaling_comparison.py

N scaling experiment: test how Naive RL vs Node RL vs OC-Guided
scale with network size N.

Theory predicts Naive RL gradient variance ∝ N, so performance
should degrade as N increases. Node RL should remain stable.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch

from config import PARAMS_NODE_RL, to_params_global, D, S
from graph import build_graph_and_groups
from ode_solver import solve, allocations_from_solution
from allocation import strict_priority_window_fill, cap_to_capacity
from env import make_env_from_graph
from rl.train import run_training_node_rl, run_training_naive_rl
from experiments.naive_vs_node_comparison import eval_with_plan, eval_oc_with_plan


OUT_DIR = 'results/naive_vs_node/n_scaling'


def run_n_scaling():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Keep K=10 fixed, vary N
    n_values = [500, 1000, 2000, 5000, 10000]
    MAX_EP = 300
    N_EVAL = 10

    all_results = []

    for N in n_values:
        print(f"\n\n{'='*62}")
        print(f"  N = {N},  K = 10")
        print(f"{'='*62}")

        params = dict(PARAMS_NODE_RL)
        params['N'] = N
        params['V_MAX_DAILY'] = 10

        # Scale initial infections proportionally (6% of N)
        frac = N / 5000.0
        params['INITIAL_INFECTED'] = max(10, int(300 * frac))
        params['INIT_INFECTED_X'] = max(5, int(150 * frac))
        params['INIT_INFECTED_Y'] = max(2, int(50 * frac))
        params['INIT_INFECTED_Z'] = max(3, int(100 * frac))

        capacity = params['V_MAX_DAILY']
        seed_counts = {
            1: params['INIT_INFECTED_X'],
            2: params['INIT_INFECTED_Y'],
            3: params['INIT_INFECTED_Z'],
        }

        sub_dir = os.path.join(OUT_DIR, f'N_{N}')
        os.makedirs(sub_dir, exist_ok=True)

        # Build graph
        print(f"  Building BA graph N={N} ...")
        G, groups, deg_dict = build_graph_and_groups(
            n=N, m=params['BA_M'], seed=params['SEED'],
            high_risk_prob=params['HIGH_RISK_PROB'],
            alpha_std=params['ALPHA_STD'],
        )
        params_global = to_params_global(params)

        # --- OC-Guided ---
        print(f"  OC-Guided ...")
        t0 = time.time()
        states_opt, ctrl_opt, _ = solve(params, init_pattern='hcp')
        ax, ay, az = allocations_from_solution(states_opt, ctrl_opt)
        ax, ay, az = strict_priority_window_fill(ax, ay, az, capacity, priority='Z')
        ax, ay, az = cap_to_capacity(ax, ay, az, capacity)
        doses_seq = np.stack([ax, ay, az], axis=1)
        oc_time = time.time() - t0

        oc_deaths, oc_plans = eval_oc_with_plan(
            G, groups, deg_dict, params_global, capacity,
            seed_counts, doses_seq, groups, N_EVAL)

        all_results.append({
            'N': N, 'method': 'OC-Guided',
            'deaths_mean': round(np.mean(oc_deaths), 1),
            'deaths_std': round(np.std(oc_deaths), 1),
            'runtime_sec': round(oc_time, 1),
        })
        print(f"    OC: {np.mean(oc_deaths):.1f} ± {np.std(oc_deaths):.1f}  ({oc_time:.1f}s)")

        with open(os.path.join(sub_dir, f'plan_oc_N{N}.json'), 'w') as f:
            json.dump(oc_plans, f)

        # --- Node RL ---
        print(f"  Node RL (max_ep={MAX_EP}) ...")
        t0 = time.time()
        node_policy, node_hist = run_training_node_rl(
            G=G, groups=groups, deg_dict=deg_dict,
            params_global=params_global, capacity_daily=capacity,
            max_episodes=MAX_EP, seed_counts=seed_counts,
            gamma=0.99, terminal_reward_scale=1.0,
            patience=9999,
            label=f'node_N{N}', out_dir=sub_dir,
        )
        node_time = time.time() - t0

        node_deaths, node_plans = eval_with_plan(
            node_policy, 'node', G, groups, deg_dict, params_global,
            capacity, seed_counts, N_EVAL)

        all_results.append({
            'N': N, 'method': 'Node RL',
            'deaths_mean': round(np.mean(node_deaths), 1),
            'deaths_std': round(np.std(node_deaths), 1),
            'runtime_sec': round(node_time, 1),
        })
        print(f"    Node RL: {np.mean(node_deaths):.1f} ± {np.std(node_deaths):.1f}  ({node_time:.1f}s)")

        with open(os.path.join(sub_dir, f'plan_node_N{N}.json'), 'w') as f:
            json.dump(node_plans, f)

        np.save(os.path.join(sub_dir, f'convergence_node_N{N}.npy'), np.array(node_hist))

        # --- Naive RL ---
        print(f"  Naive RL (max_ep={MAX_EP}) ...")
        t0 = time.time()
        naive_policy, naive_hist = run_training_naive_rl(
            G=G, groups=groups, deg_dict=deg_dict,
            params_global=params_global, capacity_daily=capacity,
            max_episodes=MAX_EP, seed_counts=seed_counts,
            gamma=0.99, terminal_reward_scale=1.0,
            patience=9999,
            label=f'naive_N{N}', out_dir=sub_dir,
        )
        naive_time = time.time() - t0

        naive_deaths, naive_plans = eval_with_plan(
            naive_policy, 'naive', G, groups, deg_dict, params_global,
            capacity, seed_counts, N_EVAL)

        all_results.append({
            'N': N, 'method': 'Naive RL',
            'deaths_mean': round(np.mean(naive_deaths), 1),
            'deaths_std': round(np.std(naive_deaths), 1),
            'runtime_sec': round(naive_time, 1),
        })
        print(f"    Naive RL: {np.mean(naive_deaths):.1f} ± {np.std(naive_deaths):.1f}  ({naive_time:.1f}s)")

        with open(os.path.join(sub_dir, f'plan_naive_N{N}.json'), 'w') as f:
            json.dump(naive_plans, f)

        np.save(os.path.join(sub_dir, f'convergence_naive_N{N}.npy'), np.array(naive_hist))

    # Save summary
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUT_DIR, 'results.csv'), index=False)

    print(f"\n\n{'='*62}")
    print(f"  N SCALING SUMMARY  (K=10 fixed)")
    print(f"{'='*62}")
    pivot = df.pivot(index='N', columns='method', values='deaths_mean')
    print(pivot.to_string())

    # Compute gap: Naive - Node
    if 'Naive RL' in pivot.columns and 'Node RL' in pivot.columns:
        print(f"\nNaive RL - Node RL gap:")
        for n in pivot.index:
            gap = pivot.loc[n, 'Naive RL'] - pivot.loc[n, 'Node RL']
            print(f"  N={n:>6d}: {gap:+.1f} deaths")

    print(f"\nResults saved to {OUT_DIR}/results.csv")


if __name__ == '__main__':
    run_n_scaling()
