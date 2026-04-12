# -*- coding: utf-8 -*-
"""
experiments/runtime_comparison.py

Runtime & performance comparison at larger scale: N=10000, K=20.
Tests Naive RL vs Node RL vs OC-Guided with more episodes (600)
to see if Naive RL needs longer to converge.

Also records per-episode eval history to plot convergence curves.
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


OUT_DIR = 'results/runtime_comparison'


def run_runtime_experiment():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Parameters: N=10000, K=20 ---
    params = dict(PARAMS_NODE_RL)
    params['N'] = 10000
    params['V_MAX_DAILY'] = 20
    # Scale initial infections proportionally (6% of N)
    params['INITIAL_INFECTED'] = 600
    params['INIT_INFECTED_X'] = 300
    params['INIT_INFECTED_Y'] = 100
    params['INIT_INFECTED_Z'] = 200

    capacity = params['V_MAX_DAILY']
    seed_counts = {
        1: params['INIT_INFECTED_X'],
        2: params['INIT_INFECTED_Y'],
        3: params['INIT_INFECTED_Z'],
    }

    MAX_EP = 1000
    N_EVAL = 10

    print(f"Building graph: N={params['N']}, BA(m={params['BA_M']})")
    t0 = time.time()
    G, groups, deg_dict = build_graph_and_groups(
        n=params['N'], m=params['BA_M'], seed=params['SEED'],
        high_risk_prob=params['HIGH_RISK_PROB'],
        alpha_std=params['ALPHA_STD'],
    )
    graph_time = time.time() - t0
    print(f"Graph built in {graph_time:.1f}s  "
          f"(nodes={G.number_of_nodes()}, edges={G.number_of_edges()})")

    params_global = to_params_global(params)
    results = []

    # =====================================================================
    # 1. OC-Guided
    # =====================================================================
    print(f"\n{'='*62}")
    print(f"OC-GUIDED  (N={params['N']}, K={capacity})")
    print(f"{'='*62}")

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

    results.append({
        'method': 'OC-Guided',
        'deaths_mean': round(np.mean(oc_deaths), 1),
        'deaths_std': round(np.std(oc_deaths), 1),
        'runtime_sec': round(oc_time, 1),
        'episodes': '-',
        'N': params['N'],
        'K': capacity,
    })
    print(f"OC-Guided: {np.mean(oc_deaths):.1f} ± {np.std(oc_deaths):.1f}  "
          f"(solve time: {oc_time:.1f}s)")

    with open(os.path.join(OUT_DIR, 'plan_oc.json'), 'w') as f:
        json.dump(oc_plans, f)

    # =====================================================================
    # 2. Node RL (with terminal reward)
    # =====================================================================
    print(f"\n{'='*62}")
    print(f"NODE RL  (N={params['N']}, K={capacity}, max_ep={MAX_EP})")
    print(f"{'='*62}")

    t0 = time.time()
    node_policy, node_hist = run_training_node_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=MAX_EP, seed_counts=seed_counts,
        gamma=0.99,
        terminal_reward_scale=1.0,
        patience=9999,
        label='node_N10000', out_dir=OUT_DIR,
    )
    node_time = time.time() - t0
    node_episodes = len(node_hist) * 10  # episodes_per_update=10

    node_deaths, node_plans = eval_with_plan(
        node_policy, 'node', G, groups, deg_dict, params_global,
        capacity, seed_counts, N_EVAL)

    results.append({
        'method': 'Node RL',
        'deaths_mean': round(np.mean(node_deaths), 1),
        'deaths_std': round(np.std(node_deaths), 1),
        'runtime_sec': round(node_time, 1),
        'episodes': node_episodes,
        'N': params['N'],
        'K': capacity,
    })
    print(f"\nNode RL: {np.mean(node_deaths):.1f} ± {np.std(node_deaths):.1f}  "
          f"(time: {node_time:.1f}s, episodes: {node_episodes})")

    with open(os.path.join(OUT_DIR, 'plan_node.json'), 'w') as f:
        json.dump(node_plans, f)

    # Save convergence history
    np.save(os.path.join(OUT_DIR, 'convergence_node.npy'), np.array(node_hist))

    # =====================================================================
    # 3. Naive RL
    # =====================================================================
    print(f"\n{'='*62}")
    print(f"NAIVE RL  (N={params['N']}, K={capacity}, max_ep={MAX_EP})")
    print(f"{'='*62}")

    t0 = time.time()
    naive_policy, naive_hist = run_training_naive_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=MAX_EP, seed_counts=seed_counts,
        gamma=0.99,
        terminal_reward_scale=1.0,
        patience=9999,
        label='naive_N10000', out_dir=OUT_DIR,
    )
    naive_time = time.time() - t0
    naive_episodes = len(naive_hist) * 10

    naive_deaths, naive_plans = eval_with_plan(
        naive_policy, 'naive', G, groups, deg_dict, params_global,
        capacity, seed_counts, N_EVAL)

    results.append({
        'method': 'Naive RL',
        'deaths_mean': round(np.mean(naive_deaths), 1),
        'deaths_std': round(np.std(naive_deaths), 1),
        'runtime_sec': round(naive_time, 1),
        'episodes': naive_episodes,
        'N': params['N'],
        'K': capacity,
    })
    print(f"\nNaive RL: {np.mean(naive_deaths):.1f} ± {np.std(naive_deaths):.1f}  "
          f"(time: {naive_time:.1f}s, episodes: {naive_episodes})")

    with open(os.path.join(OUT_DIR, 'plan_naive.json'), 'w') as f:
        json.dump(naive_plans, f)

    np.save(os.path.join(OUT_DIR, 'convergence_naive.npy'), np.array(naive_hist))

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n\n{'='*62}")
    print(f"RUNTIME COMPARISON SUMMARY  (N={params['N']}, K={capacity})")
    print(f"{'='*62}")

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT_DIR, 'results.csv'), index=False)

    # Per-episode cost
    if node_episodes and node_episodes != '-':
        print(f"\nPer-episode cost:")
        print(f"  Node RL:  {node_time/node_episodes:.2f} s/episode")
    if naive_episodes and naive_episodes != '-':
        print(f"  Naive RL: {naive_time/naive_episodes:.2f} s/episode")
    if node_time > 0:
        print(f"\nNaive/Node time ratio: {naive_time/node_time:.2f}x")

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    run_runtime_experiment()
