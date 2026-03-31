# -*- coding: utf-8 -*-
"""
experiments/ablation_terminal_reward.py — Ablation: is terminal reward useful?

Compares 4 variants to isolate the effect of warm-start vs terminal reward:
  1. Node RL cold (no warm, no terminal)        — baseline
  2. Node RL warm (warm, no terminal)            — warm-start only
  3. NodeHorizon cold (no warm, terminal α=3)    — terminal reward only
  4. NodeHorizon warm (warm + terminal α=3)      — both (current NodeHorizon)

Tests on two scenarios where NodeHorizon previously performed well:
  - baseline (standard params)
  - long_asymp (wA=0.5, tauA=1/14)
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


def run_ablation(out_dir='results/ablation_terminal', n_eval=10, max_episodes=300):
    os.makedirs(out_dir, exist_ok=True)
    params_base = dict(PARAMS_NODE_RL)

    G, groups, deg_dict = build_graph_and_groups(
        n=params_base['N'], m=params_base['BA_M'], seed=params_base['SEED'],
        high_risk_prob=params_base['HIGH_RISK_PROB'],
        alpha_std=params_base['ALPHA_STD'],
    )

    scenarios = [
        {'label': 'baseline', 'wA': 0.5, 'tauA': 1/5},
        {'label': 'long_asymp', 'wA': 0.5, 'tauA': 1/14},
    ]

    all_results = []

    for sc in scenarios:
        params = dict(params_base)
        params['wA'] = sc['wA']
        params['tauA'] = sc['tauA']
        capacity = params['V_MAX_DAILY']
        seed_counts = {
            1: int(params.get('INIT_INFECTED_X', params['INITIAL_INFECTED'])),
            2: int(params.get('INIT_INFECTED_Y', 0)),
            3: int(params.get('INIT_INFECTED_Z', 0)),
        }
        params_global = to_params_global(params)
        sub_dir = os.path.join(out_dir, sc['label'])
        os.makedirs(sub_dir, exist_ok=True)

        # ODE for warm-start
        states_opt, ctrl_opt, _ = solve(params, init_pattern='hcp')
        ax, ay, az = allocations_from_solution(states_opt, ctrl_opt)
        ax, ay, az = strict_priority_window_fill(ax, ay, az, capacity, priority='Z')
        ax, ay, az = cap_to_capacity(ax, ay, az, capacity)
        doses_seq = np.stack([ax, ay, az], axis=1)

        print(f"\n{'='*62}")
        print(f"SCENARIO: {sc['label']} (wA={sc['wA']}, tauA={sc['tauA']:.4f})")
        print(f"{'='*62}")

        # 4 variants: 2x2 (warm/cold × terminal/no-terminal)
        variants = [
            {'name': 'cold_no_terminal',  'warm': False, 'terminal': 0.0},
            {'name': 'warm_no_terminal',  'warm': True,  'terminal': 0.0},
            {'name': 'cold_terminal',     'warm': False, 'terminal': 3.0},
            {'name': 'warm_terminal',     'warm': True,  'terminal': 3.0},
        ]

        for v in variants:
            tag = f"{v['name']}_{sc['label']}"
            print(f"\n  Training {v['name']} ...")

            kwargs = dict(
                G=G, groups=groups, deg_dict=deg_dict,
                params_global=params_global, capacity_daily=capacity,
                max_episodes=max_episodes, seed_counts=seed_counts,
                label=tag, out_dir=sub_dir,
                terminal_reward_scale=v['terminal'],
            )
            if v['warm']:
                kwargs['doses_seq'] = doses_seq
                kwargs['bias_strength'] = 1.0
                kwargs['bias_decay_episodes'] = 20

            policy, _ = run_training_node_rl(**kwargs)
            deaths = eval_node_policy(policy, G, groups, deg_dict, params_global,
                                      capacity, seed_counts, n_eval)

            all_results.append({
                'scenario': sc['label'],
                'warm_start': v['warm'],
                'terminal_reward': v['terminal'] > 0,
                'method': v['name'],
                'deaths_mean': round(np.mean(deaths), 1),
                'deaths_std': round(np.std(deaths), 1),
            })
            print(f"  {v['name']}: {np.mean(deaths):.1f} +/- {np.std(deaths):.1f}")

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 62)
    print("ABLATION: TERMINAL REWARD")
    print("=" * 62)
    for sc in df['scenario'].unique():
        print(f"\n--- {sc.upper()} ---")
        sub = df[df['scenario'] == sc][['method', 'deaths_mean', 'deaths_std']]
        print(sub.to_string(index=False))

    csv_path = os.path.join(out_dir, 'results_ablation.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved -> {csv_path}")
    return df


if __name__ == '__main__':
    run_ablation()
