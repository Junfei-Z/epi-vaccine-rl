# -*- coding: utf-8 -*-
"""
experiments/ablation_terminal_reward_all.py — Terminal Reward Ablation

Compare all 3 RL methods (Group RL cold, Group RL warm, Node RL)
with and without terminal reward, plus OC-Guided as reference.

Tests on two scenarios: baseline (dY=0.27) and moderate (dY=0.40).
Terminal reward scale: 0.0 (off) vs 1.0 (on).
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
from rl.train import run_training, run_training_node_rl


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


def eval_group_policy(ppo, G, groups, deg_dict, params_global,
                      capacity, seed_counts, n_eval=10):
    deaths = []
    ppo.policy.eval()
    for i in range(n_eval):
        env = _make_stochastic_env(G, groups, deg_dict, params_global,
                                   capacity, seed_counts, rng_seed=2000 + i)
        state = env.obs_with_pressure()
        done = False
        while not done:
            s_t = torch.from_numpy(state).float()
            with torch.no_grad():
                action, _ = ppo.policy.sample_with_temp(
                    s_t, ppo.policy_old, sample_temp=0.01)
            state, _, done, _ = env.step(action.numpy())
        deaths.append(int(np.sum(env.status == D)))
    ppo.policy.train()
    return deaths


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


def run_ablation(out_dir='results/ablation_terminal', n_eval=10):
    os.makedirs(out_dir, exist_ok=True)
    params_base = dict(PARAMS_NODE_RL)

    G, groups, deg_dict = build_graph_and_groups(
        n=params_base['N'], m=params_base['BA_M'], seed=params_base['SEED'],
        high_risk_prob=params_base['HIGH_RISK_PROB'],
        alpha_std=params_base['ALPHA_STD'],
    )

    # ODE prior for warm-start
    prior_params = dict(params_base)
    states_opt, ctrl_opt, _ = solve(prior_params, init_pattern='hcp')
    ax, ay, az = allocations_from_solution(states_opt, ctrl_opt)
    ax, ay, az = strict_priority_window_fill(ax, ay, az,
                                             params_base['V_MAX_DAILY'],
                                             priority='Z')
    ax, ay, az = cap_to_capacity(ax, ay, az, params_base['V_MAX_DAILY'])
    oc_doses = np.stack([ax, ay, az], axis=1)
    # save prior for Group RL warm
    prior_path = os.path.join(out_dir, 'oc_prior.npy')
    np.save(prior_path, oc_doses / np.maximum(oc_doses.sum(axis=1, keepdims=True), 1))

    scenarios = [
        {'label': 'baseline', 'dY': 0.27},
        {'label': 'moderate', 'dY': 0.40},
    ]
    terminal_scales = [0.0, 1.0]

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

        # OC-Guided (reference, no terminal reward needed)
        print(f"\n  Evaluating OC-Guided ...")
        oc_deaths = eval_oc_guided(G, groups, deg_dict, params_global, params,
                                   capacity, seed_counts, n_eval)
        all_results.append({
            'scenario': sc['label'], 'method': 'OC-Guided',
            'terminal_scale': '-',
            'deaths_mean': round(np.mean(oc_deaths), 1),
            'deaths_std': round(np.std(oc_deaths), 1),
        })
        print(f"  OC-Guided: {np.mean(oc_deaths):.1f} +/- {np.std(oc_deaths):.1f}")

        for ts in terminal_scales:
            ts_label = f"ts{ts:.1f}"

            # --- Group RL cold ---
            print(f"\n  Training Group RL cold (terminal={ts}) ...")
            ppo_cold, _ = run_training(
                G=G, groups=groups, deg_dict=deg_dict,
                params_global=params_global, capacity_daily=capacity,
                prior_path=None, max_episodes=200,
                seed_counts=seed_counts,
                label=f'grl_cold_{sc["label"]}_{ts_label}',
                out_dir=sub_dir,
                terminal_reward_scale=ts,
            )
            cold_deaths = eval_group_policy(ppo_cold, G, groups, deg_dict,
                                            params_global, capacity, seed_counts, n_eval)
            all_results.append({
                'scenario': sc['label'], 'method': 'Group RL cold',
                'terminal_scale': ts,
                'deaths_mean': round(np.mean(cold_deaths), 1),
                'deaths_std': round(np.std(cold_deaths), 1),
            })
            print(f"  Group RL cold (ts={ts}): {np.mean(cold_deaths):.1f} +/- {np.std(cold_deaths):.1f}")

            # --- Group RL warm ---
            print(f"\n  Training Group RL warm (terminal={ts}) ...")
            ppo_warm, _ = run_training(
                G=G, groups=groups, deg_dict=deg_dict,
                params_global=params_global, capacity_daily=capacity,
                prior_path=prior_path, max_episodes=200,
                seed_counts=seed_counts,
                label=f'grl_warm_{sc["label"]}_{ts_label}',
                out_dir=sub_dir,
                terminal_reward_scale=ts,
            )
            warm_deaths = eval_group_policy(ppo_warm, G, groups, deg_dict,
                                            params_global, capacity, seed_counts, n_eval)
            all_results.append({
                'scenario': sc['label'], 'method': 'Group RL warm',
                'terminal_scale': ts,
                'deaths_mean': round(np.mean(warm_deaths), 1),
                'deaths_std': round(np.std(warm_deaths), 1),
            })
            print(f"  Group RL warm (ts={ts}): {np.mean(warm_deaths):.1f} +/- {np.std(warm_deaths):.1f}")

            # --- Node RL ---
            print(f"\n  Training Node RL (terminal={ts}) ...")
            policy_node, _ = run_training_node_rl(
                G=G, groups=groups, deg_dict=deg_dict,
                params_global=params_global, capacity_daily=capacity,
                max_episodes=300, seed_counts=seed_counts,
                label=f'node_{sc["label"]}_{ts_label}',
                out_dir=sub_dir,
                terminal_reward_scale=ts,
            )
            node_deaths = eval_node_policy(policy_node, G, groups, deg_dict,
                                           params_global, capacity, seed_counts, n_eval)
            all_results.append({
                'scenario': sc['label'], 'method': 'Node RL',
                'terminal_scale': ts,
                'deaths_mean': round(np.mean(node_deaths), 1),
                'deaths_std': round(np.std(node_deaths), 1),
            })
            print(f"  Node RL (ts={ts}): {np.mean(node_deaths):.1f} +/- {np.std(node_deaths):.1f}")

    # Summary
    df = pd.DataFrame(all_results)
    print("\n" + "=" * 70)
    print("ABLATION: Terminal Reward (all 3 RL methods)")
    print("=" * 70)
    for sc_label in df['scenario'].unique():
        print(f"\n--- {sc_label.upper()} ---")
        sub = df[df['scenario'] == sc_label][['method', 'terminal_scale',
                                               'deaths_mean', 'deaths_std']]
        print(sub.to_string(index=False))

    csv_path = os.path.join(out_dir, 'results_terminal_ablation.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved -> {csv_path}")
    return df


if __name__ == '__main__':
    run_ablation()
