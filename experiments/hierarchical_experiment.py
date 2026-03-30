# -*- coding: utf-8 -*-
"""
experiments/hierarchical_experiment.py — Hierarchical RL experiment.

Compares 6 methods:
  1. OC-Guided (degree-greedy)
  2. Group RL (cold)
  3. Group RL (warm)
  4. Node RL (cold)
  5. NodeHorizon (α=3, warm)
  6. Hierarchical RL = Group RL (layer 1) + Node RL (layer 2)

Tests across baseline and high-mortality scenarios.
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
from prior import build_feasible_prior_from_doses
from rl.train import run_training, run_training_node_rl
from rl.hierarchical import train_hierarchical, HierarchicalPolicy
from rl.model import NodeScoringPolicy


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


def eval_oc(G, groups, deg_dict, params_global, capacity,
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


def eval_group(ppo, G, groups, deg_dict, params_global,
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


def eval_node(policy, G, groups, deg_dict, params_global,
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


def eval_hierarchical(group_policy, node_scorer, G, groups, deg_dict,
                      params_global, capacity, seed_counts, n_eval=10):
    deaths = []
    node_scorer.eval()
    group_policy.eval()
    for i in range(n_eval):
        env = _make_stochastic_env(G, groups, deg_dict, params_global,
                                   capacity, seed_counts, rng_seed=2000 + i)
        hier = HierarchicalPolicy(group_policy, node_scorer, env)
        done = False
        while not done:
            s_ids, feats = env.node_features()
            if len(s_ids) == 0:
                _, _, done, _ = env.step_node_ids([])
                continue
            selected, _, _ = hier.act(deterministic=True)
            _, _, done, _ = env.step_node_ids(selected)
        deaths.append(int(np.sum(env.status == D)))
    node_scorer.train()
    return deaths


def run_scenario(scenario_name, params, out_dir, n_eval=10,
                 max_ep_node=300, max_ep_group=200):
    os.makedirs(out_dir, exist_ok=True)
    capacity = params['V_MAX_DAILY']
    seed_counts = {
        1: int(params.get('INIT_INFECTED_X', params['INITIAL_INFECTED'])),
        2: int(params.get('INIT_INFECTED_Y', 0)),
        3: int(params.get('INIT_INFECTED_Z', 0)),
    }

    print(f"\n{'='*62}")
    print(f"SCENARIO: {scenario_name}")
    print(f"  beta={params['beta']}, pY={params['pY']}, dY={params['dY']}, V_MAX={capacity}")
    print(f"{'='*62}")

    G, groups, deg_dict = build_graph_and_groups(
        n=params['N'], m=params['BA_M'], seed=params['SEED'],
        high_risk_prob=params['HIGH_RISK_PROB'], alpha_std=params['ALPHA_STD'],
    )
    params_global = to_params_global(params)

    # ODE
    states_opt, ctrl_opt, _ = solve(params, init_pattern='hcp')
    ax, ay, az = allocations_from_solution(states_opt, ctrl_opt)
    ax, ay, az = strict_priority_window_fill(ax, ay, az, capacity, priority='Z')
    ax, ay, az = cap_to_capacity(ax, ay, az, capacity)
    doses_seq = np.stack([ax, ay, az], axis=1)

    results = []

    # 1. OC-Guided
    oc_d = eval_oc(G, groups, deg_dict, params_global, capacity,
                   seed_counts, doses_seq, n_eval)
    results.append({'scenario': scenario_name, 'method': 'OC_Guided',
                    'deaths_mean': round(np.mean(oc_d), 1),
                    'deaths_std': round(np.std(oc_d), 1)})
    print(f"  OC-Guided: {np.mean(oc_d):.1f} +/- {np.std(oc_d):.1f}")

    # 2. Group RL (cold)
    print(f"\n  Training Group RL (cold) ...")
    ppo_cold, _ = run_training(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        prior_path=None, max_episodes=max_ep_group,
        label=f'group_cold_{scenario_name}', out_dir=out_dir,
        seed_counts=seed_counts,
    )
    gc_d = eval_group(ppo_cold, G, groups, deg_dict, params_global,
                      capacity, seed_counts, n_eval)
    results.append({'scenario': scenario_name, 'method': 'Group_RL (cold)',
                    'deaths_mean': round(np.mean(gc_d), 1),
                    'deaths_std': round(np.std(gc_d), 1)})
    print(f"  Group RL (cold): {np.mean(gc_d):.1f} +/- {np.std(gc_d):.1f}")

    # 3. Group RL (warm)
    print(f"\n  Training Group RL (warm) ...")
    doses_path = os.path.join(out_dir, 'ode_doses.npy')
    prior_path = os.path.join(out_dir, 'ode_prior.npy')
    np.save(doses_path, doses_seq)
    build_feasible_prior_from_doses(
        doses_path=doses_path,
        args=(G, groups, deg_dict, params_global, capacity, seed_counts),
        label=scenario_name, bias=[0, 0, 1], save_path=prior_path,
    )
    ppo_warm, _ = run_training(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        prior_path=prior_path, max_episodes=max_ep_group,
        label=f'group_warm_{scenario_name}', out_dir=out_dir,
        seed_counts=seed_counts,
    )
    gw_d = eval_group(ppo_warm, G, groups, deg_dict, params_global,
                      capacity, seed_counts, n_eval)
    results.append({'scenario': scenario_name, 'method': 'Group_RL (warm)',
                    'deaths_mean': round(np.mean(gw_d), 1),
                    'deaths_std': round(np.std(gw_d), 1)})
    print(f"  Group RL (warm): {np.mean(gw_d):.1f} +/- {np.std(gw_d):.1f}")

    # 4. Node RL (cold)
    print(f"\n  Training Node RL (cold) ...")
    p_node, _ = run_training_node_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=max_ep_node, seed_counts=seed_counts,
        label=f'node_cold_{scenario_name}', out_dir=out_dir,
    )
    nc_d = eval_node(p_node, G, groups, deg_dict, params_global,
                     capacity, seed_counts, n_eval)
    results.append({'scenario': scenario_name, 'method': 'Node_RL (cold)',
                    'deaths_mean': round(np.mean(nc_d), 1),
                    'deaths_std': round(np.std(nc_d), 1)})
    print(f"  Node RL (cold): {np.mean(nc_d):.1f} +/- {np.std(nc_d):.1f}")

    # 5. NodeHorizon (α=3, warm)
    print(f"\n  Training NodeHorizon (a=3, warm) ...")
    p_hz, _ = run_training_node_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=max_ep_node, seed_counts=seed_counts,
        label=f'horizon_{scenario_name}', out_dir=out_dir,
        terminal_reward_scale=3.0,
        doses_seq=doses_seq, bias_strength=1.0, bias_decay_episodes=20,
    )
    hz_d = eval_node(p_hz, G, groups, deg_dict, params_global,
                     capacity, seed_counts, n_eval)
    results.append({'scenario': scenario_name, 'method': 'NodeHorizon (a=3,warm)',
                    'deaths_mean': round(np.mean(hz_d), 1),
                    'deaths_std': round(np.std(hz_d), 1)})
    print(f"  NodeHorizon: {np.mean(hz_d):.1f} +/- {np.std(hz_d):.1f}")

    # 6. Hierarchical RL = Group RL (warm) + Node RL scorer
    print(f"\n  Training Hierarchical RL (Group warm + Node scorer) ...")
    hier_scorer, hier_hist = train_hierarchical(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        group_ppo=ppo_warm,
        max_episodes=max_ep_node, seed_counts=seed_counts,
        terminal_reward_scale=3.0,
        label=f'hier_{scenario_name}', out_dir=out_dir,
    )
    hier_d = eval_hierarchical(
        ppo_warm.policy, hier_scorer, G, groups, deg_dict,
        params_global, capacity, seed_counts, n_eval,
    )
    results.append({'scenario': scenario_name, 'method': 'Hierarchical_RL',
                    'deaths_mean': round(np.mean(hier_d), 1),
                    'deaths_std': round(np.std(hier_d), 1)})
    print(f"  Hierarchical RL: {np.mean(hier_d):.1f} +/- {np.std(hier_d):.1f}")

    return results


def run_hierarchical_experiment(
    out_dir='results/hierarchical',
    n_eval=10,
    max_ep_node=300,
    max_ep_group=200,
):
    # Baseline
    params_base = dict(PARAMS_NODE_RL)

    # Moderate risk (where Node RL previously beat OC)
    params_moderate = dict(PARAMS_NODE_RL)
    params_moderate['pY'] = 0.30
    params_moderate['dY'] = 0.40

    # Severe risk
    params_severe = dict(PARAMS_NODE_RL)
    params_severe['pY'] = 0.40
    params_severe['dY'] = 0.50

    all_results = []
    for name, params in [('baseline', params_base),
                         ('moderate', params_moderate),
                         ('severe', params_severe)]:
        r = run_scenario(name, params,
                         os.path.join(out_dir, name),
                         n_eval, max_ep_node, max_ep_group)
        all_results.extend(r)

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 72)
    print("HIERARCHICAL RL EXPERIMENT RESULTS")
    print("=" * 72)
    for sc in df['scenario'].unique():
        print(f"\n--- {sc.upper()} ---")
        sub = df[df['scenario'] == sc][['method', 'deaths_mean', 'deaths_std']]
        print(sub.to_string(index=False))

    csv_path = os.path.join(out_dir, 'results_hierarchical.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved -> {csv_path}")
    return df


if __name__ == '__main__':
    run_hierarchical_experiment()
