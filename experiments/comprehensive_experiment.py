# -*- coding: utf-8 -*-
"""
experiments/comprehensive_experiment.py — Comprehensive multi-scenario comparison.

Compares ALL methods across two parameter scenarios:
  Baseline: current PARAMS_NODE_RL (beta=0.08, low death rates, ~25 deaths)
  Hard:     higher beta=0.15, higher mortality, more initial infections
            → more deaths, more stochastic variance, RL's network info advantage larger

Methods compared in each scenario:
  1. OC-Guided              — ODE optimal control (open-loop)
  2. Warm Group RL          — Dirichlet PPO with OC prior (group-level)
  3. Node RL (cold)         — NodeScoringPolicy from random init
  4. Node RL (warm)         — NodeScoringPolicy with OC score bias
  5. NodeHorizon (α=3,warm) — Node RL + terminal death penalty + warm-start
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


def run_scenario(scenario_name, params, out_dir, n_eval=10,
                 max_episodes_node=300, max_episodes_group=200):
    """Run all 5 methods for one parameter scenario."""
    os.makedirs(out_dir, exist_ok=True)
    capacity    = params['V_MAX_DAILY']
    seed_counts = {
        1: int(params.get('INIT_INFECTED_X', params['INITIAL_INFECTED'])),
        2: int(params.get('INIT_INFECTED_Y', 0)),
        3: int(params.get('INIT_INFECTED_Z', 0)),
    }

    print(f"\n{'='*62}")
    print(f"SCENARIO: {scenario_name}")
    print(f"  N={params['N']}, beta={params['beta']}, V_MAX={capacity}")
    print(f"  dX={params['dX']}, dY={params['dY']}, dZ={params['dZ']}")
    print(f"  seeds: X={seed_counts[1]}, Y={seed_counts[2]}, Z={seed_counts[3]}")
    print(f"{'='*62}")

    # Build graph
    print(f"\n[{scenario_name}] Building graph ...")
    G, groups, deg_dict = build_graph_and_groups(
        n=params['N'], m=params['BA_M'], seed=params['SEED'],
        high_risk_prob=params['HIGH_RISK_PROB'], alpha_std=params['ALPHA_STD'],
    )
    params_global = to_params_global(params)

    # ODE solution
    print(f"[{scenario_name}] Solving ODE ...")
    states_opt, ctrl_opt, _ = solve(params, init_pattern='hcp')
    ax, ay, az = allocations_from_solution(states_opt, ctrl_opt)
    ax, ay, az = strict_priority_window_fill(ax, ay, az, capacity, priority='Z')
    ax, ay, az = cap_to_capacity(ax, ay, az, capacity)
    doses_seq  = np.stack([ax, ay, az], axis=1)

    results = []

    # ------ 1. OC-Guided ------
    print(f"\n[{scenario_name}] Evaluating OC-Guided ...")
    oc_deaths = eval_oc_stochastic(
        G, groups, deg_dict, params_global, capacity,
        seed_counts, doses_seq, n_eval=n_eval,
    )
    results.append({
        'scenario': scenario_name,
        'method': 'OC_Guided',
        'deaths_mean': round(np.mean(oc_deaths), 1),
        'deaths_std':  round(np.std(oc_deaths), 1),
    })
    print(f"  OC-Guided: {np.mean(oc_deaths):.1f} +/- {np.std(oc_deaths):.1f}")

    # ------ 2. Warm Group RL ------
    print(f"\n[{scenario_name}] Training Warm Group RL ...")
    doses_path = os.path.join(out_dir, 'ode_doses.npy')
    prior_path = os.path.join(out_dir, 'ode_prior.npy')
    np.save(doses_path, doses_seq)
    build_feasible_prior_from_doses(
        doses_path=doses_path,
        args=(G, groups, deg_dict, params_global, capacity, seed_counts),
        label=scenario_name, bias=[0, 0, 1], save_path=prior_path,
    )
    ppo_group, hist_group = run_training(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        prior_path=prior_path, max_episodes=max_episodes_group,
        label=f'group_warm_{scenario_name}', out_dir=out_dir,
        seed_counts=seed_counts,
    )
    group_deaths = eval_group_policy(
        ppo_group, G, groups, deg_dict, params_global,
        capacity, seed_counts, n_eval=n_eval,
    )
    results.append({
        'scenario': scenario_name,
        'method': 'Group_RL (warm)',
        'deaths_mean': round(np.mean(group_deaths), 1),
        'deaths_std':  round(np.std(group_deaths), 1),
    })
    print(f"  Group RL (warm): {np.mean(group_deaths):.1f} +/- {np.std(group_deaths):.1f}")

    # ------ 3. Node RL (cold) ------
    print(f"\n[{scenario_name}] Training Node RL (cold) ...")
    p_cold, h_cold = run_training_node_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=max_episodes_node, seed_counts=seed_counts,
        label=f'cold_{scenario_name}', out_dir=out_dir,
        terminal_reward_scale=0.0,
    )
    cold_deaths = eval_node_policy(
        p_cold, G, groups, deg_dict, params_global,
        capacity, seed_counts, n_eval=n_eval,
    )
    results.append({
        'scenario': scenario_name,
        'method': 'Node_RL (cold)',
        'deaths_mean': round(np.mean(cold_deaths), 1),
        'deaths_std':  round(np.std(cold_deaths), 1),
    })
    print(f"  Node RL (cold): {np.mean(cold_deaths):.1f} +/- {np.std(cold_deaths):.1f}")

    # ------ 4. Node RL (warm) ------
    print(f"\n[{scenario_name}] Training Node RL (warm) ...")
    p_warm, h_warm = run_training_node_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=max_episodes_node, seed_counts=seed_counts,
        label=f'warm_{scenario_name}', out_dir=out_dir,
        terminal_reward_scale=0.0,
        doses_seq=doses_seq, bias_strength=1.5, bias_decay_episodes=30,
    )
    warm_deaths = eval_node_policy(
        p_warm, G, groups, deg_dict, params_global,
        capacity, seed_counts, n_eval=n_eval,
    )
    results.append({
        'scenario': scenario_name,
        'method': 'Node_RL (warm)',
        'deaths_mean': round(np.mean(warm_deaths), 1),
        'deaths_std':  round(np.std(warm_deaths), 1),
    })
    print(f"  Node RL (warm): {np.mean(warm_deaths):.1f} +/- {np.std(warm_deaths):.1f}")

    # ------ 5. NodeHorizon (α=3, warm) ------
    print(f"\n[{scenario_name}] Training NodeHorizon (a=3, warm) ...")
    p_hz, h_hz = run_training_node_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=max_episodes_node, seed_counts=seed_counts,
        label=f'horizon_a3_warm_{scenario_name}', out_dir=out_dir,
        terminal_reward_scale=3.0,
        doses_seq=doses_seq, bias_strength=1.0, bias_decay_episodes=20,
    )
    hz_deaths = eval_node_policy(
        p_hz, G, groups, deg_dict, params_global,
        capacity, seed_counts, n_eval=n_eval,
    )
    results.append({
        'scenario': scenario_name,
        'method': 'NodeHorizon (a=3,warm)',
        'deaths_mean': round(np.mean(hz_deaths), 1),
        'deaths_std':  round(np.std(hz_deaths), 1),
    })
    print(f"  NodeHorizon (a=3,warm): {np.mean(hz_deaths):.1f} +/- {np.std(hz_deaths):.1f}")

    # Save curves
    np.save(os.path.join(out_dir, f'hist_group_{scenario_name}.npy'), np.array(hist_group))
    np.save(os.path.join(out_dir, f'hist_cold_{scenario_name}.npy'), np.array(h_cold))
    np.save(os.path.join(out_dir, f'hist_warm_{scenario_name}.npy'), np.array(h_warm))
    np.save(os.path.join(out_dir, f'hist_hz_{scenario_name}.npy'), np.array(h_hz))

    return results


def run_comprehensive_experiment(
    out_dir: str = 'results/comprehensive',
    n_eval: int = 10,
    max_episodes_node: int = 300,
    max_episodes_group: int = 200,
) -> pd.DataFrame:

    # ============================================================
    # Scenario 1: BASELINE (current params, low epidemic severity)
    # ============================================================
    params_baseline = dict(PARAMS_NODE_RL)  # copy

    # ============================================================
    # Scenario 2: HARD (higher beta + mortality → more deaths)
    #   - beta 0.08 → 0.15: faster spread, stochasticity matters more
    #   - death rates doubled: more deaths → more room for RL to optimise
    #   - V_MAX 10 → 20: more vaccine capacity → more impactful decisions
    #   - More initial infections: epidemic starts stronger
    # ============================================================
    params_hard = dict(PARAMS_NODE_RL)
    params_hard['beta'] = 0.15
    params_hard['dX']   = 0.04   # 0.02 → 0.04
    params_hard['dY']   = 0.40   # 0.27 → 0.40
    params_hard['dZ']   = 0.08   # 0.04 → 0.08
    params_hard['V_MAX_DAILY'] = 20
    params_hard['INITIAL_INFECTED'] = 500
    params_hard['INIT_INFECTED_X']  = 250
    params_hard['INIT_INFECTED_Y']  = 100
    params_hard['INIT_INFECTED_Z']  = 150

    all_results = []

    # Run baseline
    r1 = run_scenario(
        'baseline', params_baseline,
        os.path.join(out_dir, 'baseline'),
        n_eval=n_eval,
        max_episodes_node=max_episodes_node,
        max_episodes_group=max_episodes_group,
    )
    all_results.extend(r1)

    # Run hard
    r2 = run_scenario(
        'hard', params_hard,
        os.path.join(out_dir, 'hard'),
        n_eval=n_eval,
        max_episodes_node=max_episodes_node,
        max_episodes_group=max_episodes_group,
    )
    all_results.extend(r2)

    # Summary
    df = pd.DataFrame(all_results)
    print("\n" + "=" * 72)
    print("COMPREHENSIVE EXPERIMENT RESULTS")
    print("=" * 72)
    for sc in df['scenario'].unique():
        print(f"\n--- {sc.upper()} ---")
        sub = df[df['scenario'] == sc][['method', 'deaths_mean', 'deaths_std']]
        print(sub.to_string(index=False))

    csv_path = os.path.join(out_dir, 'results_comprehensive.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved -> {csv_path}")
    return df


if __name__ == '__main__':
    run_comprehensive_experiment()
