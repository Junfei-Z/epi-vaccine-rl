# -*- coding: utf-8 -*-
"""
experiments/nodehorizon_experiment.py — NodeHorizon: node RL + terminal reward.

Tests whether adding a terminal death penalty (aligning RL with the global
D(T) minimisation objective) improves node-level RL performance.

Configurations tested:
  1. OC-Guided                         — ODE optimal control baseline
  2. Node RL (cold, no terminal)       — baseline node RL
  3. NodeHorizon (α=1)                 — per-step + 1× terminal deaths
  4. NodeHorizon (α=3)                 — per-step + 3× terminal deaths
  5. NodeHorizon (α=5)                 — per-step + 5× terminal deaths
  6. NodeHorizon (α=3, warm)           — best α + OC warm-start
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
    """Create a stochastic env with a specific RNG seed."""
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
    """Evaluate a NodeScoringPolicy with stochastic env, different seeds."""
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
    """Evaluate OC-Guided with same stochastic dynamics as RL."""
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


def run_nodehorizon_experiment(
    out_dir: str = 'results/nodehorizon',
    n_eval: int = 10,
    max_episodes: int = 300,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    params      = PARAMS_NODE_RL
    capacity    = params['V_MAX_DAILY']
    seed_counts = {
        1: int(params.get('INIT_INFECTED_X', params['INITIAL_INFECTED'])),
        2: int(params.get('INIT_INFECTED_Y', 0)),
        3: int(params.get('INIT_INFECTED_Z', 0)),
    }

    # ------------------------------------------------------------------ #
    # Build graph & ODE solution                                           #
    # ------------------------------------------------------------------ #
    print("\n[horizon] Building graph (N=5000) ...")
    G, groups, deg_dict = build_graph_and_groups(
        n=params['N'], m=params['BA_M'], seed=params['SEED'],
        high_risk_prob=params['HIGH_RISK_PROB'], alpha_std=params['ALPHA_STD'],
    )
    params_global = to_params_global(params)

    print("[horizon] Solving ODE ...")
    states_opt, ctrl_opt, _ = solve(params, init_pattern='hcp')
    ax, ay, az = allocations_from_solution(states_opt, ctrl_opt)
    ax, ay, az = strict_priority_window_fill(ax, ay, az, capacity, priority='Z')
    ax, ay, az = cap_to_capacity(ax, ay, az, capacity)
    doses_seq  = np.stack([ax, ay, az], axis=1)

    results = []

    # ------------------------------------------------------------------ #
    # 1. OC-Guided baseline                                                #
    # ------------------------------------------------------------------ #
    oc_deaths = eval_oc_stochastic(
        G, groups, deg_dict, params_global, capacity,
        seed_counts, doses_seq, n_eval=n_eval,
    )
    results.append({
        'method': 'OC_Guided',
        'deaths_mean': round(np.mean(oc_deaths), 1),
        'deaths_std':  round(np.std(oc_deaths), 1),
    })
    print(f"[horizon] OC-Guided: {np.mean(oc_deaths):.1f} +/- {np.std(oc_deaths):.1f}")

    # ------------------------------------------------------------------ #
    # 2. Node RL cold (no terminal) — baseline                             #
    # ------------------------------------------------------------------ #
    print("\n[horizon] Training Node RL (cold, no terminal) ...")
    p0, h0 = run_training_node_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=max_episodes, seed_counts=seed_counts,
        label='cold_noterm', out_dir=out_dir,
        terminal_reward_scale=0.0,
    )
    d0 = eval_node_policy(p0, G, groups, deg_dict, params_global,
                          capacity, seed_counts, n_eval)
    results.append({
        'method': 'Node_RL (cold)',
        'deaths_mean': round(np.mean(d0), 1),
        'deaths_std':  round(np.std(d0), 1),
    })
    print(f"[horizon] Node RL (cold): {np.mean(d0):.1f} +/- {np.std(d0):.1f}")

    # ------------------------------------------------------------------ #
    # 3-5. NodeHorizon with different α values                             #
    # ------------------------------------------------------------------ #
    for alpha in [1.0, 3.0, 5.0]:
        tag = f'horizon_a{alpha:.0f}'
        print(f"\n[horizon] Training NodeHorizon (alpha={alpha}) ...")
        p, h = run_training_node_rl(
            G=G, groups=groups, deg_dict=deg_dict,
            params_global=params_global, capacity_daily=capacity,
            max_episodes=max_episodes, seed_counts=seed_counts,
            label=tag, out_dir=out_dir,
            terminal_reward_scale=alpha,
        )
        d = eval_node_policy(p, G, groups, deg_dict, params_global,
                             capacity, seed_counts, n_eval)
        results.append({
            'method': f'NodeHorizon (a={alpha:.0f})',
            'deaths_mean': round(np.mean(d), 1),
            'deaths_std':  round(np.std(d), 1),
        })
        np.save(os.path.join(out_dir, f'hist_{tag}.npy'), np.array(h))
        print(f"[horizon] NodeHorizon (a={alpha}): {np.mean(d):.1f} +/- {np.std(d):.1f}")

    # ------------------------------------------------------------------ #
    # 6. NodeHorizon (α=3, warm) — combining both improvements             #
    # ------------------------------------------------------------------ #
    print("\n[horizon] Training NodeHorizon (a=3, warm) ...")
    pw, hw = run_training_node_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=max_episodes, seed_counts=seed_counts,
        label='horizon_a3_warm', out_dir=out_dir,
        terminal_reward_scale=3.0,
        doses_seq=doses_seq,
        bias_strength=1.0,
        bias_decay_episodes=20,
    )
    dw = eval_node_policy(pw, G, groups, deg_dict, params_global,
                          capacity, seed_counts, n_eval)
    results.append({
        'method': 'NodeHorizon (a=3, warm)',
        'deaths_mean': round(np.mean(dw), 1),
        'deaths_std':  round(np.std(dw), 1),
    })
    np.save(os.path.join(out_dir, 'hist_horizon_a3_warm.npy'), np.array(hw))
    print(f"[horizon] NodeHorizon (a=3, warm): {np.mean(dw):.1f} +/- {np.std(dw):.1f}")

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(results)
    print("\n" + "=" * 62)
    print("NODEHORIZON EXPERIMENT  (N=5000, HCP, V_MAX=10)")
    print("=" * 62)
    print(df.to_string(index=False))

    csv_path = os.path.join(out_dir, 'results_nodehorizon.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved -> {csv_path}")
    return df


if __name__ == '__main__':
    run_nodehorizon_experiment()
