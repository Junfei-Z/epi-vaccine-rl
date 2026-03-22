# -*- coding: utf-8 -*-
"""
experiments/node_rl_experiment.py — Verify node-aware RL on HCP setting.

Compares three methods on N=1000, V_MAX=5, HCP priority:
  1. OC-Guided  — ODE optimal control (open-loop, group-level)
  2. Warm RL    — standard PPO with Dirichlet group shares (warm-start)
  3. Node RL    — NodeScoringPolicy: learned per-node scoring via PPO

Key question: does node-level information give Node RL a genuine edge
over group-level Warm RL and ODE in a stochastic epidemic?
"""

import os
import numpy as np
import pandas as pd
import torch

from config import PARAMS_NODE_RL, to_params_global, D, S, P, A, I as I_STATE
from graph import build_graph_and_groups
from ode_solver import solve, allocations_from_solution
from allocation import strict_priority_window_fill, cap_to_capacity
from env import make_env_from_graph
from prior import build_feasible_prior_from_doses
from simulate import simulate_with_ode_doses
from rl.train import run_training, run_training_node_rl


def run_node_rl_experiment(
    out_dir: str = 'results/node_rl',
    n_eval:  int = 5,
) -> pd.DataFrame:
    """
    Run OC-Guided, Warm RL (group-level), and Node RL on HCP setting.

    Parameters
    ----------
    out_dir : directory for saved models and CSV
    n_eval  : number of stochastic episodes for final evaluation

    Returns
    -------
    DataFrame with columns: method, deaths_mean, deaths_std
    """
    os.makedirs(out_dir, exist_ok=True)
    params      = PARAMS_NODE_RL
    tag         = 'node_exp'
    capacity    = params['V_MAX_DAILY']
    seed_counts = {
        1: int(params.get('INIT_INFECTED_X', params['INITIAL_INFECTED'])),
        2: int(params.get('INIT_INFECTED_Y', 0)),
        3: int(params.get('INIT_INFECTED_Z', 0)),
    }

    # ------------------------------------------------------------------ #
    # Build graph                                                          #
    # ------------------------------------------------------------------ #
    print("\n[node_rl] Building graph (N=5000) ...")
    G, groups, deg_dict = build_graph_and_groups(
        n=params['N'], m=params['BA_M'], seed=params['SEED'],
        high_risk_prob=params['HIGH_RISK_PROB'], alpha_std=params['ALPHA_STD'],
    )
    params_global = to_params_global(params)

    # ------------------------------------------------------------------ #
    # 1. ODE optimal control                                               #
    # ------------------------------------------------------------------ #
    print("\n[node_rl] Solving ODE optimal control ...")
    states_opt, ctrl_opt, _ = solve(params, init_pattern='hcp')
    ax, ay, az = allocations_from_solution(states_opt, ctrl_opt)
    ax, ay, az = strict_priority_window_fill(ax, ay, az, capacity, priority='Z')
    ax, ay, az = cap_to_capacity(ax, ay, az, capacity)
    doses_seq   = np.stack([ax, ay, az], axis=1)

    oc_deaths = []
    for _ in range(n_eval):
        env_oc, _, _, _, _ = make_env_from_graph(
            G=G, groups=groups, deg_dict=deg_dict,
            params_global=params_global, capacity_daily=capacity,
            seed_counts=seed_counts, deterministic=False,
        )
        env_oc.reset(seed_counts=seed_counts)
        _, _, d = simulate_with_ode_doses(
            env_oc, doses_seq, priority_order=[3, 2, 1],
            seed_counts=seed_counts,
        )
        oc_deaths.append(d)
    print(f"[node_rl] OC-Guided:  {np.mean(oc_deaths):.1f} ± {np.std(oc_deaths):.1f}")

    # ------------------------------------------------------------------ #
    # 2. Warm RL (group-level, standard PPO with ODE prior)               #
    # ------------------------------------------------------------------ #
    print("\n[node_rl] Building ODE prior for Warm RL ...")
    doses_path = os.path.join(out_dir, f'ode_doses_{tag}.npy')
    prior_path = os.path.join(out_dir, f'ode_prior_{tag}.npy')
    np.save(doses_path, doses_seq)
    build_feasible_prior_from_doses(
        doses_path=doses_path,
        args=(G, groups, deg_dict, params_global, capacity, seed_counts),
        label=tag, bias=[0, 0, 1], save_path=prior_path,
    )

    print("\n[node_rl] Training Warm RL (group-level PPO) ...")
    ppo_warm, _ = run_training(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        prior_path=prior_path, max_episodes=200,
        label=f'{tag}_warm', out_dir=out_dir, seed_counts=seed_counts,
    )

    warm_deaths = []
    for _ in range(n_eval):
        env_w, _, _, _, _ = make_env_from_graph(
            G=G, groups=groups, deg_dict=deg_dict,
            params_global=params_global, capacity_daily=capacity,
            seed_counts=seed_counts, deterministic=False,
        )
        state = env_w.reset(seed_counts=seed_counts)
        done  = False
        while not done:
            with torch.no_grad():
                s_t  = torch.from_numpy(state).float()
                dist = ppo_warm.policy.dist(s_t)
                act  = torch.clamp(dist.mean, min=1e-6)
                shares = (act / act.sum()).numpy()
            state, _, done, _ = env_w.step(shares)
        warm_deaths.append(int(np.sum(env_w.status == D)))
    print(f"[node_rl] Warm RL:    {np.mean(warm_deaths):.1f} ± {np.std(warm_deaths):.1f}")

    # ------------------------------------------------------------------ #
    # 3. Node RL (NodeScoringPolicy)                                       #
    # ------------------------------------------------------------------ #
    print("\n[node_rl] Training Node RL (node scoring PPO) ...")
    node_policy, _ = run_training_node_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=300, label=tag, out_dir=out_dir,
        seed_counts=seed_counts,
    )

    node_deaths = []
    for _ in range(n_eval):
        env_n, _, _, _, _ = make_env_from_graph(
            G=G, groups=groups, deg_dict=deg_dict,
            params_global=params_global, capacity_daily=capacity,
            seed_counts=seed_counts, deterministic=False,
        )
        env_n.reset(seed_counts=seed_counts)
        done = False
        while not done:
            g_state = torch.from_numpy(env_n.obs_with_pressure()).float()
            s_ids, feats = env_n.node_features()
            if len(s_ids) == 0:
                _, _, done, _ = env_n.step_node_ids([])
                continue
            f_t = torch.from_numpy(feats).float()
            with torch.no_grad():
                idxs, _ = node_policy.select(
                    g_state, f_t, capacity, deterministic=True,
                )
            selected = [s_ids[i] for i in idxs.tolist()]
            _, _, done, _ = env_n.step_node_ids(selected)
        node_deaths.append(int(np.sum(env_n.status == D)))
    print(f"[node_rl] Node RL:    {np.mean(node_deaths):.1f} ± {np.std(node_deaths):.1f}")

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    rows = [
        {'method': 'OC_Guided',
         'deaths_mean': round(np.mean(oc_deaths),   1),
         'deaths_std':  round(np.std(oc_deaths),    1)},
        {'method': 'Warm_RL (group-level)',
         'deaths_mean': round(np.mean(warm_deaths), 1),
         'deaths_std':  round(np.std(warm_deaths),  1)},
        {'method': 'Node_RL',
         'deaths_mean': round(np.mean(node_deaths), 1),
         'deaths_std':  round(np.std(node_deaths),  1)},
    ]
    df = pd.DataFrame(rows)
    print("\n" + "=" * 55)
    print("NODE-AWARE RL EXPERIMENT  (N=5000, HCP, V_MAX=10)")
    print("=" * 55)
    print(df.to_string(index=False))

    csv_path = os.path.join(out_dir, 'results_node_rl.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
    return df


if __name__ == '__main__':
    run_node_rl_experiment()
