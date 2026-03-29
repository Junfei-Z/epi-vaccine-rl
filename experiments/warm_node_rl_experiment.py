# -*- coding: utf-8 -*-
"""
experiments/warm_node_rl_experiment.py — Compare cold vs warm node-level RL.

Compares four methods on N=5000, V_MAX=10, HCP priority:
  1. OC-Guided  — ODE optimal control (open-loop, group-level)
  2. Node RL (cold)  — NodeScoringPolicy, random init
  3. Node RL (warm)  — NodeScoringPolicy, pre-trained via OC behavioral cloning
  4. Warm RL (group) — standard Dirichlet PPO with ODE prior (baseline)

Key question: does OC warm-start help node-level RL converge faster and
reach a better final policy than cold-start node RL?
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
    """Create a stochastic env with a specific RNG seed for fair comparison."""
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
    """Evaluate a NodeScoringPolicy over n_eval stochastic episodes with different seeds."""
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
    """Evaluate OC-Guided using the SAME stochastic dynamics as RL."""
    from allocation import allocate_by_priority
    from simulate import vaccinate_by_priority

    deaths = []
    T = len(doses_seq)
    for i in range(n_eval):
        env = _make_stochastic_env(G, groups, deg_dict, params_global,
                                   capacity, seed_counts, rng_seed=2000 + i)
        for t in range(T):
            # apply OC doses via env.step (uses stochastic disease progression)
            total = max(1, int(doses_seq[t].sum()))
            shares = doses_seq[t].astype(float) / total
            _, _, done, _ = env.step(shares)
            if done:
                break
        deaths.append(int(np.sum(env.status == D)))
    return deaths


def run_warm_node_rl_experiment(
    out_dir: str = 'results/warm_node_rl',
    n_eval: int = 10,
    max_episodes_node: int = 300,
    bias_strength: float = 2.0,
    bias_decay_episodes: int = 50,
) -> pd.DataFrame:
    """
    Run the four-way comparison experiment.

    Parameters
    ----------
    out_dir             : directory for saved models and CSV
    n_eval              : stochastic evaluation episodes per method
    max_episodes_node   : max PPO episodes for node RL methods
    bias_strength       : initial OC score bias scale for warm-start
    bias_decay_episodes : episodes over which bias decays to 0

    Returns
    -------
    DataFrame with columns: method, deaths_mean, deaths_std
    """
    os.makedirs(out_dir, exist_ok=True)
    params      = PARAMS_NODE_RL
    capacity    = params['V_MAX_DAILY']
    seed_counts = {
        1: int(params.get('INIT_INFECTED_X', params['INITIAL_INFECTED'])),
        2: int(params.get('INIT_INFECTED_Y', 0)),
        3: int(params.get('INIT_INFECTED_Z', 0)),
    }

    # ------------------------------------------------------------------ #
    # Build graph                                                          #
    # ------------------------------------------------------------------ #
    print("\n[warm_exp] Building graph (N=5000) ...")
    G, groups, deg_dict = build_graph_and_groups(
        n=params['N'], m=params['BA_M'], seed=params['SEED'],
        high_risk_prob=params['HIGH_RISK_PROB'], alpha_std=params['ALPHA_STD'],
    )
    params_global = to_params_global(params)

    # ------------------------------------------------------------------ #
    # 1. ODE optimal control                                               #
    # ------------------------------------------------------------------ #
    print("\n[warm_exp] Solving ODE optimal control ...")
    states_opt, ctrl_opt, _ = solve(params, init_pattern='hcp')
    ax, ay, az = allocations_from_solution(states_opt, ctrl_opt)
    ax, ay, az = strict_priority_window_fill(ax, ay, az, capacity, priority='Z')
    ax, ay, az = cap_to_capacity(ax, ay, az, capacity)
    doses_seq  = np.stack([ax, ay, az], axis=1)

    # Fair stochastic evaluation: same RNG seeds for all methods
    oc_deaths = eval_oc_stochastic(
        G, groups, deg_dict, params_global, capacity,
        seed_counts, doses_seq, n_eval=n_eval,
    )
    print(f"[warm_exp] OC-Guided:  {np.mean(oc_deaths):.1f} +/- {np.std(oc_deaths):.1f}")

    # ------------------------------------------------------------------ #
    # 2. Node RL — cold start                                              #
    # ------------------------------------------------------------------ #
    print("\n[warm_exp] Training Node RL (cold start) ...")
    policy_cold, hist_cold = run_training_node_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=max_episodes_node,
        seed_counts=seed_counts,
        label='cold', out_dir=out_dir,
    )

    cold_deaths = eval_node_policy(
        policy_cold, G, groups, deg_dict, params_global,
        capacity, seed_counts, n_eval=n_eval,
    )
    print(f"[warm_exp] Node RL (cold): {np.mean(cold_deaths):.1f} +/- {np.std(cold_deaths):.1f}")

    # ------------------------------------------------------------------ #
    # 3. Node RL — warm start (OC behavioral cloning + PPO)                #
    # ------------------------------------------------------------------ #
    print("\n[warm_exp] Training Node RL (warm start from OC) ...")
    policy_warm, hist_warm = run_training_node_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=max_episodes_node,
        seed_counts=seed_counts,
        label='warm', out_dir=out_dir,
        # warm-start: OC score bias
        doses_seq=doses_seq,
        bias_strength=bias_strength,
        bias_decay_episodes=bias_decay_episodes,
    )

    warm_deaths = eval_node_policy(
        policy_warm, G, groups, deg_dict, params_global,
        capacity, seed_counts, n_eval=n_eval,
    )
    print(f"[warm_exp] Node RL (warm): {np.mean(warm_deaths):.1f} +/- {np.std(warm_deaths):.1f}")

    # ------------------------------------------------------------------ #
    # 4. Warm RL — group-level baseline                                    #
    # ------------------------------------------------------------------ #
    print("\n[warm_exp] Training Warm RL (group-level PPO) ...")
    doses_path = os.path.join(out_dir, 'ode_doses.npy')
    prior_path = os.path.join(out_dir, 'ode_prior.npy')
    np.save(doses_path, doses_seq)
    build_feasible_prior_from_doses(
        doses_path=doses_path,
        args=(G, groups, deg_dict, params_global, capacity, seed_counts),
        label='warm_exp', bias=[0, 0, 1], save_path=prior_path,
    )

    ppo_group, _ = run_training(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        prior_path=prior_path, max_episodes=200,
        label='group_warm', out_dir=out_dir, seed_counts=seed_counts,
    )

    group_deaths = []
    for i in range(n_eval):
        env_g = _make_stochastic_env(G, groups, deg_dict, params_global,
                                     capacity, seed_counts, rng_seed=2000 + i)
        state = env_g._obs()
        done = False
        while not done:
            with torch.no_grad():
                s_t = torch.from_numpy(state).float()
                dist = ppo_group.policy.dist(s_t)
                act = torch.clamp(dist.mean, min=1e-6)
                shares = (act / act.sum()).numpy()
            state, _, done, _ = env_g.step(shares)
        group_deaths.append(int(np.sum(env_g.status == D)))
    print(f"[warm_exp] Group RL (warm): {np.mean(group_deaths):.1f} +/- {np.std(group_deaths):.1f}")

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    rows = [
        {'method': 'OC_Guided',
         'deaths_mean': round(np.mean(oc_deaths), 1),
         'deaths_std':  round(np.std(oc_deaths), 1)},
        {'method': 'Node_RL (cold)',
         'deaths_mean': round(np.mean(cold_deaths), 1),
         'deaths_std':  round(np.std(cold_deaths), 1)},
        {'method': 'Node_RL (warm)',
         'deaths_mean': round(np.mean(warm_deaths), 1),
         'deaths_std':  round(np.std(warm_deaths), 1)},
        {'method': 'Group_RL (warm)',
         'deaths_mean': round(np.mean(group_deaths), 1),
         'deaths_std':  round(np.std(group_deaths), 1)},
    ]
    df = pd.DataFrame(rows)

    # save learning curves for convergence comparison
    np.save(os.path.join(out_dir, 'hist_cold.npy'), np.array(hist_cold))
    np.save(os.path.join(out_dir, 'hist_warm.npy'), np.array(hist_warm))

    print("\n" + "=" * 62)
    print("WARM-START NODE RL EXPERIMENT  (N=5000, HCP, V_MAX=10)")
    print("=" * 62)
    print(df.to_string(index=False))

    csv_path = os.path.join(out_dir, 'results_warm_node_rl.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved -> {csv_path}")
    return df


if __name__ == '__main__':
    run_warm_node_rl_experiment()
