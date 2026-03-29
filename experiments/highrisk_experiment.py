# -*- coding: utf-8 -*-
"""
experiments/highrisk_experiment.py — High-risk scenario where degree-greedy is suboptimal.

Hypothesis: when high-risk group (Y) has very high mortality, the env's built-in
degree-greedy node selection becomes suboptimal. A low-degree Y node surrounded
by infectious neighbours should be vaccinated BEFORE a high-degree X node in a
clean neighbourhood. Node RL can see this local pressure — OC+degree-greedy cannot.

Scenarios:
  moderate_risk: pY=0.35, dY=0.50 (Y death prob ~17.5%)
  extreme_risk:  pY=0.50, dY=0.70 (Y death prob ~35%)

Each scenario compares:
  1. OC-Guided (degree-greedy within groups)
  2. OC-Guided (random within groups) — ablation
  3. Group RL (warm, degree-greedy)
  4. Node RL (cold)
  5. NodeHorizon (α=3, warm)
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


def _make_random_select_env(G, groups, deg_dict, params_global,
                            capacity, seed_counts, rng_seed):
    """Create env where _choose_to_vaccinate uses RANDOM selection instead of degree-greedy."""
    env = _make_stochastic_env(G, groups, deg_dict, params_global,
                               capacity, seed_counts, rng_seed)

    # monkey-patch: random selection within group instead of degree-first
    def _choose_random(self_env, group, k):
        if k <= 0:
            return []
        cand = [n for n in self_env.group_nodes[group] if self_env.status[n] == S]
        if len(cand) <= k:
            return cand
        return list(self_env.rng.choice(cand, size=k, replace=False))

    import types
    env._choose_to_vaccinate = types.MethodType(_choose_random, env)
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
                       seed_counts, doses_seq, n_eval=10, random_select=False):
    """Evaluate OC with degree-greedy or random within-group selection."""
    deaths = []
    T = len(doses_seq)
    for i in range(n_eval):
        if random_select:
            env = _make_random_select_env(G, groups, deg_dict, params_global,
                                          capacity, seed_counts, rng_seed=2000 + i)
        else:
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
    print(f"  pY={params['pY']}, dY={params['dY']}  (Y death prob ~{params['pY']*params['dY']*100:.1f}%)")
    print(f"  pX={params['pX']}, dX={params['dX']}  (X death prob ~{params['pX']*params['dX']*100:.1f}%)")
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

    # 1. OC-Guided (degree-greedy)
    print(f"\n[{scenario_name}] Evaluating OC-Guided (degree-greedy) ...")
    oc_deg = eval_oc_stochastic(G, groups, deg_dict, params_global, capacity,
                                seed_counts, doses_seq, n_eval=n_eval, random_select=False)
    results.append({'scenario': scenario_name, 'method': 'OC (degree-greedy)',
                    'deaths_mean': round(np.mean(oc_deg), 1),
                    'deaths_std': round(np.std(oc_deg), 1)})
    print(f"  OC (degree-greedy): {np.mean(oc_deg):.1f} +/- {np.std(oc_deg):.1f}")

    # 2. OC-Guided (random selection) — ablation
    print(f"\n[{scenario_name}] Evaluating OC-Guided (random select) ...")
    oc_rnd = eval_oc_stochastic(G, groups, deg_dict, params_global, capacity,
                                seed_counts, doses_seq, n_eval=n_eval, random_select=True)
    results.append({'scenario': scenario_name, 'method': 'OC (random select)',
                    'deaths_mean': round(np.mean(oc_rnd), 1),
                    'deaths_std': round(np.std(oc_rnd), 1)})
    print(f"  OC (random select): {np.mean(oc_rnd):.1f} +/- {np.std(oc_rnd):.1f}")

    # 3. Group RL (warm)
    print(f"\n[{scenario_name}] Training Warm Group RL ...")
    doses_path = os.path.join(out_dir, 'ode_doses.npy')
    prior_path = os.path.join(out_dir, 'ode_prior.npy')
    np.save(doses_path, doses_seq)
    build_feasible_prior_from_doses(
        doses_path=doses_path,
        args=(G, groups, deg_dict, params_global, capacity, seed_counts),
        label=scenario_name, bias=[0, 0, 1], save_path=prior_path,
    )
    ppo_group, _ = run_training(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        prior_path=prior_path, max_episodes=max_episodes_group,
        label=f'group_{scenario_name}', out_dir=out_dir, seed_counts=seed_counts,
    )
    group_deaths = eval_group_policy(ppo_group, G, groups, deg_dict, params_global,
                                     capacity, seed_counts, n_eval=n_eval)
    results.append({'scenario': scenario_name, 'method': 'Group_RL (warm)',
                    'deaths_mean': round(np.mean(group_deaths), 1),
                    'deaths_std': round(np.std(group_deaths), 1)})
    print(f"  Group RL (warm): {np.mean(group_deaths):.1f} +/- {np.std(group_deaths):.1f}")

    # 4. Node RL (cold)
    print(f"\n[{scenario_name}] Training Node RL (cold) ...")
    p_cold, h_cold = run_training_node_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=max_episodes_node, seed_counts=seed_counts,
        label=f'cold_{scenario_name}', out_dir=out_dir,
        terminal_reward_scale=0.0,
    )
    cold_deaths = eval_node_policy(p_cold, G, groups, deg_dict, params_global,
                                   capacity, seed_counts, n_eval=n_eval)
    results.append({'scenario': scenario_name, 'method': 'Node_RL (cold)',
                    'deaths_mean': round(np.mean(cold_deaths), 1),
                    'deaths_std': round(np.std(cold_deaths), 1)})
    print(f"  Node RL (cold): {np.mean(cold_deaths):.1f} +/- {np.std(cold_deaths):.1f}")

    # 5. NodeHorizon (α=3, warm)
    print(f"\n[{scenario_name}] Training NodeHorizon (a=3, warm) ...")
    p_hz, h_hz = run_training_node_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=max_episodes_node, seed_counts=seed_counts,
        label=f'horizon_{scenario_name}', out_dir=out_dir,
        terminal_reward_scale=3.0,
        doses_seq=doses_seq, bias_strength=1.0, bias_decay_episodes=20,
    )
    hz_deaths = eval_node_policy(p_hz, G, groups, deg_dict, params_global,
                                 capacity, seed_counts, n_eval=n_eval)
    results.append({'scenario': scenario_name, 'method': 'NodeHorizon (a=3,warm)',
                    'deaths_mean': round(np.mean(hz_deaths), 1),
                    'deaths_std': round(np.std(hz_deaths), 1)})
    print(f"  NodeHorizon (a=3,warm): {np.mean(hz_deaths):.1f} +/- {np.std(hz_deaths):.1f}")

    # Save curves
    np.save(os.path.join(out_dir, f'hist_cold_{scenario_name}.npy'), np.array(h_cold))
    np.save(os.path.join(out_dir, f'hist_hz_{scenario_name}.npy'), np.array(h_hz))

    return results


def run_highrisk_experiment(
    out_dir: str = 'results/highrisk',
    n_eval: int = 10,
    max_episodes_node: int = 300,
    max_episodes_group: int = 200,
) -> pd.DataFrame:

    # Scenario 1: moderate risk — pY=0.35, dY=0.50
    params_moderate = dict(PARAMS_NODE_RL)
    params_moderate['pY'] = 0.35
    params_moderate['dY'] = 0.50
    params_moderate['sY'] = 0.9   # higher symptomatic rate for Y

    # Scenario 2: extreme risk — pY=0.50, dY=0.70
    params_extreme = dict(PARAMS_NODE_RL)
    params_extreme['pY'] = 0.50
    params_extreme['dY'] = 0.70
    params_extreme['sY'] = 0.95

    all_results = []

    r1 = run_scenario('moderate_risk', params_moderate,
                      os.path.join(out_dir, 'moderate'),
                      n_eval=n_eval,
                      max_episodes_node=max_episodes_node,
                      max_episodes_group=max_episodes_group)
    all_results.extend(r1)

    r2 = run_scenario('extreme_risk', params_extreme,
                      os.path.join(out_dir, 'extreme'),
                      n_eval=n_eval,
                      max_episodes_node=max_episodes_node,
                      max_episodes_group=max_episodes_group)
    all_results.extend(r2)

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 72)
    print("HIGH-RISK EXPERIMENT RESULTS")
    print("=" * 72)
    for sc in df['scenario'].unique():
        print(f"\n--- {sc.upper()} ---")
        sub = df[df['scenario'] == sc][['method', 'deaths_mean', 'deaths_std']]
        print(sub.to_string(index=False))

    csv_path = os.path.join(out_dir, 'results_highrisk.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved -> {csv_path}")
    return df


if __name__ == '__main__':
    run_highrisk_experiment()
