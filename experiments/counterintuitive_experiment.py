# -*- coding: utf-8 -*-
"""
experiments/counterintuitive_experiment.py — Test counterintuitive vaccination priority.

Based on manuscript pages 30-31 and SI Section 7:
When asymptomatic duration is long and asymptomatic transmission is high,
low-risk individuals (X) may infect MORE people than high-risk ones (Y),
because Y becomes symptomatic quickly (short infectious period) while
X stays asymptomatic for longer (sustained transmission).

In this regime, the optimal strategy may prioritize vaccinating low-risk X
over high-risk Y — counterintuitive but rational.

2x2 factorial design:
  1. baseline:          wA=0.5, tauA=1/5  (5-day asymp period)
  2. high_wA:           wA=0.8, tauA=1/5  (high asymp transmission)
  3. long_asymp:        wA=0.5, tauA=1/14 (14-day asymp period)
  4. counterintuitive:  wA=0.8, tauA=1/14 (both: should flip priority)

Each setting compares 5 methods. We also record actual vaccination counts
per group to see if the allocation priority flips.
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


# =====================================================================
# Evaluation helpers
# =====================================================================

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


def eval_node_policy_with_counts(policy, G, groups, deg_dict, params_global,
                                  capacity, seed_counts, n_eval=10):
    """Evaluate Node RL and record vaccination counts per group."""
    deaths_list = []
    vax_counts = {1: 0, 2: 0, 3: 0}  # total vaccinated per group across episodes
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
            # Count vaccinations per group
            for node in selected:
                g = env.node_to_group[node]
                vax_counts[g] += 1
            _, _, done, _ = env.step_node_ids(selected)
        deaths_list.append(int(np.sum(env.status == D)))
    policy.train()
    # Average per episode
    avg_vax = {g: vax_counts[g] / n_eval for g in [1, 2, 3]}
    return deaths_list, avg_vax


def eval_oc_with_counts(G, groups, deg_dict, params_global, capacity,
                        seed_counts, doses_seq, n_eval=10):
    """Evaluate OC and record vaccination counts per group."""
    deaths_list = []
    vax_counts = {1: 0, 2: 0, 3: 0}
    T = len(doses_seq)
    for i in range(n_eval):
        env = _make_stochastic_env(G, groups, deg_dict, params_global,
                                   capacity, seed_counts, rng_seed=2000 + i)
        for t in range(T):
            total = max(1, int(doses_seq[t].sum()))
            shares = doses_seq[t].astype(float) / total
            # Record planned doses per group
            doses = env._project_doses(shares)
            for gi, g in enumerate(env.groups):
                vax_counts[g] += int(doses[gi])
            _, _, done, _ = env.step(shares)
            if done:
                break
        deaths_list.append(int(np.sum(env.status == D)))
    avg_vax = {g: vax_counts[g] / n_eval for g in [1, 2, 3]}
    return deaths_list, avg_vax


def eval_group_with_counts(ppo, G, groups, deg_dict, params_global,
                           capacity, seed_counts, n_eval=10):
    """Evaluate Group RL and record vaccination counts per group."""
    deaths_list = []
    vax_counts = {1: 0, 2: 0, 3: 0}
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
            doses = env._project_doses(shares)
            for gi, g in enumerate(env.groups):
                vax_counts[g] += int(doses[gi])
            state, _, done, _ = env.step(shares)
        deaths_list.append(int(np.sum(env.status == D)))
    ppo.policy.train()
    avg_vax = {g: vax_counts[g] / n_eval for g in [1, 2, 3]}
    return deaths_list, avg_vax


def _get_seed_counts(params):
    return {
        1: int(params.get('INIT_INFECTED_X', params['INITIAL_INFECTED'])),
        2: int(params.get('INIT_INFECTED_Y', 0)),
        3: int(params.get('INIT_INFECTED_Z', 0)),
    }


# =====================================================================
# Run one scenario
# =====================================================================

def run_scenario(label, params, G, groups, deg_dict, out_dir,
                 n_eval=10, max_ep_node=300, max_ep_group=200):
    os.makedirs(out_dir, exist_ok=True)
    capacity = params['V_MAX_DAILY']
    seed_counts = _get_seed_counts(params)
    params_global = to_params_global(params)

    print(f"\n{'='*62}")
    print(f"SCENARIO: {label}")
    print(f"  wA={params['wA']}, tauA={params['tauA']:.4f} "
          f"(asymp period={1/params['tauA']:.0f} days)")
    print(f"  sX={params['sX']}, sY={params['sY']}")
    print(f"{'='*62}")

    # Compute risk-dependent transmission potential (from SI eq 9)
    tauP = params['tauP']  # presymptomatic duration rate
    tauI = params['tauI']  # early infection rate
    tauA_val = params['tauA']
    wP = params['wP']
    wI = params['wI']
    wA = params['wA']
    for gname, s_val in [('X', params['sX']), ('Y', params['sY']), ('Z', params['sZ'])]:
        c = s_val * (wP / tauP + wI / tauI) + (1 - s_val) * wA / tauA_val
        print(f"  Transmission potential c_{gname} = {c:.3f}")

    # ODE
    print(f"\n  [{label}] Solving ODE ...")
    states_opt, ctrl_opt, _ = solve(params, init_pattern='hcp')
    ax, ay, az = allocations_from_solution(states_opt, ctrl_opt)
    ax, ay, az = strict_priority_window_fill(ax, ay, az, capacity, priority='Z')
    ax, ay, az = cap_to_capacity(ax, ay, az, capacity)
    doses_seq = np.stack([ax, ay, az], axis=1)

    # Print ODE allocation summary
    total_doses = doses_seq.sum(axis=0)
    print(f"  ODE total doses: X={total_doses[0]:.0f}, Y={total_doses[1]:.0f}, Z={total_doses[2]:.0f}")

    results = []

    # 1. OC-Guided
    oc_d, oc_vax = eval_oc_with_counts(
        G, groups, deg_dict, params_global, capacity,
        seed_counts, doses_seq, n_eval)
    results.append({'setting': label, 'method': 'OC_Guided',
                    'deaths_mean': round(np.mean(oc_d), 1),
                    'deaths_std': round(np.std(oc_d), 1),
                    'vax_X': round(oc_vax[1], 1),
                    'vax_Y': round(oc_vax[2], 1),
                    'vax_Z': round(oc_vax[3], 1)})
    print(f"  OC-Guided: {np.mean(oc_d):.1f} +/- {np.std(oc_d):.1f}  "
          f"[X={oc_vax[1]:.0f} Y={oc_vax[2]:.0f} Z={oc_vax[3]:.0f}]")

    # 2. Group RL (cold)
    print(f"\n  [{label}] Training Group RL (cold) ...")
    ppo_cold, _ = run_training(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        prior_path=None, max_episodes=max_ep_group,
        label=f'group_cold_{label}', out_dir=out_dir, seed_counts=seed_counts,
    )
    gc_d, gc_vax = eval_group_with_counts(
        ppo_cold, G, groups, deg_dict, params_global,
        capacity, seed_counts, n_eval)
    results.append({'setting': label, 'method': 'Group_RL (cold)',
                    'deaths_mean': round(np.mean(gc_d), 1),
                    'deaths_std': round(np.std(gc_d), 1),
                    'vax_X': round(gc_vax[1], 1),
                    'vax_Y': round(gc_vax[2], 1),
                    'vax_Z': round(gc_vax[3], 1)})
    print(f"  Group RL (cold): {np.mean(gc_d):.1f} +/- {np.std(gc_d):.1f}  "
          f"[X={gc_vax[1]:.0f} Y={gc_vax[2]:.0f} Z={gc_vax[3]:.0f}]")

    # 3. Group RL (warm)
    print(f"\n  [{label}] Training Group RL (warm) ...")
    doses_path = os.path.join(out_dir, 'ode_doses.npy')
    prior_path = os.path.join(out_dir, 'ode_prior.npy')
    np.save(doses_path, doses_seq)
    build_feasible_prior_from_doses(
        doses_path=doses_path,
        args=(G, groups, deg_dict, params_global, capacity, seed_counts),
        label=label, bias=[0, 0, 1], save_path=prior_path,
    )
    ppo_warm, _ = run_training(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        prior_path=prior_path, max_episodes=max_ep_group,
        label=f'group_warm_{label}', out_dir=out_dir, seed_counts=seed_counts,
    )
    gw_d, gw_vax = eval_group_with_counts(
        ppo_warm, G, groups, deg_dict, params_global,
        capacity, seed_counts, n_eval)
    results.append({'setting': label, 'method': 'Group_RL (warm)',
                    'deaths_mean': round(np.mean(gw_d), 1),
                    'deaths_std': round(np.std(gw_d), 1),
                    'vax_X': round(gw_vax[1], 1),
                    'vax_Y': round(gw_vax[2], 1),
                    'vax_Z': round(gw_vax[3], 1)})
    print(f"  Group RL (warm): {np.mean(gw_d):.1f} +/- {np.std(gw_d):.1f}  "
          f"[X={gw_vax[1]:.0f} Y={gw_vax[2]:.0f} Z={gw_vax[3]:.0f}]")

    # 4. Node RL (cold)
    print(f"\n  [{label}] Training Node RL (cold) ...")
    p_node, _ = run_training_node_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=max_ep_node, seed_counts=seed_counts,
        label=f'node_cold_{label}', out_dir=out_dir,
    )
    nc_d, nc_vax = eval_node_policy_with_counts(
        p_node, G, groups, deg_dict, params_global,
        capacity, seed_counts, n_eval)
    results.append({'setting': label, 'method': 'Node_RL (cold)',
                    'deaths_mean': round(np.mean(nc_d), 1),
                    'deaths_std': round(np.std(nc_d), 1),
                    'vax_X': round(nc_vax[1], 1),
                    'vax_Y': round(nc_vax[2], 1),
                    'vax_Z': round(nc_vax[3], 1)})
    print(f"  Node RL (cold): {np.mean(nc_d):.1f} +/- {np.std(nc_d):.1f}  "
          f"[X={nc_vax[1]:.0f} Y={nc_vax[2]:.0f} Z={nc_vax[3]:.0f}]")

    # 5. NodeHorizon (α=3, warm)
    print(f"\n  [{label}] Training NodeHorizon (a=3, warm) ...")
    p_hz, _ = run_training_node_rl(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        max_episodes=max_ep_node, seed_counts=seed_counts,
        label=f'horizon_{label}', out_dir=out_dir,
        terminal_reward_scale=3.0,
        doses_seq=doses_seq, bias_strength=1.0, bias_decay_episodes=20,
    )
    hz_d, hz_vax = eval_node_policy_with_counts(
        p_hz, G, groups, deg_dict, params_global,
        capacity, seed_counts, n_eval)
    results.append({'setting': label, 'method': 'NodeHorizon (a=3,warm)',
                    'deaths_mean': round(np.mean(hz_d), 1),
                    'deaths_std': round(np.std(hz_d), 1),
                    'vax_X': round(hz_vax[1], 1),
                    'vax_Y': round(hz_vax[2], 1),
                    'vax_Z': round(hz_vax[3], 1)})
    print(f"  NodeHorizon: {np.mean(hz_d):.1f} +/- {np.std(hz_d):.1f}  "
          f"[X={hz_vax[1]:.0f} Y={hz_vax[2]:.0f} Z={hz_vax[3]:.0f}]")

    return results


# =====================================================================
# Main experiment
# =====================================================================

def run_counterintuitive_experiment(
    out_dir='results/counterintuitive',
    n_eval=10,
    max_ep_node=300,
    max_ep_group=200,
):
    params_base = dict(PARAMS_NODE_RL)

    # Build graph once
    G, groups, deg_dict = build_graph_and_groups(
        n=params_base['N'], m=params_base['BA_M'], seed=params_base['SEED'],
        high_risk_prob=params_base['HIGH_RISK_PROB'],
        alpha_std=params_base['ALPHA_STD'],
    )

    # 2x2 factorial: wA x tauA
    settings = [
        {'label': 'baseline',
         'wA': 0.5, 'tauA': 1/5},     # 5-day asymp, normal transmission
        {'label': 'high_wA',
         'wA': 0.8, 'tauA': 1/5},     # 5-day asymp, high transmission
        {'label': 'long_asymp',
         'wA': 0.5, 'tauA': 1/14},    # 14-day asymp, normal transmission
        {'label': 'counterintuitive',
         'wA': 0.8, 'tauA': 1/14},    # 14-day asymp, high transmission
    ]

    all_results = []
    for s in settings:
        params = dict(params_base)
        params['wA'] = s['wA']
        params['tauA'] = s['tauA']
        sub_dir = os.path.join(out_dir, s['label'])
        r = run_scenario(s['label'], params, G, groups, deg_dict, sub_dir,
                         n_eval, max_ep_node, max_ep_group)
        all_results.extend(r)

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("COUNTERINTUITIVE VACCINATION PRIORITY EXPERIMENT")
    print("=" * 80)
    for s in df['setting'].unique():
        print(f"\n--- {s.upper()} ---")
        sub = df[df['setting'] == s]
        print(sub[['method', 'deaths_mean', 'deaths_std',
                    'vax_X', 'vax_Y', 'vax_Z']].to_string(index=False))

    csv_path = os.path.join(out_dir, 'results_counterintuitive.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved -> {csv_path}")
    return df


if __name__ == '__main__':
    run_counterintuitive_experiment()
