# -*- coding: utf-8 -*-
"""
experiments/base.py — Single-scenario full pipeline runner.

Orchestrates the complete ODE → prior → RL → ODE-guided pipeline for one
scenario (HCP or HRP), then prints a comparison table.

run_one_scenario()  — end-to-end pipeline for one param set
main()              — runs both HCP and HRP using canonical config params

Warm-RL tuning notes
--------------------
If warm RL underperforms cold RL, consider these levers (passed to run_training):

  Prior influence (reduce if warm RL is stuck near ODE solution):
    prior_weight_warm  : 0.8  → 0.5   (default 1.6 — often too strong)
    prior_decay_warm   : 0.98 → 0.97  (faster decay = RL takes over sooner)
    prior_alpha_warm   : 30.0 → 20.0  (softer prior Dirichlet)

  Training length (increase if warm RL stops too early):
    max_episodes       : 400  (default 300)
    min_episodes       : 80   (default 40)
    patience           : 8    (default 4)
    rel_std_thresh     : 0.03 (default 0.05 — stricter = harder to stop early)
    window_size        : 50   (default 40)

  Exploration schedule (warm phase should explore MORE, not less):
    warm_mean_episodes : 25   (default 12 — spend more time escaping prior)
    sample_temp_warm   : 3.0  (in model.py — higher temp = flatter Dirichlet)
    sample_temp_cold   : 1.5  (lower temp later = sharper exploitation)

  Update frequency:
    episodes_per_update: 10   (default 12 — more frequent updates)
"""

import os
import numpy as np
import pandas as pd

from config import PARAMS_HCP, PARAMS_HRP, to_params_global
from graph import build_graph_and_groups, build_graph_nlpa
from ode_solver import solve, allocations_from_solution
from allocation import strict_priority_window_fill, cap_to_capacity, to_simplex
from env import make_env_from_graph
from prior import build_feasible_prior_from_doses, simulate_episode_prior
from simulate import simulate_with_ode_doses, evaluate_and_export
from plot import (plot_stacked, plot_convergence, summarize_daily,
                  plot_inf_neighbor, share_inf_neighbor)
from rl.train import run_training


# ---------------------------------------------------------------------------
# Full pipeline for one scenario
# ---------------------------------------------------------------------------

def run_one_scenario(
    params: dict,
    scenario_tag: str,
    priority: str,
    priority_order: list,
    bias: list,
    out_dir: str = 'results',
    # --- warm-RL hyperparams (tune these if warm < cold) ---
    warm_max_episodes: int = 300,
    warm_episodes_per_update: int = 12,
    warm_mean_episodes: int = 12,
    warm_window_size: int = 40,
    warm_rel_std_thresh: float = 0.05,
    warm_patience: int = 4,
    warm_min_episodes: int = 40,
    # --- cold-RL hyperparams ---
    cold_max_episodes: int = 300,
    cold_episodes_per_update: int = 18,
    cold_window_size: int = 40,
    cold_rel_std_thresh: float = 0.08,
    cold_patience: int = 3,
    cold_min_episodes: int = 40,
) -> dict:
    """
    Run the full ODE → prior → warm-RL → cold-RL → ODE-guided pipeline.

    Steps
    -----
    1. Build BA contact network from params
    2. Solve ODE optimal control (CasADi/IPOPT)
    3. Post-process ODE doses: priority window fill → cap → simplex
    4. Build feasible prior from ODE doses
    5. Train warm-start PPO (prior-guided)
    6. Train cold-start PPO (no prior)
    7. Run ODE-guided simulation
    8. Export CSVs + plots

    Parameters
    ----------
    params         : scenario parameter dict (e.g. PARAMS_HCP)
    scenario_tag   : short label, e.g. 'hcp' or 'hrp'
    priority       : 'Y' or 'Z' — which group gets the priority window
    priority_order : e.g. [3,2,1] for HCP or [2,3,1] for HRP
    bias           : day-0 bias for prior blending, e.g. [0,0,1] or [0,1,0]
    out_dir        : directory for all output files
    warm_*/cold_*  : hyperparameters for warm/cold training runs

    Returns
    -------
    dict with keys:
        deaths_warm_rl, deaths_cold_rl, deaths_ocg,
        ppo_warm, ppo_cold, hist_warm, hist_cold
    """
    os.makedirs(out_dir, exist_ok=True)
    tag = scenario_tag  # shorthand

    # ------------------------------------------------------------------ #
    # 1. Build graph                                                       #
    # ------------------------------------------------------------------ #
    if 'ALPHA_PA' in params:
        G, groups, deg_dict = build_graph_nlpa(
            n=params['N'], m=params['BA_M'], alpha_pa=params['ALPHA_PA'],
            seed=params['SEED'], high_risk_prob=params['HIGH_RISK_PROB'],
            alpha_std=params['ALPHA_STD'],
        )
    else:
        G, groups, deg_dict = build_graph_and_groups(
            n=params['N'], m=params['BA_M'], seed=params['SEED'],
            high_risk_prob=params['HIGH_RISK_PROB'], alpha_std=params['ALPHA_STD'],
        )
    params_global = to_params_global(params)
    capacity      = params['V_MAX_DAILY']

    seed_counts = {
        1: int(params.get('INIT_INFECTED_X', params.get('INITIAL_INFECTED', 0))),
        2: int(params.get('INIT_INFECTED_Y', 0)),
        3: int(params.get('INIT_INFECTED_Z', 0)),
    }
    # for HCP, all seeds in group 1 (X) unless explicitly specified
    if 'INIT_INFECTED_X' not in params and 'INIT_INFECTED_Y' not in params:
        seed_counts = {1: int(params['INITIAL_INFECTED']), 2: 0, 3: 0}

    env_args = (G, groups, deg_dict, params_global, capacity, seed_counts)

    # ------------------------------------------------------------------ #
    # 2. ODE optimal control                                               #
    # ------------------------------------------------------------------ #
    print(f"\n[{tag}] Solving ODE optimal control ...")
    states_opt, ctrl_opt, _ = solve(params, init_pattern=tag)
    ax, ay, az = allocations_from_solution(states_opt, ctrl_opt)

    # ------------------------------------------------------------------ #
    # 3. Post-process ODE doses                                            #
    # ------------------------------------------------------------------ #
    ax_pf, ay_pf, az_pf = strict_priority_window_fill(
        ax, ay, az, capacity, priority=priority,
    )
    ax_f, ay_f, az_f = cap_to_capacity(ax_pf, ay_pf, az_pf, capacity)
    doses = np.column_stack([ax_f, ay_f, az_f]).astype(np.float32)
    shares_ode = to_simplex(doses)

    doses_path  = os.path.join(out_dir, f'ode_warm_start_doses_{tag}.npy')
    policy_path = os.path.join(out_dir, f'ode_warm_start_policy_{tag}.npy')
    np.save(doses_path,  doses)
    np.save(policy_path, shares_ode)

    # plot ODE allocation
    import pandas as pd
    df_ode_days = pd.DataFrame({
        'day': np.arange(len(ax_f)),
        'X': ax_f, 'Y': ay_f, 'Z': az_f,
        'V_MAX_DAILY': capacity,
    })
    plot_stacked(df_ode_days, f'{tag.upper()} ODE Allocation')

    # ------------------------------------------------------------------ #
    # 4. Build feasible prior                                              #
    # ------------------------------------------------------------------ #
    print(f"[{tag}] Building feasible prior ...")
    prior, prior_path = build_feasible_prior_from_doses(
        doses_path=doses_path,
        args=env_args,
        label=tag,
        window_K=8,
        bias=bias,
        eta0=0.5,
        save_path=os.path.join(out_dir, f'ode_feasible_prior_{tag}.npy'),
    )
    simulate_episode_prior(f"{tag.upper()} prior", env_args, prior)

    # ------------------------------------------------------------------ #
    # 5. Warm-start PPO                                                    #
    # ------------------------------------------------------------------ #
    print(f"[{tag}] Training warm-start PPO ...")
    ppo_warm, hist_warm = run_training(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        prior_path=prior_path,
        max_episodes=warm_max_episodes,
        reward_scale=1.0,
        episodes_per_update=warm_episodes_per_update,
        warm_mean_episodes=warm_mean_episodes,
        window_size=warm_window_size,
        rel_std_thresh=warm_rel_std_thresh,
        patience=warm_patience,
        min_episodes=warm_min_episodes,
        seed_counts=seed_counts,
        substeps=1, dt=1.0,
        label=f'{tag}_warm',
        out_dir=out_dir,
    )

    # ------------------------------------------------------------------ #
    # 6. Cold-start PPO                                                    #
    # ------------------------------------------------------------------ #
    print(f"[{tag}] Training cold-start PPO ...")
    ppo_cold, hist_cold = run_training(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        prior_path=None,
        max_episodes=cold_max_episodes,
        reward_scale=1.0,
        episodes_per_update=cold_episodes_per_update,
        warm_mean_episodes=8,
        window_size=cold_window_size,
        rel_std_thresh=cold_rel_std_thresh,
        patience=cold_patience,
        min_episodes=cold_min_episodes,
        seed_counts=seed_counts,
        substeps=1, dt=1.0,
        label=f'{tag}_cold',
        out_dir=out_dir,
    )

    # ------------------------------------------------------------------ #
    # 7. Evaluate RL agents                                                #
    # ------------------------------------------------------------------ #
    df_nodes_warm, df_days_warm, deaths_warm = evaluate_and_export(
        agent=ppo_warm, G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        label=f'{tag}_warm', seed_counts=seed_counts,
        out_dir=out_dir,
    )
    df_nodes_cold, df_days_cold, deaths_cold = evaluate_and_export(
        agent=ppo_cold, G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity,
        label=f'{tag}_cold', seed_counts=seed_counts,
        out_dir=out_dir,
    )

    print(f"[{tag}] Warm-RL final deaths: {deaths_warm}")
    print(f"[{tag}] Cold-RL final deaths: {deaths_cold}")
    plot_convergence(hist_warm, hist_cold)
    summarize_daily(df_nodes_warm, T_horizon=params['T_HORIZON'])
    summarize_daily(df_nodes_cold, T_horizon=params['T_HORIZON'])
    plot_inf_neighbor(df_nodes_warm, f'{tag.upper()} Warm-RL: infected neighbour share')

    # ------------------------------------------------------------------ #
    # 8. ODE-guided simulation                                             #
    # ------------------------------------------------------------------ #
    print(f"[{tag}] Running ODE-guided simulation ...")
    env_ocg, _, _, _, _ = make_env_from_graph(
        G, groups, deg_dict, params_global, capacity,
        reward_scale=1.0, seed_counts=seed_counts,
        substeps=1, dt=1.0, deterministic=True,
    )
    doses_ocg = np.load(doses_path)
    df_nodes_ocg, df_days_ocg, deaths_ocg = simulate_with_ode_doses(
        env=env_ocg,
        doses_seq=doses_ocg,
        priority_order=priority_order,
        seed_counts=seed_counts,
    )
    print(f"[{tag}] OC-guided final deaths: {deaths_ocg}")

    # save ODE-guided CSVs
    df_nodes_ocg.to_csv(os.path.join(out_dir, f'ode_nodes_{tag}.csv'), index=False)
    df_days_ocg.to_csv( os.path.join(out_dir, f'ode_days_{tag}.csv'),  index=False)
    plot_stacked(df_days_ocg, f'{tag.upper()} ODE-Guided Allocation')
    plot_inf_neighbor(df_nodes_ocg, f'{tag.upper()} ODE-Guided: infected neighbour share')

    return {
        'deaths_warm_rl': deaths_warm,
        'deaths_cold_rl': deaths_cold,
        'deaths_ocg':     deaths_ocg,
        'ppo_warm':       ppo_warm,
        'ppo_cold':       ppo_cold,
        'hist_warm':      hist_warm,
        'hist_cold':      hist_cold,
    }


# ---------------------------------------------------------------------------
# Main: run both HCP and HRP
# ---------------------------------------------------------------------------

def main(out_dir: str = 'results'):
    """Run HCP and HRP scenarios and print a comparison table."""

    results_hcp = run_one_scenario(
        params=PARAMS_HCP,
        scenario_tag='hcp',
        priority='Z',
        priority_order=[3, 2, 1],
        bias=[0, 0, 1],
        out_dir=out_dir,
        warm_max_episodes=500,
        warm_episodes_per_update=10,
        warm_mean_episodes=30,
        warm_window_size=50,
        warm_rel_std_thresh=0.03,
        warm_patience=10,
        warm_min_episodes=100,
    )

    results_hrp = run_one_scenario(
        params=PARAMS_HRP,
        scenario_tag='hrp',
        priority='Y',
        priority_order=[2, 3, 1],
        bias=[0, 1, 0],
        out_dir=out_dir,
        warm_max_episodes=500,
        warm_episodes_per_update=10,
        warm_mean_episodes=30,
        warm_window_size=50,
        warm_rel_std_thresh=0.03,
        warm_patience=10,
        warm_min_episodes=100,
    )

    # comparison table
    rows = []
    for tag, res in [('HCP', results_hcp), ('HRP', results_hrp)]:
        rows.append({
            'Scenario':     tag,
            'Warm_RL':      res['deaths_warm_rl'],
            'Cold_RL':      res['deaths_cold_rl'],
            'OC_Guided':    res['deaths_ocg'],
        })
    df = pd.DataFrame(rows)
    print("\n" + "="*50)
    print("FINAL COMPARISON TABLE")
    print("="*50)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(out_dir, 'comparison_base.csv'), index=False)
    return df


# ---------------------------------------------------------------------------
# Multi-seed evaluation
# ---------------------------------------------------------------------------

def run_multiseed(
    n_seeds: int = 5,
    seeds: list = None,
    out_dir: str = 'results/multiseed',
) -> pd.DataFrame:
    """
    Run HCP and HRP across multiple random seeds and report mean ± std.

    Each seed generates a different BA network and initial infection pattern,
    giving a statistically meaningful comparison between the three methods.

    Parameters
    ----------
    n_seeds : number of seeds to run (ignored if `seeds` is given)
    seeds   : explicit list of integer seeds; defaults to [42,123,456,789,999]
    out_dir : root output directory; each seed gets its own sub-folder

    Returns
    -------
    DataFrame with columns:
        Scenario, Method, Mean_Deaths, Std_Deaths, Seeds
    and one row per (scenario, method) combination.
    Also saves per-seed raw results to out_dir/multiseed_raw.csv
    and summary to out_dir/multiseed_summary.csv.
    """
    import copy

    if seeds is None:
        seeds = [42, 123, 456, 789, 999][:n_seeds]

    os.makedirs(out_dir, exist_ok=True)
    raw_rows = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"[multiseed] SEED = {seed}")
        print(f"{'='*60}")

        seed_dir = os.path.join(out_dir, f'seed{seed}')

        # HCP
        params_hcp = copy.deepcopy(PARAMS_HCP)
        params_hcp['SEED'] = seed
        res_hcp = run_one_scenario(
            params=params_hcp,
            scenario_tag='hcp',
            priority='Z',
            priority_order=[3, 2, 1],
            bias=[0, 0, 1],
            out_dir=os.path.join(seed_dir, 'hcp'),
            warm_max_episodes=500,
            warm_episodes_per_update=10,
            warm_mean_episodes=30,
            warm_window_size=50,
            warm_rel_std_thresh=0.03,
            warm_patience=10,
            warm_min_episodes=100,
        )
        raw_rows.append({'Seed': seed, 'Scenario': 'HCP',
                         'Warm_RL': res_hcp['deaths_warm_rl'],
                         'Cold_RL': res_hcp['deaths_cold_rl'],
                         'OC_Guided': res_hcp['deaths_ocg']})

        # HRP
        params_hrp = copy.deepcopy(PARAMS_HRP)
        params_hrp['SEED'] = seed
        res_hrp = run_one_scenario(
            params=params_hrp,
            scenario_tag='hrp',
            priority='Y',
            priority_order=[2, 3, 1],
            bias=[0, 1, 0],
            out_dir=os.path.join(seed_dir, 'hrp'),
            warm_max_episodes=500,
            warm_episodes_per_update=10,
            warm_mean_episodes=30,
            warm_window_size=50,
            warm_rel_std_thresh=0.03,
            warm_patience=10,
            warm_min_episodes=100,
        )
        raw_rows.append({'Seed': seed, 'Scenario': 'HRP',
                         'Warm_RL': res_hrp['deaths_warm_rl'],
                         'Cold_RL': res_hrp['deaths_cold_rl'],
                         'OC_Guided': res_hrp['deaths_ocg']})

    df_raw = pd.DataFrame(raw_rows)
    df_raw.to_csv(os.path.join(out_dir, 'multiseed_raw.csv'), index=False)

    # summary: mean ± std per (scenario, method)
    summary_rows = []
    for scenario in ['HCP', 'HRP']:
        sub = df_raw[df_raw['Scenario'] == scenario]
        for method in ['Warm_RL', 'Cold_RL', 'OC_Guided']:
            vals = sub[method].values
            summary_rows.append({
                'Scenario':    scenario,
                'Method':      method,
                'Mean_Deaths': round(vals.mean(), 2),
                'Std_Deaths':  round(vals.std(),  2),
                'Seeds':       str(seeds),
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_dir, 'multiseed_summary.csv'), index=False)

    print("\n" + "="*60)
    print("MULTI-SEED SUMMARY  (mean ± std)")
    print("="*60)
    for scenario in ['HCP', 'HRP']:
        print(f"\n  {scenario}")
        sub = df_summary[df_summary['Scenario'] == scenario]
        for _, row in sub.iterrows():
            print(f"    {row['Method']:12s}  {row['Mean_Deaths']:.1f} ± {row['Std_Deaths']:.1f}")

    print("\nRaw per-seed results:")
    print(df_raw.to_string(index=False))

    return df_summary


if __name__ == '__main__':
    main()
