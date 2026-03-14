# -*- coding: utf-8 -*-
"""
experiments/sensitivity_infected.py — Sensitivity analysis over INITIAL_INFECTED.

Sweeps INITIAL_INFECTED ∈ [400, 500, 600, 700] using the HCP scenario.
Each value is run through the full ODE → prior → RL → ODE-guided pipeline
via run_one_scenario().

Fixes vs original 3_4.py (lines 1954-2071):
  - Local params copy instead of relying on global INITIAL_INFECTED
  - Per-run out_dir to avoid output file overwrites across iterations
  - Result key renamed 'Initial_Infected' → 'INITIAL_INFECTED' (consistent
    with the params dict key convention used everywhere else)
"""

import os
import copy
import pandas as pd

from config import PARAMS_HCP
from experiments.base import run_one_scenario


VARY_LIST = [400, 500, 600, 700]


def run_sensitivity_infected(base_out_dir: str = 'results/sensitivity_infected') -> pd.DataFrame:
    """
    Sweep INITIAL_INFECTED and collect Warm_RL / Cold_RL / OC_Guided death counts.

    Parameters
    ----------
    base_out_dir : root directory; each INITIAL_INFECTED value gets its own sub-folder

    Returns
    -------
    DataFrame with columns: INITIAL_INFECTED, Warm_RL_Deaths, Cold_RL_Deaths, OC_Guided_Deaths
    """
    os.makedirs(base_out_dir, exist_ok=True)
    rows = []

    for n_inf in VARY_LIST:
        print(f"\n{'='*60}")
        print(f"[sensitivity_infected] INITIAL_INFECTED = {n_inf}")
        print(f"{'='*60}")

        # local copy — never mutates global PARAMS_HCP
        params = copy.deepcopy(PARAMS_HCP)
        params['INITIAL_INFECTED'] = n_inf

        out_dir = os.path.join(base_out_dir, f'INF{n_inf}')
        results = run_one_scenario(
            params=params,
            scenario_tag='hcp',
            priority='Z',
            priority_order=[3, 2, 1],
            bias=[0, 0, 1],
            out_dir=out_dir,
        )

        rows.append({
            'INITIAL_INFECTED':  n_inf,
            'Warm_RL_Deaths':    results['deaths_warm_rl'],
            'Cold_RL_Deaths':    results['deaths_cold_rl'],
            'OC_Guided_Deaths':  results['deaths_ocg'],
        })
        print(f"[sensitivity_infected] DONE: INITIAL_INFECTED = {n_inf}")

    df = pd.DataFrame(rows)

    print("\n" + "=" * 50)
    print("SENSITIVITY — INITIAL INFECTED")
    print("=" * 50)
    print(df.to_string(index=False))

    csv_path = os.path.join(base_out_dir, 'method_comparison_results_initial_infected.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
    return df


if __name__ == '__main__':
    run_sensitivity_infected()
