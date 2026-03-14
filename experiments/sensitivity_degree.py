# -*- coding: utf-8 -*-
"""
experiments/sensitivity_degree.py — Sensitivity analysis over BA_M (network degree).

Sweeps BA_M ∈ [2, 4, 6, 8, 10] using the HCP scenario.
Each value is run through the full ODE → prior → RL → ODE-guided pipeline
via run_one_scenario().

Fixes vs original 3_4.py (lines 1832-1950):
  - Local params copy instead of mutating global BA_M
  - Per-run out_dir to avoid output file overwrites across iterations
  - Key renamed from 'Initial_Infected' → 'BA_M' (correct semantic)
"""

import os
import copy
import pandas as pd

from config import PARAMS_HCP
from experiments.base import run_one_scenario


VARY_LIST = [2, 4, 6, 8, 10]


def run_sensitivity_degree(base_out_dir: str = 'results/sensitivity_degree') -> pd.DataFrame:
    """
    Sweep BA_M and collect Warm_RL / Cold_RL / OC_Guided death counts.

    Parameters
    ----------
    base_out_dir : root directory; each BA_M value gets its own sub-folder

    Returns
    -------
    DataFrame with columns: BA_M, Warm_RL_Deaths, Cold_RL_Deaths, OC_Guided_Deaths
    """
    os.makedirs(base_out_dir, exist_ok=True)
    rows = []

    for ba_m in VARY_LIST:
        print(f"\n{'='*60}")
        print(f"[sensitivity_degree] BA_M = {ba_m}")
        print(f"{'='*60}")

        # local copy — never mutates global PARAMS_HCP
        params = copy.deepcopy(PARAMS_HCP)
        params['BA_M'] = ba_m

        out_dir = os.path.join(base_out_dir, f'BAM{ba_m}')
        results = run_one_scenario(
            params=params,
            scenario_tag='hcp',
            priority='Z',
            priority_order=[3, 2, 1],
            bias=[0, 0, 1],
            out_dir=out_dir,
        )

        rows.append({
            'BA_M':             ba_m,
            'Warm_RL_Deaths':   results['deaths_warm_rl'],
            'Cold_RL_Deaths':   results['deaths_cold_rl'],
            'OC_Guided_Deaths': results['deaths_ocg'],
        })
        print(f"[sensitivity_degree] DONE: BA_M = {ba_m}")

    df = pd.DataFrame(rows)

    print("\n" + "=" * 50)
    print("SENSITIVITY — DEGREE (BA_M)")
    print("=" * 50)
    print(df.to_string(index=False))

    csv_path = os.path.join(base_out_dir, 'method_comparison_results_degree.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
    return df


if __name__ == '__main__':
    run_sensitivity_degree()
