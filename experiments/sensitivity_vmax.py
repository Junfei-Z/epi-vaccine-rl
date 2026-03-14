# -*- coding: utf-8 -*-
"""
experiments/sensitivity_vmax.py — Sensitivity analysis over V_MAX_DAILY.

Sweeps maximum daily vaccine doses ∈ [10, 20, 40, 60, 80] using the HCP scenario.
Each value is run through the full ODE → prior → RL → ODE-guided pipeline
via run_one_scenario().
"""

import os
import copy
import pandas as pd

from config import PARAMS_HCP
from experiments.base import run_one_scenario


VARY_LIST = [10, 20, 40, 60, 80]


def run_sensitivity_vmax(base_out_dir: str = 'results/sensitivity_vmax') -> pd.DataFrame:
    """
    Sweep V_MAX_DAILY and collect Warm_RL / Cold_RL / OC_Guided death counts.

    Parameters
    ----------
    base_out_dir : root directory; each V_MAX_DAILY value gets its own sub-folder

    Returns
    -------
    DataFrame with columns: V_MAX_DAILY, Warm_RL_Deaths, Cold_RL_Deaths, OC_Guided_Deaths
    """
    os.makedirs(base_out_dir, exist_ok=True)
    rows = []

    for vmax in VARY_LIST:
        print(f"\n{'='*60}")
        print(f"[sensitivity_vmax] V_MAX_DAILY = {vmax}")
        print(f"{'='*60}")

        params = copy.deepcopy(PARAMS_HCP)
        params['V_MAX_DAILY'] = vmax

        out_dir = os.path.join(base_out_dir, f'VMAX{vmax}')
        results = run_one_scenario(
            params=params,
            scenario_tag='hcp',
            priority='Z',
            priority_order=[3, 2, 1],
            bias=[0, 0, 1],
            out_dir=out_dir,
        )

        rows.append({
            'V_MAX_DAILY':      vmax,
            'Warm_RL_Deaths':   results['deaths_warm_rl'],
            'Cold_RL_Deaths':   results['deaths_cold_rl'],
            'OC_Guided_Deaths': results['deaths_ocg'],
        })
        print(f"[sensitivity_vmax] DONE: V_MAX_DAILY = {vmax}")

    df = pd.DataFrame(rows)

    print("\n" + "=" * 50)
    print("SENSITIVITY — MAX DAILY DOSES (V_MAX_DAILY)")
    print("=" * 50)
    print(df.to_string(index=False))

    csv_path = os.path.join(base_out_dir, 'method_comparison_results_vmax.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
    return df


if __name__ == '__main__':
    run_sensitivity_vmax()
