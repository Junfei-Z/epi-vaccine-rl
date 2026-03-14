# -*- coding: utf-8 -*-
"""
experiments/sensitivity_groupsize.py — Sensitivity analysis over N (population size).

Sweeps N ∈ [5000, 7000, 9000, 11000, 13000] using the HCP scenario.
Each value is run through the full ODE → prior → RL → ODE-guided pipeline
via run_one_scenario().
"""

import os
import copy
import pandas as pd

from config import PARAMS_HCP
from experiments.base import run_one_scenario


VARY_LIST = [5000, 7000, 9000, 11000, 13000]


def run_sensitivity_groupsize(base_out_dir: str = 'results/sensitivity_groupsize') -> pd.DataFrame:
    """
    Sweep N (total population) and collect Warm_RL / Cold_RL / OC_Guided death counts.

    Parameters
    ----------
    base_out_dir : root directory; each N value gets its own sub-folder

    Returns
    -------
    DataFrame with columns: N, Warm_RL_Deaths, Cold_RL_Deaths, OC_Guided_Deaths
    """
    os.makedirs(base_out_dir, exist_ok=True)
    rows = []

    for n in VARY_LIST:
        print(f"\n{'='*60}")
        print(f"[sensitivity_groupsize] N = {n}")
        print(f"{'='*60}")

        params = copy.deepcopy(PARAMS_HCP)
        params['N'] = n

        out_dir = os.path.join(base_out_dir, f'N{n}')
        results = run_one_scenario(
            params=params,
            scenario_tag='hcp',
            priority='Z',
            priority_order=[3, 2, 1],
            bias=[0, 0, 1],
            out_dir=out_dir,
        )

        rows.append({
            'N':                n,
            'Warm_RL_Deaths':   results['deaths_warm_rl'],
            'Cold_RL_Deaths':   results['deaths_cold_rl'],
            'OC_Guided_Deaths': results['deaths_ocg'],
        })
        print(f"[sensitivity_groupsize] DONE: N = {n}")

    df = pd.DataFrame(rows)

    print("\n" + "=" * 50)
    print("SENSITIVITY — POPULATION SIZE (N)")
    print("=" * 50)
    print(df.to_string(index=False))

    csv_path = os.path.join(base_out_dir, 'method_comparison_results_groupsize.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
    return df


if __name__ == '__main__':
    run_sensitivity_groupsize()
