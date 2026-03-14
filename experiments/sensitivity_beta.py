# -*- coding: utf-8 -*-
"""
experiments/sensitivity_beta.py — Sensitivity analysis over beta (transmissibility).

Sweeps beta ∈ [0.06, 0.08, 0.10, 0.12, 0.15] using the HCP scenario.
Each value is run through the full ODE → prior → RL → ODE-guided pipeline
via run_one_scenario().

Fixes vs original 3_4.py (lines 2076-2193):
  - Removed unused import: `from networkx.algorithms.tree.mst import ALGORITHMS`
  - Local params copy instead of relying on global beta
  - Per-run out_dir to avoid output file overwrites across iterations
  - Result key renamed 'Initial_Infected' → 'beta' (correct semantic)
"""

import os
import copy
import pandas as pd

from config import PARAMS_HCP
from experiments.base import run_one_scenario


VARY_LIST = [0.06, 0.08, 0.10, 0.12, 0.15]


def run_sensitivity_beta(base_out_dir: str = 'results/sensitivity_beta') -> pd.DataFrame:
    """
    Sweep beta and collect Warm_RL / Cold_RL / OC_Guided death counts.

    Parameters
    ----------
    base_out_dir : root directory; each beta value gets its own sub-folder

    Returns
    -------
    DataFrame with columns: beta, Warm_RL_Deaths, Cold_RL_Deaths, OC_Guided_Deaths
    """
    os.makedirs(base_out_dir, exist_ok=True)
    rows = []

    for beta in VARY_LIST:
        print(f"\n{'='*60}")
        print(f"[sensitivity_beta] beta = {beta}")
        print(f"{'='*60}")

        # local copy — never mutates global PARAMS_HCP
        params = copy.deepcopy(PARAMS_HCP)
        params['beta'] = beta

        # tag sub-folder with 2-decimal representation to avoid dot in path
        tag = f'{beta:.2f}'.replace('.', 'p')
        out_dir = os.path.join(base_out_dir, f'beta{tag}')
        results = run_one_scenario(
            params=params,
            scenario_tag='hcp',
            priority='Z',
            priority_order=[3, 2, 1],
            bias=[0, 0, 1],
            out_dir=out_dir,
        )

        rows.append({
            'beta':             beta,
            'Warm_RL_Deaths':   results['deaths_warm_rl'],
            'Cold_RL_Deaths':   results['deaths_cold_rl'],
            'OC_Guided_Deaths': results['deaths_ocg'],
        })
        print(f"[sensitivity_beta] DONE: beta = {beta}")

    df = pd.DataFrame(rows)

    print("\n" + "=" * 50)
    print("SENSITIVITY — TRANSMISSIBILITY (beta)")
    print("=" * 50)
    print(df.to_string(index=False))

    csv_path = os.path.join(base_out_dir, 'method_comparison_results_beta.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
    return df


if __name__ == '__main__':
    run_sensitivity_beta()
