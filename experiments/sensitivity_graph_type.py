# -*- coding: utf-8 -*-
"""
experiments/sensitivity_graph_type.py — Sensitivity analysis over degree
distribution via the non-linear preferential attachment (NLPA) exponent.

Background (Wikipedia — "Barabási–Albert model, Non-linear preferential
attachment"):

    Standard BA uses linear PA:  p_i ∝ k_i
    NLPA generalises to:         p_i ∝ k_i^α / Σ_j k_j^α

    α < 1  — sub-linear PA → stretched exponential degree distribution;
              hubs still exist but the network is more homogeneous (flatter
              degree distribution → smaller gap between high/low-degree nodes)
    α = 1  — standard BA → power-law γ ≈ 3  (baseline)
    α > 1  — super-linear PA → winner-takes-all / gelation; one or a few
              nodes accumulate almost all edges, creating extreme hubs

Epidemiological question
------------------------
How does the shape of the degree distribution (controlled by α) affect the
relative performance of the three vaccine allocation methods?

Hypotheses:
  • At α > 1 (extreme hubs), prioritising high-contact nodes (Z) should give
    the largest marginal benefit — the OC and warm-RL methods that can target
    hubs should pull further ahead of cold-RL.
  • At α < 1 (uniform network), the hub-targeting advantage shrinks; all
    three methods may converge in performance because there are fewer
    super-spreader nodes to protect.
  • The OC method operates at population level and may not adapt as well as
    the node-level RL agent when network structure changes.

Sweep
-----
    alpha_pa ∈ [0.5, 0.75, 1.0, 1.25, 1.5]

All other parameters held at PARAMS_HCP canonical values (BA_M = 4, N = 9000,
HCP priority).
"""

import os
import copy
import pandas as pd

from config import PARAMS_HCP
from experiments.base import run_one_scenario


ALPHA_PA_LIST = [0.5, 0.75, 1.0, 1.25, 1.5]


def run_sensitivity_graph_type(
    base_out_dir: str = 'results/sensitivity_graph_type',
) -> pd.DataFrame:
    """
    Sweep the NLPA exponent alpha_pa and collect Warm_RL / Cold_RL /
    OC_Guided death counts for the HCP scenario.

    Parameters
    ----------
    base_out_dir : root directory; each alpha value gets its own sub-folder

    Returns
    -------
    DataFrame with columns:
        alpha_pa, Warm_RL_Deaths, Cold_RL_Deaths, OC_Guided_Deaths
    """
    os.makedirs(base_out_dir, exist_ok=True)
    rows = []

    for alpha_pa in ALPHA_PA_LIST:
        print(f"\n{'='*60}")
        print(f"[sensitivity_graph_type] alpha_pa = {alpha_pa}")
        if alpha_pa < 1.0:
            print(f"  → sub-linear PA (homogeneous network, few dominant hubs)")
        elif alpha_pa == 1.0:
            print(f"  → standard BA (power-law γ≈3, baseline)")
        else:
            print(f"  → super-linear PA (winner-takes-all, extreme hubs)")
        print(f"{'='*60}")

        params = copy.deepcopy(PARAMS_HCP)
        params['ALPHA_PA'] = alpha_pa        # triggers build_graph_nlpa in base.py

        tag = f'{alpha_pa:.2f}'.replace('.', 'p')
        out_dir = os.path.join(base_out_dir, f'alpha{tag}')

        results = run_one_scenario(
            params=params,
            scenario_tag='hcp',
            priority='Z',
            priority_order=[3, 2, 1],
            bias=[0, 0, 1],
            out_dir=out_dir,
        )

        rows.append({
            'alpha_pa':          alpha_pa,
            'Warm_RL_Deaths':    results['deaths_warm_rl'],
            'Cold_RL_Deaths':    results['deaths_cold_rl'],
            'OC_Guided_Deaths':  results['deaths_ocg'],
        })
        print(f"[sensitivity_graph_type] DONE: alpha_pa = {alpha_pa}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("SENSITIVITY — DEGREE DISTRIBUTION (NLPA alpha_pa)")
    print("=" * 60)
    print(df.to_string(index=False))

    csv_path = os.path.join(base_out_dir, 'results_graph_type.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
    return df


if __name__ == '__main__':
    run_sensitivity_graph_type()
