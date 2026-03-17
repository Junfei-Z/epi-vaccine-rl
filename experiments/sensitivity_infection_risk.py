# -*- coding: utf-8 -*-
"""
experiments/sensitivity_infection_risk.py — Sensitivity analysis over infection risk.

Explores two key drivers of infection risk independently:

  Part A — Transmissibility sweep (vary beta):
    beta ∈ [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    Controls overall pathogen transmissibility.  Higher beta → more infections
    per contact → harder allocation problem.

  Part B — Asymptomatic infectiousness sweep (vary wA):
    wA ∈ [0.1, 0.3, 0.5, 0.7, 0.9]  (wP and wI held fixed)
    Controls how much "hidden" transmission occurs from asymptomatic carriers
    who do not know they are infected and cannot self-isolate.
    Low wA  → most spread from visible symptomatic cases  → easier to target
    High wA → spread dominated by invisible asymptomatics → allocation harder

Force of infection reminder
---------------------------
    lambda_g = beta * sum_h [ C[g,h] * (wA*A_h + wP*P_h + wI*I_h) / N_h ]

Both beta and wA appear directly in this expression, making them the two
most natural parameters for "infection risk" sensitivity analysis.
"""

import os
import copy
import pandas as pd

from config import PARAMS_HCP
from experiments.base import run_one_scenario


# ---------------------------------------------------------------------------
# Part A: vary beta (overall transmissibility)
# ---------------------------------------------------------------------------

BETA_LIST = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]


def run_sensitivity_beta_risk(
    base_out_dir: str = 'results/sensitivity_infection_risk/beta',
) -> pd.DataFrame:
    """
    Sweep beta and collect Warm_RL / Cold_RL / OC_Guided death counts.

    Higher beta → larger epidemic → all methods face harder allocation.
    The question is whether the gap between methods grows or shrinks with
    increasing infection pressure.

    Returns
    -------
    DataFrame with columns: beta, Warm_RL_Deaths, Cold_RL_Deaths, OC_Guided_Deaths
    """
    os.makedirs(base_out_dir, exist_ok=True)
    rows = []

    for beta in BETA_LIST:
        print(f"\n{'='*60}")
        print(f"[infection_risk/beta] beta = {beta}")
        print(f"{'='*60}")

        params = copy.deepcopy(PARAMS_HCP)
        params['beta'] = beta

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
        print(f"[infection_risk/beta] DONE: beta = {beta}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("INFECTION RISK — TRANSMISSIBILITY (beta)")
    print("=" * 60)
    print(df.to_string(index=False))
    csv_path = os.path.join(base_out_dir, 'results_beta_risk.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
    return df


# ---------------------------------------------------------------------------
# Part B: vary wA (asymptomatic infectiousness)
# ---------------------------------------------------------------------------

WA_LIST = [0.1, 0.3, 0.5, 0.7, 0.9]


def run_sensitivity_wA(
    base_out_dir: str = 'results/sensitivity_infection_risk/wA',
) -> pd.DataFrame:
    """
    Sweep wA (relative infectiousness of asymptomatic carriers) and collect
    Warm_RL / Cold_RL / OC_Guided death counts.

    wA appears in the force-of-infection as:
        lambda_g += beta * C[g,h] * wA * A_h / N_h

    Higher wA → asymptomatic carriers drive more transmission → the epidemic
    is harder to control because a larger fraction of spread is "invisible"
    (asymptomatics have no symptoms to trigger isolation or targeted vaccination).

    Returns
    -------
    DataFrame with columns: wA, Warm_RL_Deaths, Cold_RL_Deaths, OC_Guided_Deaths
    """
    os.makedirs(base_out_dir, exist_ok=True)
    rows = []

    for wA in WA_LIST:
        print(f"\n{'='*60}")
        print(f"[infection_risk/wA] wA = {wA}")
        print(f"{'='*60}")

        params = copy.deepcopy(PARAMS_HCP)
        params['wA'] = wA

        tag = f'{wA:.1f}'.replace('.', 'p')
        out_dir = os.path.join(base_out_dir, f'wA{tag}')
        results = run_one_scenario(
            params=params,
            scenario_tag='hcp',
            priority='Z',
            priority_order=[3, 2, 1],
            bias=[0, 0, 1],
            out_dir=out_dir,
        )

        rows.append({
            'wA':               wA,
            'Warm_RL_Deaths':   results['deaths_warm_rl'],
            'Cold_RL_Deaths':   results['deaths_cold_rl'],
            'OC_Guided_Deaths': results['deaths_ocg'],
        })
        print(f"[infection_risk/wA] DONE: wA = {wA}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("INFECTION RISK — ASYMPTOMATIC INFECTIOUSNESS (wA)")
    print("=" * 60)
    print(df.to_string(index=False))
    csv_path = os.path.join(base_out_dir, 'results_wA_risk.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
    return df


# ---------------------------------------------------------------------------
# Run both parts
# ---------------------------------------------------------------------------

def run_sensitivity_infection_risk(
    base_out_dir: str = 'results/sensitivity_infection_risk',
) -> tuple:
    """
    Run both Part A (beta sweep) and Part B (wA sweep).

    Returns
    -------
    (df_beta, df_wA) : tuple of DataFrames
    """
    df_beta = run_sensitivity_beta_risk(
        base_out_dir=os.path.join(base_out_dir, 'beta'),
    )
    df_wA = run_sensitivity_wA(
        base_out_dir=os.path.join(base_out_dir, 'wA'),
    )
    return df_beta, df_wA


if __name__ == '__main__':
    run_sensitivity_infection_risk()
