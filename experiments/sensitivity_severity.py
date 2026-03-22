# -*- coding: utf-8 -*-
"""
experiments/sensitivity_severity.py — Sensitivity analysis over disease
severity for the high-risk group Y (hospitalisation and death probability).

Motivation
----------
In the baseline model:
    pY = 0.20  (prob of progressing from mild L to hospitalised H)
    dY = 0.27  (prob of dying given hospitalised H)

These parameters control how lethal the disease is for the high-risk group.
Increasing them simulates a more virulent pathogen or a more vulnerable
population (e.g. older age threshold, immunocompromised individuals).

Epidemiological question
------------------------
As high-risk group Y becomes more severely affected, does HRP (prioritise Y)
pull further ahead of HCP (prioritise Z hubs)?  At some severity level, does
the optimal strategy switch from hub-targeting to direct protection of the
most vulnerable?

Sweep
-----
We scale pY and dY together by a severity multiplier, keeping X and Z fixed:

    severity level 1 (baseline): pY=0.20, dY=0.27
    severity level 2:            pY=0.30, dY=0.37
    severity level 3:            pY=0.40, dY=0.47
    severity level 4:            pY=0.50, dY=0.57

Both HCP and HRP strategies are run at each level so we can directly compare
which priority rule benefits most from the change.
"""

import os
import copy
import pandas as pd

from config import PARAMS_HCP
from experiments.base import run_one_scenario


SEVERITY_LIST = [
    {'pY': 0.20, 'dY': 0.27, 'label': 'baseline'},
    {'pY': 0.30, 'dY': 0.37, 'label': 'moderate'},
    {'pY': 0.40, 'dY': 0.47, 'label': 'severe'},
    {'pY': 0.50, 'dY': 0.57, 'label': 'critical'},
]


def run_sensitivity_severity(
    base_out_dir: str = 'results/sensitivity_severity',
) -> pd.DataFrame:
    """
    Sweep disease severity (pY, dY) for the high-risk group and collect
    Warm_RL / Cold_RL / OC_Guided death counts for both HCP and HRP.

    Parameters
    ----------
    base_out_dir : root directory; each severity level gets its own sub-folder

    Returns
    -------
    DataFrame with columns:
        pY, dY, label,
        HCP_Warm_RL, HCP_Cold_RL, HCP_OC_Guided,
        HRP_Warm_RL, HRP_Cold_RL, HRP_OC_Guided
    """
    os.makedirs(base_out_dir, exist_ok=True)
    rows = []

    for sev in SEVERITY_LIST:
        pY, dY, label = sev['pY'], sev['dY'], sev['label']

        print(f"\n{'='*60}")
        print(f"[sensitivity_severity] pY={pY}  dY={dY}  ({label})")
        print(f"{'='*60}")

        tag = f'pY{str(pY).replace(".","p")}_dY{str(dY).replace(".","p")}'

        # --- HCP: prioritise high-contact hubs (Z) ---
        params_hcp = copy.deepcopy(PARAMS_HCP)
        params_hcp['pY'] = pY
        params_hcp['dY'] = dY

        print(f"\n  [HCP] Prioritise high-contact (Z) ...")
        res_hcp = run_one_scenario(
            params=params_hcp,
            scenario_tag='hcp',
            priority='Z',
            priority_order=[3, 2, 1],
            bias=[0, 0, 1],
            out_dir=os.path.join(base_out_dir, f'{tag}_hcp'),
        )

        # --- HRP: prioritise high-risk group (Y) ---
        params_hrp = copy.deepcopy(PARAMS_HCP)
        params_hrp['pY'] = pY
        params_hrp['dY'] = dY

        print(f"\n  [HRP] Prioritise high-risk (Y) ...")
        res_hrp = run_one_scenario(
            params=params_hrp,
            scenario_tag='hrp',
            priority='Y',
            priority_order=[2, 3, 1],
            bias=[0, 1, 0],
            out_dir=os.path.join(base_out_dir, f'{tag}_hrp'),
        )

        rows.append({
            'pY':           pY,
            'dY':           dY,
            'label':        label,
            'HCP_Warm_RL':  res_hcp['deaths_warm_rl'],
            'HCP_Cold_RL':  res_hcp['deaths_cold_rl'],
            'HCP_OC_Guided': res_hcp['deaths_ocg'],
            'HRP_Warm_RL':  res_hrp['deaths_warm_rl'],
            'HRP_Cold_RL':  res_hrp['deaths_cold_rl'],
            'HRP_OC_Guided': res_hrp['deaths_ocg'],
        })
        print(f"[sensitivity_severity] DONE: pY={pY}, dY={dY}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("SENSITIVITY — DISEASE SEVERITY (pY, dY)")
    print("=" * 60)
    print(df.to_string(index=False))

    csv_path = os.path.join(base_out_dir, 'results_severity.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
    return df


if __name__ == '__main__':
    run_sensitivity_severity()
