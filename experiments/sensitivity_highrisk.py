# -*- coding: utf-8 -*-
"""
experiments/sensitivity_highrisk.py — Sensitivity analysis over high-risk
population size (high_risk_prob).

Motivation
----------
In the baseline model, high_risk_prob = 0.17 reflects the US 65+ population
share (~18%, US Census Bureau 2023).  A node is assigned to the high-risk
group Y with this probability, independently of its network degree.

If the at-risk age threshold is lowered (population "younging" of risk), the
high-risk share grows:

    65+  → ~18%  (baseline)
    55+  → ~28%  (add ~10% from 55-64 cohort)
    50+  → ~35%
    45+  → ~43%  (add 45-64 cohort: 24.6%)

Epidemiological question
------------------------
As the high-risk population grows, do the three allocation methods diverge?
Specifically:
  • HRP (prioritise Y) should gain more value as Y becomes larger.
  • The gap between HRP and HCP may widen or narrow depending on whether
    the larger Y group also becomes harder to protect within the dose budget.

Sweep
-----
    high_risk_prob ∈ [0.17, 0.25, 0.32, 0.40]

All other parameters held at PARAMS_HCP canonical values.
Both HCP (prioritise Z) and HRP (prioritise Y) strategies are run for each
value so we can see which priority rule benefits most from a larger Y group.
"""

import os
import copy
import pandas as pd

from config import PARAMS_HCP
from experiments.base import run_one_scenario


# Approximate US age-threshold mapping:
#   0.17 → 65+  (baseline)
#   0.25 → ~58+
#   0.32 → ~52+
#   0.40 → ~45+
HIGH_RISK_PROB_LIST = [0.17, 0.25, 0.32, 0.40]

AGE_LABEL = {
    0.17: '65+ (baseline, ~17%)',
    0.25: '~58+  (~25%)',
    0.32: '~52+  (~32%)',
    0.40: '~45+  (~40%)',
}


def run_sensitivity_highrisk(
    base_out_dir: str = 'results/sensitivity_highrisk',
) -> pd.DataFrame:
    """
    Sweep high_risk_prob and collect deaths for both HCP and HRP strategies.

    For each value of high_risk_prob we run:
      • HCP scenario (prioritise high-contact Z group)
      • HRP scenario (prioritise high-risk Y group)

    This lets us see how the optimal priority rule changes as the high-risk
    population grows.

    Parameters
    ----------
    base_out_dir : root directory; each prob value gets its own sub-folder

    Returns
    -------
    DataFrame with columns:
        high_risk_prob, age_threshold,
        HCP_Warm_RL, HCP_Cold_RL, HCP_OC_Guided,
        HRP_Warm_RL, HRP_Cold_RL, HRP_OC_Guided
    """
    os.makedirs(base_out_dir, exist_ok=True)
    rows = []

    for hrp in HIGH_RISK_PROB_LIST:
        print(f"\n{'='*60}")
        print(f"[sensitivity_highrisk] high_risk_prob = {hrp}  "
              f"({AGE_LABEL[hrp]})")
        print(f"{'='*60}")

        tag = f'{hrp:.2f}'.replace('.', 'p')

        # --- HCP: prioritise high-contact hubs (Z) ---
        params_hcp = copy.deepcopy(PARAMS_HCP)
        params_hcp['HIGH_RISK_PROB'] = hrp

        print(f"\n  [HCP] Prioritise high-contact (Z) ...")
        res_hcp = run_one_scenario(
            params=params_hcp,
            scenario_tag='hcp',
            priority='Z',
            priority_order=[3, 2, 1],
            bias=[0, 0, 1],
            out_dir=os.path.join(base_out_dir, f'hrp{tag}_hcp'),
        )

        # --- HRP: prioritise high-risk elderly (Y) ---
        params_hrp = copy.deepcopy(PARAMS_HCP)
        params_hrp['HIGH_RISK_PROB'] = hrp

        print(f"\n  [HRP] Prioritise high-risk (Y) ...")
        res_hrp = run_one_scenario(
            params=params_hrp,
            scenario_tag='hrp',
            priority='Y',
            priority_order=[2, 3, 1],
            bias=[0, 1, 0],
            out_dir=os.path.join(base_out_dir, f'hrp{tag}_hrp'),
        )

        rows.append({
            'high_risk_prob':  hrp,
            'age_threshold':   AGE_LABEL[hrp],
            'HCP_Warm_RL':     res_hcp['deaths_warm_rl'],
            'HCP_Cold_RL':     res_hcp['deaths_cold_rl'],
            'HCP_OC_Guided':   res_hcp['deaths_ocg'],
            'HRP_Warm_RL':     res_hrp['deaths_warm_rl'],
            'HRP_Cold_RL':     res_hrp['deaths_cold_rl'],
            'HRP_OC_Guided':   res_hrp['deaths_ocg'],
        })
        print(f"[sensitivity_highrisk] DONE: high_risk_prob = {hrp}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("SENSITIVITY — HIGH-RISK POPULATION SIZE (high_risk_prob)")
    print("=" * 60)
    print(df.to_string(index=False))

    csv_path = os.path.join(base_out_dir, 'results_highrisk.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
    return df


if __name__ == '__main__':
    run_sensitivity_highrisk()
