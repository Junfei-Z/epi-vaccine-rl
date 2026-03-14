# -*- coding: utf-8 -*-
"""
experiments/sensitivity_epsilon.py — Sensitivity analysis over vaccine efficacy (epsilon).

Sweeps epsilonX = epsilonY = epsilonZ ∈ [0.35, 0.45, 0.55, 0.65, 0.75],
where epsilon is the vaccine-induced susceptibility reduction factor
(higher = more protective vaccine).

Implementation note
-------------------
`to_params_global()` in config.py already reads p.get('epsilonX/Y/Z', 0.5),
so adding epsilon keys to the local params copy is all that is needed —
no custom params_global builder required.

Sweep mode A (used here): all three groups share the same epsilon value.
The code also supports Mode B (per-group tuples) via VARY_LIST if needed.
"""

import os
import copy
import pandas as pd

from config import PARAMS_HCP
from experiments.base import run_one_scenario


# Mode A: uniform epsilon across groups
# Swap for list-of-tuples to run Mode B (per-group)
VARY_LIST = [0.35, 0.45, 0.55, 0.65, 0.75]


def _eps_tag(ex: float, ey: float, ez: float) -> str:
    """Return a filesystem-safe tag string for a given (epsilonX, epsilonY, epsilonZ) triple."""
    def _fmt(v):
        return f'{v:.2f}'.replace('.', 'p')
    return f'ex{_fmt(ex)}_ey{_fmt(ey)}_ez{_fmt(ez)}'


def run_sensitivity_epsilon(base_out_dir: str = 'results/sensitivity_epsilon') -> pd.DataFrame:
    """
    Sweep vaccine efficacy (epsilon) and collect Warm_RL / Cold_RL / OC_Guided death counts.

    Each entry in VARY_LIST can be:
      float              → epsilonX = epsilonY = epsilonZ (Mode A, uniform)
      (float,float,float) → (epsilonX, epsilonY, epsilonZ) (Mode B, per-group)

    Parameters
    ----------
    base_out_dir : root directory; each epsilon config gets its own sub-folder

    Returns
    -------
    DataFrame with columns:
        epsilonX, epsilonY, epsilonZ,
        Warm_RL_Deaths, Cold_RL_Deaths, OC_Guided_Deaths
    """
    os.makedirs(base_out_dir, exist_ok=True)
    rows = []

    for eps in VARY_LIST:
        if isinstance(eps, (list, tuple)):
            ex, ey, ez = float(eps[0]), float(eps[1]), float(eps[2])
        else:
            ex = ey = ez = float(eps)

        tag = _eps_tag(ex, ey, ez)
        print(f"\n{'='*60}")
        print(f"[sensitivity_epsilon] epsilonX={ex}  epsilonY={ey}  epsilonZ={ez}")
        print(f"{'='*60}")

        # Inject epsilon into local params copy — to_params_global picks them up
        params = copy.deepcopy(PARAMS_HCP)
        params['epsilonX'] = ex
        params['epsilonY'] = ey
        params['epsilonZ'] = ez

        out_dir = os.path.join(base_out_dir, tag)
        results = run_one_scenario(
            params=params,
            scenario_tag='hcp',
            priority='Z',
            priority_order=[3, 2, 1],
            bias=[0, 0, 1],
            out_dir=out_dir,
        )

        rows.append({
            'epsilonX':         ex,
            'epsilonY':         ey,
            'epsilonZ':         ez,
            'Warm_RL_Deaths':   results['deaths_warm_rl'],
            'Cold_RL_Deaths':   results['deaths_cold_rl'],
            'OC_Guided_Deaths': results['deaths_ocg'],
        })
        print(f"[sensitivity_epsilon] DONE: epsilon = ({ex}, {ey}, {ez})")

    df = pd.DataFrame(rows).sort_values(['epsilonX', 'epsilonY', 'epsilonZ'])

    print("\n" + "=" * 50)
    print("SENSITIVITY — VACCINE EFFICACY (epsilon)")
    print("=" * 50)
    print(df.to_string(index=False))

    csv_path = os.path.join(base_out_dir, 'method_comparison_results_epsilon.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
    return df


if __name__ == '__main__':
    run_sensitivity_epsilon()
