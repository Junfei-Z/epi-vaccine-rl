# -*- coding: utf-8 -*-
"""
experiments/sensitivity_network_type.py — Compare three allocation methods
across four fundamentally different contact network structures.

Background (Jackson, "Social and Economic Networks")
-----------------------------------------------------
The topology of the contact network is not merely a backdrop — it shapes
which nodes are epidemiologically critical and therefore which vaccination
strategy is most effective.

Networks compared
-----------------

1. Barabási–Albert (BA)  — scale-free, power-law degree distribution
   Rich-get-richer growth produces a small number of highly-connected hubs
   (group Z) that drive disproportionate transmission.  Hub-targeting (HCP)
   is expected to be most valuable here.

2. Erdős–Rényi (ER)  — random graph, Poisson degree distribution
   Edges are placed uniformly at random.  No structural hubs exist; the
   degree distribution is narrow and approximately normal.  The advantage of
   hub-targeting shrinks because there are no clear super-spreaders.

3. Watts–Strogatz (WS)  — small-world network
   Starts as a ring lattice (high local clustering) then rewires a fraction
   of edges to create long-range shortcuts.  Captures the "friend-of-a-friend"
   clustering seen in real social networks while keeping short average path
   lengths.  Epidemics spread fast globally but cluster locally — the optimal
   strategy may differ from both BA and ER.

4. Random Regular  — all nodes have identical degree
   The most homogeneous possible network.  Because every node has the same
   number of contacts, there are no hubs and group Z is empty.  Serves as
   a null model where hub-targeting has no meaning.

All networks are calibrated to the same average degree (~6) so that
differences in epidemic outcomes come from network structure, not connectivity.

Epidemiological hypotheses
--------------------------
• BA:      OC / Warm-RL benefit most from hub targeting → largest gap vs Cold-RL
• ER:      All methods converge (no hubs to exploit)
• WS:      Intermediate — clustering slows spread locally but shortcuts maintain
           global connectivity
• Regular: All methods nearly identical (no structural heterogeneity)
"""

import os
import copy
import pandas as pd

from config import PARAMS_HCP
from experiments.base import run_one_scenario


# Average degree ≈ 6 across all networks (BA with m=3 gives avg degree ~6)
AVG_DEGREE = 6

NETWORKS = [
    {
        'name':         'BA',
        'label':        'Barabási–Albert (scale-free)',
        'params_extra': {},          # default: uses BA_M=3 from PARAMS_HCP
    },
    {
        'name':         'ER',
        'label':        'Erdős–Rényi (random)',
        'params_extra': {'NETWORK_TYPE': 'ER', 'AVG_DEGREE': AVG_DEGREE},
    },
    {
        'name':         'WS',
        'label':        'Watts–Strogatz (small-world, p=0.1)',
        'params_extra': {'NETWORK_TYPE': 'WS', 'AVG_DEGREE': AVG_DEGREE,
                         'WS_P_REWIRE': 0.1},
    },
    {
        'name':         'Regular',
        'label':        'Random Regular (d=6)',
        'params_extra': {'NETWORK_TYPE': 'Regular', 'AVG_DEGREE': AVG_DEGREE},
    },
]


def run_sensitivity_network_type(
    base_out_dir: str = 'results/sensitivity_network_type',
) -> pd.DataFrame:
    """
    Run HCP scenario across four network types and collect death counts.

    Parameters
    ----------
    base_out_dir : root directory; each network type gets its own sub-folder

    Returns
    -------
    DataFrame with columns:
        network, label, Warm_RL_Deaths, Cold_RL_Deaths, OC_Guided_Deaths
    """
    os.makedirs(base_out_dir, exist_ok=True)
    rows = []

    for net in NETWORKS:
        print(f"\n{'='*60}")
        print(f"[sensitivity_network_type] {net['name']}: {net['label']}")
        print(f"{'='*60}")

        params = copy.deepcopy(PARAMS_HCP)
        params.update(net['params_extra'])

        results = run_one_scenario(
            params=params,
            scenario_tag='hcp',
            priority='Z',
            priority_order=[3, 2, 1],
            bias=[0, 0, 1],
            out_dir=os.path.join(base_out_dir, net['name']),
        )

        rows.append({
            'network':          net['name'],
            'label':            net['label'],
            'Warm_RL_Deaths':   results['deaths_warm_rl'],
            'Cold_RL_Deaths':   results['deaths_cold_rl'],
            'OC_Guided_Deaths': results['deaths_ocg'],
        })
        print(f"[sensitivity_network_type] DONE: {net['name']}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("SENSITIVITY — NETWORK TYPE")
    print("=" * 60)
    print(df.to_string(index=False))

    csv_path = os.path.join(base_out_dir, 'results_network_type.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
    return df


if __name__ == '__main__':
    run_sensitivity_network_type()
