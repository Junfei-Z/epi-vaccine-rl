# -*- coding: utf-8 -*-
"""
experiments/analyze_vaccination_plans.py

Analyze vaccination plans from Naive RL vs Node RL vs OC-Guided.
Extracts patterns: who gets vaccinated, when, degree targeting, group priority.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PLAN_DIR = 'results/naive_vs_node/severity/baseline'
OUT_DIR = 'results/naive_vs_node/plan_analysis'


def load_plans(plan_dir, label):
    """Load plan JSON for a method."""
    path = os.path.join(plan_dir, f'plan_{label}.json')
    if not os.path.exists(path):
        # try with sweep suffix
        import glob
        candidates = glob.glob(os.path.join(plan_dir, f'plan_{label}*.json'))
        if candidates:
            path = candidates[0]
        else:
            return None
    with open(path) as f:
        return json.load(f)


def analyze_rl_plan(plans, method_name):
    """Analyze Node RL or Naive RL plans (have node_ids, groups, degrees)."""
    n_runs = len(plans)

    # Per-day stats across all runs
    daily_avg_degree = defaultdict(list)
    daily_group_counts = defaultdict(lambda: defaultdict(list))  # day -> group -> [counts]
    daily_max_degree = defaultdict(list)
    all_degrees_by_day = defaultdict(list)

    for run in plans:
        for day_data in run['days']:
            day = day_data['day']
            degrees = day_data['degrees']
            groups = day_data['groups']

            if len(degrees) > 0:
                daily_avg_degree[day].append(np.mean(degrees))
                daily_max_degree[day].append(max(degrees))
            else:
                daily_avg_degree[day].append(0)
                daily_max_degree[day].append(0)

            all_degrees_by_day[day].extend(degrees)

            # Count per group
            for g in [1, 2, 3]:
                daily_group_counts[day][g].append(sum(1 for x in groups if x == g))

    T = max(daily_avg_degree.keys()) + 1

    # Build summary dataframe
    rows = []
    for day in range(T):
        row = {
            'day': day,
            'avg_degree': np.mean(daily_avg_degree[day]),
            'max_degree': np.mean(daily_max_degree[day]),
            'group_X': np.mean(daily_group_counts[day][1]),
            'group_Y': np.mean(daily_group_counts[day][2]),
            'group_Z': np.mean(daily_group_counts[day][3]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Overall stats
    all_degrees = []
    all_groups = []
    for run in plans:
        for day_data in run['days']:
            all_degrees.extend(day_data['degrees'])
            all_groups.extend(day_data['groups'])

    group_names = {1: 'X (normal)', 2: 'Y (high mortality)', 3: 'Z (high risk)'}
    group_pcts = {}
    total = len(all_groups)
    for g in [1, 2, 3]:
        cnt = sum(1 for x in all_groups if x == g)
        group_pcts[group_names[g]] = cnt / total * 100

    print(f"\n{'='*50}")
    print(f"  {method_name}")
    print(f"{'='*50}")
    print(f"  Total vaccinations: {total} across {n_runs} runs")
    print(f"  Avg degree of vaccinated nodes: {np.mean(all_degrees):.1f}")
    print(f"  Max degree ever vaccinated: {max(all_degrees)}")
    print(f"  Group distribution:")
    for name, pct in group_pcts.items():
        print(f"    {name}: {pct:.1f}%")

    # Early vs late
    early_degrees = []
    late_degrees = []
    early_groups = defaultdict(int)
    late_groups = defaultdict(int)
    mid = T // 2
    for run in plans:
        for day_data in run['days']:
            if day_data['day'] < mid:
                early_degrees.extend(day_data['degrees'])
                for g in day_data['groups']:
                    early_groups[g] += 1
            else:
                late_degrees.extend(day_data['degrees'])
                for g in day_data['groups']:
                    late_groups[g] += 1

    print(f"\n  Early phase (day 0-{mid-1}):")
    print(f"    Avg degree: {np.mean(early_degrees):.1f}")
    total_early = sum(early_groups.values())
    for g in [1, 2, 3]:
        print(f"    Group {g}: {early_groups[g]/total_early*100:.1f}%")

    print(f"  Late phase (day {mid}-{T-1}):")
    print(f"    Avg degree: {np.mean(late_degrees):.1f}")
    total_late = sum(late_groups.values())
    for g in [1, 2, 3]:
        print(f"    Group {g}: {late_groups[g]/total_late*100:.1f}%")

    return df


def analyze_oc_plan(plans, method_name):
    """Analyze OC-Guided plans (have doses_X, doses_Y, doses_Z)."""
    n_runs = len(plans)

    daily_doses = defaultdict(lambda: defaultdict(list))

    for run in plans:
        for day_data in run['days']:
            day = day_data['day']
            daily_doses[day]['X'].append(day_data['doses_X'])
            daily_doses[day]['Y'].append(day_data['doses_Y'])
            daily_doses[day]['Z'].append(day_data['doses_Z'])

    T = max(daily_doses.keys()) + 1

    rows = []
    for day in range(T):
        row = {
            'day': day,
            'doses_X': np.mean(daily_doses[day]['X']),
            'doses_Y': np.mean(daily_doses[day]['Y']),
            'doses_Z': np.mean(daily_doses[day]['Z']),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    total_X = sum(df['doses_X'])
    total_Y = sum(df['doses_Y'])
    total_Z = sum(df['doses_Z'])
    total = total_X + total_Y + total_Z

    print(f"\n{'='*50}")
    print(f"  {method_name}")
    print(f"{'='*50}")
    print(f"  Total dose-days: {total:.0f} across {n_runs} runs")
    print(f"  Group distribution:")
    print(f"    X (normal):        {total_X/total*100:.1f}%")
    print(f"    Y (high mortality): {total_Y/total*100:.1f}%")
    print(f"    Z (high risk):     {total_Z/total*100:.1f}%")

    mid = T // 2
    early = df[df['day'] < mid]
    late = df[df['day'] >= mid]

    print(f"\n  Early phase (day 0-{mid-1}):")
    te = early[['doses_X', 'doses_Y', 'doses_Z']].sum().sum()
    print(f"    X: {early['doses_X'].sum()/te*100:.1f}%  "
          f"Y: {early['doses_Y'].sum()/te*100:.1f}%  "
          f"Z: {early['doses_Z'].sum()/te*100:.1f}%")

    print(f"  Late phase (day {mid}-{T-1}):")
    tl = late[['doses_X', 'doses_Y', 'doses_Z']].sum().sum()
    if tl > 0:
        print(f"    X: {late['doses_X'].sum()/tl*100:.1f}%  "
              f"Y: {late['doses_Y'].sum()/tl*100:.1f}%  "
              f"Z: {late['doses_Z'].sum()/tl*100:.1f}%")
    else:
        print(f"    No vaccinations (all susceptible already vaccinated?)")

    return df


def compute_overlap(plans_a, plans_b, method_a, method_b):
    """Compute daily node overlap between two RL methods."""
    n_runs = min(len(plans_a), len(plans_b))
    daily_overlap = defaultdict(list)

    for i in range(n_runs):
        days_a = {d['day']: set(d['node_ids']) for d in plans_a[i]['days']}
        days_b = {d['day']: set(d['node_ids']) for d in plans_b[i]['days']}

        for day in days_a:
            if day in days_b:
                sa, sb = days_a[day], days_b[day]
                if len(sa) > 0 and len(sb) > 0:
                    overlap = len(sa & sb)
                    jaccard = overlap / len(sa | sb)
                    daily_overlap[day].append(jaccard)

    T = max(daily_overlap.keys()) + 1
    print(f"\n{'='*50}")
    print(f"  Node overlap: {method_a} vs {method_b}")
    print(f"{'='*50}")

    overall = [v for vals in daily_overlap.values() for v in vals]
    print(f"  Overall Jaccard similarity: {np.mean(overall):.3f}")

    # Early vs late
    mid = T // 2
    early = [v for d, vals in daily_overlap.items() for v in vals if d < mid]
    late = [v for d, vals in daily_overlap.items() for v in vals if d >= mid]
    print(f"  Early (day 0-{mid-1}): {np.mean(early):.3f}")
    print(f"  Late  (day {mid}-{T-1}): {np.mean(late):.3f}")

    return daily_overlap


def degree_targeting_analysis(plans_node, plans_naive):
    """Compare degree targeting between Node RL and Naive RL."""
    print(f"\n{'='*50}")
    print(f"  Degree Targeting Comparison")
    print(f"{'='*50}")

    for method_name, plans in [('Node RL', plans_node), ('Naive RL', plans_naive)]:
        # Collect degree of vaccinated nodes per day
        day_degrees = defaultdict(list)
        for run in plans:
            for day_data in run['days']:
                day_degrees[day_data['day']].extend(day_data['degrees'])

        T = max(day_degrees.keys()) + 1

        # Degree percentiles per phase
        early = [d for day, degs in day_degrees.items() for d in degs if day < T//2]
        late = [d for day, degs in day_degrees.items() for d in degs if day >= T//2]

        print(f"\n  {method_name}:")
        print(f"    Early — mean: {np.mean(early):.1f}, "
              f"median: {np.median(early):.0f}, "
              f"p90: {np.percentile(early, 90):.0f}, "
              f"p99: {np.percentile(early, 99):.0f}")
        print(f"    Late  — mean: {np.mean(late):.1f}, "
              f"median: {np.median(late):.0f}, "
              f"p90: {np.percentile(late, 90):.0f}, "
              f"p99: {np.percentile(late, 99):.0f}")

        # Top-10 most vaccinated degree values
        from collections import Counter
        deg_counter = Counter(d for degs in day_degrees.values() for d in degs)
        print(f"    Most common degrees: {deg_counter.most_common(8)}")


def run_analysis(plan_dir, sweep_label):
    """Run full analysis for one sweep setting."""
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n\n{'#'*60}")
    print(f"  VACCINATION PLAN ANALYSIS: {sweep_label}")
    print(f"  Data: {plan_dir}")
    print(f"{'#'*60}")

    # Load plans
    import glob
    node_files = glob.glob(os.path.join(plan_dir, 'plan_node*.json'))
    naive_files = glob.glob(os.path.join(plan_dir, 'plan_naive*.json'))
    oc_files = glob.glob(os.path.join(plan_dir, 'plan_oc*.json'))

    plans_node = json.load(open(node_files[0])) if node_files else None
    plans_naive = json.load(open(naive_files[0])) if naive_files else None
    plans_oc = json.load(open(oc_files[0])) if oc_files else None

    # 1. Per-method analysis
    if plans_node:
        df_node = analyze_rl_plan(plans_node, 'Node RL')
    if plans_naive:
        df_naive = analyze_rl_plan(plans_naive, 'Naive RL')
    if plans_oc:
        df_oc = analyze_oc_plan(plans_oc, 'OC-Guided')

    # 2. Node overlap
    if plans_node and plans_naive:
        compute_overlap(plans_node, plans_naive, 'Node RL', 'Naive RL')

    # 3. Degree targeting
    if plans_node and plans_naive:
        degree_targeting_analysis(plans_node, plans_naive)

    return plans_node, plans_naive, plans_oc


if __name__ == '__main__':
    # Analyze baseline severity (the default / most important setting)
    run_analysis(
        'results/naive_vs_node/severity/baseline',
        'Severity: Baseline (pY=0.2, dY=0.27)'
    )

    # Also analyze critical severity (largest gap between methods)
    run_analysis(
        'results/naive_vs_node/severity/critical',
        'Severity: Critical (pY=0.5, dY=0.65)'
    )

    # Analyze vmax=20 (biggest K scaling gap)
    run_analysis(
        'results/naive_vs_node/vmax/vmax_20',
        'V_MAX=20 (K scaling)'
    )

    # Analyze discount gamma=1.0 (best Node RL)
    run_analysis(
        'results/naive_vs_node/discount/gamma_1.00',
        'Gamma=1.0 (no discount)'
    )
