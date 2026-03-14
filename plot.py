# -*- coding: utf-8 -*-
"""
plot.py — Visualisation utilities.

All plotting functions are collected here. No epidemic logic lives in this
file — it only consumes DataFrames and arrays produced by other modules.

Functions
---------
plot_stacked          — stacked bar chart of daily dose allocations
plot_lines            — line chart of daily dose allocations per group
plot_convergence      — warm-start vs cold-start learning curves
summarize_daily       — median degree + infected-neighbour share per day
plot_inf_neighbor     — infected-neighbour share curve for one scenario
share_inf_neighbor    — compute per-day infected-neighbour share (utility)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# 1. Daily allocation — stacked bar  (merged from lines 1430 & 1748)
# ---------------------------------------------------------------------------

def plot_stacked(df_days: pd.DataFrame, title: str, vmax: int = None) -> None:
    """
    Stacked bar chart of daily vaccine allocations for groups X, Y, Z.

    The y-axis upper limit is max(vmax, actual_max) + 5 so bars never touch
    the top. If `vmax` is None and the DataFrame contains a 'V_MAX_DAILY'
    column, that value is used as the reference ceiling.

    Parameters
    ----------
    df_days : DataFrame with columns day, X, Y, Z and optionally V_MAX_DAILY
    title   : plot title
    vmax    : optional y-axis ceiling override
    """
    colors = cm.viridis(np.linspace(0, 1, 3))
    days   = df_days['day'].values
    x      = df_days['X'].values
    y      = df_days['Y'].values
    z      = df_days['Z'].values

    # determine y ceiling
    actual_max = int((x + y + z).max()) if len(x) > 0 else 0
    if vmax is None and 'V_MAX_DAILY' in df_days.columns and len(df_days) > 0:
        vmax = int(df_days['V_MAX_DAILY'].iloc[0])
    ylim_top = (max(vmax, actual_max) if vmax is not None else actual_max) + 5

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(days, x,     label='Baseline (X)',     color=colors[0], edgecolor='white', linewidth=0.1)
    ax.bar(days, y,     label='High-risk (Y)',     color=colors[1], edgecolor='white', linewidth=0.1, bottom=x)
    ax.bar(days, z,     label='High-contact (Z)',  color=colors[2], edgecolor='white', linewidth=0.1, bottom=x + y)
    ax.set_title(title)
    ax.set_xlabel('Days')
    ax.set_ylabel('Number of doses')
    ax.set_ylim(0, ylim_top)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. Daily allocation — line chart  (source: lines 1447-1455)
# ---------------------------------------------------------------------------

def plot_lines(df_days: pd.DataFrame, title: str) -> None:
    """
    Line chart of daily vaccine allocations for groups X, Y, Z.

    Parameters
    ----------
    df_days : DataFrame with columns day, X, Y, Z
    title   : plot title
    """
    days = df_days['day'].values
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(days, df_days['X'].values, label='Baseline (X)')
    ax.plot(days, df_days['Y'].values, label='High-risk (Y)')
    ax.plot(days, df_days['Z'].values, label='High-contact (Z)')
    ax.set_title(title)
    ax.set_xlabel('Days')
    ax.set_ylabel('Number of doses')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3. Training convergence  (source: lines 1170-1177)
# ---------------------------------------------------------------------------

def plot_convergence(hist_warm: list, hist_cold: list) -> None:
    """
    Plot warm-start vs cold-start final-death learning curves.

    Each list entry is the deterministic evaluation death count after one
    PPO update round.

    Parameters
    ----------
    hist_warm : list of floats — warm-start eval deaths per update
    hist_cold : list of floats — cold-start eval deaths per update
    """
    plt.figure(figsize=(9, 4))
    plt.plot(hist_warm, label='Warm Start', color='tab:purple')
    plt.plot(hist_cold, label='Cold Start',  color='tab:gray')
    plt.xlabel('Update rounds')
    plt.ylabel('Final deaths (deterministic eval)')
    plt.title('Convergence: Warm vs Cold Start')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. Daily vaccination quality summary  (source: lines 1179-1225)
# ---------------------------------------------------------------------------

def summarize_daily(
    df_nodes: pd.DataFrame,
    T_horizon: int = None,
    smooth_window: int = None,
) -> None:
    """
    Plot median degree and infected-neighbour share of vaccinated nodes per day.

    Parameters
    ----------
    df_nodes      : DataFrame with columns day, degree, inf_nbr_count
    T_horizon     : if given, reindex to fill missing days with NaN
    smooth_window : rolling window size for smoothing (>1 to activate)
    """
    if df_nodes is None or len(df_nodes) == 0:
        print("[summarize_daily] No per-node details to summarise.")
        return

    df = df_nodes[df_nodes['degree'] >= 0].copy()
    if len(df) == 0:
        print("[summarize_daily] No valid node details (degree >= 0).")
        return

    grp        = df.groupby('day')
    deg_median = grp['degree'].median()
    inf_ratio  = grp['inf_nbr_count'].apply(lambda x: (x > 0).mean())

    if T_horizon is not None:
        idx        = pd.RangeIndex(0, T_horizon)
        deg_median = deg_median.reindex(idx)
        inf_ratio  = inf_ratio.reindex(idx)

    if smooth_window and smooth_window > 1:
        deg_median = deg_median.rolling(smooth_window, min_periods=1).median()
        inf_ratio  = inf_ratio.rolling(smooth_window, min_periods=1).mean()

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    deg_median.plot(color='tab:blue')
    plt.title('Median Degree of Vaccinated per Day')
    plt.xlabel('Day')
    plt.ylabel('Median degree')
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    inf_ratio.plot(color='tab:red')
    plt.title('Share with Infected Neighbour per Day')
    plt.xlabel('Day')
    plt.ylabel('Share')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5. Infected-neighbour share curve  (source: lines 1816-1828)
# ---------------------------------------------------------------------------

def plot_inf_neighbor(df_nodes: pd.DataFrame, title: str) -> None:
    """
    Plot the fraction of vaccinated nodes that had at least one infected
    neighbour on each day.

    Parameters
    ----------
    df_nodes : DataFrame with columns day, inf_nbr_count
    title    : plot title
    """
    sw = share_inf_neighbor(df_nodes)
    plt.figure(figsize=(8, 4))
    plt.plot(sw['day'], sw['share'], color='tab:red')
    plt.title(title)
    plt.xlabel('Day')
    plt.ylabel('Share with infected neighbour')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6. Utility: compute share with infected neighbour  (source: lines 1770-1773)
# ---------------------------------------------------------------------------

def share_inf_neighbor(df_nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-day fraction of vaccinated nodes with at least one
    infected (P, A, or I) neighbour at time of vaccination.

    Parameters
    ----------
    df_nodes : DataFrame with columns day, inf_nbr_count

    Returns
    -------
    DataFrame with columns day, share
    """
    s = (
        df_nodes
        .groupby('day')['inf_nbr_count']
        .apply(lambda x: (x > 0).mean())
        .reset_index()
    )
    s.rename(columns={'inf_nbr_count': 'share'}, inplace=True)
    return s
