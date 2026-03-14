# -*- coding: utf-8 -*-
"""
prior.py — Feasible prior policy construction for warm-start PPO.

Takes ODE-optimal dose sequences and converts them into simplex-valued
prior policies that are compatible with the node-level environment
(i.e. they respect integer capacity and per-group susceptible availability).

Functions
---------
load_prior                    — load and normalise a saved .npy prior
build_feasible_prior_from_doses — main prior builder (ODE doses → shares)
simulate_episode_prior        — run one episode with a fixed prior policy
trace_doses_from_prior        — record actual doses applied from a prior
"""

import numpy as np
import pandas as pd

from config import S, D
from allocation import cap_int
from env import build_env


# ---------------------------------------------------------------------------
# 1. Load prior
# ---------------------------------------------------------------------------

def load_prior(path: str):
    """
    Load a saved prior policy from a .npy file and normalise rows to simplex.

    Parameters
    ----------
    path : file path to a (T, 3) numpy array

    Returns
    -------
    arr : np.ndarray of shape (T, 3) with rows summing to 1, or None on error
    """
    try:
        arr = np.load(path)
        arr = arr / (arr.sum(axis=1, keepdims=True) + 1e-12)
        return arr
    except Exception as exc:
        print(f"[load_prior] Could not load {path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# 2. Build feasible prior from ODE doses
# ---------------------------------------------------------------------------

def build_feasible_prior_from_doses(
    doses_path: str,
    args: tuple,
    label: str,
    window_K: int = 0,
    bias=None,
    eta0: float = 0.5,
    save_path: str = None,
) -> tuple:
    """
    Convert ODE-optimal integer doses into a feasible simplex prior policy.

    The prior is built by rolling the ODE allocation forward through the
    node-level environment one day at a time:
      1. Cap doses to V_MAX and zero out groups with no susceptibles.
      2. Project shares through env._project_doses() to get feasible integers.
      3. Optionally blend with a bias vector during the first window_K days
         (linear decay from eta0 to 0), so the policy starts with a strong
         priority signal.
      4. Advance the environment one step with these shares.

    This ensures the prior reflects the same dynamics the RL agent will face.

    Parameters
    ----------
    doses_path : path to .npy file of shape (T, 3) — ODE integer doses
    args       : env args tuple (G, groups, deg, params_global, capacity, seed_counts)
    label      : tag for the saved file name (e.g. 'hcp', 'hrp')
    window_K   : number of days to apply bias blending (0 = no blending)
    bias       : length-3 list, priority direction e.g. [0,0,1] for Z-first
    eta0       : initial blending strength (decays linearly to 0 over window_K)
    save_path  : override output path; defaults to 'ode_feasible_prior_{label}.npy'

    Returns
    -------
    prior    : np.ndarray of shape (T, 3)
    out_path : str, path where the prior was saved
    """
    doses = np.load(doses_path)
    env, _, _ = build_env(args, deterministic=True)
    T   = doses.shape[0]
    cap = int(env.V_MAX_DAILY)

    prior = []
    for t in range(T):
        # available susceptibles per group
        s1 = int(np.sum(env.status[env.group_nodes[1]] == S))
        s2 = int(np.sum(env.status[env.group_nodes[2]] == S))
        s3 = int(np.sum(env.status[env.group_nodes[3]] == S))
        mask = np.array([s1 > 0, s2 > 0, s3 > 0], dtype=int)

        # cap raw ODE doses to daily capacity and zero unavailable groups
        x, y, z = int(doses[t][0]), int(doses[t][1]), int(doses[t][2])
        x, y, z = cap_int(x, y, z, cap)
        x *= mask[0]; y *= mask[1]; z *= mask[2]

        # convert to shares
        total = max(1, x + y + z)
        shares = np.array([x / total, y / total, z / total], dtype=float)

        # project through env to get truly feasible integer doses, then re-normalise
        proj   = env._project_doses(shares)
        total3 = max(1, int(proj.sum()))
        shares2 = proj.astype(float) / float(total3)

        # optional bias blending during priority window
        if window_K > 0 and bias is not None and t < window_K:
            eta     = eta0 * (1.0 - t / float(max(1, window_K)))
            shares2 = (1.0 - eta) * shares2 + eta * np.asarray(bias, dtype=float)
            shares2 = shares2 / (shares2.sum() + 1e-12)

        prior.append(shares2)
        env.step(shares2)

    prior    = np.asarray(prior, dtype=float)
    out_path = save_path or f'ode_feasible_prior_{label}.npy'
    np.save(out_path, prior)
    print(f"[build_feasible_prior] Saved {out_path}  shape={prior.shape}")
    return prior, out_path


# ---------------------------------------------------------------------------
# 3. Simulate episode with prior policy
# ---------------------------------------------------------------------------

def simulate_episode_prior(name: str, args: tuple, prior_seq: np.ndarray) -> int:
    """
    Run one deterministic episode using a fixed prior policy sequence.

    Parameters
    ----------
    name      : label for console output
    args      : env args tuple
    prior_seq : np.ndarray of shape (T, 3) — per-day share vectors

    Returns
    -------
    final_deaths : int
    """
    env, _, _ = build_env(args, deterministic=True)
    done = False
    while not done:
        t      = int(env.day)
        shares = prior_seq[t] if t < len(prior_seq) else prior_seq[-1]
        shares = shares / (shares.sum() + 1e-12)
        _, _, done, _ = env.step(shares)

    final_deaths = int(np.sum(env.status == D))
    print(f"[{name}] prior episode final deaths: {final_deaths}")
    return final_deaths


# ---------------------------------------------------------------------------
# 4. Trace actual doses from prior (for visualisation)
# ---------------------------------------------------------------------------

def trace_doses_from_prior(prior_seq: np.ndarray, args: tuple) -> pd.DataFrame:
    """
    Run one deterministic episode and record the actual doses applied each day.

    Useful for verifying that the feasible prior matches the ODE allocation.

    Parameters
    ----------
    prior_seq : np.ndarray of shape (T, 3)
    args      : env args tuple

    Returns
    -------
    pd.DataFrame with columns: day, X, Y, Z, unused, V_MAX_DAILY
    """
    env, _, _ = build_env(args, deterministic=True)
    rows = []
    for t in range(len(prior_seq)):
        shares = prior_seq[t]
        shares = shares / (shares.sum() + 1e-12)
        _, _, done, info = env.step(shares)

        doses  = info.get('doses', np.array([0, 0, 0], dtype=int))
        unused = int(info.get('unused', 0))
        rows.append({
            'day':         int(env.day - 1),
            'X':           int(doses[0]),
            'Y':           int(doses[1]),
            'Z':           int(doses[2]),
            'unused':      unused,
            'V_MAX_DAILY': int(env.V_MAX_DAILY),
        })
        if done:
            break

    return pd.DataFrame(rows)
