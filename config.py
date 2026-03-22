# -*- coding: utf-8 -*-
"""
config.py — Global constants, state indices, parameter dictionaries.

All other modules import from here. Never redefine these values elsewhere.
"""

# ---------------------------------------------------------------------------
# Epidemic compartment indices  (single canonical definition)
# ---------------------------------------------------------------------------
S, E, P, A, I, L, H, R, V, D = range(10)

# ---------------------------------------------------------------------------
# Network & simulation globals
# ---------------------------------------------------------------------------
N               = 10_000   # total population
BA_M            = 3        # Barabasi-Albert attachment parameter
T_HORIZON       = 60       # simulation horizon (days)
V_MAX_DAILY     = 40       # maximum vaccine doses per day
HIGH_RISK_PROB  = 0.17     # probability a non-hub node is high-risk (Y group)
ALPHA_STD       = 0.7      # hub threshold: mean + ALPHA_STD * std of degree
INITIAL_INFECTED = 300     # default seed infections

# ---------------------------------------------------------------------------
# RL / ActorCritic constants
# ---------------------------------------------------------------------------
MIN_CONC  = 0.01   # minimum Dirichlet concentration (numerical stability)
CONC_SCALE = 10    # scaling factor applied to softplus output

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def idx(g: int, c: int) -> int:
    """Flat index into the 30-dim state vector for group g (0-2), compartment c (0-9)."""
    return g * 10 + c


# ---------------------------------------------------------------------------
# Canonical parameter dictionaries
# ---------------------------------------------------------------------------
# Disease parameters shared by both scenarios:
#   beta   — transmission rate
#   wA/wP/wI — infectiousness weights (asymptomatic / pre-symptomatic / symptomatic)
#   tauE/P/A/I/L/H — transition rates
#   sX/Y/Z — prob of becoming symptomatic (vs asymptomatic) per group
#   pX/Y/Z — prob of progressing to severe (H) after mild (L) per group
#   dX/Y/Z — prob of dying given severe (H) per group

PARAMS_HCP = {
    # Scenario identity
    'N': N, 'BA_M': BA_M, 'SEED': 42,
    'HIGH_RISK_PROB': HIGH_RISK_PROB, 'ALPHA_STD': ALPHA_STD,
    'T_HORIZON': T_HORIZON, 'V_MAX_DAILY': V_MAX_DAILY,
    'INITIAL_INFECTED': INITIAL_INFECTED,
    # Transmission
    'beta': 0.08,
    # Infectiousness weights
    'wA': 0.5, 'wP': 0.8, 'wI': 1.0,
    # Transition rates
    'tauE': 1/3, 'tauP': 1/2, 'tauA': 1/5,
    'tauI': 1/3, 'tauL': 1/5, 'tauH': 1/10,
    # Group-specific: symptomatic probability
    'sX': 0.5, 'sY': 0.8, 'sZ': 0.6,
    # Group-specific: hospitalisation probability
    'pX': 0.05, 'pY': 0.2, 'pZ': 0.08,
    # Group-specific: death probability given hospitalised
    'dX': 0.02, 'dY': 0.27, 'dZ': 0.04,
}

# HRP scenario: higher initial infections seeded entirely in Y group,
# and dY=0.47 (intentionally harder — higher mortality for high-risk group).
PARAMS_HRP = {
    'N': N, 'BA_M': BA_M, 'SEED': 42,
    'HIGH_RISK_PROB': HIGH_RISK_PROB, 'ALPHA_STD': ALPHA_STD,
    'T_HORIZON': T_HORIZON, 'V_MAX_DAILY': V_MAX_DAILY,
    'INITIAL_INFECTED': 2 * INITIAL_INFECTED,
    'INIT_INFECTED_X': 0,
    'INIT_INFECTED_Y': 2 * INITIAL_INFECTED,
    'INIT_INFECTED_Z': 0,
    'beta': 0.08,
    'wA': 0.5, 'wP': 0.8, 'wI': 1.0,
    'tauE': 1/3, 'tauP': 1/2, 'tauA': 1/5,
    'tauI': 1/3, 'tauL': 1/5, 'tauH': 1/10,
    'sX': 0.5, 'sY': 0.8, 'sZ': 0.6,
    'pX': 0.05, 'pY': 0.2, 'pZ': 0.08,
    'dX': 0.02, 'dY': 0.47, 'dZ': 0.04,   # dY=0.47 (canonical HRP value)
}


# Node-aware RL experiment: smaller population for tractable node-level scoring
PARAMS_NODE_RL = {
    'N': 1000, 'BA_M': 3, 'SEED': 42,
    'HIGH_RISK_PROB': HIGH_RISK_PROB, 'ALPHA_STD': ALPHA_STD,
    'T_HORIZON': T_HORIZON, 'V_MAX_DAILY': 2,
    'INITIAL_INFECTED': 50,
    'beta': 0.08,
    'wA': 0.5, 'wP': 0.8, 'wI': 1.0,
    'tauE': 1/3, 'tauP': 1/2, 'tauA': 1/5,
    'tauI': 1/3, 'tauL': 1/5, 'tauH': 1/10,
    'sX': 0.5, 'sY': 0.8, 'sZ': 0.6,
    'pX': 0.05, 'pY': 0.2, 'pZ': 0.08,
    'dX': 0.02, 'dY': 0.27, 'dZ': 0.04,
}


def to_params_global(p: dict) -> dict:
    """
    Convert a raw scenario param dict into the group-structured format
    expected by EpidemicNodeEnv.

    Adds per-group epsilon (vaccine efficacy reduction factor, default 0.5)
    and the T_HORIZON key required by the environment.
    """
    return {
        'beta': p['beta'],
        'wA': p['wA'], 'wP': p['wP'], 'wI': p['wI'],
        'tauE': p['tauE'], 'tauP': p['tauP'], 'tauA': p['tauA'],
        'tauI': p['tauI'], 'tauL': p['tauL'], 'tauH': p['tauH'],
        'sX': p['sX'], 'sY': p['sY'], 'sZ': p['sZ'],
        'pX': p['pX'], 'pY': p['pY'], 'pZ': p['pZ'],
        'dX': p['dX'], 'dY': p['dY'], 'dZ': p['dZ'],
        'epsilonX': p.get('epsilonX', 0.5),
        'epsilonY': p.get('epsilonY', 0.5),
        'epsilonZ': p.get('epsilonZ', 0.5),
        'T_HORIZON': p['T_HORIZON'],
    }
