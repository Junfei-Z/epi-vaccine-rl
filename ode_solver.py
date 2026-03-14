# -*- coding: utf-8 -*-
"""
ode_solver.py — ODE-based optimal control for vaccine allocation.

Uses CasADi + IPOPT to solve a finite-horizon NLP that minimises total
deaths over T_HORIZON days subject to a daily vaccine capacity constraint.

State vector per group (10 compartments, 3 groups → 30 total):
    S, E, P, A, I, L, H, R, V, D
    indices 0-9 for group X, 10-19 for group Y, 20-29 for group Z.

Control vector: u = [ux, uy, uz] per day — vaccination rate for each group.
"""

import numpy as np
import casadi as ca

from config import idx
from graph import build_graph_and_groups, get_contact_matrix


# ---------------------------------------------------------------------------
# Seed reconciliation
# ---------------------------------------------------------------------------

def _reconcile_seeds(
    Nx: int, Ny: int, Nz: int,
    total: int,
    ix, iy, iz,
    strategy_hint: str,
) -> tuple:
    """
    Distribute `total` initial infections across groups X, Y, Z.

    If no per-group counts are provided (all None), all seeds go to X.
    If the provided counts don't sum to `total`, they are rescaled
    proportionally; the remainder goes to the priority group indicated
    by `strategy_hint` ('hrp' → Y, otherwise → Z).

    Returns
    -------
    (ix, iy, iz) : int triple, each capped at group size
    """
    ix = 0 if ix is None else ix
    iy = 0 if iy is None else iy
    iz = 0 if iz is None else iz

    s = ix + iy + iz
    if s == 0:
        ix = total
        s = total

    if s != total:
        scale = total / max(s, 1)
        ix = int(np.floor(ix * scale))
        iy = int(np.floor(iy * scale))
        iz = int(np.floor(iz * scale))
        rest = total - (ix + iy + iz)
        if strategy_hint == 'hrp':
            iy += rest
        else:
            iz += rest

    ix = min(ix, Nx)
    iy = min(iy, Ny)
    iz = min(iz, Nz)
    return ix, iy, iz


# ---------------------------------------------------------------------------
# ODE optimal control solver
# ---------------------------------------------------------------------------

def solve(params: dict, init_pattern: str) -> tuple:
    """
    Solve the vaccine-allocation optimal control problem via CasADi/IPOPT.

    The objective is to minimise total deaths D_x + D_y + D_z at time T.
    Dynamics are discretised with RK4 (dt=1 day). The daily capacity
    constraint is:  S_x*u_x + S_y*u_y + S_z*u_z <= V_MAX_DAILY.

    Parameters
    ----------
    params       : scenario parameter dict (see config.PARAMS_HCP / PARAMS_HRP)
    init_pattern : 'hcp' or 'hrp' — controls initial warm-start and seed strategy

    Returns
    -------
    states_opt : np.ndarray, shape (T+1, 30)
    ctrl_opt   : np.ndarray, shape (T, 3)
    meta       : dict with keys 'Nx', 'Ny', 'Nz', 'C'
    """
    # --- build graph & contact matrix ---
    G, groups, _ = build_graph_and_groups(
        params['N'], params['BA_M'], params['SEED'],
        params['HIGH_RISK_PROB'], params['ALPHA_STD'],
    )
    C_mat, g_sizes, _ = get_contact_matrix(G, groups)
    print(g_sizes)

    C  = ca.DM(C_mat)
    Nx = int(g_sizes[1])
    Ny = int(g_sizes[2])
    Nz = int(g_sizes[3])

    dt      = 1.0
    K       = params['T_HORIZON']
    n_state = 30

    # --- unpack disease parameters ---
    beta  = params['beta']
    wA    = params['wA'];  wP = params['wP'];  wI = params['wI']
    tauE  = params['tauE']; tauP = params['tauP']; tauA = params['tauA']
    tauI  = params['tauI']; tauL = params['tauL']; tauH = params['tauH']
    sX    = params['sX'];  sY = params['sY'];  sZ = params['sZ']
    pX    = params['pX'];  pY = params['pY'];  pZ = params['pZ']
    dX    = params['dX'];  dY = params['dY'];  dZ = params['dZ']

    # --- decision variables ---
    w_states = ca.MX.sym('states', (K + 1) * n_state)
    w_ctrl   = ca.MX.sym('ctrl', K * 3)

    g_cons = []; lbg = []; ubg = []
    lbx    = []; ubx = []; x0  = []

    # --- initial infections ---
    total  = params.get('INITIAL_INFECTED', 0)
    ix_raw = params.get('INIT_INFECTED_X', None)
    iy_raw = params.get('INIT_INFECTED_Y', None)
    iz_raw = params.get('INIT_INFECTED_Z', None)
    ix, iy, iz = _reconcile_seeds(
        Nx, Ny, Nz, total, ix_raw, iy_raw, iz_raw,
        'hrp' if init_pattern == 'hrp' else 'hcp',
    )

    Sx0 = float(Nx - ix); Ix0 = float(ix)
    Sy0 = float(Ny - iy); Iy0 = float(iy)
    Sz0 = float(Nz - iz); Iz0 = float(iz)

    # Initial state: [S,E,P,A,I,L,H,R,V,D] × 3 groups
    x0_group = [
        Sx0, 0, 0, 0, Ix0, 0, 0, 0, 0, 0,
        Sy0, 0, 0, 0, Iy0, 0, 0, 0, 0, 0,
        Sz0, 0, 0, 0, Iz0, 0, 0, 0, 0, 0,
    ]

    # bounds for state variables (all non-negative)
    for _ in range(n_state):
        lbx.append(0); ubx.append(ca.inf); x0.append(x0_group[_])
    for _ in range(n_state * K):
        lbx.append(0); ubx.append(ca.inf); x0.append(0)
    # bounds for control variables ∈ [0, 1]
    for _ in range(K * 3):
        lbx.append(0); ubx.append(1); x0.append(0)

    # --- helper slicers ---
    def x_slice(k):
        return w_states[k * n_state:(k + 1) * n_state]

    def u_slice(k):
        return w_ctrl[k * 3:(k + 1) * 3]

    def group_vars(x, g):
        base = g * 10
        return [x[base + c] for c in range(10)]

    # --- ODE right-hand side (one step) ---
    def f_ode(x, u):
        Sx, Ex, Px, Ax, Ix, Lx, Hx, Rx, Vx, Dx = group_vars(x, 0)
        Sy, Ey, Py, Ay, Iy, Ly, Hy, Ry, Vy, Dy = group_vars(x, 1)
        Sz, Ez, Pz, Az, Iz, Lz, Hz, Rz, Vz, Dz = group_vars(x, 2)

        Icx = wA * Ax + wP * Px + wI * Ix
        Icy = wA * Ay + wP * Py + wI * Iy
        Icz = wA * Az + wP * Pz + wI * Iz

        lamx = beta * (C[0,0]*(Icx/max(Nx,1)) + C[0,1]*(Icy/max(Ny,1)) + C[0,2]*(Icz/max(Nz,1)))
        lamy = beta * (C[1,0]*(Icx/max(Nx,1)) + C[1,1]*(Icy/max(Ny,1)) + C[1,2]*(Icz/max(Nz,1)))
        lamz = beta * (C[2,0]*(Icx/max(Nx,1)) + C[2,1]*(Icy/max(Ny,1)) + C[2,2]*(Icz/max(Nz,1)))

        ux, uy, uz = u[0], u[1], u[2]

        # Group X
        dSx = -lamx*Sx - ux*Sx
        dEx = lamx*Sx - tauE*Ex
        dPx = tauE*Ex - tauP*Px
        dAx = (1-sX)*tauP*Px - tauA*Ax
        dIx = sX*tauP*Px - tauI*Ix
        dLx = tauI*Ix - tauL*Lx
        dHx = pX*tauL*Lx - tauH*Hx
        dRx = (1-pX)*tauL*Lx + tauA*Ax + (1-dX)*tauH*Hx
        dVx = ux*Sx
        dDx = dX*tauH*Hx

        # Group Y
        dSy = -lamy*Sy - uy*Sy
        dEy = lamy*Sy - tauE*Ey
        dPy = tauE*Ey - tauP*Py
        dAy = (1-sY)*tauP*Py - tauA*Ay
        dIy = sY*tauP*Py - tauI*Iy
        dLy = tauI*Iy - tauL*Ly
        dHy = pY*tauL*Ly - tauH*Hy
        dRy = (1-pY)*tauL*Ly + tauA*Ay + (1-dY)*tauH*Hy
        dVy = uy*Sy
        dDy = dY*tauH*Hy

        # Group Z
        dSz = -lamz*Sz - uz*Sz
        dEz = lamz*Sz - tauE*Ez
        dPz = tauE*Ez - tauP*Pz
        dAz = (1-sZ)*tauP*Pz - tauA*Az
        dIz = sZ*tauP*Pz - tauI*Iz
        dLz = tauI*Iz - tauL*Lz
        dHz = pZ*tauL*Lz - tauH*Hz
        dRz = (1-pZ)*tauL*Lz + tauA*Az + (1-dZ)*tauH*Hz
        dVz = uz*Sz
        dDz = dZ*tauH*Hz

        return ca.vertcat(
            dSx, dEx, dPx, dAx, dIx, dLx, dHx, dRx, dVx, dDx,
            dSy, dEy, dPy, dAy, dIy, dLy, dHy, dRy, dVy, dDy,
            dSz, dEz, dPz, dAz, dIz, dLz, dHz, dRz, dVz, dDz,
        )

    # --- RK4 collocation constraints ---
    for k in range(K):
        xk = x_slice(k)
        uk = u_slice(k)
        k1 = f_ode(xk, uk)
        k2 = f_ode(xk + 0.5*dt*k1, uk)
        k3 = f_ode(xk + 0.5*dt*k2, uk)
        k4 = f_ode(xk + dt*k3, uk)
        xnext = xk + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # continuity constraint: x(k+1) == xnext
        g_cons.append(x_slice(k + 1) - xnext)
        lbg += [0] * n_state
        ubg += [0] * n_state

        # daily capacity constraint: S_x*u_x + S_y*u_y + S_z*u_z <= V_MAX_DAILY
        Sx_k = xk[idx(0, 0)]
        Sy_k = xk[idx(1, 0)]
        Sz_k = xk[idx(2, 0)]
        g_cons.append(Sx_k*uk[0] + Sy_k*uk[1] + Sz_k*uk[2] - params['V_MAX_DAILY'])
        lbg.append(-ca.inf)
        ubg.append(0)

    # initial condition constraint
    g_cons.append(x_slice(0) - ca.DM(x0_group))
    lbg += [0] * n_state
    ubg += [0] * n_state

    # --- objective: minimise total deaths at terminal time ---
    xT = x_slice(K)
    J = xT[idx(0, 9)] + xT[idx(1, 9)] + xT[idx(2, 9)]

    # --- assemble and solve NLP ---
    nlp = {
        'x': ca.vertcat(w_states, w_ctrl),
        'f': J,
        'g': ca.vertcat(*g_cons),
    }
    solver = ca.nlpsol(
        'solver', 'ipopt', nlp,
        {'ipopt.print_level': 0, 'print_time': 0},
    )

    # warm-start control: bias toward priority group
    w0_states = ca.DM(np.array([*x0_group] + [0] * (K * n_state)))
    w0_ctrl   = np.zeros((K, 3))
    if init_pattern == 'hcp':
        w0_ctrl[:] = [0.0, 0.0, 1.0]
    elif init_pattern == 'hrp':
        w0_ctrl[:] = [0.0, 1.0, 0.0]
    w0_ctrl *= 0.3
    w0 = ca.vertcat(w0_states, ca.DM(w0_ctrl.reshape(-1)))

    sol  = solver(lbx=ca.DM(lbx), ubx=ca.DM(ubx),
                  lbg=ca.DM(lbg), ubg=ca.DM(ubg), x0=w0)
    wopt = sol['x']

    states_opt = np.array(wopt[:(K + 1) * n_state]).reshape((K + 1, n_state))
    ctrl_opt   = np.array(wopt[(K + 1) * n_state:]).reshape((K, 3))

    return states_opt, ctrl_opt, {'Nx': Nx, 'Ny': Ny, 'Nz': Nz, 'C': C_mat}


# ---------------------------------------------------------------------------
# Extract allocations from optimal solution
# ---------------------------------------------------------------------------

def allocations_from_solution(states: np.ndarray, ctrl: np.ndarray) -> tuple:
    """
    Convert optimal control rates to daily vaccine dose counts.

    doses_g[t] = S_g[t] * u_g[t]   (actual individuals vaccinated in group g on day t)

    Parameters
    ----------
    states : (T+1, 30) array from solve()
    ctrl   : (T, 3)   array from solve()

    Returns
    -------
    ax, ay, az : np.ndarray of shape (T,) — doses for groups X, Y, Z
    """
    K  = ctrl.shape[0]
    Sx = states[:K, idx(0, 0)]
    Sy = states[:K, idx(1, 0)]
    Sz = states[:K, idx(2, 0)]

    ax = Sx * ctrl[:, 0]
    ay = Sy * ctrl[:, 1]
    az = Sz * ctrl[:, 2]
    return ax, ay, az
