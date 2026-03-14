# -*- coding: utf-8 -*-
"""
allocation.py — Vaccine dose post-processing utilities.

Converts continuous ODE outputs into integer, capacity-feasible daily
allocations and provides priority-based assignment helpers.

Functions
---------
strict_priority_window_fill  — enforce strict priority window then forward-fill
cap_to_capacity              — hard-cap daily totals to V_MAX
to_simplex                   — normalise dose array to probability simplex
cap_int                      — integer allocation with proportional leftover
allocate_by_priority         — priority-ordered allocation with carry-over
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Priority window fill  (source: 3_4.py lines 269-366)
# ---------------------------------------------------------------------------

def strict_priority_window_fill(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    V_MAX: int,
    priority: str = 'Y',
) -> tuple:
    """
    Enforce a strict priority window then forward-fill remaining capacity.

    Phase 1 — Priority window (days 0 … win_end-1):
        Only the priority group receives doses. If the day's allocation is
        below V_MAX, doses are pulled forward from future days of the same
        group until the daily cap is reached.

    Phase 2 — Post-window (days win_end … T-1):
        Any day whose total is below V_MAX pulls doses proportionally from
        the nearest future days (all three groups).

    Parameters
    ----------
    ax, ay, az : dose arrays for groups X, Y, Z  (length T, float or int)
    V_MAX      : daily capacity ceiling
    priority   : 'Y' (high-risk first) or 'Z' (high-contact first)

    Returns
    -------
    ax, ay, az : integer arrays of length T, each day summing to <= V_MAX
    """
    ax = np.floor(np.asarray(ax, dtype=float)).astype(int).copy()
    ay = np.floor(np.asarray(ay, dtype=float)).astype(int).copy()
    az = np.floor(np.asarray(az, dtype=float)).astype(int).copy()
    T  = len(ax)

    # identify end of priority window: first day any non-priority group > 0
    if priority == 'Y':
        nonpr = np.asarray(ax) + np.asarray(az)
    else:  # 'Z'
        nonpr = np.asarray(ax) + np.asarray(ay)
    win_end = next((t for t in range(T) if nonpr[t] > 0), T)

    # pull doses forward from future days of the priority group
    def _take_from_future(pr_arr, k, need):
        take = min(need, pr_arr[k])
        pr_arr[k] -= take
        return take

    for t in range(win_end):
        if priority == 'Y':
            ax[t] = 0; az[t] = 0
            need = max(0, V_MAX - ay[t])
            k = t + 1
            while need > 0 and k < T:
                got = _take_from_future(ay, k, need)
                ay[t] += got; need -= got; k += 1
            if ay[t] > V_MAX:
                ay[t] = V_MAX
        else:  # Z
            ax[t] = 0; ay[t] = 0
            need = max(0, V_MAX - az[t])
            k = t + 1
            while need > 0 and k < T:
                got = _take_from_future(az, k, need)
                az[t] += got; need -= got; k += 1
            if az[t] > V_MAX:
                az[t] = V_MAX

    # post-window: proportional forward-fill across all three groups
    def _forward_fill_strict(ax2, ay2, az2, V_MAX, start):
        for t in range(start, T - 1):
            total_t = ax2[t] + ay2[t] + az2[t]
            if total_t >= V_MAX:
                continue
            need = V_MAX - total_t
            k = t + 1
            while need > 0 and k < T:
                availX, availY, availZ = ax2[k], ay2[k], az2[k]
                avail = availX + availY + availZ
                if avail <= 0:
                    k += 1; continue
                take = min(need, avail)
                # proportional integer split
                pX = availX / avail; pY = availY / avail; pZ = availZ / avail
                bx = min(availX, int(np.floor(take * pX)))
                by = min(availY, int(np.floor(take * pY)))
                bz = min(availZ, int(np.floor(take * pZ)))
                rem = take - (bx + by + bz)
                # distribute remainder by largest fractional part
                frac  = np.array([take*pX - bx, take*pY - by, take*pZ - bz])
                order = np.argsort(-frac)
                for gi in order:
                    if rem == 0: break
                    if   gi == 0 and bx < availX: bx += 1; rem -= 1
                    elif gi == 1 and by < availY: by += 1; rem -= 1
                    elif gi == 2 and bz < availZ: bz += 1; rem -= 1
                # safety fallback: fill any leftover greedily
                if rem > 0:
                    for gi in (0, 1, 2):
                        if rem == 0: break
                        if gi == 0:
                            add = min(rem, max(0, availX - bx)); bx += add; rem -= add
                        elif gi == 1:
                            add = min(rem, max(0, availY - by)); by += add; rem -= add
                        else:
                            add = min(rem, max(0, availZ - bz)); bz += add; rem -= add
                ax2[k] -= bx; ay2[k] -= by; az2[k] -= bz
                ax2[t] += bx; ay2[t] += by; az2[t] += bz
                need -= (bx + by + bz)
                k += 1
            # safety cap (should not trigger in practice)
            tot2 = ax2[t] + ay2[t] + az2[t]
            if tot2 > V_MAX:
                overflow = tot2 - V_MAX
                parts = np.array([ax2[t], ay2[t], az2[t]])
                for gi in np.argsort(-parts):
                    if overflow == 0: break
                    cut = min(overflow, parts[gi])
                    if   gi == 0: ax2[t] -= cut
                    elif gi == 1: ay2[t] -= cut
                    else:         az2[t] -= cut
                    overflow -= cut
        return ax2, ay2, az2

    ax, ay, az = _forward_fill_strict(ax, ay, az, V_MAX, start=win_end)
    return ax, ay, az


# ---------------------------------------------------------------------------
# 2. Hard cap  (source: 3_4.py lines 369-393)
# ---------------------------------------------------------------------------

def cap_to_capacity(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    V_MAX: int,
) -> tuple:
    """
    Trim daily totals that exceed V_MAX by cutting the largest group first.

    Parameters
    ----------
    ax, ay, az : integer dose arrays of length T
    V_MAX      : daily capacity ceiling

    Returns
    -------
    ax, ay, az : trimmed integer arrays
    """
    ax = np.asarray(ax).astype(int).copy()
    ay = np.asarray(ay).astype(int).copy()
    az = np.asarray(az).astype(int).copy()
    T  = len(ax)

    for t in range(T):
        tot = ax[t] + ay[t] + az[t]
        if tot <= V_MAX:
            continue
        overflow = tot - V_MAX
        parts = np.array([ax[t], ay[t], az[t]])
        for gi in np.argsort(-parts):   # cut largest group first
            if overflow == 0: break
            cut = min(overflow, parts[gi])
            if   gi == 0: ax[t] -= cut
            elif gi == 1: ay[t] -= cut
            else:         az[t] -= cut
            overflow -= cut

    return ax, ay, az


# ---------------------------------------------------------------------------
# 3. Simplex normalisation  (source: 3_4.py lines 441-448)
# ---------------------------------------------------------------------------

def to_simplex(d: np.ndarray) -> np.ndarray:
    """
    Normalise a (T, 3) dose matrix so each row sums to 1.

    Days with zero total doses default to [1, 0, 0] (allocate to X).

    Parameters
    ----------
    d : np.ndarray of shape (T, 3), dtype float

    Returns
    -------
    shares : np.ndarray of shape (T, 3), each row summing to 1
    """
    d = np.asarray(d, dtype=np.float32)
    s = d.sum(axis=1, keepdims=True)
    shares = np.divide(d, np.maximum(s, 1e-8), dtype=np.float32)
    zero_days = (s.squeeze() < 1e-8)
    if np.any(zero_days):
        shares[zero_days] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return shares


# ---------------------------------------------------------------------------
# 4. Integer cap with proportional leftover  (source: 3_4.py lines 1335-1346)
# ---------------------------------------------------------------------------

def cap_int(x: float, y: float, z: float, cap: int) -> tuple:
    """
    Distribute up to `cap` doses across x, y, z proportionally (integers).

    If x+y+z <= cap the values are returned as-is (cast to int).
    Otherwise they are scaled down proportionally; the integer remainder
    is distributed to the groups with the largest fractional parts.

    Parameters
    ----------
    x, y, z : raw dose counts (float or int)
    cap      : maximum total allowed

    Returns
    -------
    (int, int, int) summing to <= cap
    """
    total = x + y + z
    if total <= cap:
        return int(x), int(y), int(z)

    ratios = np.array([x, y, z], dtype=float) * cap / total
    alloc  = np.floor(ratios).astype(int)
    leftover = int(cap - alloc.sum())
    fracs  = ratios - alloc
    order  = np.argsort(-fracs)
    for i in range(leftover):
        alloc[order[i % 3]] += 1
    return int(alloc[0]), int(alloc[1]), int(alloc[2])


# ---------------------------------------------------------------------------
# 5. Priority-ordered allocation with carry-over  (source: 3_4.py lines 1598-1632)
# ---------------------------------------------------------------------------

def allocate_by_priority(
    request: np.ndarray,
    avail: np.ndarray,
    cap: int,
    order: list,
) -> np.ndarray:
    """
    Allocate doses to groups following a strict priority order.

    Each group is served up to min(requested, available, remaining_capacity).
    Any unmet demand from the priority group is carried over and offered to
    subsequent groups in order.

    Parameters
    ----------
    request : array-like of length 3 — requested doses per group (int or shares)
              if shares (sum ≈ 1), they are first scaled by cap
    avail   : array-like of length 3 — available susceptibles per group
    cap     : daily capacity ceiling
    order   : list of group indices in priority order, e.g. [3, 2, 1]

    Returns
    -------
    out : np.ndarray of shape (3,), integer doses allocated per group
    """
    req     = np.array(request, dtype=int).copy()
    out     = np.zeros(3, dtype=int)
    rem_cap = int(cap)

    # if shares input (row sums to ~1), convert to integer requests
    if req.sum() <= 0 and np.isclose(np.sum(request), 1.0, atol=1e-6):
        req = np.floor(np.array(request, dtype=float) * cap).astype(int)

    for g in order:
        gi   = g - 1
        want = int(req[gi])
        if want <= 0 or rem_cap <= 0:
            continue
        can     = int(min(want, avail[gi], rem_cap))
        out[gi] += can
        rem_cap -= can
        carry    = want - can
        # redistribute unmet demand to lower-priority groups
        if carry > 0:
            for gg in order:
                if gg == g or rem_cap <= 0:
                    continue
                j    = gg - 1
                room = int(min(avail[j] - out[j], rem_cap))
                add  = int(min(carry, room))
                if add > 0:
                    out[j]  += add
                    rem_cap -= add
                    carry   -= add
                if carry <= 0:
                    break
        if rem_cap <= 0:
            break

    return out
