# -*- coding: utf-8 -*-
"""
graph.py — Network construction and contact matrix computation.

Builds a Barabasi-Albert contact network and partitions nodes into
three risk/contact groups (X, Y, Z), then computes the inter-group
contact matrix used by the epidemic model.
"""

import numpy as np
import networkx as nx


def build_graph_nlpa(
    n: int,
    m: int,
    alpha_pa: float,
    seed: int,
    high_risk_prob: float,
    alpha_std: float,
) -> tuple:
    """
    Build a graph using non-linear preferential attachment (NLPA) and
    partition nodes into three groups.

    Standard BA uses linear PA: p_i ∝ k_i.
    NLPA generalises to:  p_i ∝ k_i^alpha_pa / Σ_j k_j^alpha_pa

    Regime summary (Wikipedia — "Non-linear preferential attachment"):
      alpha_pa < 1  — sub-linear PA → stretched exponential degree dist.
                       hubs still exist but are less dominant
      alpha_pa = 1  — standard BA → power law with γ ≈ 3  (same as BA)
      alpha_pa > 1  — super-linear PA → winner-takes-all, one or few
                       nodes accumulate almost all edges ("gelation")

    Implementation notes
    --------------------
    NetworkX does not expose the alpha exponent natively, so we grow the
    graph manually using the Barabási–Albert growth protocol:
      • Start with a complete graph on (m+1) nodes.
      • At each step add one node; connect it to m existing nodes sampled
        without replacement with probability ∝ k_i^alpha_pa.
      • Self-loops are forbidden; multi-edges are avoided.

    Parameters
    ----------
    n            : total number of nodes in the final graph
    m            : number of edges each new node attaches to
    alpha_pa     : PA non-linearity exponent
    seed         : RNG seed
    high_risk_prob : probability a non-hub node is assigned to group Y
    alpha_std    : hub threshold multiplier (Z if degree ≥ μ + alpha_std·σ)

    Returns
    -------
    G        : networkx Graph
    groups   : dict with keys 'X', 'Y', 'Z' mapping to sets of node ids
    deg_dict : dict mapping node id → degree
    """
    rng = np.random.default_rng(seed)

    # --- grow graph with NLPA ---
    G = nx.complete_graph(m + 1)
    # ensure degrees are ints in a mutable dict
    deg = dict(G.degree())

    for new_node in range(m + 1, n):
        existing = list(G.nodes())
        # compute NLPA weights
        weights = np.array([deg[v] ** alpha_pa for v in existing], dtype=float)
        total = weights.sum()
        if total == 0:
            probs = np.ones(len(existing)) / len(existing)
        else:
            probs = weights / total

        # sample m distinct targets
        targets = rng.choice(existing, size=m, replace=False, p=probs)

        G.add_node(new_node)
        deg[new_node] = 0
        for t in targets:
            G.add_edge(new_node, t)
            deg[new_node] += 1
            deg[t] += 1

    deg_dict = deg

    # --- group assignment (same logic as build_graph_and_groups) ---
    deg_vals  = np.array(list(deg_dict.values()), dtype=float)
    mu, sigma = deg_vals.mean(), deg_vals.std()
    threshold = mu + alpha_std * sigma

    groups = {'X': set(), 'Y': set(), 'Z': set()}
    for node in G.nodes():
        if deg_dict[node] >= threshold:
            groups['Z'].add(node)
        elif rng.random() <= high_risk_prob:
            groups['Y'].add(node)
        else:
            groups['X'].add(node)

    return G, groups, deg_dict


def build_graph_and_groups(
    n: int,
    m: int,
    seed: int,
    high_risk_prob: float,
    alpha_std: float,
) -> tuple:
    """
    Build a Barabasi-Albert graph and partition nodes into three groups.

    Groups
    ------
    Z — high-contact: degree >= mean + alpha_std * std  (network hubs)
    Y — high-risk:    non-hub nodes drawn with probability high_risk_prob
    X — baseline:     all remaining nodes

    Parameters
    ----------
    n            : number of nodes
    m            : BA attachment parameter (edges per new node)
    seed         : random seed for reproducibility
    high_risk_prob : probability a non-hub node is assigned to Y
    alpha_std    : hub threshold multiplier (Z if degree >= mu + alpha_std*sigma)

    Returns
    -------
    G        : networkx Graph
    groups   : dict with keys 'X', 'Y', 'Z' mapping to sets of node ids
    deg_dict : dict mapping node id -> degree
    """
    G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    rng = np.random.default_rng(seed)

    deg_dict = dict(G.degree())
    deg_vals = np.array(list(deg_dict.values()), dtype=float)
    mu, sigma = deg_vals.mean(), deg_vals.std()
    threshold = mu + alpha_std * sigma

    groups = {'X': set(), 'Y': set(), 'Z': set()}
    for node in G.nodes():
        if deg_dict[node] >= threshold:
            groups['Z'].add(node)
        elif rng.random() <= high_risk_prob:
            groups['Y'].add(node)
        else:
            groups['X'].add(node)

    return G, groups, deg_dict


def get_contact_matrix(G, groups: dict) -> tuple:
    """
    Compute the 3x3 inter-group contact matrix from graph edges.

    Each entry C[i, j] is the average number of contacts that a node in
    group i+1 has with nodes in group j+1, normalised by group i size.

    Parameters
    ----------
    G      : networkx Graph
    groups : dict with keys 'X', 'Y', 'Z' (sets of node ids)

    Returns
    -------
    C            : np.ndarray of shape (3, 3)
    N_g          : dict {1: |X|, 2: |Y|, 3: |Z|}
    node_to_group: dict mapping node id -> group index (1, 2, or 3)
    """
    node_to_group = {}
    for node in groups['X']:
        node_to_group[node] = 1
    for node in groups['Y']:
        node_to_group[node] = 2
    for node in groups['Z']:
        node_to_group[node] = 3

    N_g = {1: len(groups['X']), 2: len(groups['Y']), 3: len(groups['Z'])}

    C = np.zeros((3, 3))
    for u, v in G.edges():
        gu, gv = node_to_group[u], node_to_group[v]
        C[gu - 1, gv - 1] += 1
        C[gv - 1, gu - 1] += 1

    for i in range(3):
        gid = i + 1
        if N_g[gid] > 0:
            C[i] = C[i] / N_g[gid]

    return C, N_g, node_to_group
