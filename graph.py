# -*- coding: utf-8 -*-
"""
graph.py — Network construction and contact matrix computation.

Builds a Barabasi-Albert contact network and partitions nodes into
three risk/contact groups (X, Y, Z), then computes the inter-group
contact matrix used by the epidemic model.
"""

import numpy as np
import networkx as nx


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
