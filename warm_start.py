# -*- coding: utf-8 -*-
"""
warm_start.py — OC-guided warm-start for node-level RL.

Bridges the gap between population-level ODE optimal control and node-level
RL by generating teacher trajectories and performing behavioral cloning.

The ODE gives group-level doses [a_X(t), a_Y(t), a_Z(t)] per day.
Within each group, the env already vaccinates highest-degree nodes first.
This module:
  1. Rolls the ODE solution through the node-level env to record which
     specific nodes get vaccinated each day (the "teacher trajectory").
  2. Pre-trains the NodeScoringPolicy scorer via cross-entropy loss so
     it learns to reproduce the OC node selections.
  3. After BC, the policy is handed off to PPO for free exploration.

Functions
---------
generate_oc_teacher_trajectory — roll ODE doses through env, record per-day targets
behavioral_cloning             — pre-train scorer to match OC node selections
"""

import numpy as np
import torch
import torch.nn.functional as F

from config import S, D, P, A, I
from env import make_env_from_graph
from allocation import allocate_by_priority


def generate_oc_teacher_trajectory(
    G,
    groups: dict,
    deg_dict: dict,
    params_global: dict,
    capacity_daily: int,
    doses_seq: np.ndarray,
    seed_counts: dict,
    priority_order: list = None,
) -> list:
    """
    Roll ODE dose allocations through the node-level env and record the
    teacher's per-day decisions.

    At each day t the ODE prescribes integer doses [a_X, a_Y, a_Z].
    The env applies these using high-degree-first ordering within each group.
    We record (global_state, node_feats, s_node_ids, selected_mask) so the
    scorer can be trained to reproduce these selections.

    Parameters
    ----------
    G, groups, deg_dict, params_global, capacity_daily : env spec
    doses_seq      : np.ndarray (T, 3) — ODE integer doses per day
    seed_counts    : dict {group -> initial infected count}
    priority_order : list of group indices in priority order (default [3,2,1])

    Returns
    -------
    trajectory : list of dicts, one per day, each containing:
        'g_state'      : np.ndarray (34,) — global observation with pressure
        'node_feats'   : np.ndarray (n_s, 6) — features of susceptible nodes
        's_node_ids'   : list of int — susceptible node IDs
        'target_idxs'  : np.ndarray of int — indices into s_node_ids that OC selected
        'day'          : int
    """
    if priority_order is None:
        priority_order = [3, 2, 1]

    env, _, _, _, _ = make_env_from_graph(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity_daily,
        seed_counts=seed_counts, deterministic=True,
    )
    env.reset(seed_counts=seed_counts)

    trajectory = []
    T = min(len(doses_seq), int(params_global.get('T_HORIZON', 60)))

    for t in range(T):
        # record pre-vaccination state
        g_state = env.obs_with_pressure()               # (34,)
        s_node_ids, node_feats = env.node_features()     # lists, (n_s, 6)

        if len(s_node_ids) == 0:
            # no susceptibles left — advance env with empty action
            env.step_node_ids([])
            continue

        # determine which nodes OC would vaccinate
        avail = np.array([
            np.sum(env.status[env.group_nodes[g]] == S)
            for g in [1, 2, 3]
        ], dtype=int)
        req = doses_seq[t].astype(int)
        final_doses = allocate_by_priority(req, avail, capacity_daily, priority_order)

        # collect selected node IDs using env's high-degree-first logic
        selected_nodes = []
        for gi, g in enumerate([1, 2, 3]):
            k = int(final_doses[gi])
            if k > 0:
                sel = env._choose_to_vaccinate(g, k)
                selected_nodes.extend(sel)

        # map selected node IDs to indices in s_node_ids
        id_to_idx = {nid: i for i, nid in enumerate(s_node_ids)}
        target_idxs = np.array(
            [id_to_idx[nid] for nid in selected_nodes if nid in id_to_idx],
            dtype=int,
        )

        trajectory.append({
            'g_state':    g_state.copy(),
            'node_feats': node_feats.copy(),
            's_node_ids': list(s_node_ids),
            'target_idxs': target_idxs,
            'day':        t,
        })

        # advance env with the OC-selected nodes
        env.step_node_ids(selected_nodes)

    return trajectory


def behavioral_cloning(
    policy,
    trajectory: list,
    n_epochs: int = 50,
    lr: float = 1e-3,
    verbose: bool = True,
) -> list:
    """
    Pre-train the NodeScoringPolicy scorer to reproduce OC node selections.

    For each day in the teacher trajectory:
      - Compute scores for all susceptible nodes
      - Apply cross-entropy loss: the target is a multi-label classification
        where selected nodes have label 1 and others have label 0
      - This teaches the scorer to assign high scores to the nodes that
        OC would vaccinate (high-degree, priority-group nodes)

    After BC the scorer outputs roughly match OC priorities, giving PPO a
    strong initialisation without constraining the action space.

    Parameters
    ----------
    policy     : NodeScoringPolicy instance (will be modified in-place)
    trajectory : output of generate_oc_teacher_trajectory
    n_epochs   : number of passes over the trajectory
    lr         : learning rate for the BC optimizer
    verbose    : print loss every 10 epochs

    Returns
    -------
    losses : list of float — mean loss per epoch
    """
    # filter out days with no targets (e.g. all susceptibles already gone)
    valid = [d for d in trajectory if len(d['target_idxs']) > 0]
    if len(valid) == 0:
        if verbose:
            print("[BC] No valid teacher data — skipping behavioral cloning")
        return []

    optimizer = torch.optim.Adam(policy.scorer.parameters(), lr=lr)
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for day_data in valid:
            g_state = torch.from_numpy(day_data['g_state']).float()
            feats   = torch.from_numpy(day_data['node_feats']).float()
            n_nodes = feats.shape[0]

            # target: binary vector — 1 for nodes OC selected
            target = torch.zeros(n_nodes, dtype=torch.float32)
            for idx in day_data['target_idxs']:
                if idx < n_nodes:
                    target[idx] = 1.0

            # forward: score all susceptible nodes
            scores = policy.score(g_state, feats)   # (n_nodes,)

            # binary cross-entropy loss: teach scorer to assign high scores
            # to OC-selected nodes and low scores to others
            loss = F.binary_cross_entropy_with_logits(scores, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.scorer.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        mean_loss = epoch_loss / len(valid)
        losses.append(mean_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"[BC] epoch {epoch+1:3d}/{n_epochs}  loss={mean_loss:.4f}")

    return losses
