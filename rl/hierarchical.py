# -*- coding: utf-8 -*-
"""
rl/hierarchical.py — Hierarchical RL: Group RL (inter-group) + Node RL (intra-group).

Layer 1: A trained Group RL policy (ActorCritic with Dirichlet) decides
         the allocation shares for X, Y, Z groups.
Layer 2: A Node RL scorer (NodeScoringPolicy) selects which specific nodes
         to vaccinate WITHIN each group, replacing the degree-greedy heuristic.

This decomposes the hard problem (choose k from ~4500 nodes) into two
easier sub-problems:
  - Group allocation: 3-dim simplex (same as before)
  - Within-group ranking: ~1500 candidates per group (3x smaller)
"""

import numpy as np
import torch

from config import S, D
from rl.model import NodeScoringPolicy


class HierarchicalPolicy:
    """
    Combines a frozen Group RL policy with a trainable Node RL scorer.

    At each step:
      1. Group policy outputs shares → env._project_doses() → integer doses per group
      2. Node scorer ranks susceptible nodes within each group
      3. Top-k_g nodes selected per group g (where k_g = doses allocated to g)
      4. Combined node list passed to env.step_node_ids()
    """

    def __init__(self, group_policy, node_scorer: NodeScoringPolicy, env):
        """
        Parameters
        ----------
        group_policy : trained PPO agent (ppo.policy is ActorCritic)
        node_scorer  : NodeScoringPolicy (trainable)
        env          : EpidemicNodeEnv (for _project_doses and group info)
        """
        self.group_policy = group_policy  # ActorCritic, frozen
        self.node_scorer = node_scorer
        self.env = env

    def act(self, deterministic=False):
        """
        Hierarchical action: group allocation → within-group node selection.

        Returns
        -------
        selected_ids : list of int — node ids to vaccinate
        group_shares : np.ndarray (3,) — group allocation shares
        node_log_prob : Tensor — log-prob of node selection (for PPO)
        """
        env = self.env

        # --- Layer 1: Group allocation (frozen) ---
        obs = env._obs()
        with torch.no_grad():
            s_t = torch.from_numpy(obs).float()
            dist = self.group_policy.dist(s_t)
            act = torch.clamp(dist.mean, min=1e-6)
            shares = (act / act.sum()).numpy()

        # Convert shares to integer doses per group
        doses = env._project_doses(shares)  # (3,) int array

        # --- Layer 2: Within-group node scoring ---
        g_state = torch.from_numpy(env.obs_with_pressure()).float()
        s_ids, feats = env.node_features()

        if len(s_ids) == 0:
            return [], shares, None

        s_ids = np.array(s_ids)
        feats_t = torch.from_numpy(feats).float()

        # Score all susceptible nodes
        scores = self.node_scorer.score(g_state, feats_t)

        # Group membership from one-hot features (cols 2,3,4)
        group_labels = np.argmax(feats[:, 2:5], axis=1)  # 0=X, 1=Y, 2=Z

        # Select top-k_g within each group
        selected_indices = []
        for gi in range(3):
            k_g = int(doses[gi])
            if k_g <= 0:
                continue
            mask = (group_labels == gi)
            group_idx = np.where(mask)[0]
            if len(group_idx) == 0:
                continue
            group_scores = scores[group_idx]
            k_actual = min(k_g, len(group_idx))

            if deterministic:
                top_k = torch.topk(group_scores, k_actual).indices
            else:
                # Gumbel-top-k within group
                gumbel = -torch.log(-torch.log(torch.rand_like(group_scores) + 1e-10) + 1e-10)
                perturbed = group_scores + gumbel
                top_k = torch.topk(perturbed, k_actual).indices

            selected_indices.extend(group_idx[top_k.numpy()].tolist())

        # Compute log-prob for PPO (using unbiased scores)
        log_prob = None
        if not deterministic and len(selected_indices) > 0:
            log_probs = torch.log_softmax(scores, dim=0)
            log_prob = log_probs[selected_indices].sum()

        selected_node_ids = s_ids[selected_indices].tolist()
        return selected_node_ids, shares, log_prob


def train_hierarchical(
    G, groups, deg_dict, params_global, capacity_daily,
    group_ppo,
    max_episodes=300,
    episodes_per_update=10,
    lr=3e-4,
    gamma=0.99,
    K_epochs=8,
    eps_clip=0.2,
    seed_counts=None,
    terminal_reward_scale=3.0,
    label=None,
    out_dir='.',
):
    """
    Train the Node RL scorer with a frozen Group RL policy.

    The Group RL policy provides inter-group allocation (Layer 1).
    Only the Node scorer is trained via PPO (Layer 2).

    Parameters
    ----------
    group_ppo       : trained PPO agent (frozen, provides group allocation)
    max_episodes    : training episodes for node scorer
    terminal_reward_scale : terminal death penalty (default 3.0)

    Returns
    -------
    node_scorer : trained NodeScoringPolicy
    hist_eval   : list of eval death counts
    """
    import os
    from env import make_env_from_graph

    os.makedirs(out_dir, exist_ok=True)

    env, _, _, _, _ = make_env_from_graph(
        G=G, groups=groups, deg_dict=deg_dict,
        params_global=params_global, capacity_daily=capacity_daily,
        seed_counts=seed_counts, deterministic=False,
    )

    node_scorer = NodeScoringPolicy(hidden=64)
    group_policy = group_ppo.policy  # frozen ActorCritic
    group_policy.eval()

    hier = HierarchicalPolicy(group_policy, node_scorer, env)

    optimizer = torch.optim.Adam(node_scorer.parameters(), lr=lr)
    MSE = torch.nn.MSELoss()

    # Buffers (same structure as run_training_node_rl)
    buf_g_states = []
    buf_feats = []
    buf_sel_idxs = []
    buf_old_logp = []
    buf_rewards = []
    buf_dones = []

    best_death = float('inf')
    best_state = None
    hist_eval = []
    patience_counter = 0

    def eval_det(n_eval=3):
        node_scorer.eval()
        deaths = []
        for _ in range(n_eval):
            env_e, _, _, _, _ = make_env_from_graph(
                G=G, groups=groups, deg_dict=deg_dict,
                params_global=params_global, capacity_daily=capacity_daily,
                seed_counts=seed_counts, deterministic=True,
            )
            env_e.reset(seed_counts=seed_counts)
            hier_e = HierarchicalPolicy(group_policy, node_scorer, env_e)
            done = False
            while not done:
                selected, _, _ = hier_e.act(deterministic=True)
                _, _, done, _ = env_e.step_node_ids(selected)
            deaths.append(int(np.sum(env_e.status == D)))
        node_scorer.train()
        return float(np.mean(deaths))

    for ep in range(max_episodes):
        env.reset(seed_counts=seed_counts)
        hier.env = env
        done = False

        while not done:
            g_np = env.obs_with_pressure()
            s_ids, feats = env.node_features()

            if len(s_ids) == 0:
                _, reward, done, _ = env.step_node_ids([])
                if done and terminal_reward_scale > 0:
                    total_deaths = int(np.sum(env.status == D))
                    reward += -total_deaths * terminal_reward_scale
                buf_g_states.append(g_np)
                buf_feats.append(None)
                buf_sel_idxs.append(torch.tensor([], dtype=torch.long))
                buf_old_logp.append(0.0)
                buf_rewards.append(reward)
                buf_dones.append(float(done))
                continue

            selected_ids, shares, log_prob = hier.act(deterministic=False)
            _, reward, done, _ = env.step_node_ids(selected_ids)

            if done and terminal_reward_scale > 0:
                total_deaths = int(np.sum(env.status == D))
                reward += -total_deaths * terminal_reward_scale

            # Store transition — need to record which indices were selected
            # relative to the node_features output
            s_ids_arr = np.array(s_ids)
            sel_mask = np.isin(s_ids_arr, selected_ids)
            sel_idxs = torch.tensor(np.where(sel_mask)[0], dtype=torch.long)

            buf_g_states.append(g_np)
            buf_feats.append(feats)
            buf_sel_idxs.append(sel_idxs)
            buf_old_logp.append(float(log_prob.item()) if log_prob is not None else 0.0)
            buf_rewards.append(reward)
            buf_dones.append(float(done))

        # PPO update
        if (ep + 1) % episodes_per_update == 0:
            T = len(buf_rewards)
            rewards = torch.tensor(buf_rewards, dtype=torch.float32)
            dones = torch.tensor(buf_dones, dtype=torch.float32)
            old_logp = torch.tensor(buf_old_logp, dtype=torch.float32)

            with torch.no_grad():
                values = torch.tensor([
                    node_scorer.value(
                        torch.from_numpy(buf_g_states[t]).float()
                    ).item()
                    for t in range(T)
                ], dtype=torch.float32)

            next_v = torch.cat([values[1:], torch.tensor([0.0])])
            adv = torch.zeros(T, dtype=torch.float32)
            gae = 0.0
            for t in reversed(range(T)):
                delta = rewards[t] + gamma * next_v[t] * (1 - dones[t]) - values[t]
                gae = delta + gamma * 0.95 * (1 - dones[t]) * gae
                adv[t] = gae
            returns = (adv + values).detach()
            if adv.std() > 1e-6:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv = adv.detach()

            for _ in range(K_epochs):
                logps = []
                v_preds = []
                for t in range(T):
                    g_t = torch.from_numpy(buf_g_states[t]).float()
                    v_preds.append(node_scorer.value(g_t).squeeze())

                    if buf_feats[t] is None or len(buf_sel_idxs[t]) == 0:
                        logps.append(torch.tensor(0.0))
                        continue

                    f_t = torch.from_numpy(buf_feats[t]).float()
                    scores = node_scorer.score(g_t, f_t)
                    lp = torch.log_softmax(scores, dim=0)
                    logps.append(lp[buf_sel_idxs[t]].sum())

                logps = torch.stack(logps)
                v_preds = torch.stack(v_preds).squeeze()

                ratios = torch.exp(logps - old_logp)
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * adv

                loss = (-torch.min(surr1, surr2)
                        + 0.5 * MSE(v_preds, returns)).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(node_scorer.parameters(), 0.5)
                optimizer.step()

            buf_g_states.clear(); buf_feats.clear(); buf_sel_idxs.clear()
            buf_old_logp.clear(); buf_rewards.clear(); buf_dones.clear()

            eval_deaths = eval_det(n_eval=3)
            hist_eval.append(eval_deaths)
            if eval_deaths < best_death:
                best_death = eval_deaths
                best_state = {k: v.cpu().clone()
                              for k, v in node_scorer.state_dict().items()}

            print(f"[hier_rl] ep={ep+1:3d}  eval_deaths={eval_deaths:.1f}")

        # Early stopping
        if len(hist_eval) >= 30 and ep >= 40:
            recent = np.array(hist_eval[-30:])
            rel_std = recent.std() / max(1.0, recent.mean())
            if rel_std < 0.05:
                patience_counter += 1
                if patience_counter >= 4:
                    print(f"[hier_rl] Early stop at episode {ep}")
                    break
            else:
                patience_counter = 0

    if best_state is not None:
        node_scorer.load_state_dict(best_state)
        if label is not None:
            save_path = os.path.join(out_dir, f'best_hier_scorer_{label}.pt')
            torch.save(node_scorer.state_dict(), save_path)
            print(f"[hier_rl] Best scorer saved -> {save_path} (deaths={best_death:.1f})")

    return node_scorer, hist_eval
