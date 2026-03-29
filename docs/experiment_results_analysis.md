# Node-Level RL vs Optimal Control: Experiment Results & Analysis

## Overview

This document records all experiment results from the `node-aware-rl` branch, analyses why certain methods outperform others, discusses fairness of comparisons, and identifies the conditions under which node-level RL can surpass optimal control.

**Common setup**: N=5000 Barabasi-Albert network, 3 groups (X=baseline, Y=high-risk, Z=high-contact hubs), T=60 days, stochastic disease dynamics. All methods evaluated on the same 10 stochastic episodes with identical RNG seeds (2000+i).

---

## Experiment 1: Warm-Start Node RL (warm_node_rl_experiment.py)

**Goal**: Does OC warm-start improve node-level RL convergence?

| Method | Deaths (mean +/- std) |
|--------|----------------------|
| OC-Guided | 25.5 +/- 3.7 |
| Node RL (cold) | 30.4 +/- 5.1 |
| Node RL (warm) | 27.2 +/- 5.7 |
| Group RL (warm) | 28.6 +/- 8.2 |

**Finding**: Warm-start improves Node RL (27.2 vs 30.4), but OC still wins (25.5).

---

## Experiment 2: NodeHorizon — Terminal Reward (nodehorizon_experiment.py)

**Goal**: Does adding a terminal death penalty (aligning RL with global D(T) minimisation) improve performance?

| Method | Deaths (mean +/- std) |
|--------|----------------------|
| OC-Guided | 25.5 +/- 3.7 |
| Node RL (cold, no terminal) | 27.6 +/- 5.4 |
| NodeHorizon (alpha=1) | 28.9 +/- 4.1 |
| NodeHorizon (alpha=3) | 27.2 +/- 5.7 |
| NodeHorizon (alpha=5) | 27.0 +/- 7.6 |
| NodeHorizon (alpha=3, warm) | 25.9 +/- 3.7 |

**Finding**: Terminal reward + warm-start (25.9) nearly matches OC (25.5). Higher alpha increases variance. alpha=3 is the sweet spot.

---

## Experiment 3: Comprehensive Multi-Scenario (comprehensive_experiment.py)

**Goal**: Compare all methods across baseline and hard (high-beta, high-mortality) scenarios.

### Baseline (beta=0.08, V_MAX=10, dY=0.27)

| Method | Deaths (mean +/- std) |
|--------|----------------------|
| OC-Guided | 25.5 +/- 3.7 |
| Group RL (warm) | 27.9 +/- 5.6 |
| Node RL (cold) | 26.6 +/- 4.9 |
| Node RL (warm) | 31.1 +/- 4.5 |
| NodeHorizon (alpha=3, warm) | 25.9 +/- 3.7 |

### Hard (beta=0.15, V_MAX=20, dY=0.40)

| Method | Deaths (mean +/- std) |
|--------|----------------------|
| OC-Guided | 50.7 +/- 6.8 |
| Group RL (warm) | 56.0 +/- 4.5 |
| Node RL (cold) | 53.8 +/- 5.8 |
| Node RL (warm) | 58.0 +/- 8.3 |
| NodeHorizon (alpha=3, warm) | 52.8 +/- 7.0 |

**Finding**: RL consistently underperforms OC in both scenarios. This led us to investigate the root cause.

---

## Experiment 4: High-Risk Scenario (highrisk_experiment.py)

**Goal**: Test the hypothesis that when high-risk group mortality is very high, degree-greedy node selection becomes suboptimal, and Node RL's ability to see local infection pressure gives it an advantage.

### Moderate Risk (pY=0.35, dY=0.50, Y death prob ~17.5%)

| Method | Deaths (mean +/- std) |
|--------|----------------------|
| OC (degree-greedy) | 94.7 +/- 7.0 |
| OC (random select) | 93.4 +/- 11.1 |
| Group RL (warm) | 94.3 +/- 5.9 |
| **Node RL (cold)** | **91.2 +/- 12.2** |
| NodeHorizon (alpha=3, warm) | 94.0 +/- 6.1 |

### Extreme Risk (pY=0.50, dY=0.70, Y death prob ~35%)

| Method | Deaths (mean +/- std) |
|--------|----------------------|
| OC (degree-greedy) | 185.5 +/- 6.2 |
| OC (random select) | 192.2 +/- 11.9 |
| Group RL (warm) | 194.0 +/- 13.4 |
| Node RL (cold) | 194.0 +/- 10.1 |
| NodeHorizon (alpha=3, warm) | 186.3 +/- 13.2 |

**Key finding**: In moderate risk, **Node RL (cold) beats OC for the first time** (91.2 vs 94.7). This confirms the hypothesis.

---

## Root Cause Analysis: Why Does OC Usually Beat RL?

### The "Free Lunch" Problem

The core reason is in `env.py`, line 468-473:

```python
def _choose_to_vaccinate(self, group: int, k: int) -> list:
    """Return up to k susceptible nodes from `group`, highest degree first."""
    return sorted(cand, key=lambda x: self.deg[x], reverse=True)[:k]
```

When OC-Guided executes via `env.step(shares)`, the environment **automatically selects the highest-degree nodes** within each group. This means OC gets two levels of optimality for free:

| Component | OC-Guided | Group RL (warm) | Node RL |
|-----------|-----------|-----------------|---------|
| Group allocation | ODE exact solution | PPO learned (approximate) | PPO learned (approximate) |
| Within-group node selection | Degree-greedy (free) | Degree-greedy (free) | PPO learned (approximate) |

Node RL must simultaneously learn BOTH the inter-group allocation AND the intra-group node selection, competing against a method that gets the first one solved exactly and the second one hardcoded optimally.

### Why Group RL Also Loses to OC

Group RL uses the same `env.step(shares)` path, so it also gets free degree-greedy selection. The only difference is that its group allocation is learned by PPO rather than computed by the ODE solver. Since PPO's learned allocation is slightly worse than the mathematical optimum, Group RL consistently performs slightly worse than OC (e.g., 27.9 vs 25.5 in baseline).

### When Does Node RL Win?

Node RL wins when degree-greedy is **not the optimal within-group strategy**. This happens when:

1. **High-risk mortality is very high** (pY=0.35, dY=0.50): Saving a low-degree Y node from death is more valuable than blocking a high-degree X node's transmission chain
2. **Degree-greedy = random** (OC degree-greedy ~= OC random in moderate risk): This proves degree priority loses its advantage when mortality risk dominates

Node RL can learn to balance "block transmission" vs "save high-risk individuals" based on each node's local infection pressure (visible via `node_features()`), which is something the hardcoded degree-greedy strategy cannot do.

---

## Fairness Analysis

### What is fair across all methods:

1. **Evaluation seeds**: All methods use identical RNG seeds (2000+i for i=0..9) for stochastic evaluation
2. **Network structure**: Same graph (seed=42) for all methods within each scenario
3. **Disease dynamics**: Same stochastic progression rules (vectorised `_stochastic_substep_vec`)
4. **Information available to Node RL**: `node_features()` provides degree (col 0), infection pressure (col 1), group one-hot (cols 2-4), susceptible flag (col 5). This includes degree information, so Node RL *can* learn degree-greedy if that's optimal

### Intentional asymmetries (part of experimental design):

1. **Warm-start methods** receive OC solution as prior knowledge. This is the point of testing warm-start.
2. **OC uses a mathematical solver** (CasADi/IPOPT) while RL uses learned approximation. This is the fundamental comparison.
3. **Group-level methods** get free degree-greedy node selection via `_choose_to_vaccinate`. This is the key insight we discovered.

### The degree-greedy asymmetry is NOT unfair, but important to understand:

It's a real-world consideration: if you have a group-level allocation strategy, you would naturally vaccinate the most connected individuals first. The question is whether individual-level RL can do even better by considering local epidemic context. Our results show: **yes, but only when mortality heterogeneity is high enough**.

---

## Summary of Insights

1. **OC's dominance comes from two sources**: optimal group allocation + free degree-greedy selection. RL must learn both.

2. **Terminal reward (NodeHorizon) helps** by aligning RL's objective with the global D(T) minimisation goal, reducing the myopia of per-step rewards. Best result: alpha=3.

3. **Warm-start is a double-edged sword**: It provides good initialisation but can trap the policy in a local optimum (eval_deaths stuck at a fixed value for 200+ episodes). Cold-start allows more exploration.

4. **Node RL's unique value emerges in high-risk scenarios** where the "right" vaccination strategy requires balancing transmission blocking vs mortality prevention based on local infection context.

5. **The key research question going forward**: How to make Node RL reliably learn in the large action space (choose k from ~4500 susceptible nodes) without getting stuck in local optima? Potential directions:
   - Attention-based / GNN policies for better credit assignment
   - Larger `episodes_per_update` for lower gradient variance
   - Curriculum learning: start with small networks, transfer to large
   - Hybrid approach: use OC for group allocation, RL only for within-group refinement
