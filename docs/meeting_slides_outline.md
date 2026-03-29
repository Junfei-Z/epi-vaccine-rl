# Meeting Slides Outline — 2026-03-30

## Prompt for Gemini / AI PPT Tool

> 请根据以下提纲，生成一个学术风格的 PPT（英文），用于跟导师的周会汇报。风格简洁，每页不超过5个要点，配合表格展示实验数据。共约12页。主题是：在网络上的疫苗分配问题中，分析强化学习为什么在大多数场景下不如最优控制，以及在什么条件下强化学习能超过最优控制。

---

## Slide 1: Title

**Node-Level RL vs Optimal Control for Vaccine Allocation on Networks**

Subtitle: Why OC dominates, and when RL can win

Name, Date: 2026-03-30

---

## Slide 2: Background & Problem Setup

- Epidemic simulation on Barabasi-Albert network (N=5000, 3 groups)
  - X: baseline, Y: high-risk (high mortality), Z: high-contact hubs
- Daily vaccine budget V_MAX, horizon T=60 days
- Goal: minimise total deaths D(T)
- Two levels of decision:
  - Inter-group: how many doses to X, Y, Z?
  - Intra-group: which specific nodes to vaccinate?

---

## Slide 3: What I Found — Group RL is Not Individual-Level

- Previous Group RL only decides inter-group allocation (3-dim simplex)
- Within each group, env automatically uses **degree-greedy** heuristic
- So Group RL vs OC is really just comparing **who does group allocation better**
- OC uses ODE exact solver → mathematically optimal → RL can never beat it

| Component | OC-Guided | Group RL | Node RL |
|-----------|-----------|----------|---------|
| Group allocation | ODE exact | PPO learned | PPO learned |
| Node selection | Degree-greedy (free) | Degree-greedy (free) | PPO learned |

---

## Slide 4: Node-Level RL — True Individual Decision-Making

- Implemented **NodeScoringPolicy**: scores every susceptible node
- Input features (per node): degree, infection pressure from neighbours, group, susceptible status
- Selection: Gumbel-top-k (stochastic training), argmax-top-k (deterministic eval)
- Training: PPO with per-step death reward
- Key advantage over OC: can see **which specific nodes are under infection threat right now**

---

## Slide 5: Improvement Attempts

Three improvements to help Node RL:

1. **Warm-start**: use OC solution as decaying score bias in early episodes
2. **Terminal reward (NodeHorizon)**: add -total_deaths * alpha at episode end to reduce myopia
3. **Fair evaluation**: all methods evaluated on same stochastic dynamics with identical RNG seeds

---

## Slide 6: Results — Baseline Scenario (beta=0.08, dY=0.27)

| Method | Deaths (mean +/- std) |
|--------|----------------------|
| OC-Guided | **25.5 +/- 3.7** |
| Group RL (warm) | 27.9 +/- 5.6 |
| Node RL (cold) | 26.6 +/- 4.9 |
| NodeHorizon (alpha=3, warm) | 25.9 +/- 3.7 |

- NodeHorizon nearly matches OC, but does not surpass it
- All RL variants <= OC

---

## Slide 7: Why OC Always Wins — The "Free Lunch" Analysis

OC's advantage comes from **two layers**:

1. **Layer 1 — Group allocation**: ODE solver computes mathematically optimal inter-group doses
2. **Layer 2 — Node selection**: `_choose_to_vaccinate()` automatically picks highest-degree nodes

**Why is degree-greedy so good?**
- When mortality is low, indirect deaths from transmission >> direct deaths from not vaccinating high-risk
- Vaccinating one hub node prevents a chain of 10-50 downstream infections
- This makes degree-greedy nearly optimal in low-mortality settings

---

## Slide 8: Key Insight — When Does Degree-Greedy Fail?

- Degree-greedy is optimal when: **preventing transmission > saving high-risk individuals**
- This assumption breaks when **mortality rate is very high**:
  - If a Y-group node has 35% chance of dying once infected, saving them directly is more valuable than blocking a hub
- Hypothesis: increase Y-group mortality → degree-greedy loses advantage → Node RL can win

---

## Slide 9: Experiment — High-Risk Scenario

Moderate Risk: pY=0.35, dY=0.50 (Y death prob ~17.5%)

| Method | Deaths (mean +/- std) |
|--------|----------------------|
| OC (degree-greedy) | 94.7 +/- 7.0 |
| OC (random select) | 93.4 +/- 11.1 |
| Group RL (warm) | 94.3 +/- 5.9 |
| **Node RL (cold)** | **91.2 +/- 12.2** |
| NodeHorizon (alpha=3, warm) | 94.0 +/- 6.1 |

**Node RL beats OC for the first time!** (91.2 vs 94.7)

---

## Slide 10: Ablation — OC (degree-greedy) vs OC (random)

| Scenario | OC (degree) | OC (random) | Difference |
|----------|-------------|-------------|------------|
| Baseline (dY=0.27) | 25.5 | — | degree matters a lot |
| Moderate risk (dY=0.50) | 94.7 | 93.4 | **~0, degree doesn't help** |
| Extreme risk (dY=0.70) | 185.5 | 192.2 | degree helps again (different reason) |

- In moderate risk: degree-greedy ≈ random → confirms degree priority loses its edge
- Node RL's advantage: it can adaptively balance "block transmission" vs "save high-risk" based on local infection pressure

---

## Slide 11: Summary of Findings

1. Group RL was never truly individual-level — it shared degree-greedy with OC
2. OC's strength = optimal group allocation + free degree-greedy (two-layer advantage)
3. Node RL must learn both layers simultaneously — hard optimisation problem
4. Degree-greedy is nearly optimal in low-mortality settings → hard to beat
5. **When mortality heterogeneity is high enough, Node RL surpasses OC** by leveraging local infection pressure information
6. OC and RL are complementary, not competing

---

## Slide 12: Next Steps

- Explore hybrid approach: OC for group allocation + RL for within-group refinement
- Test on more realistic mortality distributions (age-stratified COVID-19 data)
- Improve RL training: GNN/attention policies, curriculum learning
- Investigate whether RL advantage grows with network size / heterogeneity
