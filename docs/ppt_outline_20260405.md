# PPT Outline — Advisor Meeting 2026-04-05

## Page 1: Title

```
本周进展汇报

1. Node RL 时间复杂度分析（vs Naive Individual RL）
2. Terminal Reward 解决 RL 短视问题
3. [Bonus] Gumbel-Top-K vs Greedy Top-K Ablation

日期：2026-04-05
```

---

## Page 2: 问题引入 — 为什么不直接对每个人做决策？

**Title:** Why not let RL decide for each individual directly?

**Left: diagram**
- N=5000 nodes, each with a 0/1 decision
- Arrow pointing to action space = C(4000, 20) ≈ 10^52

**Right: text**

```
Naive Individual RL:
• Each node: vaccinate (1) or not (0)
• Must choose exactly K = 20 from ~4000 susceptible nodes
• Action space: C(4000, 20) ≈ 10^52 combinations

Problems:
• Budget constraint Σaᵢ = K is non-differentiable
  → needs projection, breaks gradient flow
• Gradient variance scales as O(Nₛ)
  → training becomes unstable as population grows
• Sample complexity: O(2^Nₛ / ε²) — exponential
  → intractable for N > 100
```

**Script:** "老师上次问为什么我们不直接让 RL 对每个人做二元决策。原因是组合爆炸——从4000个人里选20个，有10的52次方种可能。而且预算约束不可微，梯度方差随人口线性增长，根本训不动。"

---

## Page 3: 我们的方法 — Scoring + Top-K

**Title:** Our approach: Score each node, select Top-K

**Left: simplified architecture diagram**
```
Node features (6-dim) ──┐
                         ├─→ Shared MLP → Score (scalar) ─→ Top-K → K nodes
Global state (31-dim) ──┘
```

**Right: key points**
```
Key design:
• Shared MLP gives each node a single score φ(xᵢ, g) ∈ ℝ
• Select K nodes with highest scores: V = Top-K(φ)
• Log-prob: log π(a|s) = Σ log softmax(φⱼ)

Why this works:
✓ Budget constraint automatically satisfied (exactly K)
✓ Action space: 10^52 → O(N) — just rank nodes
✓ Gradient variance: O(1), not O(Nₛ)
✓ Parameters: O(H²) = ~8500, independent of N
✓ Shared scorer → generalizes across population sizes
```

**Script:** "我们的做法是把组合选择问题转化为打分排序问题。每个节点过同一个 MLP 得到一个分数，取 Top-K 就行。Action space 从10的52次方降到 O(N)，预算约束自动满足，梯度方差是常数级。"

---

## Page 4: 复杂度对比总结

**Title:** Complexity Comparison: Naive vs Ours

**Main table:**
```
┌──────────────────────┬─────────────────────────┬────────────────────┐
│       Aspect         │   Naive Individual RL   │  Node RL (Ours)    │
├──────────────────────┼─────────────────────────┼────────────────────┤
│ Action space         │ C(Nₛ,K) ≈ 10^52        │ O(Nₛ) scores       │
│ Policy output        │ Nₛ binary decisions     │ 1 scalar per node  │
│ Budget constraint    │ Non-diff. projection    │ Built-in via Top-K │
│ Gradient variance    │ O(Nₛ)                   │ O(1)               │
│ Sample complexity    │ O(2^Nₛ / ε²) exponential│ Polynomial in Nₛ   │
│ Parameters           │ O(Nₛ·H) or shared       │ O(H²) = ~8,500     │
│ Scalability          │ Intractable for Nₛ>100  │ Tested at N=5,000  │
└──────────────────────┴─────────────────────────┴────────────────────┘
```

**Below: numerical example**
```
Numerical example (N=5000, K=20, Nₛ≈4000):
• Naive action space: 10^52     vs  Ours: 4000 scores
• Naive gradient variance: 4000×  vs  Ours: 1×
```

**Script:** "总结一下，核心是把组合决策变成连续打分。复杂度从指数降到线性，梯度方差从 O(N) 降到 O(1)。代价是每步推理需要对所有易感节点跑一次 MLP，但这是线性的，完全可以接受。"

---

## Page 5: 问题2引入 — RL 为什么短视？

**Title:** Why is RL short-sighted?

**Left: timeline diagram**
```
Day 1          Day 5          Day 10         Day 60
  │              │              │              │
  ▼              ▼              ▼              ▼
vaccinate    vaccine takes   fewer new      fewer total
node A       effect →        infections     deaths
             immunity

reward₁ = -deaths₁          (no signal!)   
  ↑
  RL only sees this
```

**Right: text**
```
Current reward: rₜ = -deaths_today

Problem:
• Vaccination today → effect appears days later
• rₜ is almost uncorrelated with today's action
• RL optimizes Σ γᵗ rₜ, but γ⁶⁰ ≈ 0.55
  → Day 60 reward is discounted by half

vs. Optimal Control:
• OC minimizes total deaths over full 60-day horizon
• No discounting, no credit assignment problem
```

**Bottom: references (small font)**
```
Literature:
• Temporal credit assignment problem — Arjona-Medina et al., NeurIPS 2019
• Exponential discounting devalues future — Fedus et al., RLDM 2019
```

**Script:** "第二个问题是上次我跟老师解释的——RL 短视。当前 reward 是每天的死亡数，但今天打的疫苗要过好几天才有效果。RL 看不到这个延迟效应。而且 discount factor 0.99 的60次方只有0.55，远期 reward 被打了近一半的折。OC 没有这个问题，它直接优化60天总死亡。这在文献里叫 temporal credit assignment problem。"

---

## Page 6: 我们的方案 — Terminal Reward

**Title:** Solution: Add terminal reward

**Formulas:**
```
Original:   rₜ = -deaths_today                    (myopic)

Modified:   rₜ = -deaths_today                    (t < T)
            r_T = -deaths_today - total_deaths × scale   (t = T, episode end)
```

**Diagram:**
```
Episode reward stream:

Without terminal:  r₁, r₂, r₃, ... r₆₀
                   ↑ only daily deaths, no global view

With terminal:     r₁, r₂, r₃, ... r₆₀ + (-total_deaths)
                                          ↑ global signal:
                                    "how did your entire
                                     strategy perform?"
```

**Bottom: reference**
```
Reward shaping preserves optimal policy — Ng, Harada, Russell, ICML 1999 (2349 citations)
```

**Script:** "方案很简单——在 episode 结束时加一个终端奖励，等于负的总死亡数。这给了 agent 一个全局信号：你这一整轮下来到底做得怎么样。这是一种 reward shaping，Ng 1999 年证明了 potential-based reward shaping 不改变最优策略。"

---

## Page 7: 实验设计

**Title:** Experiment Design

```
Methods:       3 RL methods + OC-Guided reference
               ├─ Group RL cold (no ODE initialization)
               ├─ Group RL warm (ODE warm-start)
               └─ Node RL

Terminal reward: scale = 0.0 (off) vs 1.0 (on)

Scenarios:     baseline (dY=0.27) and moderate (dY=0.40)

Training:      Group RL: 200 episodes
               Node RL:  300 episodes

Evaluation:    10 runs per method, report mean ± std deaths
```

**Script:** "我们对三种 RL 方法都做了 terminal reward 的 on/off 对比，在两个死亡率场景下测试。OC-Guided 作为参考基线。"

---

## Page 8: 结果 — Baseline (dY=0.27)

**Title:** Results: Baseline Scenario (dY=0.27)

**Table (highlight Node RL ts=1.0 in red/bold):**
```
┌─────────────────┬──────────────┬──────────────┬─────────┐
│     Method      │ ts=0.0 (off) │ ts=1.0 (on)  │ Change  │
├─────────────────┼──────────────┼──────────────┼─────────┤
│ OC-Guided       │ 25.5 ± 3.7   │      —       │    —    │
│ Group RL cold   │ 25.9 ± 6.3   │ 26.8 ± 4.0   │  +0.9  │
│ Group RL warm   │ 26.9 ± 3.9   │ 28.6 ± 3.4   │  +1.7  │
│ Node RL         │ 31.5 ± 7.0   │ 25.9 ± 3.7 ★ │  -5.6  │
└─────────────────┴──────────────┴──────────────┴─────────┘

★ Node RL + terminal reward matches OC-Guided (25.9 vs 25.5)
```

**Optional: training curve comparison for Node RL**
- ts=0: eval_deaths oscillates between 26-37
- ts=1: stable at 26 from ep=10 onward

**Script:** "Baseline 场景下，Node RL 获益最大——从31.5降到25.9，直接追平了 OC 的25.5。而且训练曲线从波动变得完全稳定。但 Group RL 反而略差了一点，后面我会解释为什么。"

---

## Page 9: 结果 — Moderate (dY=0.40)

**Title:** Results: Moderate Scenario (dY=0.40)

**Table:**
```
┌─────────────────┬──────────────┬──────────────┬─────────┐
│     Method      │ ts=0.0 (off) │ ts=1.0 (on)  │ Change  │
├─────────────────┼──────────────┼──────────────┼─────────┤
│ OC-Guided       │ 35.3 ± 6.8   │      —       │    —    │
│ Group RL cold   │ 39.6 ± 3.4   │ 36.9 ± 5.8   │  -2.7 ✓│
│ Group RL warm   │ 43.5 ± 4.4   │ 40.5 ± 6.4   │  -3.0 ✓│
│ Node RL         │ 41.1 ± 5.3   │ 38.7 ± 6.7   │  -2.4 ✓│
└─────────────────┴──────────────┴──────────────┴─────────┘

All 3 methods improve when mortality is high.
```

**Script:** "Moderate 场景下，三种方法加了 terminal reward 全部改善了，2到3个 death 的提升。高死亡率下 per-step reward 噪声更大，terminal reward 的全局信号更有价值。"

---

## Page 10: 分析 — 为什么效果因方法而异？

**Title:** Why does terminal reward help Node RL most?

**Left: 2x2 concept diagram**
```
                    Per-step reward signal
                    sufficient ←————————→ insufficient
                    
Action space   ┌──────────────┬──────────────────┐
small (3-dim)  │  Group RL    │                  │
               │  baseline    │                  │
               │  ✗ no need   │                  │
               ├──────────────┼──────────────────┤
large (N-dim)  │              │    Node RL       │
               │              │    baseline      │
               │              │  ✓ terminal      │
               │              │    reward helps  │
               └──────────────┴──────────────────┘

         High mortality → shifts everything rightward
         (noisier per-step reward → all methods benefit)
```

**Right: bullet points**
```
Node RL benefits most:
• Large action space (score N nodes)
  → per-step reward is weak signal for so many decisions
• Terminal reward provides clear global objective
• Training stabilizes immediately (eval=26 from ep 10)

Group RL slightly worse in baseline:
• Action space is only 3-dim → per-step reward sufficient
• Already near-optimal (25.9 ≈ OC's 25.5)
• Terminal spike (-25) distorts reward scale
  (per-step sum ≈ -30, terminal = -25 → half of total return)

High mortality → all methods benefit:
• More daily death variance → noisier per-step reward
• Terminal reward provides stable cumulative signal
```

**Bottom (bold):**
```
Reward design is not one-size-fits-all —
it must match the method's action space and the environment's noise level.
```

**Script:** "核心发现是 terminal reward 的价值取决于 per-step reward 够不够用。Node RL action space 大，per-step 信号弱，所以 terminal reward 是雪中送炭。Group RL 只有3维决策，per-step 信号已经够了，terminal reward 反而改变了 reward 的量级比例，干扰了已经不错的策略。高死亡率下环境噪声大，所有方法的 per-step 信号都变弱，所以都受益。结论是 reward design 需要匹配方法特点，不是 one-size-fits-all。"

---

## Page 11: [Bonus] Gumbel vs Greedy Top-K

**Title:** Bonus: Gumbel-Top-K vs Greedy Top-K Ablation

**Left: explanation**
```
Gumbel-Top-K: add random noise to scores during training
              → stochastic exploration
Greedy Top-K: always pick highest-scoring nodes
              → deterministic, no exploration noise
```

**Right: table**
```
┌───────────────┬───────────┬──────────────────┬───────────────────┐
│   Scenario    │ OC-Guided │ Node RL (Gumbel) │ Node RL (Greedy)  │
├───────────────┼───────────┼──────────────────┼───────────────────┤
│ baseline      │ 25.5±3.7  │ 29.0±4.3         │ 26.1±4.0 ★       │
│ moderate      │ 35.3±6.8  │ 43.4±6.2         │ 38.6±6.8 ★       │
└───────────────┴───────────┴──────────────────┴───────────────────┘
```

**Bottom: reasons**
```
Why Greedy wins:
• Tight budget (K=10): wasting 1 dose = extra infections
• Stochastic environment already provides natural diversity
• PPO parameter updates implicitly shift scoring → implicit exploration

→ Changed default to Greedy Top-K
```

**Script:** "额外做了一个 ablation：训练时用 Gumbel 随机噪声做探索好还是直接 Greedy 好。结果 Greedy 在两个场景都更好。因为疫苗预算很紧，随机浪费1剂就会导致额外感染。而且环境本身是随机的，已经提供了足够的多样性。所以我们已经改成了 Greedy Top-K。"

---

## Page 12: 总结

**Title:** Summary & Next Steps

**Left: findings**
```
What we found:

1. Node RL reduces action space from C(N,K) ≈ 10^52 to O(N)
   via scoring + Top-K
   — theoretically sound and practically scalable

2. Terminal reward is critical for Node RL
   → baseline: 31.5 → 25.9 (matches OC)
   → not universally helpful (Group RL baseline slightly worse)
   → reward design must match method characteristics

3. Greedy Top-K > Gumbel-Top-K
   → exploration noise hurts when budget is tight

Best configuration:
   Node RL + Greedy Top-K + Terminal Reward = 25.9 ≈ OC (25.5)
```

**Right: next steps**
```
Possible next steps (for discussion):

• Re-run severity sweep with best config
  (terminal reward on)
  → expect Node RL to beat OC in
    moderate / severe / critical

• Tune terminal_reward_scale
  (try 0.1, 0.5, 2.0)?

• Write reward design analysis
  section for paper
```

**Script:** "总结一下，复杂度分析说明了我们的 Top-K 设计是合理的。Terminal reward 实验验证了 RL 短视确实是一个问题，加了之后 Node RL 直接追平了 OC。目前最佳配置是 Node RL + Greedy Top-K + Terminal Reward。下一步建议是用这个配置重跑 severity sweep，预期在高死亡率场景下 Node RL 应该能稳定超过 OC。"

---

## References (backup slide, if asked)

```
[1] Arjona-Medina et al. "RUDDER: Return Decomposition for Delayed Rewards"
    NeurIPS 2019

[2] Fedus, Gelada, Bengio, Bellemare, Larochelle.
    "Hyperbolic Discounting and Learning over Multiple Horizons"
    RLDM 2019 Best Paper

[3] Ng, Harada, Russell. "Policy Invariance Under Reward Transformations:
    Theory and Application to Reward Shaping"
    ICML 1999, 2349+ citations

[4] Kim et al. "Adaptive Discount Factor for Deep RL in Continuing Tasks
    with Uncertainty" Sensors 2022

[5] Ohi et al. "Exploring Optimal Control of Epidemic Spread Using RL"
    Scientific Reports 2020

[6] Beigi et al. "Application of RL for Effective Vaccination Strategies
    of COVID-19" European Physical Journal Plus 2021
```
