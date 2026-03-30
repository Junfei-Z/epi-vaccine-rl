# ODE vs BA Network: Model Mismatch and Its Impact on Method Performance

## The Two Models

### ODE Model (used by Optimal Control)

The ODE solver treats each group as a continuous, well-mixed compartment:

- **Homogeneous mixing within groups**: any two nodes in the same group have equal contact probability
- **Deterministic transitions**: dS/dt = -βSI/N ... (continuous flows)
- **Inter-group coupling via 3×3 contact matrix C**: captures average cross-group interaction
- **Continuous population**: can have 3.7 infected individuals
- **No network structure**: no concept of degree, neighbours, or local clustering

### BA Network Model (used by RL training environment)

The simulation runs on a Barabasi-Albert scale-free network:

- **Heterogeneous contacts**: degree follows power-law distribution (hubs with degree 50+, periphery with degree 3)
- **Stochastic transitions**: each node independently samples its state transition
- **Transmission along edges only**: infection can only spread to direct neighbours
- **Integer population**: vaccination is discrete (whole individuals)
- **Rich local structure**: clustering, communities, bridge nodes between groups

### The ODE is a mean-field approximation of the BA network

The ODE essentially applies the law of large numbers to average out:
- Network topology → replaced by average contact rates
- Individual stochasticity → replaced by expected values
- Degree heterogeneity → ignored within groups

## How Model Mismatch Affects Each Method

### When ODE ≈ Reality (low mortality, low beta)

The mean-field approximation is accurate. OC's two-layer advantage holds:
- Layer 1 (group allocation): ODE solution ≈ true optimal → OC wins
- Layer 2 (node selection): degree-greedy is nearly optimal because blocking hub transmission prevents more deaths than protecting any individual

**Result**: OC dominates all RL methods.

### When ODE starts diverging (moderate mortality)

Higher mortality means:
- Y-group population shrinks faster than ODE predicts (discrete death events, not continuous flow)
- Saving one high-risk individual (17.5% death probability) becomes more valuable than blocking one hub's transmission chain
- Degree-greedy loses its edge → OC's Layer 2 advantage disappears

**Result**: Node RL can win because it sees local infection pressure and adapts node selection beyond degree-greedy.

### When ODE severely diverges (critical mortality)

At extreme mortality (pY=0.50, dY=0.65):
- Y-group undergoes rapid population collapse that ODE cannot model accurately
- The optimal group allocation shifts dramatically from what ODE computes
- Even OC's Layer 1 (group allocation) becomes wrong
- Warm-start from OC prior is actively harmful (anchors to wrong solution)

**Result**: Group RL cold start wins because it explores freely on the true stochastic environment without being anchored to the wrong ODE solution.

## Severity Sweep Results

Parameters: N=5000, beta=0.08, V_MAX=10. Only pY and dY vary.

### Baseline (pY=0.20, dY=0.27, Y death prob ~5.4%)

| Method | Deaths (mean ± std) |
|--------|-------------------|
| **OC_Guided** | **25.5 ± 3.7** |
| Group_RL (cold) | 28.5 ± 4.7 |
| Group_RL (warm) | 27.1 ± 6.3 |
| Node_RL (cold) | 29.5 ± 4.8 |
| NodeHorizon (α=3,warm) | 25.9 ± 3.7 |

OC dominates. ODE approximation is accurate at low mortality.

### Moderate (pY=0.30, dY=0.40, Y death prob ~12%)

| Method | Deaths (mean ± std) |
|--------|-------------------|
| OC_Guided | 60.5 ± 6.5 |
| Group_RL (cold) | 58.1 ± 6.1 |
| Group_RL (warm) | 59.6 ± 9.3 |
| **Node_RL (cold)** | **56.7 ± 8.3** |
| NodeHorizon (α=3,warm) | 63.2 ± 7.5 |

Node RL cold wins. Degree-greedy loses advantage; local infection pressure matters.

### Severe (pY=0.40, dY=0.50, Y death prob ~20%)

| Method | Deaths (mean ± std) |
|--------|-------------------|
| OC_Guided | 91.2 ± 7.3 |
| Group_RL (cold) | 97.3 ± 5.3 |
| Group_RL (warm) | 95.7 ± 8.3 |
| Node_RL (cold) | 101.5 ± 8.8 |
| **NodeHorizon (α=3,warm)** | **89.7 ± 7.8** |

NodeHorizon wins. Terminal reward helps align RL with global D(T) minimisation.

### Critical (pY=0.50, dY=0.65, Y death prob ~32.5%)

| Method | Deaths (mean ± std) |
|--------|-------------------|
| OC_Guided | 149.4 ± 7.0 |
| **Group_RL (cold)** | **142.5 ± 10.8** |
| Group_RL (warm) | 149.5 ± 7.9 |
| Node_RL (cold) | 152.0 ± 8.6 |
| NodeHorizon (α=3,warm) | 148.8 ± 6.6 |

Group RL cold wins. ODE model is so wrong that even group allocation from OC is suboptimal. Cold start explores freely without ODE anchoring.

## Beta Sweep Results

Parameters: N=5000, pY=0.20, dY=0.27, V_MAX=10. Only beta varies.

*(To be filled after beta sweep completes)*

## Key Insight: Winner Depends on Model Mismatch Level

| ODE-Reality Gap | Winner | Why |
|-----------------|--------|-----|
| Small (low mortality/beta) | OC-Guided | ODE is accurate; exact solution + degree-greedy is hard to beat |
| Medium (moderate mortality) | Node RL | Degree-greedy becomes suboptimal; individual infection pressure matters |
| Large (high mortality) | NodeHorizon | Terminal reward helps; OC's node selection fails but group allocation still reasonable |
| Very large (critical mortality) | Group RL cold | Even OC's group allocation is wrong; free exploration on true dynamics wins |

## Improvement to NodeHorizon Warm-Start

After analyzing these results, we identified that the warm-start score bias contained a **degree component** that pushed Node RL toward degree-greedy behavior:

```python
# OLD: group share + degree (biases toward degree-greedy)
sb = (group_score + degree_score) * cur_bias_scale

# NEW: group share only (lets RL decide node selection freely)
sb = group_score * cur_bias_scale
```

This explains why NodeHorizon (warm) performed worse than Node RL (cold) in the moderate scenario (63.2 vs 56.7) — the degree bias anchored the policy toward degree-greedy, which is exactly the wrong strategy when mortality is high enough that protecting high-risk individuals matters more.

The fix removes the degree term so warm-start only communicates OC's temporal group allocation pattern ("vaccinate more Y early, shift to X later") without imposing a within-group selection strategy.
