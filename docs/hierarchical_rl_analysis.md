# Hierarchical RL: Design, Framework, and Why It Doesn't Work

## Design Motivation

After discovering that OC's advantage comes from two layers (optimal group allocation + degree-greedy node selection), a natural idea emerged: **decompose the problem into two RL sub-problems**:

- Layer 1 (inter-group): Group RL decides how many doses to allocate to X, Y, Z
- Layer 2 (intra-group): Node RL selects which specific nodes to vaccinate within each group

The hope was that decomposition would make each sub-problem easier to learn:
- Group RL: only 3-dim simplex (same as before)
- Node RL: ~1500 candidates per group instead of ~4500 globally (3x smaller action space)

## Framework

```
                    ┌──────────────┐
  Global State ───> │  Group RL    │ ──> shares (X%, Y%, Z%)
  (31-dim)          │  (frozen)    │
                    └──────────────┘
                           │
                    _project_doses()
                           │
                    ┌──────────────┐
                    │  doses per   │  k_X=3, k_Y=5, k_Z=2
                    │  group       │
                    └──────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Node RL  │ │ Node RL  │ │ Node RL  │
        │ score X  │ │ score Y  │ │ score Z  │
        │ top-3    │ │ top-5    │ │ top-2    │
        └──────────┘ └──────────┘ └──────────┘
              │            │            │
              └────────────┼────────────┘
                           ▼
                env.step_node_ids([...10 nodes...])
```

Training was sequential:
1. Train Group RL first (using env.step with degree-greedy)
2. Freeze Group RL, train Node RL scorer only (PPO with terminal reward)

## Why It Doesn't Work

### Problem 1: No advantage over standalone Node RL

Node RL alone already:
- Reduces action space from 2^N to O(N) via scoring + top-K
- Can implicitly learn optimal group allocation (if Y nodes should be prioritized, the scorer naturally gives them higher scores)
- Has no artificial constraints on which groups to select from

Hierarchical RL adds a group-level constraint that **limits the search space without benefit**. If Group RL says "give Z 3 doses", Node RL cannot vaccinate a high-risk Y node even if it's about to be infected — the budget is already allocated.

### Problem 2: Co-adaptation mismatch

Group RL was trained with degree-greedy node selection (via `env.step(shares)` → `_choose_to_vaccinate()`). Its learned allocation is optimized FOR degree-greedy, not for Node RL's selection strategy.

For example:
- Group RL may have learned "allocate heavily to Z" because degree-greedy efficiently protects Z hubs
- But Node RL might be better at protecting Y nodes based on local infection pressure
- The frozen Group RL keeps sending doses to Z while Node RL would prefer Y

Joint training could fix this, but then we arrive at Problem 1 again — joint training of both layers to minimize deaths is exactly what standalone Node RL already does, just without the artificial decomposition.

### Problem 3: Added complexity without gain

Hierarchical approach introduces:
- Two separate networks to manage
- Sequential or joint training complexity
- A hard group-allocation boundary that prevents cross-group optimization

All for solving the same objective (minimize total deaths) that Node RL solves directly.

## Conclusion

**The hierarchical decomposition is an unnecessary bottleneck.** The scoring + top-K formulation of Node RL already provides an efficient O(N) search over all candidates globally. Adding a group-level allocation layer on top only restricts the solution space without improving learning efficiency.

The right approach is to improve Node RL directly (better training, better exploration) rather than adding architectural constraints.
