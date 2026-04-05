# 2026-04-05 — Add Terminal Reward to Group RL + Ablation Experiment

## What Changed

- `rl/train.py`: Added `terminal_reward_scale` parameter to `run_training()` (Group RL). At episode end, adds `-total_deaths * terminal_reward_scale` to the final step's reward. Default is 0.0 (off), matching previous behavior. Node RL already had this parameter.
- `experiments/ablation_terminal_reward_all.py`: New experiment script comparing all 3 RL methods (Group RL cold, Group RL warm, Node RL) with terminal reward on (scale=1.0) vs off (scale=0.0), plus OC-Guided as reference.

## Why

The advisor suggested that all RL methods may suffer from myopic (short-sighted) behavior: the per-step reward `r_t = -deaths_today` only reflects immediate mortality, but vaccination decisions have delayed effects (a vaccine given today prevents infections days later). Adding a terminal reward `r_T += -total_deaths * scale` at episode end provides a global optimization signal aligned with the true objective: minimizing total deaths over the entire epidemic horizon.

## Experimental Evidence

Experiment: `experiments/ablation_terminal_reward_all.py` (Group RL 200 eps, Node RL 300 eps, 10 eval runs)

### Baseline (dY=0.27)

| Method | ts=0.0 (off) | ts=1.0 (on) | Change |
|--------|-------------|-------------|--------|
| OC-Guided | 25.5 +/- 3.7 | - | - |
| Group RL cold | **25.9** +/- 6.3 | 26.8 +/- 4.0 | +0.9 (worse) |
| Group RL warm | **26.9** +/- 3.9 | 28.6 +/- 3.4 | +1.7 (worse) |
| Node RL | 31.5 +/- 7.0 | **25.9** +/- 3.7 | **-5.6 (major improvement)** |

### Moderate (dY=0.40)

| Method | ts=0.0 (off) | ts=1.0 (on) | Change |
|--------|-------------|-------------|--------|
| OC-Guided | 35.3 +/- 6.8 | - | - |
| Group RL cold | 39.6 +/- 3.4 | **36.9** +/- 5.8 | **-2.7 (improved)** |
| Group RL warm | 43.5 +/- 4.4 | **40.5** +/- 6.4 | **-3.0 (improved)** |
| Node RL | 41.1 +/- 5.3 | **38.7** +/- 6.7 | **-2.4 (improved)** |

## Analysis

### Node RL benefits most (baseline: 31.5 -> 25.9, matching OC)

Node RL scores thousands of nodes per step — a large action space. The per-step reward (-deaths_today) is nearly uncorrelated with today's vaccination choice because vaccines take days to show effect. Without terminal reward, Node RL gets mostly noise as learning signal, causing unstable training (eval_deaths oscillating between 26-37). With terminal reward, training stabilizes immediately (constant 26 from ep=10 onward) because the agent receives a clear, holistic signal: "how many people died in total because of your decisions."

### Group RL gets slightly worse in baseline

Group RL's action space is only 3-dimensional (group allocation ratios). Per-step reward is already informative enough for such a small decision space — Group RL cold already achieves 25.9 without terminal reward, nearly matching OC (25.5). Adding terminal reward (-25 at episode end) changes the reward scale: per-step rewards sum to ~-30 over 60 steps, but the terminal bonus is a single -25 spike. This causes the value function to over-focus on predicting total deaths rather than learning fine-grained daily allocation, slightly degrading a policy that was already near-optimal.

### All methods improve in moderate scenario

When mortality is high, daily death counts fluctuate more (higher variance in per-step reward), and the ODE model mismatch is larger. Terminal reward provides a more stable, cumulative optimization signal that helps all methods converge in this noisier environment.

### Key insight

Terminal reward's value depends on whether per-step reward is sufficient. Larger action space (Node RL) and noisier environment (high mortality) make per-step reward less informative, increasing the marginal value of terminal reward. When per-step reward is already sufficient (Group RL + baseline), terminal reward becomes interference rather than help.

## Files Modified

- `rl/train.py` — `run_training()` added `terminal_reward_scale` parameter

## Files Created

- `experiments/ablation_terminal_reward_all.py` — ablation experiment script

## Results Data

- `results/ablation_terminal/results_terminal_ablation.csv`
