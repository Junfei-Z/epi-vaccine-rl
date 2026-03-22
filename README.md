# 💉 epi-vaccine-rl

> Optimal vaccine allocation on contact networks — combining ODE optimal control with warm-start reinforcement learning.

We model an epidemic spreading across a **Barabási-Albert contact network** and ask: *given a limited daily vaccine budget, how should we split doses across population groups to minimise deaths?* Three methods compete: classical optimal control, RL from scratch, and RL warm-started from the optimal control solution.

---

## 📋 Table of Contents

- [Epidemic Model](#-epidemic-model)
- [ODE Formulation](#-ode-formulation)
- [Three Methods](#-three-methods)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Sensitivity Experiments](#-sensitivity-experiments)
- [Warm-RL Tuning](#-warm-rl-tuning)
- [Output Files](#-output-files)

---

## 🦠 Epidemic Model

### Population groups

The network population is split into three groups with different risk profiles:

| Group | Label | Description |
|---|---|---|
| **X** | Baseline | General population — average risk and contact rate |
| **Y** | High-Risk | Elderly / comorbid — higher hospitalisation & death rates |
| **Z** | High-Contact | Network hubs — more connections, higher transmission potential |

### Compartments

Each individual node carries one of **10 disease states**:

```
S → E → P → A → R
              ↘
               I → L → H → D
                       ↘
                        R
S → V  (vaccinated, reduced susceptibility)
```

| State | Name | Meaning |
|---|---|---|
| **S** | Susceptible | Healthy, can be infected |
| **E** | Exposed | Infected but not yet infectious (latent) |
| **P** | Pre-symptomatic | Infectious, no symptoms yet |
| **A** | Asymptomatic | Infectious, never shows symptoms |
| **I** | Symptomatic | Infectious with symptoms |
| **L** | Mild illness | Post-infectious, mild recovery |
| **H** | Hospitalised | Severe illness, risk of death |
| **R** | Recovered | Immune, no longer infectious |
| **V** | Vaccinated | Susceptibility reduced by factor ε |
| **D** | Dead | Terminal |

### Group-specific parameters

| Parameter | X | Y | Z | Meaning |
|---|---|---|---|---|
| `s` | 0.5 | 0.8 | 0.6 | Prob. of becoming symptomatic (vs asymptomatic) |
| `p` | 0.05 | 0.20 | 0.08 | Prob. of progressing to severe (H) |
| `d` | 0.02 | 0.27 | 0.04 | Prob. of dying given hospitalised |
| `ε` | 0.5 | 0.5 | 0.5 | Vaccine efficacy (susceptibility reduction) |

---

## 📐 ODE Formulation

The population-level ODE tracks the compartment sizes for each group g ∈ {X, Y, Z}.

### Force of infection

Each group faces a group-specific force of infection driven by infectious contacts across all groups:

```
λ_g(t) = β · Σ_h  C_{g,h} · (wA·A_h + wP·P_h + wI·I_h) / N_h
```

where `C` is the **inter-group contact matrix** derived from the BA network, and `wA < wP < wI` are infectiousness weights for each infectious stage.

### Transition equations (per group g)

```
dS_g/dt = − λ_g · S_g  − u_g · S_g
dE_g/dt = + λ_g · S_g  − τE · E_g
dP_g/dt = + τE · E_g   − τP · P_g
dA_g/dt = + (1 − s_g) · τP · P_g  − τA · A_g
dI_g/dt = + s_g · τP · P_g         − τI · I_g
dL_g/dt = + τI · I_g               − τL · L_g
dH_g/dt = + p_g · τL · L_g         − τH · H_g
dR_g/dt = + (1 − p_g) · τL · L_g  + τA · A_g  + (1 − d_g) · τH · H_g
dV_g/dt = + u_g · S_g
dD_g/dt = + d_g · τH · H_g
```

`u_g(t)` is the **vaccination control**: fraction of the susceptible pool in group g vaccinated per day, subject to:

```
S_X · u_X(t) + S_Y · u_Y(t) + S_Z · u_Z(t) ≤ V_MAX_DAILY   ∀ t
```

### Objective

Minimise total deaths at the end of the horizon T:

```
min  D_X(T) + D_Y(T) + D_Z(T)
```

Discretised with **RK4** (dt = 1 day) and solved as an NLP by **CasADi / IPOPT**.

---

## 🚀 Three Methods

### Method 1 — ODE Optimal Control (OC-Guided) 📊

> *"Solve the maths, then apply it."*

1. Solve the population-level NLP with CasADi/IPOPT → get optimal per-day dose fractions `u_g*(t)`
2. Convert to actual integer dose counts: `a_g(t) = S_g(t) · u_g*(t)`
3. Post-process: priority-window fill → cap to `V_MAX_DAILY` → integer rounding
4. Apply doses to the **node-level** stochastic environment in priority order (highest-degree susceptibles first within each group)

**Strengths:** globally optimal at population level, no training needed.
**Weakness:** population-level solution may not adapt well to node-level stochasticity.

---

### Method 2 — Cold-Start PPO 🧊

> *"Learn from scratch — no hints."*

The RL agent outputs a **Dirichlet distribution** over the 3-group allocation simplex. At each day it receives a 31-dimensional observation:

```
obs = [S_X/N, E_X/N, ..., D_Z/N,  day/T]   (30 compartment fracs + normalised day)
```

and outputs shares `(α_X, α_Y, α_Z)` summing to 1. Doses are converted to integers via the **Largest Remainder Method** respecting available susceptibles per group.

**Training loop:**
1. Collect `episodes_per_update` rollout episodes using the old policy
2. Estimate advantages with **Generalised Advantage Estimation** (GAE, λ = 0.95)
3. Run K PPO gradient steps with clipped surrogate + entropy bonus
4. Evaluate deterministically every update round; save best checkpoint
5. Stop early when the rolling-window relative std of eval deaths drops below threshold

**Reward per step:** `−deaths_today · reward_scale − 0.01 · unused_doses`

---

### Method 3 — Warm-Start PPO 🔥

> *"Let the optimal control solution give RL a head start."*

Same PPO architecture as cold-start, but the ODE solution is turned into a **feasible prior** that guides early training:

**Prior construction:**
1. Run the node-level environment following ODE doses → record actual daily simplex shares
2. Blend with a group bias vector (e.g. `[0,0,1]` for HCP) to nudge toward the priority group
3. Save as a `(T × 3)` prior array

**Warm training phases:**
- **Episode 0** — imitate prior exactly (pure imitation, no exploration)
- **Episodes 1 … warm_mean_episodes** — high-temperature Dirichlet sampling (flat = more exploration)
- **Episodes warm_mean_episodes+** — lower temperature (sharper = more exploitation)

**KL regularisation:**
An analytic KL-divergence penalty pulls the current policy toward the ODE prior:

```
loss += prior_weight · KL( Dirichlet(conc_current) ‖ Dirichlet(prior_vec · α + ε) )
```

`prior_weight` decays by `prior_decay` each gradient step so the prior influence fades as the agent gains experience and can deviate from the ODE solution.

**Why warm-start?** The ODE solution is a strong initialisation but ignores network stochasticity and individual node heterogeneity. Warm-start PPO uses it as scaffolding, then adapts — ideally converging faster and to a better solution than cold-start.

---

## 📁 Project Structure

```
epi-vaccine-rl/
├── config.py                    # Constants, state indices, PARAMS_HCP/HRP, to_params_global()
├── graph.py                     # BA network construction, contact matrix
├── ode_solver.py                # ODE optimal control (CasADi/IPOPT + RK4)
├── allocation.py                # Dose post-processing (priority fill, cap, simplex)
├── env.py                       # Node-level EpidemicNodeEnv + make_env_from_graph()
├── prior.py                     # Feasible prior construction from ODE doses
├── simulate.py                  # ODE-guided simulation, evaluate_and_export()
├── plot.py                      # All visualisation functions
├── requirements.txt
├── colab_run.txt                # Ready-to-paste Google Colab runner
├── rl/
│   ├── model.py                 # ActorCritic (Dirichlet policy head)
│   ├── ppo.py                   # PPO + PPOBuffer, GAE, KL-prior regularisation
│   └── train.py                 # run_training(), quick_eval_det(), early stopping
└── experiments/
    ├── base.py                  # HCP + HRP full pipeline, run_one_scenario()
    ├── sensitivity_degree.py    # BA_M sweep [2, 4, 6, 8, 10]
    ├── sensitivity_infected.py  # INITIAL_INFECTED sweep [400, 500, 600, 700]
    ├── sensitivity_beta.py      # beta sweep [0.06, 0.08, 0.10, 0.12, 0.15]
    ├── sensitivity_vmax.py      # V_MAX_DAILY sweep [10, 20, 40, 60, 80]
    ├── sensitivity_groupsize.py # N sweep [5000, 7000, 9000, 11000, 13000]
    ├── sensitivity_epsilon.py   # Vaccine efficacy sweep [0.35, 0.45, 0.55, 0.65, 0.75]
    ├── sensitivity_infection_risk.py  # beta sweep [0.04→0.15] + wA sweep [0.1→0.9]
    ├── sensitivity_graph_type.py      # NLPA exponent sweep [0.5, 0.75, 1.0, 1.25, 1.5]
    ├── sensitivity_highrisk.py        # high_risk_prob sweep [0.17, 0.25, 0.32, 0.40]
    └── sensitivity_severity.py        # pY/dY sweep [baseline → critical]
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** `numpy`, `pandas`, `matplotlib`, `networkx`, `casadi`, `torch`

> CasADi's pip package bundles IPOPT — no separate solver installation needed.

---

## ▶️ Quick Start

### Run locally

```bash
git clone https://github.com/Junfei-Z/epi-vaccine-rl.git
cd epi-vaccine-rl
pip install -r requirements.txt

# Run HCP + HRP baseline (ODE → prior → warm PPO → cold PPO → OC-guided)
python -m experiments.base
```

### Run on Google Colab ☁️

1. Copy the contents of `colab_run.txt` into a Colab notebook
2. In **Cell 4**, set `RUN_*` flags to `True` for the experiments you want
3. Run all cells in order
4. Uncomment **Cell 7** to zip and download all result CSVs

---

## 🔬 Sensitivity Experiments

Each experiment sweeps one parameter while holding everything else at the HCP canonical values. All results go to `results/sensitivity_*/`.

| Script | Parameter | Values | Question |
|---|---|---|---|
| `sensitivity_degree.py` | `BA_M` | 2, 4, 6, 8, 10 | How does network connectedness affect the value of prioritising hubs? |
| `sensitivity_infected.py` | `INITIAL_INFECTED` | 400, 500, 600, 700 | Does a larger outbreak seed change the optimal strategy? |
| `sensitivity_beta.py` | `beta` | 0.06 → 0.15 | How does transmissibility shift the balance between methods? |
| `sensitivity_vmax.py` | `V_MAX_DAILY` | 10, 20, 40, 60, 80 | Does scarcity or abundance of doses change who wins? |
| `sensitivity_groupsize.py` | `N` | 5k → 13k | Does population scale affect relative performance? |
| `sensitivity_epsilon.py` | `epsilon` | 0.35 → 0.75 | How much does vaccine efficacy matter for each method? |
| `sensitivity_infection_risk.py` | `beta`, `wA` | beta 0.04→0.15 / wA 0.1→0.9 | How do transmissibility and hidden asymptomatic spread interact with allocation? |
| `sensitivity_graph_type.py` | NLPA `alpha_pa` | 0.5, 0.75, 1.0, 1.25, 1.5 | Does network heterogeneity (hub dominance) change which allocation method wins? |
| `sensitivity_highrisk.py` | `high_risk_prob` | 0.17 → 0.40 | If the at-risk age threshold lowers (more high-risk people), does HRP pull ahead of HCP? |
| `sensitivity_severity.py` | `pY`, `dY` | pY 0.20→0.50 / dY 0.27→0.57 | As the high-risk group becomes more severely affected, does direct protection of Y dominate hub-targeting? |

Run any experiment individually:

```bash
python -m experiments.sensitivity_degree
python -m experiments.sensitivity_beta
# etc.
```

---

## 🔧 Warm-RL Tuning

If warm-start RL underperforms cold-start RL, the prior is likely too strong or training stops too early. Try these levers (passed as keyword args to `run_one_scenario`):

| Parameter | Default | Try | Effect |
|---|---|---|---|
| `warm_max_episodes` | 300 | 400 | More training time |
| `warm_min_episodes` | 40 | 80 | Can't stop before this many episodes |
| `warm_patience` | 4 | 8 | Harder to trigger early stop |
| `warm_mean_episodes` | 12 | 25 | Longer high-exploration warm-up phase |
| `prior_weight` (ppo.py) | 1.6 | 0.8 | Weaker prior pull → more RL freedom |
| `prior_decay` (ppo.py) | 0.995 | 0.98 | Prior fades faster each gradient step |
| `prior_alpha` (ppo.py) | 60.0 | 30.0 | Softer prior Dirichlet → less concentrated |

---

## 📦 Output Files

Each `run_one_scenario()` call saves everything to its `out_dir`:

| File | Contents |
|---|---|
| `ode_warm_start_doses_{tag}.npy` | ODE optimal integer doses, shape (T, 3) |
| `ode_warm_start_policy_{tag}.npy` | ODE doses as simplex shares, shape (T, 3) |
| `ode_feasible_prior_{tag}.npy` | Feasible prior after node-level rollout, shape (T, 3) |
| `best_policy_{tag}_warm.pt` | Best warm-start PPO checkpoint (PyTorch) |
| `best_policy_{tag}_cold.pt` | Best cold-start PPO checkpoint (PyTorch) |
| `nodes_{tag}_warm/cold.csv` | Per-node vaccination records (day, group, degree, …) |
| `days_{tag}_warm/cold.csv` | Per-day dose allocation records (X, Y, Z counts) |
| `ode_nodes_{tag}.csv` | ODE-guided per-node records |
| `ode_days_{tag}.csv` | ODE-guided per-day allocation records |
| `comparison_base.csv` | HCP vs HRP final death count summary |
