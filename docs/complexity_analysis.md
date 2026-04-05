# Time Complexity Analysis: Node RL vs Naive Individual RL

## Problem Setting

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| $N$ | Total population (nodes in contact network) | 5,000–10,000 |
| $N_s$ | Number of susceptible nodes at time $t$ | $\leq N$ (decreases over time) |
| $K$ | Daily vaccine budget ($V_{\text{MAX}}$) | 10–60 |
| $T$ | Epidemic horizon (days) | 60 |
| $d$ | Per-node feature dimension | 6 |
| $d_g$ | Global state dimension | 31 |
| $\bar{k}$ | Average node degree | ~6 (BA graph) |
| $|E|$ | Number of edges | $N \bar{k} / 2$ |

---

## Method 1: Naive Individual RL (Binary Action per Node)

The most straightforward formulation: at each time step, the agent outputs a binary decision $a_i \in \{0, 1\}$ for every susceptible node $i$, where $a_i = 1$ means "vaccinate node $i$".

### Action Space

$$|\mathcal{A}| = \binom{N_s}{K}$$

This is the number of ways to choose exactly $K$ nodes from $N_s$ susceptible nodes. Using Stirling's approximation:

$$\binom{N_s}{K} \approx \left(\frac{N_s \cdot e}{K}\right)^K$$

For $N_s = 4000, K = 20$: $\binom{4000}{20} \approx 10^{52}$.

Even if relaxed to independent binary decisions (ignoring the budget constraint), the action space is $2^{N_s}$, which is exponentially large.

### Policy Representation Options

**Option A: Combinatorial output (direct enumeration)**

The policy would need to output a probability distribution over $\binom{N_s}{K}$ actions. This is:
- **Infeasible** — the output layer alone requires $O\left(\binom{N_s}{K}\right)$ parameters
- Cannot generalize across population sizes

**Option B: Independent Bernoulli per node + projection**

Output $\pi_\theta(a_i = 1 \mid s)$ for each node independently, then project to satisfy the budget constraint $\sum_i a_i = K$.

| Component | Complexity |
|-----------|-----------|
| Forward pass (one output per node) | $O(N_s \cdot H)$ where $H$ = hidden dim |
| Constraint satisfaction (sort + truncate) | $O(N_s \log N_s)$ |
| **Per-step total** | $O(N_s \cdot H + N_s \log N_s)$ |

**Problem**: The budget constraint $\sum_i a_i = K$ is **not differentiable**. The projection step (sorting and truncating) breaks the gradient flow. Training requires:
- REINFORCE with $N_s$-dimensional discrete actions → **variance scales as $O(N_s)$**
- Or relaxation methods (e.g., continuous relaxation + rounding) with biased gradients

**Per-step PPO update complexity**:
- Must recompute log-probabilities for $N_s$ independent Bernoulli decisions
- Log-probability of selected action: $\log \pi(a \mid s) = \sum_{i=1}^{N_s} \log \pi(a_i \mid s)$
- Gradient variance: $\text{Var}[\nabla \log \pi] = O(N_s)$ (sum of independent terms)
- **Training instability** grows linearly with population size

### Convergence

The sample complexity of policy gradient methods scales with the action space entropy. For $N_s$ independent Bernoulli variables:

$$H(\pi) = \sum_{i=1}^{N_s} H(\pi_i) = O(N_s)$$

The number of episodes needed to achieve $\epsilon$-optimal policy scales as:

$$\text{Sample complexity} = \tilde{O}\left(\frac{|\mathcal{A}|}{\epsilon^2}\right) = \tilde{O}\left(\frac{2^{N_s}}{\epsilon^2}\right) \text{ (worst case)}$$

This is **exponential** in $N_s$, making naive individual RL fundamentally intractable for large populations.

---

## Method 2: Node RL with Scoring + Gumbel-Top-K (Ours)

Instead of outputting a combinatorial action, our method scores each susceptible node independently and selects the top-$K$ via differentiable Gumbel-Top-K sampling.

### Action Representation

The policy outputs a **scalar score** $\phi_\theta(x_i, g) \in \mathbb{R}$ for each susceptible node $i$, where:
- $x_i \in \mathbb{R}^d$ is the per-node feature (6-dim)
- $g \in \mathbb{R}^{d_g}$ is the global state (31-dim)

The action is the set of $K$ nodes with the highest (perturbed) scores:

$$\mathcal{V}_t = \text{Top-}K\left(\phi_\theta(x_i, g) + G_i\right), \quad G_i \sim \text{Gumbel}(0, 1)$$

### Per-Step Complexity Breakdown

| Component | Operation | Complexity |
|-----------|-----------|-----------|
| **1. Feature construction** | Sparse adjacency multiply for neighbor infection pressure | $O(N + |E|)$ |
| **2. Global observation** | Count compartments per group | $O(N)$ |
| **3. Scoring network** | Shared MLP: $(d + d_g) \to H \to H \to 1$ applied to $N_s$ nodes | $O(N_s \cdot H \cdot (d + d_g))$ |
| **4. Gumbel-Top-K selection** | Sample Gumbel noise + `torch.topk` | $O(N_s \log K)$ |
| **5. Log-probability** | `log_softmax` over $N_s$ scores + index $K$ entries | $O(N_s)$ |
| **Total per step** | | $O(N + |E| + N_s \cdot H \cdot (d+d_g))$ |

With $H = 64, d = 6, d_g = 31$:

$$\text{Per-step} = O(N + |E| + 37 \cdot 64 \cdot N_s) = O(N + |E| + 2368 \cdot N_s)$$

Since $|E| = O(N \bar{k})$ and $N_s \leq N$:

$$\boxed{\text{Per-step complexity} = O(N)}$$

### Training (PPO Update) Complexity

Per PPO epoch over a batch of $T$ transitions:

| Component | Complexity |
|-----------|-----------|
| Recompute scores for all transitions | $O(T \cdot N_s \cdot H \cdot (d + d_g))$ |
| Recompute log-probabilities | $O(T \cdot N_s)$ |
| GAE advantage estimation | $O(T)$ |
| Backward pass | $O(T \cdot N_s \cdot H \cdot (d + d_g))$ |
| **Per epoch** | $O(T \cdot N_s \cdot H \cdot (d + d_g))$ |

With $K_{\text{epochs}} = 8$:

$$\text{Per update} = O(K_{\text{epochs}} \cdot T \cdot N_s \cdot H \cdot (d + d_g)) = O(T \cdot N_s)$$

### Why Gumbel-Top-K is Key

The Gumbel-Top-K trick provides:

1. **Exact budget satisfaction**: Always selects exactly $K$ nodes — no projection needed
2. **Differentiable sampling**: Log-probability $\log \pi(a \mid s) = \sum_{j \in \mathcal{V}_t} \log \text{softmax}(\phi_j)$ has well-defined gradients
3. **Low variance**: Single scalar score per node → gradient variance is $O(1)$ w.r.t. $N_s$ (vs $O(N_s)$ for independent Bernoulli)
4. **Permutation equivariance**: Shared scorer treats nodes symmetrically, enabling generalization

---

## Side-by-Side Comparison

| Aspect | Naive Individual RL | Node RL (Ours) |
|--------|-------------------|----------------|
| **Action space** | $2^{N_s}$ or $\binom{N_s}{K}$ | $O(N_s)$ scores → Top-K |
| **Policy output dim** | $N_s$ (or $\binom{N_s}{K}$) | $1$ (shared scorer) |
| **Budget constraint** | Requires projection (non-differentiable) | Built-in via Top-K |
| **Per-step forward** | $O(N_s \cdot H)$ | $O(N_s \cdot H)$ |
| **Gradient variance** | $O(N_s)$ | $O(1)$ w.r.t. $N_s$ |
| **Sample complexity** | $\tilde{O}(2^{N_s} / \epsilon^2)$ worst case | Polynomial in $N_s$ |
| **Parameters** | $O(N_s \cdot H)$ or shared | $O(H^2)$ (shared, size-invariant) |
| **Scalability** | Intractable for $N_s > 100$ | Tested at $N = 5{,}000$ |

### Numerical Example ($N = 5{,}000$, $K = 20$, $N_s \approx 4{,}000$)

| Metric | Naive Individual RL | Node RL (Ours) |
|--------|-------------------|----------------|
| Action space size | $\binom{4000}{20} \approx 10^{52}$ | 4,000 scores |
| Forward pass FLOPs | ~512,000 | ~512,000 |
| Gradient variance (relative) | ~4,000× | 1× |
| Budget constraint | Post-hoc projection | Exact |
| Trainable parameters | ~4,000× (if node-specific) or same (if shared) | ~8,500 (shared MLP) |

---

## Summary

The key insight is that our Node RL method **reformulates the combinatorial action selection problem into a continuous scoring problem**:

$$\underbrace{\text{Choose } K \text{ from } N_s}_{\binom{N_s}{K} \text{ combinations}} \longrightarrow \underbrace{\text{Score each node, select Top-}K}_{O(N_s) \text{ evaluations}}$$

This reduces:
- **Decision complexity** from exponential $O(2^{N_s})$ to linear $O(N_s)$
- **Gradient variance** from $O(N_s)$ to $O(1)$
- **Parameter count** from $O(N_s \cdot H)$ (node-specific) to $O(H^2)$ (shared scorer)

while maintaining the ability to make **heterogeneous, node-specific decisions** based on local network information (neighbor infection pressure, degree centrality) that group-level methods cannot access.
