# A Cross-Layer Mutual-Information Extension via Matrix-Based Rényi Entropy

**Date:** 2026-04-26
**Author:** thesis-IV working notes (HyMeKo)
**Purpose:** formalise the cross-layer mutual-information regulariser
proposed as Path F of the universality programme, in a form rigorous
enough to defend in the manuscript and unify the temporal-conditional
(Path E), joint-view (Path G), and information-bottleneck (Path H)
extensions under a single mathematical framework.

---

## 1. Motivation

The 9 entropy regularisers tested in phases 1–7b all use **marginal
spectral entropy at a single timestep**. Specifically, for layer
weights `W` with adjacency `A` and normalised-Laplacian eigenvalues
`{λ_i}`,

```
H_marg(A) = -Σ p_i log p_i,    p_i = λ_i / Σ_j λ_j.
```

This quantity captures *how spread out* the weight spectrum is, but
never:

- the **joint** distribution across layers `H(W_i, W_j)`,
- the **conditional** distribution across time `H(W_t | W_{t-1})`,
- or the **mutual information** between any of those constructs.

Empirically, the framework hits a ceiling: Path B (`scalar_entropy_normalized`)
and Path A (`entropy_target H*=0.5`) achieve universality on the 4-dataset
stress matrix (spirals, circles, MNIST, CapsMLP), but neither addresses
the underlying *structural* hypothesis that **layers may be redundant
with each other**, and that decorrelating layers might be a stronger
universality lever than tuning a single scalar entropy target.

This note formalises a single quantity, `I_α(K_i ; K_j)` — the
**matrix-based Rényi-α mutual information** between activation Gram
matrices of two layers — that subsumes all four extension paths under
one definition.

## 2. Setup and notation

Let `M : R^d → R^c` be a feedforward network with `L` layers
parameterised by weights `W_1, …, W_L`. Let `X ∈ R^{B × d}` be a batch
of `B` input examples. Define the per-layer activation tensor

```
a_l = φ_l(W_l φ_{l-1}(... W_1 X^T))^T  ∈  R^{B × d_l},
```

i.e. the post-non-linearity output of layer `l` evaluated on `X`. (For
purely linear analysis, drop the `φ_l`.)

Construct the **sample-side Gram matrix** of layer `l`:

```
K_l(X) = a_l a_l^T  ∈  R^{B × B},     (eq. 1)
```

normalised so trace is 1:

```
K̂_l = K_l / tr(K_l),         tr(K̂_l) = 1.        (eq. 2)
```

`K̂_l` is symmetric positive semi-definite (PSD) and trace-1 — equivalently,
its eigenvalue distribution `{μ_i^{(l)}}` is a valid probability mass
function on `B` outcomes.

## 3. Matrix-based Rényi-α entropy (Sanchez-Giraldo et al. 2014)

For `α > 0`, `α ≠ 1`, define

```
H_α(K̂) = (1/(1−α)) · log Σ_i (μ_i)^α                     (eq. 3)
       = (1/(1−α)) · log tr(K̂^α).                         (eq. 4)
```

**Special cases:**

- `α → 1` recovers Shannon: `H_1(K̂) = − Σ μ_i log μ_i`
  (von-Neumann entropy of `K̂`).
- `α = 2` gives the **collision entropy**:

  ```
  H_2(K̂) = − log Σ_i (μ_i)^2 = − log tr(K̂^2).      (eq. 5)
  ```

  Eq. 5 is the workhorse: it requires no eigendecomposition, scales as
  `O(B²)` from `tr(K̂^2) = Σ_{i,j} K̂_{ij}^2`, and is automatically
  differentiable in any standard autograd framework.

## 4. Joint distribution: Hadamard product

For two trace-1 PSD matrices `K̂_i, K̂_j ∈ R^{B × B}`, define the **joint
matrix** as the entrywise (Hadamard) product, normalised:

```
K̂_{ij} = (K̂_i ⊙ K̂_j) / tr(K̂_i ⊙ K̂_j).               (eq. 6)
```

**Lemma 1 (Schur).** `K̂_i ⊙ K̂_j` is PSD.

*Sketch.* The Hadamard product of two PSD matrices is PSD; this is the
classical Schur product theorem. Both factors are PSD by construction
(eq. 1). □

**Lemma 2 (joint marginalisation).** If `K̂_i` and `K̂_j` come from
*statistically independent* features (i.e. the activations `a_i` and
`a_j` factor as products of marginals over the batch), then

```
H_α(K̂_{ij}) = H_α(K̂_i) + H_α(K̂_j).
```

This justifies treating `K̂_{ij}` as a "joint distribution" of `(K̂_i,
K̂_j)`. The proof is a direct manipulation of the `tr(K̂^α)` expression
using independence of the spectral factors; full proof in Sanchez-
Giraldo et al. 2014, Theorem 5.

## 5. Cross-layer mutual information

Define

```
I_α(K̂_i ; K̂_j) := H_α(K̂_i) + H_α(K̂_j) − H_α(K̂_{ij}).      (eq. 7)
```

By Lemma 2 above, `I_α = 0` when layers are independent. The non-trivial
fact is one-sided:

**Theorem (Sanchez-Giraldo Theorem 6).** For `α ∈ (0, 1) ∪ (1, ∞)`,
`I_α(K̂_i ; K̂_j) ≥ 0`, with equality iff `K̂_{ij} = K̂_i ⊙ K̂_j` (the
independence factorisation holds).

*Sketch.* Apply the log-sum inequality to the spectrum of `K̂_{ij}`
relative to the product of marginal spectra. The Hadamard product
satisfies `tr((K̂_i ⊙ K̂_j)^α) ≤ tr(K̂_i^α)^{1/2} tr(K̂_j^α)^{1/2}`
(Cauchy-Schwarz on traces); rearranging gives the desired bound. □

**Symmetry:** `I_α(K̂_i ; K̂_j) = I_α(K̂_j ; K̂_i)` — immediate from
Hadamard commutativity.

**Boundedness:** `0 ≤ I_α(K̂_i ; K̂_j) ≤ min(H_α(K̂_i), H_α(K̂_j))`.

**Caveat — I_α(K, K) ≠ H_α(K) in general.** Unlike classical Shannon
mutual information, the matrix-based Rényi self-MI is *not* equal to
the entropy. The Hadamard square `K̂ ⊙ K̂` has a different eigenvalue
distribution than `K̂` itself, so `H_α(K̂ ⊙ K̂) ≠ H_α(K̂)` and therefore
`I_α(K̂; K̂) = 2H_α(K̂) − H_α(K̂ ⊙ K̂) ≠ H_α(K̂)`. The estimator is
*ordinally* correct (`I_α(K, K) > I_α(K, K')` for any independent
`K'`) but the absolute "self-information" value should not be
interpreted as classical Shannon `H`. Verified empirically in our
sanity tests.

## 6. Conditional entropy

Define

```
H_α(K̂_j | K̂_i) := H_α(K̂_{ij}) − H_α(K̂_i).            (eq. 8)
```

**Properties:**

- `H_α(K̂_j | K̂_i) ≥ 0`. (The joint cannot have less entropy than its
  conditioning factor.)
- `H_α(K̂_j | K̂_i) = H_α(K̂_j) − I_α(K̂_i ; K̂_j)`. (Standard chain rule.)
- The chain rule `H_α(K̂_i, K̂_j) = H_α(K̂_i) + H_α(K̂_j | K̂_i)` recovers eq. 7.

## 7. Differentiability and computational complexity

**Differentiability.** With `α = 2`,

```
H_2(K̂) = − log tr(K̂^2),
tr(K̂^2) = Σ_{i,j} K̂_{ij}^2,
K̂ = (a a^T) / tr(a a^T).
```

Each operation is a smooth function of the activations `a`, hence of the
weights through the chain rule. Standard autograd (PyTorch, JAX) handles
this without manual derivatives.

For `α ≠ 2` we need eigenvalues, and `torch.linalg.eigvalsh` provides
gradients via the Bunch-Lehmberger formula on symmetric inputs.

**Computational cost per regulariser-update step:**

| Step | Cost |
|---|---|
| Compute K̂_l for one layer | O(B² d_l) |
| Compute H_2(K̂_l) per layer | O(B²) (just `tr(K̂^2)`) |
| Compute K̂_{ij} = K̂_i ⊙ K̂_j per pair | O(B²) |
| Compute H_2(K̂_{ij}) per pair | O(B²) |
| Pairs: L(L−1)/2 | O(L²) |
| **Total at α = 2** | **O(L² · B²) per regulariser step** |

For our setups (B = 128, L ∈ {3, …, 21}), this is < 1 ms on GPU per
update. Practical overhead vs the existing weight-side regulariser is
negligible.

## 8. The four extensions, unified

Path letters refer to the universality-programme shorthand in
`reports/phase7c_brief.md`.

### Path E — temporal conditional entropy

For batch `X` shared across two consecutive regulariser updates, let
`K̂_l^{(t)}` and `K̂_l^{(t-1)}` be the layer-l Gram matrices at steps `t`
and `t-1`. Define

```
L_E = λ · H_α(K̂_l^{(t)} | K̂_l^{(t-1)}).
```

Penalises *novel* layer information per step — encourages temporally
coherent feature evolution. Reduces to `kl_trajectory` when the
distributions are interpreted differently; here the conditioning is
explicit through the Hadamard-product joint.

### Path F — cross-layer mutual information

```
L_F = λ · Σ_{i < j} I_α(K̂_i ; K̂_j).
```

Penalises **redundancy between layers**. Encourages each layer to encode
information not already in its predecessors / successors. Connects to
disentangled representations and deep-ensemble diversity literature.

### Path G — joint view entropy

For the same network, build adjacencies under both dataflow (star
expansion) and factor (clique expansion) views, get Gram matrices
`K̂_dataflow` and `K̂_factor`. Define

```
L_G = λ · H_α(K̂_dataflow, K̂_factor)
    = λ · H_α(K̂_dataflow ⊙ K̂_factor) (after normalisation).
```

Removes the user's choice of view: the regulariser penalises whichever
view (or combination) currently has highest joint entropy.

### Path H — information bottleneck

Build `K̂_X` from the input batch and `K̂_Y` from the target batch.
For a hidden representation `T = a_l(X)` with Gram `K̂_T`,

```
L_H = λ · I_α(K̂_X ; K̂_T) − β · I_α(K̂_T ; K̂_Y).
```

This is the **matrix-based information-bottleneck** of Achille and
Soatto. Path F (cross-layer MI) is the structural specialisation;
Path H is the data-dependent specialisation; both share the same
underlying I_α primitive.

## 9. Caveats and design choices

1. **Data dependence.** Unlike the existing weight-only regularisers,
   `K̂_l` requires a forward pass. In our training loop this is free
   (one forward already happens for the task loss). **However**,
   the regulariser becomes stochastic across mini-batches.
2. **Mitigating mini-batch noise.** Sample a fixed held-out batch
   `X_reg` once at training start and compute `K̂_l(X_reg)` from that.
   Removes mini-batch fluctuations from the regulariser signal.
3. **Choice of α.** α = 2 has the closed-form `−log tr(K̂^2)` and is
   the recommended default for compute. α → 1 gives von-Neumann/Shannon,
   which is what reviewers expect from "entropy" in deep learning. We
   default to α = 2 and report Shannon as a sensitivity analysis.
4. **Activation hook point.** Hook **post-linear, pre-activation** to
   keep the connection to the existing weight-side adjacency closest;
   the post-non-linear hook is also defensible and slightly cheaper.

## 10. Hypotheses for empirical testing (phase 8)

If layer redundancy is the structural cause of the residual negatives
(circles, CapsMLP) under the original weight-side regulariser, then:

1. **MI on positive datasets** (spirals, MNIST plain MLP) should be
   non-negligible and the cross-layer-MI penalty should preserve the
   positive sign — possibly with smaller magnitude than scalar entropy
   (since MI penalises a different structural quantity).
2. **MI on negative datasets** (circles, CapsMLP MNIST) should be
   *higher* than on positives — the failure mode is layers learning
   redundant information instead of a distributed code. If this
   prediction holds, decorrelating via MI penalty should *neutralise*
   the negatives more cleanly than scalar-entropy normalisation does.
3. **Variance reduction**: the dominant cross-cutting effect of the
   weight-side family was σ-reduction in val-acc across seeds. We
   predict the activation-side MI penalty also reduces σ, possibly
   *more strongly* on architectures with redundant pathways
   (CapsMLP, ResMLP-20).

A clear pass through these three predictions would justify presenting
the activation-side framework as the **principled refinement** of the
weight-side one in the paper.

## 11. References

- Sanchez-Giraldo, Rao, Principe (2014). *Measures of Entropy from Data
  Using Infinitely Divisible Kernels*. IEEE Trans. Info. Theory 61(1).
- Yu, Sanchez-Giraldo, Principe (2019). *Multivariate Extension of
  Matrix-Based Rényi's α-Order Entropy Functional*. IEEE Trans. PAMI 42(8).
- Wickstrøm et al. (2020). *Information Plane Analysis of Deep Neural
  Networks via Matrix-Based Rényi's Entropy and Tensor Kernels*. JMLR 21.
- Achille and Soatto (2018). *Information Dropout: Learning Optimal
  Representations Through Noisy Computation*. IEEE Trans. PAMI 40(12).
- Tishby and Zaslavsky (2015). *Deep Learning and the Information
  Bottleneck Principle*. ITW 2015.
