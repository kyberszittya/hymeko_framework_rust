# Concept: Tensor decomposition

`hymeko_core::tensor::decomposition` ships:

- **Mode-k unfolding** of 3D tensors
- **Truncated SVD** via power iteration (Lanczos-like)
- **HOSVD / Tucker decomposition** (the Higher-Order SVD)
- **Structural entropy** from singular-value spectrum

For runnable examples and how to call from Python, see [Quickstart: Compute structural entropy + HOSVD](../quickstart/13-tensor-decomposition.md).

## Why it matters for HyMeKo

Two strands:

**1. IR analysis.** The clique-tensor expansion of a HyMeKo IR produces a 3D sparse tensor (vertex × vertex × hyperedge). HOSVD compresses this into a small Tucker core + per-mode basis matrices. Structural entropy of the spectrum gives a basis-invariant scalar measuring how much "structure" is in the IR.

**2. Polytopic decomposition for hypergraph operators.** Sparse `M_e^{(k)}` for k=4 on 220K-edge graphs has 55M cycle columns — too large to materialise. HOSVD-based polytopic decomposition (Baranyi-style TP transformation) approximates `M_e ≈ Σᵢ wᵢ(p) M_i` where `wᵢ` are basis functions over a graph parameter and `M_i` are small vertex tensors. At inference, evaluate r small `M_i` instead of one huge mm.

## See also

- [Quickstart: Compute structural entropy + HOSVD](../quickstart/13-tensor-decomposition.md) — runnable
- `hymeko_core/src/tensor/decomposition.rs` — implementation
- `docs/plans_rl_al_hsikan_2026_05_06.md` — TP-transformation roadmap
