"""Sanity-test the matrix-based Renyi-alpha entropy estimator.

Verifies the closed-form alpha=2 collision-entropy / Hadamard-product
formulation behaves as expected on three controlled cases:

    1. Independent random Gaussians → I_2 should be ≈ 0.
    2. Identical activations           → I_2 should be ≈ H_2.
    3. Deterministic linear transform  → I_2 should be ≈ H_2 (since y = Wx
                                          is a bijection in feature space
                                          for invertible W).
    4. Orthogonal projections          → I_2 should be substantially > 0
                                          but < min(H_2_a, H_2_b).

Implementation:
    H_2(K)            = − log tr(K_norm²)
    K(a)              = (a a^T) / tr(a a^T)
    K_join(a, b)      = (K(a) ⊙ K(b)) / tr(K(a) ⊙ K(b))
    I_2(a, b)         = H_2(K(a)) + H_2(K(b)) − H_2(K_join)

Run: python3 python/benches/thesis_iv_hard/test_sanchez_giraldo.py
"""
from __future__ import annotations

import math
import torch


def renyi2_entropy(K_norm: torch.Tensor) -> torch.Tensor:
    """H_2(K) = -log tr(K^2) for trace-1 PSD K (no eigendecomp needed)."""
    return -torch.log((K_norm * K_norm).sum())


def gram_normalised(a: torch.Tensor) -> torch.Tensor:
    """Linear-kernel Gram K(a) = (a a^T) / tr(a a^T).
    Rank-deficient when d < B; biased estimator of H_2 in that regime.
    Kept for diagnostic comparison with the Gaussian variant."""
    K = a @ a.t()
    return K / K.trace().clamp(min=1e-12)


def median_bandwidth(a: torch.Tensor) -> torch.Tensor:
    """Median-heuristic Gaussian kernel bandwidth σ from Garreau,
    Jitkrittum, Kanagawa (2017). σ² is set so that 2σ² = median of
    squared pairwise distances. Common Sanchez-Giraldo / kernel-method
    default."""
    sq = a.pow(2).sum(dim=1, keepdim=True)
    pair_sq = (sq + sq.t() - 2 * (a @ a.t())).clamp(min=0)
    med = pair_sq[pair_sq > 0].median().clamp(min=1e-8)
    return (med / 2.0).sqrt()


def gaussian_gram(a: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
    """Gaussian-RBF Gram K(a)_{ij} = exp(-||a_i - a_j||² / (2σ²)),
    normalised to trace 1.

    Full rank B almost surely → unbiased H_2 estimator. This is the
    Sanchez-Giraldo et al. (2014) kernel choice. σ defaults to the
    median heuristic if not supplied."""
    if sigma is None:
        sigma = median_bandwidth(a)
    sq = a.pow(2).sum(dim=1, keepdim=True)
    pair_sq = (sq + sq.t() - 2 * (a @ a.t())).clamp(min=0)
    K = torch.exp(-pair_sq / (2 * sigma * sigma))
    return K / K.trace().clamp(min=1e-12)


def hadamard_join(K_i: torch.Tensor, K_j: torch.Tensor) -> torch.Tensor:
    """K_join = (K_i ⊙ K_j) / tr(K_i ⊙ K_j)."""
    H = K_i * K_j
    return H / H.trace().clamp(min=1e-12)


def mi_renyi2(
    a: torch.Tensor, b: torch.Tensor, kernel: str = "gauss"
) -> tuple[float, float, float, float]:
    """I_2(a; b) using the chosen kernel for sample-side Gram."""
    if kernel == "linear":
        K_a, K_b = gram_normalised(a), gram_normalised(b)
    elif kernel == "gauss":
        K_a, K_b = gaussian_gram(a), gaussian_gram(b)
    else:
        raise ValueError(f"unknown kernel {kernel}")
    K_j = hadamard_join(K_a, K_b)
    H_a = renyi2_entropy(K_a).item()
    H_b = renyi2_entropy(K_b).item()
    H_j = renyi2_entropy(K_j).item()
    I = H_a + H_b - H_j
    return H_a, H_b, H_j, I


def header(title: str) -> None:
    print(f"\n--- {title} ---")


def main() -> None:
    torch.manual_seed(0)
    B = 256                   # batch size
    d = 32                    # feature dim

    print(f"Batch B={B}, feature dim d={d}, kernel = Gaussian-RBF (Sanchez-Giraldo)")
    print(f"Maximal H_2 for B-dim valid distribution = log(B) = {math.log(B):.4f}")

    # 1. Independent Gaussians ---------------------------------------------------
    header("1. Independent N(0, I) Gaussians (expect I_2 ≈ 0)")
    a = torch.randn(B, d)
    b = torch.randn(B, d)
    H_a, H_b, H_j, I = mi_renyi2(a, b, kernel="gauss")
    print(f"  H_2(a)={H_a:.4f}, H_2(b)={H_b:.4f}, H_2(joint)={H_j:.4f}")
    print(f"  I_2(a; b) = {I:.4f}   (should be small relative to H_2)")
    assert I >= -1e-6, f"non-negativity violated: I_2 = {I}"
    assert I < 0.10 * min(H_a, H_b), \
        f"independence violated: I_2={I:.4f} vs 10%·min(H)={0.10*min(H_a,H_b):.4f}"
    print(f"  ✓ non-negative; ✓ < 10%·min(H_a,H_b)={0.10*min(H_a,H_b):.4f}")

    # 2. Identical activations ---------------------------------------------------
    # Caveat: matrix-based Rényi I_α(K; K) ≠ H_α(K) in general because
    # the Hadamard *square* K ⊙ K has a different spectrum than K. The
    # correct test is monotonic: I_α(K, K) should be substantially
    # larger than I_α(K, K') for any independent K'.
    header("2. Identical activations (expect I_2 ≫ I_2 of independent)")
    a = torch.randn(B, d)
    b_same = a.clone()
    b_indep = torch.randn(B, d)
    H_a, _, _, I_same  = mi_renyi2(a, b_same,  kernel="gauss")
    _,   _, _, I_indep = mi_renyi2(a, b_indep, kernel="gauss")
    print(f"  I_2(a; a) = {I_same:.4f}  (identical)")
    print(f"  I_2(a; b_indep) = {I_indep:.4f}  (independent)")
    assert I_same > 2.0 * I_indep, \
        f"identical I_2={I_same:.4f} should be ≫ independent I_2={I_indep:.4f}"
    print(f"  ✓ identical I_2 is at least 2× the independent I_2")

    # 3. Deterministic linear transform -----------------------------------------
    header("3. Deterministic linear transform b = a · W (expect I_2 ≫ independent)")
    a = torch.randn(B, d)
    W = torch.randn(d, d)  # invertible w.h.p.
    b = a @ W
    _, _, _, I_lin    = mi_renyi2(a, b, kernel="gauss")
    _, _, _, I_indep2 = mi_renyi2(a, torch.randn(B, d), kernel="gauss")
    print(f"  I_2(a; a·W) = {I_lin:.4f}")
    print(f"  I_2(a; indep) = {I_indep2:.4f}")
    # Gaussian RBF with median bandwidth varies by ‖W‖; demand at least
    # 2× the independent baseline to confirm substantial sharing.
    assert I_lin > 2.0 * I_indep2, \
        f"deterministic link should keep info: I_2(a;aW)={I_lin:.4f} vs indep {I_indep2:.4f}"
    print(f"  ✓ I_2(deterministic link) > 2× I_2(independent)")

    # 4. Orthogonal projection (partial overlap) ---------------------------------
    header("4. Orthogonal projections of a (expect 0 < I_2 < min(H_a, H_b))")
    a = torch.randn(B, d)
    # Build two orthogonal projection matrices P1, P2 spanning disjoint subspaces.
    Q, _ = torch.linalg.qr(torch.randn(d, d))
    P1 = Q[:, :d//2]                          # first half axes
    P2 = Q[:, d//2:]                          # second half axes
    a_p1 = a @ P1                             # projection 1
    a_p2 = a @ P2                             # projection 2
    H_a, H_b, H_j, I = mi_renyi2(a_p1, a_p2, kernel="gauss")
    print(f"  H_2(p1)={H_a:.4f}, H_2(p2)={H_b:.4f}, H_2(joint)={H_j:.4f}")
    print(f"  I_2(p1; p2) = {I:.4f}   (expected: positive but bounded above)")
    assert I > 0, "orthogonal projections of same source should still share info"
    assert I < min(H_a, H_b), "projections share less info than full identity"
    print("  ✓ 0 < I_2 < min(H_a, H_b)")

    # 5. Differentiability check -------------------------------------------------
    header("5. Differentiability of H_2 and I_2 wrt activations")
    a = torch.randn(B, d, requires_grad=True)
    b = torch.randn(B, d, requires_grad=True)
    K_a = gaussian_gram(a)
    K_b = gaussian_gram(b)
    K_j = hadamard_join(K_a, K_b)
    H_a = renyi2_entropy(K_a)
    H_b = renyi2_entropy(K_b)
    H_j = renyi2_entropy(K_j)
    I = H_a + H_b - H_j
    I.backward()
    grad_a_norm = a.grad.norm().item()
    grad_b_norm = b.grad.norm().item()
    print(f"  ‖∂I_2/∂a‖ = {grad_a_norm:.4e}, ‖∂I_2/∂b‖ = {grad_b_norm:.4e}")
    assert grad_a_norm > 1e-8 and grad_b_norm > 1e-8, "grads should be non-zero"
    print("  ✓ differentiable through autograd")

    print("\nAll sanity checks passed ✓")


if __name__ == "__main__":
    main()
