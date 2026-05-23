# Quickstart: Compute structural entropy + HOSVD

`hymeko_core::tensor::decomposition` ships:
- **Mode-k unfolding** (matricization) of 3D tensors
- **Truncated SVD** via power iteration
- **HOSVD / Tucker decomposition**
- **Structural entropy** from the singular-value spectrum

Useful for: rank-bounding sparse incidence matrices, compressing cycle ensembles, and (planned) Baranyi-style TP transformations of parameter-dependent hypergraph operators.

## Get a 3D tensor from a HyMeKo IR

```rust
use hymeko_query::transforms::TransformConfig;
use hymeko_hre::engine::HypergraphEngine;

let ir = parse_and_resolve("data/robotics_imported/wam/wam.hymeko")?;
let engine = HypergraphEngine::new(&ir, &resolver);
// Clique-tensor expansion: 3D sparse tensor (k=hyperedges, i,j=vertex pairs)
let coo = engine.compile_clique_tensor_expansion::<f32>(&ir);
println!("shape = ({}, {}, {})", coo.num_slices, coo.dim_i, coo.dim_j);
```

## Mode-k unfold + SVD

```rust
use hymeko::tensor::decomposition::{mode_k_unfold, truncated_svd_power};

// Unfold along mode 0 (each row = one hyperedge's flattened vertex pair matrix)
let (mat, m, n) = mode_k_unfold(&coo, 0);

// Top-r SVD via power iteration
let rank = 16;
let (u, sigma, v_t) = truncated_svd_power(&mat, m, n, rank);
println!("top singular values: {:?}", sigma.iter().take(5).collect::<Vec<_>>());
```

## Structural entropy

```rust
use hymeko::tensor::decomposition::structural_entropy;

let h = structural_entropy(&sigma);  // Shannon entropy of the σ²-normalized distribution
println!("structural entropy = {h:.4}");
```

Higher entropy → more uniform spectrum → less compressible. Lower entropy → most signal in a few modes → high compression potential.

## HOSVD (Tucker)

```rust
use hymeko::tensor::decomposition::hosvd;

let (core, u_factors) = hosvd(&coo, &[16, 16, 16]);  // rank-r per mode
// core is a small dense (r0, r1, r2) tensor; u_factors[i] is the basis matrix for mode i
```

The decomposition `T ≈ core ×₁ U₀ ×₂ U₁ ×₃ U₂` lets you compute approximations / measure rate-distortion.

## Why this matters for HSiKAN

Sparse signed-incidence `M_e^{(k)}` for k=4 on Slashdot is up to (220K × 55M) — too large to materialise. HOSVD-based approximation gives a polytopic decomposition `M_e ≈ Σᵢ wᵢ(p) M_i` where `M_i` are small vertex tensors and `wᵢ` are basis functions over a graph parameter `p`. At inference, evaluate `wᵢ(p)` for the test edge and sum r small `M_i` instead of the full mm.

This is the building block for the **Baranyi-style TP transformation** sketched in `docs/plans_rl_al_hsikan_2026_05_06.md` — composable with the learned-enumerator / RL-controller direction.

## See also

- `hymeko_core/src/tensor/decomposition.rs` — implementation
- `hymeko_core/src/tensor/representations/tensor_coo.rs` — the COO 3D tensor type
- [Concepts: Tensor decomposition](../concepts/tensor-decomposition.md) — deeper math + applications

## Next

- [Concepts: The IR](../concepts/ir.md) — what `coo` actually represents
- [Research: HSiKAN](../research/hsikan.md) — where TP / HOSVD plug in for learned enumeration
