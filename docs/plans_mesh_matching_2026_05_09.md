# Mesh matching via signed-cycle attention — 2026-05-09

A triangulated 3D mesh IS, structurally, a signed-3-cycle hypergraph.  Each triangle is a $k=3$ cycle; face-normal orientation gives a natural sign per face; the consistent-orientation requirement is exactly Cartwright–Harary balance.  HSiKAN's primitives — σ-masked aggregation, α-mixer, sparse Hamilton-product attention — map onto this structure with no architectural retrofitting.  The plan: take the SMC/journal HSiKAN architecture as-is, point it at FAUST / SHREC mesh-correspondence benchmarks, and see whether the signed-cycle bias gives a competitive correspondence model on consumer hardware.

## Goal

Establish HSiKAN as a competitive 3D mesh-correspondence architecture on FAUST and at least one of SHREC'19 / TOSCA, demonstrating:
1. **Architecture-natural fit**: triangles → signed 3-cycles, no retrofitting.
2. **Competitive on a real benchmark**: FAUST geodesic-error within 2× of FMNet / GeomFmaps.
3. **Consumer-GPU feasibility**: training and inference fit in 8 GiB GPU thanks to the cycle-batched + sparse-attention infrastructure from 2026-05-08.

The contribution thread: **"signed-incidence attention is the first mesh-correspondence architecture that natively treats the triangulation as a signed structure."**  Most baselines (FMNet, GeomFmaps, Universe of Functions) treat the mesh as a Laplacian-domain operator, not a signed hypergraph — different inductive bias.

## Why this works architecturally

| HSiKAN primitive | mesh interpretation |
|---|---|
| Vertex embeddings $\mathbf{H}_v$ | per-mesh-vertex 3D coordinate + heat-kernel descriptor |
| Edge signs $s : E \to \{\pm 1\}$ | edge dihedral-angle sign (convex/concave fold) |
| $k=3$ cycle | one mesh triangle |
| $\sigma$ parity per cycle vertex | face-normal-induced vertex orientation |
| Cartwright–Harary balance flag | consistent-normal triangle orientation across the mesh |
| Sparse Hamilton attention over $M_e$ | per-edge attention over incident triangles |
| α-mixer over $\{c_3, c_4, ...\}$ | mix of triangles, quadrilateral faces (in non-tri meshes), longer cycles |

The architectural fit is unusual.  Most graph-matching baselines either: (a) treat mesh as point cloud + Laplacian, losing connectivity; (b) use spectral methods that need eigendecomposition per mesh; (c) train a separate conv per mesh-resolution.  HSiKAN handles all three regimes via the same α-routed encoder.

## Datasets

| dataset | size | task | metric |
|---|---|---|---|
| FAUST (training set) | 100 meshes, 6890 vertices each | dense correspondence | mean geodesic error |
| SHREC'19 connectivity track | 44 source-target pairs | sparse correspondence | geodesic error at 1% / 5% |
| TOSCA non-rigid | 80 meshes | dense correspondence | geodesic error |

All standard, all available with prepared point-cloud + triangulation files.

## Architecture

```
mesh M = (V, E_mesh, F_triangles, signs from face normals)
  ↓
Per-vertex features:    coords (3) + HKS / WKS descriptor (16-32) + Laplacian eigvecs (8)
  ↓
Encoder: HSiKAN with K = {c_3, c_4} (triangle, quad-face)
  - σ from face-normal alignment
  - sparse Hamilton attention over (vertex, triangle) incidence
  - Highway-gated (uniform pool fallback)
  ↓
Per-vertex embedding   z_v ∈ R^d
```

For correspondence:
- Encode source mesh M_A → embeddings $Z_A \in \mathbb{R}^{V_A \times d}$
- Encode target mesh M_B → $Z_B \in \mathbb{R}^{V_B \times d}$
- Pairwise cost matrix: $C_{ij} = \|Z_{A,i} - Z_{B,j}\|^2$ (or cosine)
- Sinkhorn iterations → soft permutation matrix $P$
- Loss: BCE on ground-truth correspondence pairs (FAUST has GT) or geodesic-distance loss

## Experiments

### M1. FAUST sanity smoke

- 5 source meshes, 5 target meshes (subset)
- Encoder: $h=16$, $\mathcal{K}{=}\{c_3\}$ only (just triangles), Highway-quat
- Loss: ground-truth correspondence BCE
- Acceptance: training loss decreases, gradients flow, no OOM

### M2. FAUST full dense correspondence

- All 100 meshes, leave-one-out
- Encoder: $\mathcal{K}{=}\{c_3, c_4\}$ where c_4 is "two-triangle quad face" (joined adjacent triangles)
- 80 epochs, Adam
- Compare against FMNet, GeomFmaps at iso-param (~330K-1M)
- Metric: mean geodesic error on test mesh

**Acceptance**: mean geodesic error within 2× of FMNet (FMNet typically ~5-7 cm on FAUST).

### M3. SHREC'19 transfer

- Train on FAUST only, evaluate on SHREC'19 connectivity track (zero-shot transfer)
- Tests architectural generality

**Acceptance**: > 50% correspondence accuracy at 5% geodesic threshold.

### M4. Sign ablation

- Same as M2 with all face signs = +1 (unsigned variant)
- Tests whether the dihedral-angle sign assignment adds anything

**Acceptance**: signed > unsigned by > 1σ.

### M5. Routing readout

- Visualise α_κ for c_3 vs c_4 across meshes of different topology (genus-0 sphere-like vs genus-1 torus-like)
- Tests whether α captures mesh topology

**Acceptance**: α distribution differs across topology classes by > 0.1.

## Implementation notes

- New module `signedkan_wip/src/mesh_signed_graph.py` (~200 LOC):
  - Mesh loader (PLY / OBJ via `trimesh`)
  - Face-normal sign assignment per edge
  - Triangle → SignedNTuple converter
  - HKS / WKS descriptor (use `pyshot` or compute inline)
- Extend `run_final_cell.py` with `cell_mesh_correspondence` (~150 LOC)
- New runner `signedkan_wip/src/run_faust.py` (~100 LOC)
- Eval harness for geodesic error (~80 LOC; standard `compute_geodesic` via Dijkstra over the mesh graph)
- Sinkhorn implementation (~40 LOC; standard)
- Total: ~600 LOC new code, no new dependencies (trimesh + numpy + torch all already present)

## Cost

- M1 smoke: ~half day (1-2 GPU hours)
- M2 full FAUST: ~2-3 days, ~5-10 GPU hours per training run, ~3 runs to tune
- M3 SHREC zero-shot: ~half day eval only
- M4 ablation: 1 day, same training time as M2
- M5 routing: half day analysis

Total: ~5-7 days for a full result + write-up.

## Risk register

| risk | probability | mitigation |
|---|---|---|
| Per-vertex HKS descriptor doesn't generalise across mesh topologies | medium | use multiple descriptors (HKS + WKS + 3D coords); HSiKAN's α-mixer handles fusion |
| Sinkhorn training is unstable | low | proven technique; many open implementations |
| FAUST training time too long | medium | start with subsampled meshes (1000 vertices); FAUST has standard remeshing |
| Architecture wins on FAUST but not on real-world meshes | medium | scope this paper to standard benchmarks; real-world generalisation is its own problem |
| FMNet / GeomFmaps already too tuned to beat | high | the win is architectural-natural fit, not per-benchmark SOTA |

## Acceptance for the plan as a whole

- M1 trains smoke: minimum.
- M2 within 2× of FMNet: paper-grade result.
- M3 + M4 confirm generalisation + sign-bias contribution: rounds out the paper.
- M5 architectural-readout: interpretability handle that no other mesh-matching baseline has.

If M2 fails by >5×, mesh matching is not a natural HSiKAN target and the plan closes as a negative result.

## Order of operations

1. Mesh loader + face-sign assignment (`mesh_signed_graph.py`) — 1-2 days
2. M1 sanity smoke on a single mesh pair — 1 day
3. M2 FAUST training pipeline — 2-3 days
4. M3 + M4 evaluations — 2 days
5. Paper draft (8 pp, SIGGRAPH-style) — 1-2 weeks

Total: ~4-5 weeks for a venue-ready submission.

## What this plan does NOT do

- Doesn't claim SOTA on FAUST.  The bar is "competitive" (within 2× of best).
- Doesn't tackle non-rigid mesh deformation as an explicit task.  That's a separate problem.
- Doesn't address mesh generation / synthesis.
- Doesn't propose mesh-aware attention specific to differential-geometry — it's just HSiKAN as-is on a triangulated mesh.

## Connection to other plans

- Phase A entropy regularisers (`docs/plans_entropy_learning_2026_05_08.md`) — the α-entropy aux loss could lift mesh-correspondence training stability.
- Tabular benchmarks (`docs/plans_hsikan_tabular_benchmarks_2026_05_09.md`) — successful mesh matching is a *different* universality demonstration; together they argue HSiKAN as a unified architecture.
- BO controller (`signedkan_wip/src/run_optuna_search.py`) — directly applicable for tuning the mesh-matching hyperparameters.

## Why this is venue-grade novel

Most graph / mesh matching papers either operate on the spectral domain (Laplacian eigenvectors) or on point clouds (PointNet-style permutation equivariance).  Neither leverages the *signed* nature of triangulated mesh structure (face-normal alignment, dihedral-angle sign), which is intrinsic to oriented manifold geometry.  HSiKAN's signed-cycle aggregation is the first method that uses this signal natively.  The paper claim:
- *"Triangulated meshes are signed 3-cycle hypergraphs.  HSiKAN is the natural mesh-correspondence architecture."*

This is a categorical claim with a clean architectural argument — different from "yet another GNN beats FMNet by 0.5pp."
