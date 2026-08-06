# Redundancy audit — hymeko_neuro.core / hymeko_rl / hymeko_neuro (engine-wise reuse)

**Date:** 2026-06-24 · Goal: find duplicated logic across the three lines and reuse from the engine (`hymeko_neuro.core`).

## Fixed in this pass

| redundancy | was | now |
|---|---|---|
| **Catmull-Rom eval** | 3 copies (hymeko_neuro `_catmull_rom_eval`, hymeko_rl `_catmull_rom`, my hymeko_neuro.core port) | **one** canonical `hymeko_neuro.core.catmull_rom`; hymeko_rl re-exports it, hymeko_neuro imports it (keeps its `torch.compile` wrapper). Parity-tested. |
| **B-spline basis** (`cox_de_boor`, `make_uniform_knots`) | duplicated — my phase-3 port copied them into `hymeko_neuro.core`, hymeko_neuro still had its own | **one** in `hymeko_neuro.core.splines`; hymeko_neuro imports them (re-exported for its B-spline activations + `prune_distill`). Bit-parity-tested (`test_splines.py`); B-spline self-tests still pass. |
| **signed-adjacency-from-edges** | `build_signed_adj` local to the OTC harness | moved to `hymeko_neuro.core.build_signed_adjacency` (reusable; the harness imports it). Tested (`test_graphs.py`). |

All three were "the same atom in N places"; they now live once in the engine. No behaviour change (parity/self-tests
green; the in-flight legacy run already loaded its modules, so it is unaffected).

## Flagged — recommend, not done this pass

- **Link-sign AUC/F1 eval is reimplemented ~10×** across `hymeko_neuro` (`core/train.py`, `core/sigma_masking.py`,
  `chicken/{aggressor,unsupervised}.py`, `synthetic_signed_graphs.py`, `experiments/eval/eval_metrics_full.py`, …)
  plus my `otc_cr_ab.py`. The block is always `probs = sigmoid(logits); auc = roc_auc_score(y01, probs); f1 = …`.
  This is the §6.5 #3 "per-experiment scaffold" pattern. **Recommend** a single `link_sign_metrics(y, probs)` helper
  (in `hymeko_neuro`, not the pure-torch `hymeko_neuro.core` core — it pulls in sklearn). Not done now: ~10 call sites,
  several feed published numbers — a focused refactor with regression checks, out of scope for this pass.
- **Highway gate**: `hymeko_neuro.core.HighwaySkip` (per-layer, node-level) vs hymeko_neuro's inline `gate_inner` /
  `cr_highway` (per-arc, inside the triad gather). Same formula, **different application point** — not trivially one
  module. Keep both; the shared piece (the CR eval the `cr_highway` gate calls) is already unified.
- **Signed message passing itself**: `hymeko_neuro.core` (pairwise, dense/sparse) vs hymeko_neuro (k-uniform triad +
  dual spline) are **genuinely different algorithms** (the phase-3-halt finding), not redundancy. Unifying the
  *atoms* (splines, adjacency builder, gate formula) is the right granularity; the layers stay distinct.

## Tests added this pass
- `hymeko_neuro/core/tests/test_splines.py` (5): B-spline activation (shape / zero-coef / grad), `cox_de_boor`
  partition-of-unity, `make_activation` all kinds, **cox_de_boor + Catmull-Rom parity vs hymeko_neuro** (the dedup
  gates).
- `hymeko_neuro/core/tests/test_graphs.py` (4): `build_signed_adjacency` (sign separation + no-leakage, symmetry/row-norm,
  dense↔sparse parity, errors).
- (Existing: `test_core.py` 11, `test_graph_convention.py` 4 — total **24 hymeko_neuro.core tests green**, ruff + mypy clean.)

## Net
The engine (`hymeko_neuro.core`) now owns the genuinely-shared primitives (both splines, the adjacency builder, the
activation Strategy, the backends, the heads); the two consumer lines import them. Remaining duplication
(link-sign eval) is flagged with a concrete recommendation; the layer-level "duplication" is not duplication
(different algorithms) and is correctly left split.
