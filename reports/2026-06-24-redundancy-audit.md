# Redundancy audit — signed_kan / hymeko_rl / signedkan_wip (engine-wise reuse)

**Date:** 2026-06-24 · Goal: find duplicated logic across the three lines and reuse from the engine (`signed_kan`).

## Fixed in this pass

| redundancy | was | now |
|---|---|---|
| **Catmull-Rom eval** | 3 copies (signedkan_wip `_catmull_rom_eval`, hymeko_rl `_catmull_rom`, my signed_kan port) | **one** canonical `signed_kan.catmull_rom`; hymeko_rl re-exports it, signedkan_wip imports it (keeps its `torch.compile` wrapper). Parity-tested. |
| **B-spline basis** (`cox_de_boor`, `make_uniform_knots`) | duplicated — my phase-3 port copied them into `signed_kan`, signedkan_wip still had its own | **one** in `signed_kan.splines`; signedkan_wip imports them (re-exported for its B-spline activations + `prune_distill`). Bit-parity-tested (`test_splines.py`); B-spline self-tests still pass. |
| **signed-adjacency-from-edges** | `build_signed_adj` local to the OTC harness | moved to `signed_kan.build_signed_adjacency` (reusable; the harness imports it). Tested (`test_graphs.py`). |

All three were "the same atom in N places"; they now live once in the engine. No behaviour change (parity/self-tests
green; the in-flight legacy run already loaded its modules, so it is unaffected).

## Flagged — recommend, not done this pass

- **Link-sign AUC/F1 eval is reimplemented ~10×** across `signedkan_wip` (`core/train.py`, `core/sigma_masking.py`,
  `chicken/{aggressor,unsupervised}.py`, `synthetic_signed_graphs.py`, `experiments/eval/eval_metrics_full.py`, …)
  plus my `otc_cr_ab.py`. The block is always `probs = sigmoid(logits); auc = roc_auc_score(y01, probs); f1 = …`.
  This is the §6.5 #3 "per-experiment scaffold" pattern. **Recommend** a single `link_sign_metrics(y, probs)` helper
  (in `signedkan_wip`, not the pure-torch `signed_kan` core — it pulls in sklearn). Not done now: ~10 call sites,
  several feed published numbers — a focused refactor with regression checks, out of scope for this pass.
- **Highway gate**: `signed_kan.HighwaySkip` (per-layer, node-level) vs signedkan_wip's inline `gate_inner` /
  `cr_highway` (per-arc, inside the triad gather). Same formula, **different application point** — not trivially one
  module. Keep both; the shared piece (the CR eval the `cr_highway` gate calls) is already unified.
- **Signed message passing itself**: `signed_kan` (pairwise, dense/sparse) vs signedkan_wip (k-uniform triad +
  dual spline) are **genuinely different algorithms** (the phase-3-halt finding), not redundancy. Unifying the
  *atoms* (splines, adjacency builder, gate formula) is the right granularity; the layers stay distinct.

## Tests added this pass
- `signed_kan/tests/test_splines.py` (5): B-spline activation (shape / zero-coef / grad), `cox_de_boor`
  partition-of-unity, `make_activation` all kinds, **cox_de_boor + Catmull-Rom parity vs signedkan_wip** (the dedup
  gates).
- `signed_kan/tests/test_graphs.py` (4): `build_signed_adjacency` (sign separation + no-leakage, symmetry/row-norm,
  dense↔sparse parity, errors).
- (Existing: `test_core.py` 11, `test_graph_convention.py` 4 — total **24 signed_kan tests green**, ruff + mypy clean.)

## Net
The engine (`signed_kan`) now owns the genuinely-shared primitives (both splines, the adjacency builder, the
activation Strategy, the backends, the heads); the two consumer lines import them. Remaining duplication
(link-sign eval) is flagged with a concrete recommendation; the layer-level "duplication" is not duplication
(different algorithms) and is correctly left split.
