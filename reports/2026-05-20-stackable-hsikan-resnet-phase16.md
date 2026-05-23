# Phase 16: ResNet-style stackable HSIKAN — 2026-05-20 overnight

## Summary

Built the explicit ResNet-style stackable HSIKAN abstraction
(`SignedKANResidualBlock` + `StackedSignedKAN`), ran the 5-seed
depth-scaling experiment, **falsified the depth-helps hypothesis**
on Bitcoin Alpha. Plumbed the depth axis into the P-graph
framework so the architecture search can pick depth via ABB (which
correctly chooses L=1 as both cost-min and quality-best on this
dataset).

**Key negative result:** HSIKAN strongly degrades with depth on
Bitcoin Alpha. L=1 mean AUC = 0.770; L=8 mean AUC = 0.442
(−0.328). Both identity-residual and highway-gated skip variants
show the same monotonic degradation. ResNet-style stackability
does not transfer to HSIKAN on this dataset at the canonical
hidden=8 / n_epochs=30 config.

## Files touched

| File | Status | LOC | Notes |
| --- | --- | --- | --- |
| `docs/plans/2026-05-20-stackable-hsikan-resnet/plan.{tex,pdf,mmd,tikz}` | new | 4-format plan (3 pp PDF) | Written before code |
| `signedkan_wip/src/core/stacked_signedkan.py` | **new** | 195 | `SignedKANResidualBlock` + `StackedSignedKAN` + `StackedSignedKANConfig` (thin ResNet-style wrapper over `MultiLayerSignedKAN`) |
| `signedkan_wip/tests/test_stacked_signedkan.py` | **new** | 120 | 6 unit tests: forward/backward, shape at L=1/2/4/8, depth-1 equivalence to inner, node_embed back-compat, param-count scaling |
| `data/hsikan/sweep_msg_depth.hymeko` | **new** | 45 | 4-unit depth axis for P-graph |
| `signedkan_wip/src/hsikan_pgraph_mapping.py` | extended | +10 | `depth_l{1,2,4,8}` units + `n_layers` kwarg |

## CORE.YAML items touched

None.

## The depth-scaling experiment (5 seeds × Bitcoin Alpha × hidden=8 × n_epochs=30)

**Identity-residual skip (the `StackedSignedKAN` default):**

| L | mean AUC ± std | Δ vs L=1 | wall/seed |
| --- | --- | --- | --- |
| 1 | **0.7696 ± 0.0484** | baseline | 0.6 s |
| 2 | 0.6724 ± 0.0296 | −0.097 | 1.2 s |
| 4 | 0.5989 ± 0.0448 | −0.171 | 2.3 s |
| 8 | 0.4421 ± 0.0284 | −0.328 | 4.3 s |

**Highway-gated skip (`HighwaySignedKAN`'s default):**

| L | mean AUC ± std | Δ vs L=1 | wall/seed |
| --- | --- | --- | --- |
| 1 | 0.7546 ± 0.0222 | baseline | 0.6 s |
| 2 | 0.6938 ± 0.0411 | −0.061 | 1.2 s |
| 4 | 0.5887 ± 0.0330 | −0.166 | 2.3 s |
| 8 | 0.4682 ± 0.0482 | −0.287 | 4.4 s |

Both variants degrade monotonically with depth. The plan's
falsifier ("AUC at L∈{2,4,8} worse than L=1 by ≥ 0.05") fires for
**every depth at both skip types**. The depth-helps hypothesis is
falsified.

## Three substantive findings

### 1. ResNet-style identity residual is not a free lunch on signed-triad graphs

ResNet's identity skip lets CNN backbones go from 20 to 100+
layers without degrading. Phase 16's `SignedKANResidualBlock` is
the direct analogue: pre-norm + SignedKAN layer + identity skip on
the vertex side. On Bitcoin Alpha it does not transfer. Both the
intra-layer `inner_skip="residual"` and the between-layer
`use_residual=True` are active, and the model still degrades.

This is empirically the *pre-ResNet observation* applied to
signed-graph KAN: deeper nets without architectural change suffer
the degradation problem. ResNet's introduction of identity skip
solved this for CNN. Phase 16 confirms HSIKAN's *current*
residual-skip implementation is not solving it equivalently on
this dataset.

### 2. Bitcoin Alpha's signed-triad structure is shallow

The simplest interpretation: Bitcoin Alpha's signed triads encode
local, 1-hop signal. Each additional layer dilutes the signal
through the mean-pooled vertex-side aggregator `M_vt @ h_t`
(row-normalised triad→vertex incidence). At L=8 the signal is
mean-pooled 7 times — effectively erased. The Phase-8 SGT
finding (signed-graph transformer winning on dense walk-rich
Slashdot but losing on cycle-rich SBM) suggests Bitcoin Alpha may
not have the multi-hop walk structure that benefits from depth.

### 3. The P-graph framework still earns its keep on this negative

Even though depth hurts, plumbing the depth axis into the P-graph
fixture is useful: the framework's scalar-cost ABB **correctly
picks `depth_l1` as cost-min**, which is also the empirically best
architecture. **For once the cost-min ABB is also the
quality-max** — no need for the MO weighting (Phase 10) or the
by-product filter (Phase 11). The framework can carry both
positive levers (where they help, e.g. HSIKAN cycle-pool) and
trivially-aligned levers (where they don't, e.g. depth) without
any change to the wiring.

The depth axis is now a permanent search-space dimension; future
HSIKAN variants that *do* benefit from depth (perhaps Slashdot,
perhaps a depth-aware-init variant) plug in via the same
`hsikan_pgraph_mapping` table.

## What would make depth work (out-of-scope candidate fixes)

1. **Depth-aware initialization.** ResNet uses `kaiming_normal`
   with scale ∝ 1/√L. HSIKAN currently uses `init_scale=0.1`
   regardless of L. A fix: `init_scale = 0.1 / sqrt(L)`.
2. **Pre-activation / BN-style normalisation.** Pre-ResNet's
   `BN→ReLU→conv` order matters; HSIKAN uses post-norm
   (LayerNorm after the residual sum). Pre-norm is already
   available via `MultiLayerSignedKANConfig.layer_norm_between=True`
   but the order may differ from canonical ResNet.
3. **Stochastic depth.** Drop entire blocks at training time
   (Huang et al., 2016). Trades training-time compute for
   gradient flow at very deep nets.
4. **Test on Slashdot.** If the dataset has multi-hop walk
   structure, depth might help; Phase 8 already showed
   walks-augmented HSIKAN beats cycle-only on Slashdot (+0.075
   AUC).
5. **Bottleneck blocks.** ResNet-50's `1×1 + 3×3 + 1×1` pattern
   reduces compute. HSIKAN's signed-triad structure doesn't have
   a direct analogue, but a `d→d/2→d→d/2→d` bottleneck pattern
   might work.

Phase 16's deliverable is the substrate (block abstraction + the
falsifier-hit empirical result). The candidate fixes deserve a
separate phase if pursued.

## Test results

| Suite | Result |
| --- | --- |
| `cargo test -p hymeko_pgraph` (full) | 96 / 96 + 1 ignored doctest |
| `test_stacked_signedkan.py` | **6 / 6 pass** (new) |
| `test_hsikan_pgraph_mapping.py` | 7 / 7 pass (mapping extension is back-compat) |
| All prior Phase 1-15 suites | no regressions |

## §6.5 anti-pattern audit

No new anti-patterns. `StackedSignedKAN` is a thin Strategy-style
wrapper (delegates to `MultiLayerSignedKAN`); the
`SignedKANResidualBlock` is a single class with one forward
method; no Cartesian function-name explosion. The mapping
extension is one new dict entry per depth; not a new family.

## Open follow-ups

1. **Try the candidate fixes** — depth-aware init is the cheapest
   to test (~1 line in `SignedKANLayer.__init__`).
2. **Slashdot depth test** — the obvious next dataset to test
   depth on. Slashdot has known walk-rich structure
   ([[project-sgt-baseline]]), which is the regime where depth
   might help.
3. **Stack depth × hidden trade-off.** Phase 16 used hidden=8;
   maybe deeper + narrower wins at the same parameter budget
   (ResNet's lesson). A 2-axis sweep would surface this.
4. **Document the falsifier-hit explicitly in the framework
   book chapter** as a worked example: "the cost-min P-graph
   ABB is the empirically best architecture when the lever in
   question is monotonic-bad". Pairs with Phase 14's
   sub-additive composition finding.

## Experiment provenance

- **Git SHA:** `2ccaa4d12fae` (uncommitted: phases 1-16).
- **Depth-scaling wall:** 40 s for the 4×5 single-skip-type
  sweep; ~80 s for the residual+highway double sweep.
- **Tests:** all Phase 1-15 suites pass + 6 new Phase 16 tests.

## Acceptance check

- [x] 4-format plan + PDF compiled before code (3 pp).
- [x] No `CORE.YAML` items touched.
- [x] `SignedKANResidualBlock` + `StackedSignedKAN` shipped as a
      thin ResNet-style wrapper over `MultiLayerSignedKAN`.
- [x] 6 new tests pass (forward/backward, depths 1/2/4/8,
      depth-1 equivalence, node_embed back-compat, param-count
      scaling).
- [x] No regressions on any prior Phase 1-15 test.
- [x] Depth axis plumbed into the P-graph fixture + mapping;
      ABB picks `depth_l1` as cost-min.
- [x] **5-seed × L∈{1,2,4,8} depth-scaling A/B run honestly;
      negative result documented.**
- [x] Falsifier-hit acknowledged + candidate fixes enumerated
      for future phases.
- [x] §6.5 anti-pattern audit clean.
- [x] Report on disk.
