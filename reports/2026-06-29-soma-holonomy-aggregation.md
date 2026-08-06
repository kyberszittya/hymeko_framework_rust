# Sign-as-Connection (Holonomy) Aggregation for Gömb-Soma Walk-Conv

**Date:** 2026-06-29
**Plan:** [docs/plans/2026-06-29-soma-holonomy-aggregation/](../docs/plans/2026-06-29-soma-holonomy-aggregation/) (tex/pdf/tikz/mmd)
**Author:** Aiko (Claude Code) for Dr. Csaba Hajdu

## Summary

The Gömb-Soma walk-vision hypothesis was falsified on MNIST (2026-06-15):
base-Soma walk-conv 0.5186 ± 0.0204 vs. a linear control 0.9056 ± 0.0079. That
base-Soma encodes the walk sign **as routing** — dual weight banks `W⁺/W⁻`
selected by walk sign, then a sign-blind sum-pool `M_v @ messages`. The recent
StructuralActor / gauge-holonomy result reframes sign on a signed graph as a
**connection**: the walk's σ-product is a Z₂ holonomy that *multiplies* the
transported message, `M_v @ (σ⊙m)`, reproducing the signed L-hop operator a
sign-blind sum cannot. That operator had never been run on vision; the
2026-06-15 ablation tested the wrong hypothesis. This change adds a `HOLONOMY`
aggregation mode to the Gömb-Soma `HypergraphConv` and re-runs the *exact* MNIST
falsification harness as a single-machine A/B.

## Result (MNIST, 5000/1000, 5 epochs, Adam 3e-3, CPU)

| arm | sign handling | params | test acc (5-seed mean ± pstd) |
|---|---|---:|---|
| Linear control | — | 7850 | **0.9056 ± 0.0079** |
| base-Soma (`gomb_soma`) | routing (dual bank + sign-blind sum) | 2010 | 0.5186 ± 0.0204 |
| Soma-holonomy (`gomb_soma_holonomy`) | connection (single bank + σ⊙m pool) | 1226 | 0.4888 ± 0.0093 |

Figure: `reports/figures/soma_holonomy_ab_20260629.png`.

**Provenance check (both anchors reproduced):** linear → 0.9056 ± 0.0079 and
routing base-Soma → 0.5186 ± 0.0204 *both* reproduce the 2026-06-15 numbers to
four decimals on this machine/torch build, so the holonomy arm is measured
against a verified baseline, not a drifted one.

**Verdict — falsification CONFIRMED, and tightened.** Sign-as-connection
(holonomy, the signed Bᴸ operator) does **not** revive walk-vision on MNIST: at
0.4888 it is marginally *below* the routing base-Soma (0.5186; the routing IQR
nearly reaches it), and both are crushed by a linear classifier (0.9056). The
2026-06-15 ablation tested the wrong operator, but testing the *right* one
returns the same answer — the signed walk-holonomy prior is not load-bearing for
MNIST patch-graph classification. Holonomy was a well-motivated bet (it is the
operator that carries StructuralActor on control tasks); on vision it lands
negative. The walk-vision line stays parked; the holonomy operator's value
remains on the control/RL side, not vision. **No escalation to RicciStim /
`.hymeko` round-trip is warranted by this result.**

Distinguishing measured / inferred / hypothesis (CLAUDE.md operating principle):
*measured* — the three 5-seed means above; *inferred* — the structural prior is
not load-bearing here (from holonomy ≤ routing ≪ linear); *still open* — whether
a deeper RicciStim stack (polygons/triangles/Hodge) with holonomy pooling would
differ; not tested, and not prioritised given this null.

## Files touched

| file | change | ± |
|---|---|---|
| `hymeko_neuro/models/hymeko_gomb/soma/hg_conv.py` | `Aggregation` enum; `aggregation` config field; signs into `_aggregate`; holonomy branch; removed 2 pre-existing dead lines (`allowed`, `Optional`) | +35 / −4 |
| `hymeko_neuro/models/hymeko_gomb/soma/hg_conv_bochner.py` | `_aggregate` override matches new signature (delegates inward) | +3 / −2 |
| `hymeko_neuro/models/hymeko_gomb/soma/vision/walk_conv_classifier.py` | thread `aggregation` arg into the layer config | +16 / −3 |
| `hymeko_neuro/models/hymeko_gomb/soma/vision/train_mnist.py` | `gomb_soma_holonomy` arm | +14 / −1 |
| `hymeko_neuro/experiments/runs/soma_holonomy_ab_plot.py` | new: A/B accuracy bar (pure `summarize` + `render`) | +135 |
| `hymeko_neuro/tests/test_gomb_soma_holonomy_aggregation.py` | new: 9 aggregation/contract/integration tests | +180 |
| `hymeko_neuro/tests/test_soma_holonomy_ab_plot.py` | new: 4 plot-shaper tests | +47 |

## CORE.YAML items touched

None. All edits are in `hymeko_neuro/` (non-core, editable). No pinned
dependency, grammar, or core type touched.

## Test results

- **New holonomy tests** (`test_gomb_soma_holonomy_aggregation.py`): 9 passed —
  holonomy = manual σ⊙m pool (exact); holonomy ≠ sum on mixed signs
  (regression vs prior single-mode impl); single-primitive sign-flip negates the
  pooled contribution; sparse invariant preserved; default aggregation is SUM;
  classifier forward shape; single-bank < dual-bank params; trainable; Bochner
  delegates holonomy.
- **Plot tests** (`test_soma_holonomy_ab_plot.py`): 4 passed.
- **Regression** (signature change): `test_gomb_soma_hg_conv` (12),
  `test_gomb_soma_bochner_conv`, `test_gomb_soma_polygon_layer`,
  `test_gomb_soma_walk_layer`, `test_gomb_soma_vision_ricci_stim_backbone`,
  `test_gomb_soma_vision_ricci_stim_classifier`,
  `test_gomb_soma_vision_walk_conv_classifier`, `test_stim_geometry_cache` —
  62 passed, no regression.
- **Static analysis:** `ruff check` clean on all changed files; `radon cc` — only
  pre-existing `_check_preconditions` at C (12, < 15 fail threshold), behaviorally
  untouched; the new `_aggregate` holonomy branch is below the report threshold.

## Performance

- Production-scale smoke (§3): holonomy arm, 1 seed / 1 epoch / full 5000-1000
  split → 51.8 s/epoch, 1226 params. Full-run wall ≈ 45 min (3 arms × 5 seeds ×
  5 ep), within the plan ETA; within 2× of the closest prior baseline (§11 OK).
- Peak RSS: a 1.2k-param model on a 5k-image MNIST subset, CPU — well under the
  16 GB cap (< 1 GB).

## §6.5 anti-patterns

None introduced. Sign-handling axes live in a config enum + Strategy dispatch
(no new function-per-variant; §6.5 #1/#7). Plot shaper is a distinct schema, not
a re-implementation of `bench_to_png` (§6.1).

## Graphical output (§9)

Numerical (table above + JSONL), plotted (bar chart). No GIF: this is a static
classification A/B with no spatial/temporal/control character — the §9 animation
clause does not apply.

## Experiment provenance

- Git SHA: 9aea4f6 (working tree dirty — this feature diff; see Files touched).
- JSONL: `reports/soma_holonomy_vs_routing_mnist_20260629.jsonl` (15 per-seed
  records; routing arm re-run after a backgrounded `for`-loop dropped its first
  pass — the arm runs cleanly when invoked directly, a shell-pipe artifact, not
  a code fault).
- Seeds: 0–4 per arm. Optimiser Adam, lr 3e-3, batch 64, 5 epochs.
- Dataset: MNIST, 5000 train / 1000 test sub-sampled by seed (same as 2026-06-15).
- Env: torch 2.12.0 (cu132), CPU run, Windows 11; `.venv` interpreter.

## Open issues / follow-up

- **Resolved by this run:** holonomy does not move the number → no 2×2 factorial,
  no RicciStim holonomy wiring, no `soma_vision.hymeko` round-trip pursued. The
  `HOLONOMY` mode stays in the codebase (tested, costs nothing, available if a
  later signed-vision task wants it) but is not the default and not promoted.
- The `HOLONOMY` aggregation primitive is now reusable by any GömbSoma layer
  (Walk/Polygon/Triangle/Bochner) via `HypergraphConvConfig(aggregation=…)` —
  the operator is preserved even though the vision bet came back negative.
- Holonomy remains load-bearing on the control/RL side (StructuralActor); this
  run only closes the *vision* question, consistent with the broader Soma-vision
  falsification (`reports/2026-05-28-vision-hypergraph-vs-cnn-rebench.md`).
