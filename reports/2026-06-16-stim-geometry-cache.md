# Stimulus-graph geometry cache (RicciStim topology reuse)

**Date:** 2026-06-16 · **Plan:** `docs/plans/2026-06-16-stim-geometry-cache/`
(tex/pdf/tikz/mmd) · **Persona:** Aiko

## Summary

Training RicciStim rebuilt the entire per-image stimulus graph every epoch
(quadtree → edges → Forman curvature → walk/polygon/triangle enumeration → Hodge
Laplacian) — CPU-bound work that dominated wall time and that a GPU could not
accelerate. But the **topology is a pure function of the image geometry**; only
the edge signs depend on the learned features. This change builds the geometry
once and reuses it across epochs, recomputing only the cheap signs.

- **Builder split.** `StimulusGraphBuilder.forward` is now
  `apply_signs(build_geometry(tree), features)`:
  - `build_geometry(tree) -> StimulusGeometry` — feature-independent: edges,
    Forman curvatures, the walk/polygon/triangle families with their
    edge-index maps (`*_eidx`), incidence matrices, primitive curvatures, Hodge
    Laplacian.
  - `apply_signs(geometry, features) -> StimulusGraph` — feature-dependent and
    cheap: edge signs from `sign(⟨f_u,f_v⟩−θ)`, then per-primitive sign products
    via the stored `*_eidx`.
  - `forward` keeps its exact signature, validation, and SDRF-override behaviour.
- **Backbone cache.** `RicciStimBackbone(cache_geometry=True)` keeps a per-image
  `{content-hash: (tree, geometry)}` dict (an explicit instance attribute — no
  globals, §6.5 #11). Hit → reuse geometry, recompute features + signs; miss →
  build + store. **Bypassed when `use_sdrf=True`** (SDRF rewires topology from
  features, so it is not epoch-invariant). Threaded through `RicciStimDetector`
  and the runner (`--cache`).

## Correctness (the load-bearing part)

A cache that silently drifts is worse than no cache, so correctness was gated
before any speed claim:

- **Bit-exact split** — `apply_signs(build_geometry(t), f)` equals the old
  `forward(t, f)` field-by-field (test `test_apply_signs_matches_forward_bit_exactly`).
- **Cached == uncached** — backbone forward with cache on vs off, same image +
  weights, returns **bit-identical** features; second call hits the cache.
- **Signs still recompute on a hit** — changing the patch-encoder weights changes
  the output even though the geometry is reused (proves features/signs are not
  frozen).
- **SDRF bypass** — with `use_sdrf=True` the cache stays empty.
- **Production-scale parity** — cached vs uncached at canvas-64, n_train=300,
  3 ep, seed 0: **final mAP50_proxy identical to 18 digits**
  (`0.020357142857142855` both); wall **115.0 s → 77.7 s (1.48×)** — and the
  cache benefit grows with epoch count (epoch 1 pays the build in both; epochs
  2…N are cheap), so the full 40-epoch run amortises toward ~2×.

A test-design subtlety surfaced and was fixed in the *test*, not the code:
negating *both* edge endpoints preserves `⟨f_u,f_v⟩`, so it is not a valid
"features changed the signs" probe — replaced with independent random features.

## Files touched

| File | Note |
|---|---|
| `…/vision/stim_graph.py` | `StimulusGeometry` dataclass; `forward` split into `build_geometry` + `apply_signs`; removed one pre-existing dead variable (`lt_ab`) |
| `…/vision/ricci_stim_backbone.py` | `cache_geometry` flag, `_image_key`, cached/uncached forward branch, SDRF guard |
| `…/vision/ricci_stim_detector.py` | `cache_geometry` passthrough |
| `experiments/run_ricci_stim_cluttered_mnist.py` | `--cache` flag; records `cache` in JSONL |
| `hymeko_neuro/tests/test_stim_geometry_cache.py` | **new** — 5 correctness tests |

CORE.YAML: none (`hymeko_neuro` non-core). No new dependency.

## Test results

- `pytest test_stim_geometry_cache.py` — **5 passed** (parity, feature-independence
  of geometry + feature-dependence of signs, cached==uncached, sign-recompute on
  hit, SDRF bypass).
- `pytest test_soma_aggregators.py test_gomb_soma_vision_ricci_stim_detector.py`
  — **all pass** (the split did not regress existing behaviour).
- `ruff check` on the touched files — **clean**.

## Performance / memory

- Per-epoch graph build (the CPU bottleneck) is skipped on cache hits; measured
  1.48× at 3 epochs, asymptotically ~2× at 40.
- Cache memory ≈ 0.1–0.3 MB/image (sparse `M_v` + Hodge + index tensors) ⇒
  ~1–1.5 GB at 5000 images, GPU-resident; within the RTX 3070 8 GB budget and ≪
  the 16 GB RSS cap. If VRAM becomes tight the cache can move to CPU (noted in
  the plan); not needed at the current scale.

## The full run — RESULT (the point of the cache)

The full **"turn vision positive" run** completed on the cache:

- **Config:** RicciStim **upgraded + cached**, Cluttered-MNIST config F,
  n_train=5000, n_eval=1000, **n_epochs=40**, canvas 64, seed 0, device cuda.
- **Disk anchor:** `reports/ricci_stim_detect_full_20260616.jsonl` (full
  per-epoch history); was background task `b73ag0n9n`.
- **Result: final mAP50_proxy = 0.228**, wall 12 434 s (~3.5 h). The curve
  crossed the prior bare-sum config-F headline (~0.174, 5000 img/20 ep,
  2026-05-16) at ~epoch 15 (0.169 → 0.185) and plateaued in the 0.21–0.23 band
  from ~epoch 25 (ep25 0.211 · ep30 0.210 · ep35 0.224 · ep40 0.228) — near its
  ceiling.
- **Verdict: a genuine positive** — the upgraded model at full scale **beats the
  prior hypergraph-vision headline (0.228 > 0.174)**. The user's instinct held:
  the bare-sum configuration was under-aggregated/under-trained, and the strong
  aggregator at scale clears the old number.
- **The cache earned its keep:** 40 epochs in ~3.5 h; without it the per-epoch
  graph rebuild would have made this ~7 h+ (and it is what made 40 epochs
  affordable at all on the laptop GPU).
- **Honest caveats (stated, not buried):** (a) single seed; (b) **not a perfectly
  matched A/B** — this is upgraded/40-ep vs the old bare-sum/20-ep headline; the
  matched 2000-img A/B already showed the upgrade at +27 % over bare sum (so the
  gain is real), but a bare-sum/40-ep run would cleanly split upgrade-vs-epochs;
  (c) `mAP50_proxy` is the per-image F1-at-IoU-0.5 proxy, not COCO mAP, and 0.228
  **beats the prior hypergraph-vision number — it is not parity with a
  conventional detector.**

## Open issues / follow-up

- **Matched bare-sum/40-ep baseline** (5000 img, no upgrade, cached) — the clean
  paired comparison that isolates "the upgrade" from "more epochs." ~3.5 h GPU;
  worth it before any paper claim about the magnitude of the upgrade's effect.
- **Multi-seed** the upgraded run (≥3 seeds) for an error bar before publishing.
- Per-epoch JSONL logging: the runner is silent until completion (a 3.5 h black
  box). Add per-epoch append to the JSONL next time the detection harness is
  touched, so long runs are observable mid-flight.
- Gömb (strict cascade) signed-link numbers remain unrun (local OOM) — Komondor/GCP.
- Eval forward also benefits from the cache; the eval images are cached
  separately by content (distinct from train), as expected.
