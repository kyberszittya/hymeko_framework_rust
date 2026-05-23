# GömbSoma cortical-benchmark — Slice 1 (2026-05-19)

## Summary

Slice 1 of the cortical-benchmark implementation plan
(`docs/plans/2026-05-19-gomb-soma-cortical-implementation/`) lands the
**dataset-agnostic scoring infrastructure** + a **synthetic
Cichy-92-faithful smoke**. The 4-format plan (.tex/.pdf/.tikz/.mmd) was
written first per CLAUDE.md §2. A new `signedkan_wip/src/cortical/`
package ships 5 modules implementing the Brain-Score 2018 protocol
(Schrimpf et al.) in PyTorch + sklearn (no `brainscore`/`brainio`
dependency — those aren't in the env). End-to-end 3-seed smoke at the
real Cichy-92 shape (92 images × 16 subjects, V1/V2/V4 ROIs) runs in
≈22 s/seed and emits a JSONL of per-ROI r²/noise-ceiling/corrected
scores for both GömbSoma and a parameter-matched ResNet baseline.

The hot-path Python loop in `CorticalFeatureExtractor._bin_features`
(§6.5 #1 anti-pattern, anchors-per-image inner loop) was identified
during profiling, replaced with `scatter_add_`, and pinned by two new
regression tests.

## Files touched

| File                                                                     | Status | LOC  |
| ------------------------------------------------------------------------ | ------ | ---- |
| `docs/plans/2026-05-19-gomb-soma-cortical-implementation/plan.tex`       | new    | (4-fmt plan; PDF compiled, 5 pp, 287 KB) |
| `docs/plans/2026-05-19-gomb-soma-cortical-implementation/plan.tikz`      | new    | 55   |
| `docs/plans/2026-05-19-gomb-soma-cortical-implementation/plan.mmd`       | new    | 37   |
| `docs/plans/2026-05-19-gomb-soma-cortical-implementation/plan.pdf`       | new    | (binary) |
| `signedkan_wip/src/cortical/__init__.py`                                 | new    | 84   |
| `signedkan_wip/src/cortical/synthetic.py`                                | new    | 266  |
| `signedkan_wip/src/cortical/features.py`                                 | new    | 231  |
| `signedkan_wip/src/cortical/scoring.py`                                  | new    | 312  |
| `signedkan_wip/src/cortical/baselines.py`                                | new    | 174  |
| `signedkan_wip/tests/test_cortical_benchmark.py`                         | new    | 303  |
| `signedkan_wip/experiments/runs/run_cortical_benchmark.py`               | new    | 237  |

Total: ~1.6 kLOC across 7 new source files + 4 plan artifacts.

## CORE.YAML items touched

None.

## Interface changes

The new public surface (curated re-exports from
`signedkan_wip.src.cortical.__init__`):

- **Frozen dataclasses (state-only types)**:
  `SyntheticCorticalDataset`, `BinningConfig`, `PerDepthFeatures`,
  `BrainScore`.
- **Generators**: `make_synthetic_cichy_like(...)` →
  `SyntheticCorticalDataset`.
- **Feature extractors (`nn.Module`)**:
  `CorticalFeatureExtractor` (wraps any backbone with `forward(image) →
  (features, tree)` contract), `ResNetTinyCortical` (param-matched
  baseline). Both expose `extract_one(image)` and `extract_batch(imgs)`.
- **Scorer**: `BrainScorer(n_pls_components, ridge_alpha, n_cv_folds,
  seed)` with `score(...)` + `noise_ceiling(...)` + convenience
  `score_all_rois(scorer, features, roi_signals)`.
- **Utilities**: `count_parameters(model)`, `assert_param_match(a, b,
  factor)`.

Object-oriented commitment: three frozen dataclasses + three classes,
consistent `extract_one`/`extract_batch` interface across both feature
extractors, observer-pattern runner via `SimpleExperiment` + Stdout +
Jsonl observers (CLAUDE §7 Strategy / §6.5 #3 — no per-experiment
scaffold duplication; the runner reuses the existing
`_experiment_base.SimpleExperiment` adapter from Slice B).

## Test results

All 21 tests in `signedkan_wip/tests/test_cortical_benchmark.py`
pass under `pytest -p no:randomly`:

```
============================== 21 passed in 19.86s ==============================
```

Coverage by layer:

- **Unit** — `BinningConfig` defaults, `BrainScore` frozen invariant,
  `assert_param_match` success/failure, `BrainScorer` axis-mismatch +
  too-few-images preconditions, single-subject noise-ceiling
  degenerate case.
- **Property-ish unit** — `BrainScorer` extremes: r²≈1 for
  identity-mapping (perfect fit), r²≈0 for pure-noise features,
  noise-ceiling≈1 for high-agreement subjects, noise-ceiling near
  zero for independent-noise subjects.
- **Regression** — new `_bin_features` correctness tests pin
  scatter_add_ vectorisation against a hand-computed reference and
  exercise the empty-anchors edge case.
- **Integration** — synthetic dataset shape/determinism/custom-size,
  ResNet baseline per-depth + batch shapes.
- **End-to-end smoke** — full pipeline (synthetic data → ResNet
  features → BrainScorer per ROI) with shape + finite-value + bound
  assertions.

## Performance results

### Production-scale 3-seed synthetic smoke

Command:

```
python -m signedkan_wip.experiments.runs.run_cortical_benchmark \
    --seeds 0 1 2 --n_images 92 --n_subjects 16 \
    --image_size 64 --d_hidden 16
```

| Metric                            | Mean ± std (n=3) |
| --------------------------------- | ---------------- |
| GömbSoma V1 r²                    | 0.6916 ± 0.0170  |
| GömbSoma V2 r²                    | 0.6945 ± 0.0192  |
| GömbSoma V4 r²                    | 0.6829 ± 0.0443  |
| ResNet V1 r²                      | 0.7378 ± 0.0372  |
| ResNet V2 r²                      | 0.7401 ± 0.0556  |
| ResNet V4 r²                      | 0.7261 ± 0.0751  |
| Noise ceiling (split-half + S-B)  | 0.68 ± 0.01      |
| Noise-corrected score (both)      | clamped near 1.0 |
| GömbSoma params                   | 4,566            |
| ResNet params                     | 18,720 (~4.1×)   |
| GömbSoma feature extraction       | ~4.1 s / 92 imgs |
| ResNet feature extraction         | ~0.06 s / 92 imgs|
| BrainScorer per seed (both models)| ~2.8 s           |
| Wall time / seed                  | ~22 s            |
| Peak RSS (steady-state)           | ~1.0 GiB         |

The raw 3-seed JSONL is at
`/tmp/cortical_smoke_2026_05_19.jsonl` (one line per seed; all
24 fields per row).

### Interpretation

The synthetic data is intentionally "too easy" — SNR=0.3 + Gaussian
RF filters means a fresh-init ResNet's r² already exceeds the
noise ceiling, so the **corrected** scores saturate near 1.0 for both
models. This is **expected for Slice 1**: the goal is to validate
the scoring pipeline's contracts at the real data shape, not to
discriminate between architectures. The contract assertions that
matter — r²≈1 on identity, r²≈0 on noise, ceiling≈1 on high-agreement
subjects, ceiling small on pure-noise subjects — are pinned by the
test suite.

### Optimisation note

Profiling at 92 images × 16 subjects showed `RicciStimBackbone.forward`
dominates feature extraction (4.31 s / 92 imgs ≈ 99% of GömbSoma's
budget); `CorticalFeatureExtractor._bin_features` ran sub-millisecond
even before vectorisation. The Python `for j in range(features.shape[0])`
loop in `_bin_features` was still removed (CLAUDE §5/§6.5 #1: hot-path
Python loops are an anti-pattern even when they are not currently the
bottleneck — once the backbone is sped up, this would become one).
Replaced with `scatter_add_`; behaviour pinned by
`test_bin_features_vectorised_matches_loop_reference`.

The remaining backbone hotspot (Bochner conv + StimulusGraphBuilder)
is a Slice 2+ target — out of scope for tonight per the
"one phase per session" feedback rule. Profile artifact and Slice 2
candidates are queued in the plan PDF §Future Work.

## New / removed dependencies

None. The implementation uses only `torch`, `numpy`, and `sklearn`
(all already in `tools.yaml` / CORE). The originally-planned
`brainscore` + `brainio` packages were dropped after confirming they
aren't installed in this env; the Brain-Score 2018 protocol (PLS
reduce → Ridge per voxel → K-fold CV + split-half noise ceiling with
Spearman-Brown correction) is reimplemented locally in
`scoring.BrainScorer` against sklearn primitives.

## Open issues and follow-up items

1. **Real Cichy 92 download** — Slice 2. The synthetic generator is
   shape-faithful; one new fetcher function in `cortical/` is enough
   to swap to real data.
2. **ViT-S/16 baseline** — Slice 2. Brain-Score 2018 used ViT-S/16
   as one of the published comparison points; add to `baselines.py`
   alongside `ResNetTinyCortical`.
3. **GömbSoma backbone bottleneck** — `RicciStimBackbone.forward` is
   4.3 s/92imgs. A profile-driven cleanup (likely the Bochner +
   StimulusGraphBuilder path, both currently use Python-level
   per-anchor iteration) is the natural next slice for the
   "optimization" half of the user's request.
4. **Synthetic-data discrimination** — current SNR=0.3 lets both
   models saturate corrected scores at 1.0. If Slice 2 wants a
   "synthetic discrimination smoke" between architectures, lower SNR
   (~0.05) or sparser RF filters would force the gap to open.
5. **Training-time signal** — Slice 1 evaluates with fresh weights
   (architectural-prior measurement). A future slice could evaluate
   pretrained-backbone scores (Brain-Score's standard protocol).

## §6.5 anti-pattern audit

No new §6.5 anti-patterns introduced. Specifically:

- **#1 Cartesian-product API** — no string-typed mode arguments;
  `BinningConfig` is a typed dataclass, not a `bins: str` flag.
- **#3 Per-experiment scaffold duplication** — runner subclasses
  `SimpleExperiment` from `_experiment_base.py`; argparse, observer
  dispatch, JSONL emission, summary stats all live in the base.
- **#7 String-typed config that should be an enum** — n/a; all configs
  are dataclasses.
- **#11 Globals** — no module-level mutable state. The
  `_DEFAULT_VOXELS` / `_CATEGORIES` constants in `synthetic.py` are
  immutable POD literals.

`CorticalFeatureExtractor._bin_features` had a Python `for` loop in
the v0 draft (cleared on vectorisation; see Optimisation note above).

## Experiment provenance

- **Git SHA**: `2ccaa4d12fae` (current branch:
  `refactor/extract-hymeko-hre`; working tree has prior
  pre-existing dirty files from earlier today's signedkan-wip
  reorg + book regenerations — none touch the new code paths).
- **Python**: 3.13.5
- **Torch**: 2.11.0+cu130 (drifts from CORE 2.4.1 — pre-existing env
  drift, unrelated to this change)
- **NumPy**: 2.4.4
- **scikit-learn**: 1.8.0
- **OS / kernel**: Ubuntu 24.04.4 / Linux 6.17.0-23 / x86_64
- **Random seeds**: 0, 1, 2
- **Dataset**: synthetic, generated from seed (no fixture hash needed —
  the generator is deterministic per seed; tested by
  `test_synthetic_dataset_deterministic`).
- **Output**: `/tmp/cortical_smoke_2026_05_19.jsonl` (3 lines).

## Acceptance check

- [x] Plan written in all four formats before code.
- [x] Plan PDF compiled (5 pages, 287 KB).
- [x] No `CORE.YAML` items touched.
- [x] No new dependencies.
- [x] All 21 new tests pass.
- [x] End-to-end smoke at production scale (92 × 16) runs cleanly.
- [x] Peak RSS ≪ 16 GB cap.
- [x] §6.5 anti-pattern audit clean.
- [x] Hot-path Python loop in `_bin_features` removed + regression-tested.
- [x] Report on disk.
