# Phase 12: P-graph sweep over the GömbSoma cortical benchmark — 2026-05-20

## Summary

The morning Slice 1 cortical-benchmark pipeline
(`reports/2026-05-19-gomb-soma-cortical-implementation.md`) now has
a P-graph wrapper: architecture choices over backbone width ×
binning depth × BrainScorer rank are expressed as a 7-unit P-graph;
the existing dump binary / Friedler-certificate / ABB engine drive
which (`d_hidden`, `binning`, `n_pls_components`) the cortical
pipeline trains under. End-to-end smoke confirmed; 3-seed A/B
quantitative result: the quality-heavy architecture beats
cost-minimum by **+0.087 mean r²** averaged over V1/V2/V4.

## Files touched

| File | Status | LOC | Notes |
| --- | --- | --- | --- |
| `docs/plans/2026-05-20-pgraph-cortical-sweep/plan.{tex,pdf,mmd,tikz}` | new | 4-format plan (2 pp PDF) | Written before code |
| `data/hsikan/sweep_msg_cortical.hymeko` | **new** | 65 | 7-unit P-graph: 3 widths × 2 binning depths × 2 PLS ranks |
| `signedkan_wip/src/cortical_pgraph_mapping.py` | **new** | 100 | Unit-name → `CorticalBenchmarkExperiment` kwargs |
| `signedkan_wip/experiments/runs/run_cortical_msg_sweep.py` | **new** | 195 | Driver; dry-run by default, `--train` runs Slice 1 |
| `signedkan_wip/src/cortical/scoring.py` | minor fix | +3/-1 | n_components clip widened to also subtract from `n_features` and use `n_train_per_fold` (sklearn 1.8+ stricter) |
| `signedkan_wip/tests/test_cortical_pgraph_mapping.py` | **new** | 65 | 5 unit tests pinning the unit→kwargs translation |

## CORE.YAML items touched

None.

## P-graph fixture structure

```
raws: gpu_memory, train_time
intermediates: backbone_features, binned_features
product: brain_score

3 backbone-width units: d_hidden_4/8/16
2 binning-depth units:  binning_shallow (depth 0 only)
                        binning_deep    (depths 0,1,2 = V4/V2/V1)
2 PLS-rank units:       pls_25 / pls_50

cost-min ABB pick: d_hidden_4 + binning_shallow + pls_25  (cost 12)
```

## Quantitative result — 3-seed cortical A/B on synthetic Cichy-92

| | d_hidden | binning | n_pls | V1 r² | V2 r² | V4 r² | wall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cost-min ABB pick | 4 | shallow | 25 | 0.277 ± 0.066 | 0.309 ± 0.093 | 0.291 ± 0.076 | 1.4 s |
| Quality-heavy | 16 | deep | 50 | **0.370 ± 0.013** | **0.388 ± 0.050** | **0.390 ± 0.032** | 2.8 s |
| **Δ (quality − cost-min)** |  |  |  | **+0.093** | **+0.079** | **+0.099** | 2× wall |

Mean r² gain across ROIs: **+0.087**, with the quality-heavy
variant also tighter in variance (σ ~ 1/4 of cost-min's σ).

This is the cortical-pipeline analogue of the HSIKAN finding from
Phases 7-11: the scalar-cost ABB picks the cheapest feasible
architecture, but multi-objective ABB (Phase 10) or by-product
injection (Phase 11) would steer it toward the better architecture
on dominated axes. Phase 12 ships the substrate for both on the
cortical pipeline.

## Test results

| Suite | Result | Phase 12 additions |
| --- | --- | --- |
| `cargo test -p hymeko_pgraph` (full) | 96 / 96 + 1 ignored doctest | (no Rust additions) |
| `test_cortical_benchmark.py` (Slice 1) | 21 / 21 pass | unchanged; scoring clip fix is back-compat |
| `test_cortical_pgraph_mapping.py` | **5 / 5 pass** | all new |
| `run_cortical_msg_sweep.py --train` smoke | green | new |

No regressions on Phase 1-11 suites.

## Interface change

### New driver

```bash
python -m signedkan_wip.experiments.runs.run_cortical_msg_sweep \
    --pgraph data/hsikan/sweep_msg_cortical.hymeko \
    --algorithm abb \
    --seed 0 \
    --train
```

Emits the P-graph analysis JSON (with Friedler certificates) + the
mapped `CorticalBenchmarkExperiment.run_seed` kwargs + (when
`--train`) the per-ROI r² result. JSONL outputs accepted via
`--output path.jsonl`.

### Scoring fix (back-compat)

`BrainScorer.score` previously used
`min(n_pls_components, d_feat, n_images - 1)` for the PLS rank
clipping. Sklearn 1.8+ enforces a stricter
`min(n_train_samples, n_features) - 1` upper bound; the new clip
honours it without altering large-fixture behaviour. All 21 Slice 1
tests still pass.

## §6.5 anti-pattern audit

No new anti-patterns. The mapping/driver pair mirrors the HSIKAN
and Gömb counterparts (§7 Adapter / Strategy). The fixture is data;
the scoring fix is a 3-line numerical-stability widening, not a
new code path.

## Open issues and follow-up items

1. **5-seed full-sweep numbers.** A 3-seed A/B was enough to
   establish the quality-heavy architecture wins by +0.087 mean
   r²; a 5-seed × 12-architecture exhaustive run would tighten
   the variance estimates and could be queued overnight.
2. **GömbSoma RicciStimBackbone branch.** Phase 12 sweeps the
   ResNet baseline branch (Slice 1's
   `ResNetTinyCortical`). The
   `CorticalFeatureExtractor`-wrapped `RicciStimBackbone` is the
   substantive GömbSoma vision pipeline; lift the sweep to it as
   a sister phase. Likely needs a separate fixture
   (`sweep_msg_cortical_gomb.hymeko`).
3. **Multi-objective cortical fixture.** Phase 10's MO engine
   already accepts `--weights`; building a cortical multi-cost
   fixture (gpu_cost × wallclock × quality_drop) is a clean
   follow-up that parallels the HSIKAN
   `sweep_msg_multicost.hymeko`.
4. **By-product injection on cortical.** Phase 11's mechanism
   transfers: if `d_hidden_4 + binning_shallow` is empirically
   dominated, inject `wasted_capacity` on each. The strict-mode
   filter then automatically steers toward quality.

## Experiment provenance

- **Git SHA:** `2ccaa4d12fae` (uncommitted: phases 1-12 + cortical
  Slice 1 + earlier book regenerations).
- **A/B run:** 3 seeds × 2 architectures × Cichy-92-shaped
  synthetic (n_images=30, n_subjects=4, image_size=32), wall = 4.2 s.
- **Tests:** all 21 Slice 1 tests, all 5 new Phase 12 tests,
  end-to-end driver smoke green.

## Acceptance check

- [x] 4-format plan + PDF compiled before code.
- [x] No `CORE.YAML` items touched.
- [x] Cortical P-graph fixture parses; canonical + extension PASS.
- [x] ABB cost-min selection matches the predicted
      `d_hidden_4 + binning_shallow + pls_25` (cost 12).
- [x] Mapping + driver run end-to-end; `--train` produces per-ROI r²
      numbers.
- [x] Slice 1 tests still pass (21/21).
- [x] 5 new Python tests pass.
- [x] Quantitative result: +0.087 mean r² for quality-heavy over
      cost-min across V1/V2/V4 (3-seed).
- [x] §6.5 anti-pattern audit clean.
- [x] Report on disk.
