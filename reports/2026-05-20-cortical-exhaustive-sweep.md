# Cortical exhaustive sweep — 2026-05-20 overnight queue

## Summary

The 5-seed × 12-architecture × 2-backbone exhaustive sweep queued at
the end of the overnight rollup. **120 training cells** in 164.8 s
wall (1.37 s/cell avg). Confirms and extends the Phase 12.5 3-seed
preview, with one substantive new finding: **there's a regime
crossover.** The hypergraph backbone wins at small width (d=4); the
CNN catches up at d≥8 and edges ahead at the maximum architecture.

## Files touched

| File | Status |
| --- | --- |
| `signedkan_wip/experiments/runs/run_cortical_exhaustive_sweep.py` | **new** (180 LOC) |
| `reports/cortical_exhaustive_2026_05_20.jsonl` | **new** (120 rows) |
| `reports/cortical_exhaustive_2026_05_20.summary.json` | **new** |

## Aggregate 5-seed results per architecture

Identical-r²-by-pls-rank rows are merged (PLS rank doesn't
differentiate at this fixture scale — see Finding 5 below).

### ResNet (CNN baseline)

| arch | scalar cost | mean r² (V1/V2/V4) | mean σ |
| --- | --- | --- | --- |
| d4 + shallow | 12-17 | 0.284 | **0.127** |
| d4 + deep | 18-23 | 0.235 | **0.110** |
| d8 + shallow | 16-21 | 0.348 | 0.052 |
| d8 + deep | 22-27 | 0.357 | 0.033 |
| d16 + shallow | 24-29 | 0.330 | 0.033 |
| **d16 + deep** | **30-35** | **0.397** | 0.034 |

### GömbSoma (hypergraph machine)

| arch | scalar cost | mean r² (V1/V2/V4) | mean σ |
| --- | --- | --- | --- |
| **gomb_d4 + shallow** | **12-17** | **0.367** | 0.038 |
| gomb_d4 + deep | 18-23 | 0.349 | 0.033 |
| gomb_d8 + shallow | 16-21 | 0.322 | 0.035 |
| gomb_d8 + deep | 22-27 | 0.359 | 0.039 |
| gomb_d16 + shallow | 24-29 | 0.325 | 0.036 |
| gomb_d16 + deep | 30-35 | 0.387 | 0.031 |

## Five substantive findings

### 1. Cost-min: hypergraph **wins by +0.083 r²** (3-seed preview underestimated)

| | Phase 12.5 (3-seed) | Phase 14 (5-seed) |
| --- | --- | --- |
| ResNet cost-min mean r² | 0.292 | 0.284 |
| **Gomb cost-min mean r²** | **0.347** | **0.367** |
| **Δ in favour of hypergraph** | **+0.055** | **+0.083** |

The 5-seed reproduces the cost-min hypergraph win and tightens the
magnitude. **At the architecture the P-graph framework picks under
scalar-cost ABB**, the hypergraph machine outperforms the CNN by
0.08 r² across V1/V2/V4 — a substantively meaningful margin on a
Brain-Score-protocol benchmark.

### 2. Regime crossover at d=8: ResNet catches up

ResNet r² jumps from 0.235-0.284 at d=4 to 0.348-0.357 at d=8
(+0.07 to +0.10) — the CNN was under-fitting at width 4. Gomb is
*flat* across widths (0.32-0.37 at all widths) — the hypergraph
machine isn't capacity-starved at the smallest size.

This means the comparison's headline depends on which point of
the Pareto frontier you ask:

- **At cost-min**: hypergraph clearly better.
- **At maximum architecture (d16 + deep + pls50)**: ResNet 0.397
  vs Gomb 0.387 — **ResNet edges by +0.010**, within noise.

### 3. **Variance: hypergraph dominates uniformly**

ResNet σ ranges **0.033–0.127**; Gomb σ ranges **0.031–0.039**.
**Gomb σ is uniformly ≈ 1/3 of ResNet's worst-case σ** and matches
ResNet's best-case σ.

The CNN's σ blow-up at d=4 (σ ≈ 0.11-0.13) means **the CNN is
unstable at small architectures** while the hypergraph machine is
stable. This is the strongest case for hypergraph as a CV
substrate where reproducibility matters more than peak accuracy.

### 4. Best per-backbone architecture is structurally the same

Both backbones' best architecture is `d=16 + deep + pls=25 (or 50)`
— width 16 + the full V1/V2/V4 retinotopic mapping. The PLS rank
doesn't differentiate. **The P-graph framework would converge to
the same best architecture for both backbones** if cost-quality
weighting is applied (Phase 10 mechanism). The only divergence is
at the cost-min end of the Pareto frontier.

### 5. PLS rank is a **structural no-op** at this fixture scale

Every (`pls_25`, `pls_50`) pair returns byte-identical mean r²
because the `BrainScorer.score` clip
`min(n_pls_components, d_feat - 1, n_train_per_fold - 1)` saturates
at the smaller dimension. For `d_hidden=4, shallow binning,
n_images=30, n_cv_folds=4`, the cap is `min(25, 15, 21) = 15` =
`min(50, 15, 21) = 15`. Same n_components → same PLS regression
→ same r².

**Action item:** the PLS axis should either (a) scale `n_images` to
make pls_50 reach a higher rank, or (b) be removed from the
search space because it's effectively non-functional at the
current fixture size. The fixture's scalar-cost design treats
`pls_50` as 2× expensive but it produces no improvement; under a
multi-objective ABB with quality_drop encoded, `pls_50` is
strictly dominated.

## Pareto frontier (mean r² vs scalar cost)

```
Cost  ResNet (CNN)             Gomb (hypergraph)
 12   0.284 ± 0.127            0.367 ± 0.038   ← Pareto-front start
 16   0.348 ± 0.052            0.322 ± 0.035
 18   0.235 ± 0.110            0.349 ± 0.033
 22   0.357 ± 0.033            0.359 ± 0.039
 24   0.330 ± 0.033            0.325 ± 0.036
 30   0.397 ± 0.034            0.387 ± 0.031   ← top performers
```

ResNet's Pareto-front:  cost 12 → 22 → 30 (climbs slowly with cost).
Gomb's Pareto-front:    cost 12 (already best at d=4 + shallow);
                        d=16+deep gives a smaller +0.02 gain.

The **hypergraph backbone Pareto-dominates the CNN backbone** at
costs 12 and 18 (both axes of the small-width regime). At costs ≥
22 the two are within noise, with the CNN edging at cost 30 by
+0.010.

## Quantitative wrap-up

| metric | ResNet | Gomb (hypergraph) |
| --- | --- | --- |
| best mean r² (V1/V2/V4 avg) | 0.397 | 0.387 |
| best σ | 0.033 | 0.031 |
| worst σ | **0.127** | 0.039 |
| cost-min mean r² | 0.284 | **0.367 (+0.083)** |
| cost-min σ | **0.127** | **0.038** |
| params (typ.) | 18,720 | ~4,566 (¼ of CNN) |

## Test results

The new driver (`run_cortical_exhaustive_sweep.py`) ran end-to-end
without errors over 120 cells. No regressions on any prior test
suite. All Phase 1-13 suites still green.

## §6.5 anti-pattern audit

No new anti-patterns. The new driver script is a straightforward
nested loop over backbones × architectures × seeds; it reuses
`merge_structure_knobs`, `benchmark_kwargs`, and
`_run_one_benchmark_seed` from the Phase 12 driver. The
`_arch_cost` helper mirrors the `.hymeko` fixture scalar costs;
it's a 12-line lookup table, not a Cartesian function family.

## Open items for the next queue

1. **Phase 14 — stack MO + by-product on the same fixture.** Still
   pending. The cortical exhaustive sweep + the HSIKAN Phase 11
   result both motivate it. Plan: a fixture where the cortical
   `binning_shallow` axis emits a `wasted_capacity` by-product
   (because shallow under-uses the deep binning's V1/V2/V4
   structure when the backbone has capacity), and the MO weights
   are over (cost, sigma, quality_drop). Single sweep → Pareto
   frontier.
2. **Remove or rescale the PLS axis** in the cortical fixture. At
   current `n_images = 30` it's a structural no-op (Finding 5).
3. **Document the regime crossover** as Phase 12.5's natural
   refinement: "hypergraph at small widths, CNN at large widths,
   variance always favours hypergraph". This is a citable claim
   on the synthetic cortical benchmark.

## Experiment provenance

- **Git SHA:** `2ccaa4d12fae` (still uncommitted).
- **Hardware:** miniconda3 Python 3.13.5 / torch 2.11.0+cu130 /
  Ubuntu 24.04.4 / CPU (no GPU on this overnight).
- **Sweep wall:** 164.8 s total / 120 cells / 1.37 s/cell avg.
- **JSONL:** `reports/cortical_exhaustive_2026_05_20.jsonl` (120
  rows).
- **Summary:** `reports/cortical_exhaustive_2026_05_20.summary.json`.
- **Reproducible:** `python -m
  signedkan_wip.experiments.runs.run_cortical_exhaustive_sweep`.

## Acceptance check

- [x] 120 cells run cleanly; no errors in any cell.
- [x] Headline 5-seed numbers reproduce + tighten Phase 12.5's
      3-seed preview.
- [x] **New finding:** regime crossover at d=8; ResNet catches up
      at large architectures; hypergraph dominates variance and
      cost-min.
- [x] JSONL output written for downstream analysis.
- [x] Pareto-frontier comparison documented.
- [x] PLS-rank-is-no-op finding surfaced (Finding 5) — actionable
      follow-up.
- [x] §6.5 anti-pattern audit clean.
- [x] Report on disk.
