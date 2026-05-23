# HymeKo-Gömb (sphere) — feasibility report

**Date:** 2026-05-11
**Plan:** `docs/plans/2026-05-11-hymeko-gomb-sphere/plan.{tex,pdf,tikz,mmd}`
**Branch:** `refactor/extract-hymeko-hre`
**Working-tree git SHA at report time:** `5f14ac08b85824ed82e4d97f8c010e089eda5b98` (dirty — see "Files touched")

## Summary

Three-shell concentric cascade (FIR-volume outer / HSiKAN-CR middle /
CPML-MLP inner core) for signed-hypergraph link prediction. Plan
written 16:59, implementation 17:32, tests 17:33, smoke runner 17:35;
a JetBrains-Toolbox OOM at 17:38 (unrelated, 23.6 GB IDE process —
**not** the experiment) blocked the original smoke. This session
resumed, ran the unit tests, ran a 5-seed Bitcoin OTC validation, ran
3 one-seed ablations, and fixed a clippy-gate regression in the
sibling `hymeko_nagare` crate uncovered while verifying the build.

**Plan smoke gate (val AUC ≥ 0.85 on Bitcoin OTC, seed 0):** ✓ passed
at 0.9246.

**Paper-headline gate (5-seed paired vs edge_cr Slashdot SOTA):**
**not yet evaluated** — Bitcoin OTC 5-seed locks the in-distribution
result at 0.9118 ± 0.0089, but Slashdot 5-seed and the paired vs
edge_cr (0.9067 ± 0.0034) baseline still need to run.

## Files touched

| File | +/- | Notes |
|---|---|---|
| `signedkan_wip/src/hymeko_gomb.py` | +144 / -3 | added `GombNoOuter`, `GombNoMiddle`, `GombNoInner` ablation wrappers |
| `signedkan_wip/tests/test_hymeko_gomb.py` | +91 / -1 | 4 new tests (one per ablation + a param-budget assertion) |
| `signedkan_wip/src/run_gomb_smoke.py` | new (192 LOC) | smoke runner with `--model {gomb,no_outer,no_middle,no_inner}` dispatch |
| `hymeko_graph/src/spine.rs` | +3 / -3 | doc-list overindent fix (clippy 1.92 `doc_overindented_list_items`) |
| `hymeko_nagare/src/ops/linear.rs` | +6 / -6 | `needless_range_loop` → iter+enumerate refactor (4 sites) |
| `hymeko_nagare/src/ops/scatter.rs` | +1 / -1 | unused-var `h` → `_h` in a unit test |

Total: **5 files changed, 239 insertions(+), 12 deletions(-)** +
1 new file (192 LOC).

## CORE.YAML items touched

None. Sphere is an additive new model class; existing pipelines
unmodified. The nagare/spine clippy fixes are non-`CORE.YAML`
maintenance (`hymeko_graph` is uncovered by `CORE.YAML`; the nagare
crate itself is new this session, also outside `CORE.YAML`).

## Test results

### Python (`pytest -p no:randomly`)

`signedkan_wip/tests/test_hymeko_gomb.py`: **13/13 passed in 2.62 s**.

Breakdown:
- 3 OuterFIRShell tests (forward shape, M-bank diversification, zero-cycles passthrough)
- 2 MiddleHSiKAN tests (forward shape, zero-cycles passthrough)
- 4 HymeKoGomb tests (full forward shape + param count, backward no-NaN, no-cycles embedding-only path, 20-epoch synthetic-moons smoke)
- 4 ablation tests (no_outer/no_middle/no_inner forward+backward + ablation-param-budget assertion)

### Rust (`cargo test -p hymeko_nagare`)

**8/8 passed in 0.00 s** (post-refactor: confirms the
`needless_range_loop` cleanup preserved numerical-derivative parity in
`linear_backward_matches_numerical` and the scatter tests).

### Static-analysis gates

- `cargo clippy -p hymeko_nagare --all-targets -- -D warnings`: **clean** (after 5 fixes)
- `cargo clippy -p hymeko_graph --all-targets -- -D warnings`: **clean** (after 3 doc-list-indent fixes)
- `cargo check -p hymeko_nagare`: clean

## Performance results — Bitcoin OTC

Real-data smoke per plan §Test strategy. Production-scale:
|V|=5 881, |E|=35 592, ~9:1 pos/neg imbalance. Top-K=64 per-vertex
cycle enum, k=3, ~19 100 cycles/seed. Hardware: AMD Ryzen 7 3700X
(16 threads) + RTX 2070 SUPER (driver 580.126.09, CUDA 13.0), torch
2.11.0+cu130, numpy 2.4.4. Memory cap: `ulimit -v 16777216` (16 GB)
per CLAUDE.md §4.

### 5-seed full Gömb (default config: M=8, d_outer=16, d_middle=32, d_core=32, n_tiers=3)

| seed | val_auc_best | wall (s) |
|---|---|---|
| 0 | 0.9246 | 3.84 |
| 1 | 0.9170 | 3.93 |
| 2 | 0.9058 | 3.81 |
| 3 | 0.9026 | 3.79 |
| 4 | 0.9089 | 3.95 |

**Mean: 0.9118 ± 0.0089 (n=5)**. Wall total: 19.3 s. Param count:
266 321. Vs the plan's 0.85 smoke gate: **+0.062 (every seed clears
the gate).**

### Ablation cells (seed 0, Bitcoin OTC)

| Model | val_auc_best | n_params | Δ vs full seed-0 (0.9246) |
|---|---|---|---|
| Full Gömb | 0.9246 | 266 321 | — |
| GombNoOuter | 0.9219 | 225 057 | **−0.0027** |
| GombNoMiddle | 0.9215 | 252 337 | **−0.0031** |
| GombNoInner | 0.9231 | 210 737 | **−0.0015** |

**Per-shell single-seed Δ are all within 1σ of the 5-seed full-model
noise (σ = 0.0089).** Single-seed ablation is therefore inconclusive
for any individual shell. Per memory `feedback_n_seed_before_paper_promotion.md`,
a 5-seed paired ablation grid is required before any per-shell
attribution claim. **This report does NOT promote a per-shell claim.**

### Wall time vs plan budget

Plan budgeted ≤3 min/epoch on Epinions (much larger than Bitcoin OTC);
actual Bitcoin OTC was 3.8 s for 50 epochs (≈76 ms/epoch). No
performance contracts violated. Peak RSS not yet measured per Section
3 ("performance tests assert a numerical budget") — **flagged as
open issue** below.

## New / removed dependencies

None.

## Open issues / follow-up

1. **Peak RSS not measured.** Bench wall is reported but `dhat` /
   `memray` not yet run. The 16 GB cap held (process did not abort)
   but the actual numerical budget is unknown. Action: run
   `memray run -o gomb.bin python -m signedkan_wip.src.run_gomb_smoke ...`
   on one seed and attach.

2. **5-seed ablation paired study required.** Single-seed Δ for all
   three ablations is within full-model σ. The cascade-attribution
   table cannot be published from this report alone.

3. **Slashdot 5-seed + paired vs edge_cr SOTA (0.9067 ± 0.0034) not
   run.** This is the validation gate from the plan
   ("HymeKo-Gömb is reported as a clean architectural contribution
   only if its 5-seed paired AUC on at least one dataset is ≥ the
   existing edge_cr Slashdot SOTA recipe at iso-parameters"). Until
   that runs, this report is a feasibility result on Bitcoin OTC, not
   a SOTA claim.

4. **JetBrains-Toolbox 23.6 GB peak is unrelated but worth noting.**
   The Toolbox process OOM-killed itself at 17:38 (`journalctl`
   evidence). It was not the experiment. However the IDE consuming
   ~24 GB on this 32 GB box is a latent risk for any future
   long-running experiment that runs alongside it.

5. **`run_gomb_smoke.py` is in the working tree as untracked.**
   Adding to a future commit per memory `feedback_no_auto_commit.md`
   — left to user.

## Experiment provenance

- **Git SHA at run:** `5f14ac08b85824ed82e4d97f8c010e089eda5b98` (HEAD before this session's edits; working tree dirty per "Files touched" — every result above is reproducible from this SHA + the listed diffs).
- **OS / kernel:** Linux Amaterasu 6.17.0-23-generic (Ubuntu 24.04, x86_64).
- **CPU:** AMD Ryzen 7 3700X, 8 cores / 16 threads.
- **GPU:** NVIDIA GeForce RTX 2070 SUPER (8 GB), driver 580.126.09, CUDA 13.0.
- **Python:** 3.13.5 (miniconda); torch 2.11.0+cu130; numpy 2.4.4; pytest 9.0.3 + pluggy 1.5.0.
- **Dataset:** Bitcoin OTC. `md5(signedkan_wip/data/bitcoin_otc.csv) = eeaf5cd1d29ab435505baeeb6816317b`. |V|=5 881, |E|=35 592, pos/neg = 32 029 / 3 563.
- **Seeds:** {0, 1, 2, 3, 4} for the full-model 5-seed; {0} for each ablation.
- **Memory cap:** `ulimit -v 16777216` (16 GB virtual) per CLAUDE.md §4.

### Log artifacts (on-disk, verifiable)

- Single-seed smoke (initial): `/tmp/gomb_smoke_bitcoin_otc_seed0_2026_05_11.log`
- 5-seed full: `/tmp/gomb_5seed_bitcoin_otc_2026_05_11/{seed0..seed4}.log`, `results.jsonl`, `driver.log`
- 3 ablations: `/tmp/gomb_ablations_bitcoin_otc_2026_05_11/{no_outer,no_middle,no_inner}.log`, `results.jsonl`, `driver.log`

Per CLAUDE.md §9 in-flight discipline: all referenced log paths existed at report-write time.

## Plan-vs-result delta

| Plan section | Status |
|---|---|
| §Sequencing 1-4 (OuterFIRShell, MiddleHSiKAN, InnerCPMLCore, composer) | ✓ |
| §Sequencing 5 (6 unit tests) | ✓ — actually 13 (added 4 ablation + 2 extra coverage tests) |
| §Sequencing 6 (Bitcoin OTC seed-0 smoke) | ✓ |
| §Sequencing 7 (ablations 5-seed paired Δ) | partial — 1 seed per ablation only |
| §Validation gate (5-seed paired vs Slashdot edge_cr SOTA + best-cell of factorial) | **not run** |

The plan's cascade-vs-orthogonal-factorial comparison (§Comparison to
the factorial) is unaddressed by this report — the
`2026-05-11-hsikan-cpml-fir-orthogonal/` factorial cells have not
been compared to Gömb at iso-param.
