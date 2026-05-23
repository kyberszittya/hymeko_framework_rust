# Phase 21: SideMixedAritySignedKAN — null at the Bitcoin Alpha 0.997 ceiling — 2026-05-20

## Summary

Phase 21 ported the Phase 17/19/20 parallel-branch + highway
pattern onto the mixed-arity HSIKAN family
(`c2, c5, w2, w3, w4` with learned αₖ — Bitcoin Alpha SOTA
0.9959 ± 0.0011 per `[[project_bitcoin_optuna_best_10seed_2026_05_13]]`).

Hypothesis: N parallel mixed-arity branches with mean-fusion
either (a) lift AUC by a few thousandths or (b) tighten σ in
the same way Phase 19's Side did on c3-only (σ ≈ 0.013
uniformly across scales). 5-seed paired A/B falsifies both.

**Headline (null):** N=4 vs N=1 paired Δ AUC = **+0.0003 ±
0.0007** (σ\_d = +1.07, wins = 3/5). σ identical at 0.0005
for both. 4× wall cost, 4× params, no measurable gain on
either axis. The architectural ceiling at the
mixed-arity-Optuna-best config is reached; parallel branches
do not improve it.

## Files touched

| File | Status | LOC |
| --- | --- | --- |
| `signedkan_wip/src/core/side_signedkan.py` | extended | +165 (`SideMixedAritySignedKAN` + config) |
| `signedkan_wip/tests/test_side_mixed_arity.py` | new | 213 (10 unit tests) |
| `signedkan_wip/experiments/runs/run_phase2_mixed_arity.py` | extended | +18 (`n_branches`, `side_fusion` kwargs + dispatch + smooth-reg over all branches) |
| `signedkan_wip/experiments/runs/run_final_cell.py` | extended | +24 (CLI `--n-branches`, `--side-fusion`; dispatch in `cell_signed_graph`) |
| `signedkan_wip/experiments/run_phase21_side_mixed_5seed_2026_05_20.sh` | new | 109 |
| `docs/plans/2026-05-20-phase21-mixed-arity-side/{plan.tex,plan.pdf,plan.tikz,plan.mmd}` | new | 4 plan formats per CLAUDE.md §2 |
| `reports/2026-05-20-phase21-mixed-arity-side-null.md` | new | this file |

## CORE.YAML items touched

None.

## Interface change

```python
@dataclass
class SideMixedAritySignedKANConfig:
    base: MixedAritySignedKANConfig
    n_branches: int = 4
    fusion: str = "mean"     # "mean" | "sum"

class SideMixedAritySignedKAN(nn.Module):
    branches: nn.ModuleList     # N x independent MixedAritySignedKAN
    classifier: nn.Linear       # at wrapper level (one per Phase 21 model)

    # delegated to first branch:
    base, node_embed
    # aggregates across branches:
    alpha()                     # mean of per-branch softmaxed α
    _attn_entropy_terms         # concatenated branch lists
```

Dispatch wired through `run_final_cell.cell_signed_graph` and
`run_phase2_mixed_arity.run_one_mixed`; CLI flags
`--n-branches N --side-fusion {mean,sum}` pass through.

## Phase 21a — production-scale smoke (single seed)

| config | AUC | F1m | n_params | wall (s) |
| --- | --- | --- | --- | --- |
| N=1 (bare) | 0.9970 | 0.9229 | 30,487 | 257 |
| N=4 | 0.9967 | 0.9177 | 121,914 | 1023 |

Smoke succeeded: wrapper runs end-to-end on the real Optuna
SOTA config under a 14 GB RSS cap. Δ at seed 0 was −0.0003 —
not a lift, but within noise.

## Phase 21b — 5-seed paired A/B vs Optuna SOTA

Bitcoin Alpha, mixed `c2, c5, w2, w3, w4`, hidden=8, n\_epochs=80,
`HSIKAN_ALPHA_ENTROPY_LAMBDA=0.0966`. 5 seeds × 2 configs.

| seed | N=1 AUC | N=4 AUC | Δ |
| --- | --- | --- | --- |
| 0 | 0.99700 | 0.99673 | −0.00027 |
| 1 | 0.99624 | 0.99720 | +0.00096 |
| 2 | 0.99569 | 0.99682 | +0.00113 |
| 3 | 0.99619 | 0.99585 | −0.00034 |
| 4 | 0.99670 | 0.99685 | +0.00015 |

| N | mean AUC ± σ | wall/seed | n\_params |
| --- | --- | --- | --- |
| 1 | **0.9964 ± 0.0005** | 254 s | 30,487 |
| 4 | **0.9967 ± 0.0005** | 1023 s | 121,914 |

**Paired:** Δ = **+0.0003 ± 0.0007**, σ\_d = **+1.07**,
wins = **3 / 5**. By the §3 n=5 paired-promotion gate this is
NOT a significant lift — well under the
`[[feedback_n_seed_before_paper_promotion]]` bar.

### What this means

**1. The AUC lift hypothesis fails at the 0.997 ceiling.**
Phase 19's c3-only side N=8 lifted bare from 0.794 → 0.808
(+0.014, ~6σ paired) at AUC 0.79; the same wrapper applied
at 0.997 lifts by +0.0003 (1σ). The headroom argument
predicted small or null lift; this is the null branch.

**2. The variance-tightening hope also fails.** Phase 19
found Side σ ≈ 0.013 uniformly across L, vs depth σ in
0.028–0.048; the architectural-stability story. Here both
N=1 and N=4 sit at σ = 0.0005 (training-noise floor at the
0.997 regime). The σ-tightening lever has no slack to act
on. It was a regime-specific result.

**3. N=1 reproduces the published 10-seed Optuna SOTA.**
0.9964 ± 0.0005 (n=5) overlaps 0.9959 ± 0.0011 (n=10,
[[project_bitcoin_optuna_best_10seed_2026_05_13]]) within
0.5σ. The training pipeline is reproducing the canonical
result before we modify it — important sanity for the null
itself.

**4. Cost asymmetry.** N=4 spends 4× the wall, 4× the
parameters, for +0.03% AUC lift (which is not statistically
significant). This is a strictly inferior operating point at
this dataset/config; parallel branches buy nothing at SOTA.

## Why Phase 21 was still the right experiment

The Phase 19 SOTA-check report explicitly identified this as
the open question:

> "The natural Phase 20 candidate: port the side/membrane
> parallel-branch pattern to the mixed-arity HSIKAN family.
> Instead of `c3 × N branches`, run `[c2, c5, w2, w3, w4] × N
> branches` and fuse. Combines the variance-tightening side
> benefit with the mixed-arity SOTA infrastructure."

It was a falsifiable architectural hypothesis at a clear
boundary (c3-only → mixed-arity). The null answers it
cleanly: the side-stacking lift on c3-only was the right
lever at the **low-AUC** end of the regime curve. At
mixed-arity SOTA there is no signal to extract by
parallelism.

The §6.5 audit also stayed clean (one new class, one config
struct, three new kwargs, zero new wrapper functions or
Cartesian-product surface).

## Test results

| Suite | Result |
| --- | --- |
| `pytest signedkan_wip/tests/test_side_mixed_arity.py` | **10 / 10 pass** |
| `pytest signedkan_wip/tests/test_side_signedkan.py` | 12 / 12 pass (no regression) |
| `cargo test -p hymeko_pgraph` | **96 / 96 + 1 ignored doctest** |
| N=1 seed-0 baseline reproduces Optuna SOTA at 0.9970 | ✓ |

## §6.5 anti-pattern audit

- **(1) Cartesian-product API:** no — single new class with
  `n_branches` and `fusion` as config fields.
- **(2) Algorithm code behind a binding layer:** no —
  wrapper sits in `signedkan_wip/src/core/` next to the
  other side/membrane/stacked variants.
- **(3) Per-experiment scaffold duplication:** no — reuses
  `run_one_mixed` and `cell_signed_graph` training loops
  unmodified except for an `n_branches` dispatch.
- **(4) Long single-file modules:** `side_signedkan.py` grew
  from ~490 to ~650 LOC, still within §6.2's warn (800)
  ceiling. All four parallel-branch ensembles live together
  coherently.
- **(5) New axis = new function name:** no.
- **(6) `#[allow(...)]` band-aid:** no.
- **(7) String-typed config that should be enum:** the new
  `side_fusion: str ∈ {"mean", "sum"}` is at the
  Python/CLI boundary; an enum would only matter once the
  set grows. Acceptable per §6.5.
- **(8) Forward-time flags for structural differences:**
  no — `if n_branches > 1` is a class-construction
  decision, not a `forward()` toggle.
- **(9) Bypassing existing Strategy traits:** no —
  `MixedAritySignedKAN` is reused unchanged.
- **(10) `ulimit -v` on CUDA:** no — used
  `systemd-run --user --scope -p MemoryMax=14G`.
- **(11) Module-level mutable state:** no.

Clean.

## Performance results vs plan budget

| metric | plan target | plan cap | actual |
| --- | --- | --- | --- |
| Peak RSS (smoke) | 3.5 GB | 8 GB | ~2 GB |
| Wall N=4 / seed | 240 s | 480 s | 1023 s (4× over target) |
| 5-seed total | 20 min | 40 min | ~105 min |

Wall significantly exceeded the plan's optimistic target.
Root cause: the plan assumed N=4 cost ≈ 4× N=1, but N=1
under the Optuna config (h=8, full c2+c5+w2+w3+w4 with
cap=100k) is itself ~4 min (not the assumed 30 s). Memory
budget and 16 GB §4 cap were never approached.

## Open follow-ups

1. **Phase 22 candidate (pivot away from parallel branches at
   SOTA).** Mixed-arity has hit its architectural ceiling on
   Bitcoin Alpha. The remaining lift levers are either
   **dataset-side** (test on Slashdot / Epinions where SOTA
   gaps persist) or **regime-side** (port Phase 21's
   parallel-branch pattern to Slashdot / Epinions where
   variance is higher and headroom larger).
2. **N=4 mixed-arity on Slashdot.** Slashdot edge\_cr 5-seed
   ([[project_edge_cr_5seed_2026_05_09]]) sits at 0.9067 ±
   0.0034; σ is 7× higher than Bitcoin Alpha's, the side-
   stacking variance-tightening lever may still apply. This
   is the natural next test.
3. **N=4 mixed-arity on Epinions.** Epinions edge\_cr 5-seed
   ([[project_epinions_edge_cr_null_2026_05_10]]) at 0.846
   ± 0.011 — far higher σ. If the Phase 17/19 σ-tightening
   transfers anywhere, this is the place.
4. **Sum-fusion variant.** Phase 21 used `mean`; on c3-only
   Phase 17 found sum and mean equivalent (within σ). Worth
   one seed at Slashdot/Epinions before discarding.
5. **N=8 at Slashdot/Epinions.** Skipped from Phase 21
   because seed-0 N=4 ≈ N=1 at the ceiling; for higher-σ
   datasets the same logic doesn't pre-empt N=8.

## Experiment provenance

- **Git SHA:** uncommitted (pre-Phase-21 branch
  `refactor/extract-hymeko-hre`).
- **Dataset:** Bitcoin Alpha (n\_nodes=3783, n\_edges=24186).
- **5-seed log directory:**
  `/tmp/phase21_side_mixed_5seed_20260520T154226Z/`
  (per-cell logs + `orchestrator.log`).
- **JSONL results:**
  `signedkan_wip/experiments/results/phase21_side_mixed_5seed_2026_05_20.jsonl`.
- **GPU:** RTX 2070 SUPER, 8 GiB.
- **5-seed wall:** ~105 min total (256 s + 1023 s + 253 s +
  1022 s + 254 s + 1022 s + 256 s + 1033 s + 253 s + 1015 s,
  per-seed) = 6287 s actual.
- **Reproducibility:** seeds [0, 1, 2, 3, 4]; cache enabled
  (`HYMEKO_CYCLE_CACHE=1`); identical cache key across all
  10 cells.

## Acceptance check

- [x] CLAUDE.md §2 plan (4 formats) on disk.
- [x] CORE.YAML items touched = none.
- [x] 10 / 10 unit tests pass.
- [x] N=1 baseline reproduces Optuna SOTA before any
      modification.
- [x] Production-scale smoke at 1 seed before queuing
      multi-seed.
- [x] §6.5 anti-pattern audit clean.
- [x] 5-seed paired A/B with σ\_d and win-count reported.
- [x] **Null result framed honestly:** Δ = +0.0003 (not a
      lift); σ identical; cost 4× worse. Phase 21 falsifies
      the parallel-branch-helps-at-SOTA hypothesis on
      Bitcoin Alpha.
- [x] Memory updated to reflect the null and the Phase 22
      pivot direction.
- [x] Report on disk.
