# Report — Regime A/B/C 5-seed comparison (HSiKAN-mixed protocol sweep)

**Date:** 2026-05-27
**Plan:** `docs/plans/2026-05-27-composite-regime-abc-5seed/` (tex/pdf/tikz/mmd)
**Parent:** `reports/2026-05-27-hsikan-mixed-composite-regime.md`
**CORE.YAML items touched:** none (torch drift caveat carries over — see provenance).

## Summary

Ran the quality comparison the Composite-regime pipeline deferred: does
the P-graph regime that canonical MSG/ABB uses change which HSiKAN-mixed
architecture you should pick? Trained the **Canonical superset (12
architectures)** on Bitcoin Alpha at **5 seeds** (60 runs, 0 failures),
then sliced per regime by the *authoritative* Rust SSG admissible sets.

**Headline:** the canonical regime's larger admissible set buys **zero
quality**. The best architecture is the same non-quaternion one under both
Canonical and No-Excess, and Cost-dominance over-prunes.

| Regime | #admissible | Best architecture | 5-seed AUC |
|:--|--:|:--|:--|
| **A Canonical** | 12 | `attn=none · gate=scalar · dm=on` | **0.9766 ± 0.0136** |
| **B No-Excess ≡ Composite** | 8 | `attn=none · gate=scalar · dm=on` (same) | **0.9766 ± 0.0136** |
| **C Cost-dominance** | 2 | `attn=quaternion · gate=scalar · dm=off` | 0.9506 ± 0.0323 |

Canonical-best vs No-Excess-best: **identical architecture**, paired
Δ = 0.0 (σ = 0, n = 5). The four quaternion-bearing architectures that
**only Canonical admits** never win — the best of them
(`quaternion·scalar·dm=on`, 0.9578) sits below the global best
(0.9766), paired Δ = +0.0189 in favour of the non-quaternion arch (σ +0.66,
3/5 — nominally better, statistically a tie). Either way, keeping
quaternion costs 50 % more candidates (12 vs 8) for **no** gain.

### Why: the regime's distinguishing axis is the wrong axis

Marginal paired lever effects (over all other axes × 5 seeds):

| Lever | paired Δ | σ | wins | verdict |
|:--|--:|--:|--:|:--|
| **direct-messaging `dm=on − dm=off`** | **+0.0150** | **+2.50** | 23/30 | **real signal** |
| attention `dot − none` | −0.0111 | −0.95 | 11/20 | null→negative |
| attention `quaternion − none` | −0.0093 | −0.81 | 11/20 | null→negative |
| edge-gate `edge_cr − scalar` | −0.0053 | −0.65 | 15/30 | no effect |

The axis that distinguishes Canonical from No-Excess is **attention**
(quaternion is admitted only by Canonical). But attention is null-to-
negative on Bitcoin Alpha; the signal-carrying protocol is
**direct-messaging**. So No-Excess loses nothing by pruning the
quaternion branch — it prunes on the axis that doesn't matter. **No-Excess
(≡ Composite) is the regime to use: identical optimum, 33 % smaller search
space.** Cost-dominance is too aggressive — it dropped *every* `dm=on`
architecture (kept only `dm=off`), missing the actual winner by −0.026.

## Files touched

| File | Status | LOC |
|:--|:--|--:|
| `signedkan_wip/experiments/runs/run_regime_abc_5seed.py` | new | 297 |
| `signedkan_wip/tests/test_regime_abc.py` | new | 152 |
| `signedkan_wip/experiments/runs/run_hsikan_mixed_composite_smoke.py` | modified | +27/−13 (RSS-measurement fix; see below) |
| `signedkan_wip/tests/test_mixed_composite_regime.py` | modified | +14 (RSS parser tests) |
| `docs/plans/2026-05-27-composite-regime-abc-5seed/{tex,pdf,tikz,mmd}` | new | — |
| `reports/2026-05-27-composite-regime-abc-5seed.md` | new | — |

### Measurement-correctness fix (caught during the 1-seed smoke)

`run_cell` previously derived per-run peak RSS from
`getrusage(RUSAGE_CHILDREN).ru_maxrss` deltas — wrong for a multi-run
harness, because that counter is a **monotonic high-water over all reaped
children** (the smoke produced bogus `0.01 GiB` / repeated `1.63 GiB`
rows). Replaced with `/usr/bin/time -v` per child (reliable per-cell peak
RSS), with `nan` fallback when GNU time is absent. Unit-tested
(`_parse_time_v_rss_gib`). The parent task's single-run smoke number
(1.56 GiB) was unaffected (single child → delta correct).

## Test results

Runner: `.venv/bin/python -m pytest -p no:randomly` (torch-free).

| File | Count | Result |
|:--|--:|:--|
| `test_regime_abc.py` (clean-sig, regime-admissible 12/8/2, arch-env cap, aggregation maths, dry-run job count) | 10 | pass |
| `test_mixed_composite_regime.py` (+RSS parser) | 21 | pass |
| Combined | **31** | **pass (0.55 s)** |

`ruff check` clean on both new/modified Python files. Every new function
(`_clean_signature`, `regime_admissible_archs`, `arch_env`, `_mean_sd`,
`_paired_delta`, `aggregate`, `_parse_time_v_rss_gib`, `main` dry-run) is
unit-tested; the GPU `run_cell` path is exercised by the smoke + the
60-run experiment.

## Performance results

Production-scale 1-seed smoke (12 archs): **205 s**; 5-seed estimate
5 × 205 ≈ 17 min — reconciled (CLAUDE.md §11). Actual 5-seed run:

| Metric | Value | Budget |
|:--|:--|:--|
| Total wall (60 runs) | 1024 s (17.1 min) | ≤ 90 min ✓ |
| Mean per-run wall | 17.1 s | — |
| Max per-run peak RSS | 1.63 GiB | ≤ 7 GiB ✓ (16 GB cap) |
| GPU OOMs | 0 | — (attention inherits top-K cap) |

`systemd-run --user --scope -p MemoryMax=16G` enforced the RSS gate
(`ulimit -v` not used, §4). Per-(arch,seed) rows checkpointed to jsonl
(resumable); all 60 completed in one pass.

## New / removed dependencies

None.

## Experiment provenance

- **Git SHA:** `8fd8187` (working tree dirty: this task's new files +
  the parent task's `hsikan_pgraph_mapping.py` / sweep / driver, plus the
  RSS-fix edit to `run_hsikan_mixed_composite_smoke.py`).
- **Interpreter:** `/home/kyberszittya/miniconda3/bin/python` 3.13.5,
  **torch 2.11.0+cu130**. **DEPENDENCY DRIFT (carried from parent,
  user-approved):** CORE pins `torch==2.12.0`; no local env matches. AUCs
  are **relative comparison** values at abbreviated epochs, *not*
  CORE-reproducible SOTA (BA SOTA ≈ 0.996, memory
  `bitcoin_optuna_best_10seed`).
- **OS/kernel:** Linux 6.17.0-29-generic. **CPU:** AMD Ryzen 7 3700X.
  **RAM:** 31 GiB. **GPU:** RTX 2070 SUPER 8 GiB, driver 580.126.09.
- **Seeds:** {0,1,2,3,4}. **Epochs:** 20 (fixed across all archs).
  **Hidden:** 8. **Dataset:** bitcoin_alpha (native signed graph).
- **Sweep hash:** `md5 d0fd8010cef7473cae20c2edd8a37e3b`.
- **Artifacts:** `/tmp/regime_abc_5seed/results.jsonl` (60 rows),
  `/tmp/regime_abc_5seed/summary.json`, per-cell logs.

## §6.5 anti-patterns

None introduced. Harness reuses the parent driver's `solve_regime` /
`structure_to_env` / `run_cell` rather than duplicating them; admissible
sets derived from the authoritative Rust SSG (no hardcoded pruning rule);
aggregation is small pure functions.

## Conclusions & follow-ups

1. **Use No-Excess (≡ Composite), not Canonical, for this search.** Same
   optimum, 33 % fewer candidates. Canonical's extra (quaternion) set is
   on the null axis.
2. **Don't use Cost-dominance for architecture search** — it pruned the
   `dm=on` winner. It optimises cost-Pareto, which is orthogonal to AUC.
3. **Direct-messaging is the BA lever (+0.015, σ2.5); attention is not.**
   This is a BA-specific finding — attention helped on dense walk-rich
   Slashdot (memory `attention_cycle_batch_compose`). The regime
   conclusion generalises only while the regime-distinguishing axis stays
   null on the target dataset.
4. **Follow-up:** repeat on Slashdot, where attention *does* carry signal
   — there Canonical (admitting quaternion) may genuinely beat No-Excess,
   flipping the recommendation. That is the decisive cross-dataset test
   and is a launch (harness ready), not new code.
5. Abbreviated 20-epoch AUCs are relative; no promotion to any results
   table (memory `feedback_n_seed_before_paper_promotion`).
