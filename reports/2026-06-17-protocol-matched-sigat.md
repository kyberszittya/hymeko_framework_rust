# Protocol-matched, honest SiGAT comparison — the "0.04 gap" was a dedup artifact

**Date:** 2026-06-17
**Plan:** [docs/plans/2026-06-17-protocol-matched-sigat](../docs/plans/2026-06-17-protocol-matched-sigat/) (4 artifacts; PDF compiles).
**Status:** ✅ implemented + tested + 5-seed grid (160 arms), all shuffle-gated. **Reframing result** (honest negative-to-neutral): the headline gap was cross-protocol; on a matched protocol it inverts vs `sigat_rotor` and shrinks vs pure SiGAT.

## Summary

The 2026-06-17 session chased a ~0.04 "SiGAT gap". Provenance check: the target
`sigat_rotor` 0.880/0.900 was measured **without** the `--dedup` true-held-out
filter that the rotor-HSiKAN line uses, so target and model were scored on
**different** test sets — the deduped one (HSiKAN) drops ~65 % of bitcoin_alpha
test edges (those whose `(u,v)` also sits in train) and is strictly harder.

This task added an opt-in `--dedup` to the baseline audit, single-sourced the
dedup helper into the datasets layer, and ran every model on **both** protocols,
5 seeds, each real + label-shuffled, with the shuffle gate enforced. The result:

- **vs `sigat_rotor` (the original target): the gap was an artifact and inverts.**
  Deduped `sigat_rotor` = 0.833/0.868 — *below* our line (0.850/0.879). Like-for-like,
  the rotor-HSiKAN line is **ahead by +0.017 (alpha) / +0.012 (otc)**.
- **The real residual gap is vs *pure* SiGAT**, and it is smaller than alleged:
  deduped pure SiGAT 0.886/0.895 → **+0.036 (alpha) / +0.016 (otc)** over our line.
  Pure SiGAT is the genuine ceiling; `sigat_rotor`/`cayley_rotor` are not.
- **The non-deduped protocol leaks**: the rotor-HSiKAN line's bitcoin_alpha
  shuffled gate is 0.56 > 0.55 (⚠, not reportable). Dedup is the honest protocol.
- **Consistency:** the deduped rotor-HSiKAN numbers reproduce the prior report
  (`2026-06-17-signed-rotor-slerp-propagation.md`) exactly — 0.850 / 0.879.

## Honest gap table (5-seed mean ± pstdev; g = mean shuffled AUROC)

### bitcoin_alpha
| model | non-deduped | deduped | Δ vs rotor-HSiKAN (raw / dedup) |
|---|---|---|---|
| sigat | 0.9060±0.0055 (g0.51✓) | **0.8862±0.0146** (g0.49✓) | +0.0111 / **+0.0362** |
| sigat_rotor | 0.8803±0.0084 (g0.55✓) | 0.8326±0.0191 (g0.51✓) | −0.0146 / **−0.0174** |
| cayley_rotor | 0.8664±0.0112 (g0.54✓) | 0.8193±0.0227 (g0.51✓) | −0.0285 / −0.0308 |
| **hsikan_rotor_r2sw4** (ref) | 0.8949±0.0087 (g0.56⚠) | **0.8500±0.0128** (g0.53✓) | 0 / 0 |

### bitcoin_otc
| model | non-deduped | deduped | Δ vs rotor-HSiKAN (raw / dedup) |
|---|---|---|---|
| sigat | 0.9104±0.0049 (g0.52✓) | **0.8953±0.0106** (g0.52✓) | +0.0029 / **+0.0163** |
| sigat_rotor | 0.9004±0.0098 (g0.54✓) | 0.8675±0.0021 (g0.53✓) | −0.0071 / **−0.0115** |
| cayley_rotor | 0.8945±0.0059 (g0.51✓) | 0.8593±0.0103 (g0.52✓) | −0.0130 / −0.0196 |
| **hsikan_rotor_r2sw4** (ref) | 0.9075±0.0110 (g0.53✓) | **0.8790±0.0102** (g0.52✓) | 0 / 0 |

**Measured / inferred / hypothesis (CLAUDE.md operating principle).**
*Measured:* the means/stds above; dedup drops 1566/2420 (65 %) of bitcoin_alpha
test edges. *Inferred:* the headline gap was a dedup-mismatch artifact (target
scored non-deduped, model deduped). *Hypothesis, not yet tested:* whether the
+0.012/+0.017 lead over `sigat_rotor` is significant — at the seed-mean it is
clear on otc (gap 0.012 vs stds 0.002/0.010) and marginal on alpha (gap 0.017 vs
stds 0.019/0.013, within ~1 std). A paired bootstrap over the per-seed deltas is
the follow-up; not run here (out of plan scope).

## Files touched

**New (4):**
- `hymeko_neuro/experiments/runs/run_protocol_matched_sigat.py` (+205) — focused
  in-process driver; Strategy-closure per model (audit registry via `run_audit`,
  rotor-HSiKAN via `run_hsikan_rotor.run`) so the grid loop never branches on
  family (§6.5 #9); reuses the gate/RSS helpers from `run_no_leak_benchmark` (no
  re-definition, no train-loop dup, §6.5 #1/#3); resumable JSONL; `--smoke/--full`.
- `hymeko_neuro/experiments/eval/aggregate_protocol_match.py` (+135) — pure-stdlib
  aggregator → the model×protocol gap table above.
- `hymeko_neuro/tests/test_protocol_matched_sigat.py` (+150) — 9 tests (grid
  construction, real 3-epoch arm + resume, aggregation maths/gate/gap).
- `hymeko_neuro/experiments/results/protocol_matched_full.jsonl` — 160 result rows.

**Modified (mine; tree was already dirty from the session):**
- `hymeko_neuro/data/datasets/legacy.py` — added `undirected_pair` +
  `drop_train_pairs` (the canonical true-held-out filter) beside `split`/
  `deduplicate_pairs`; incidental: cleared 3 pre-existing ruff errors in the file
  (unused top-level `import io` — shadowed by a local re-import; two `;`-joined
  lines in the KONECT reader).
- `hymeko_neuro/data/datasets/__init__.py` — re-export the two helpers.
- `hymeko_neuro/experiments/runs/run_baseline_audit.py` — `--dedup` flag + `dedup`
  param threaded through `run_audit`; records `dedup`/`n_test`/`n_test_dropped`/
  `n_val_dropped`; hardened `_evaluate` to return `(nan, nan)` on an empty
  held-out slice (a possible dedup outcome; sklearn raises on empty).
- `hymeko_neuro/experiments/runs/run_hsikan_rotor.py` — replaced the local
  `_pair`/`_drop_train_pairs` **defs** with imports from the datasets layer, kept
  the names as thin aliases (single source of truth; §6.5 #1). `run_rotor_head_ablation`
  and `test_hsikan_rotor` import these names unchanged.
- `hymeko_neuro/tests/test_baseline_audit.py` — `--dedup` regression (drops
  train-overlapping held-out; new result keys; dedup-off no-op).
- `hymeko_neuro/tests/test_konect_datasets.py` — datasets-layer unit tests for
  the two new functions (orientation invariance, mirror/repeat drop, no-op cases).
- `hymeko_neuro/tests/test_hsikan_rotor.py` — single-source guard (`_drop_train_pairs
  is datasets.drop_train_pairs`).

**CORE.YAML items touched:** none. `run_baseline_audit`/`run_hsikan_rotor`/
`datasets` are application code; no model change, no new dependency.

## Test results

- `test_konect_datasets.py`: 10 passed (4 new). `test_baseline_audit.py`: 12
  passed (3 new). `test_hsikan_rotor.py`: 31 passed (1 new guard).
  `test_protocol_matched_sigat.py`: 9 passed. (`pytest -p no:randomly`.)
- Static gates: `ruff check` — clean on all touched files. `mypy --strict` —
  clean on all my added code (aggregator + driver pass fully; my two new
  `legacy.py` functions add no errors). `legacy.py` retains **5 pre-existing**
  strict errors in unchanged functions (`load` dict/set type-args, two KONECT
  `Reader` reassignments, `split`'s bare `tuple` return) — not introduced here,
  left out of scope.

## Performance

- Per-cell wall ≈ 2.5 s (audit baselines) / ≈ 10–12 s (rotor-HSiKAN), GPU (cuda,
  cu132). 5-seed grid (160 arms): **wall 392.7 s**, **peak RSS 1.77 GB** (11 % of
  the 16 GB cap; cap assertion in the driver passed). §11 reconcile: 1-seed smoke
  104 s × 5 ≈ 520 s predicted vs 393 s measured (seed-0 resumed) — consistent.
- No benchmark-regression claim (this is a measurement task, not an optimisation);
  no profiling required.

## §6.5 anti-patterns

None introduced. Dedup is **single-sourced** (datasets layer; the run scripts
alias it — §6.5 #1/#2, guarded by a test). The driver reuses the existing
gate/RSS machinery rather than re-implementing it (§6.5 #3) and dispatches model
families via Strategy closures, not an `if family ==` ladder (§6.5 #9). One file
per concern with a mode arg (§6.5 #13). No new globals, no env-flag reads.

## Experiment provenance

- Git SHA `7d16ad0` (working tree dirty from the ongoing session; my touched
  files listed above).
- Datasets: cached SNAP `bitcoin_alpha`, `bitcoin_otc` under `hymeko_neuro/assets/data/`
  (no network). Seeds 0–4. Device: CUDA (torch 2.12.0+cu132). Split: 80/10/10
  via `datasets.split(seed=seed)`; strict train-only message passing.
- Artifact: `hymeko_neuro/experiments/results/protocol_matched_full.jsonl` (160
  rows; on disk). Aggregate via
  `python -m hymeko_neuro.experiments.eval.aggregate_protocol_match --in <that file>`.

## Open issues / follow-ups

- **Paired-bootstrap significance** of the rotor-HSiKAN vs `sigat_rotor` deduped
  lead (clear on otc, marginal on alpha) — per-seed deltas are on disk.
- **The honest narrative for the paper:** report the **deduped** protocol only;
  the ceiling is **pure SiGAT** (+0.036/+0.016), not `sigat_rotor`. Exclude the
  published transductive-leaky SiGAT (~0.95) entirely (different protocol).
- Optional: fold the deduped pure-SiGAT row into the Nature Table-1 baseline grid
  once the tier-2/3 audit lands (BACKLOG P1).
