---
campaign: COIN-DELIVERY-RECOVERY-BASELINE-0
title: Committed recovery baseline + golden harness for Coin Delivery
date: 2026-07-20
verdict: RECOVERY_BASELINE_READY_WITH_LIMITATIONS
recovery_commit: c0f62ab (branch recovery/coin-delivery-dirty-snapshot)
no_refactor: true
---

# COIN-DELIVERY-RECOVERY-BASELINE-0

**Created-at:** 2026-07-20 17:40 JST. First stage of the architecture recovery: preserve, commit, and gate — **no BC/
SAC/TD3/PPO/tensor-RL, no new controllers/actors/scenarios, no behavior refactor**. Stops for review.

## Verdict: **RECOVERY_BASELINE_READY_WITH_LIMITATIONS**

The uncommitted arc is now a committed, hash-anchored baseline with a golden harness and provenance gates; headlines
reproduce through the real path; the state/physics machinery is bit-exact (independently confirmed). The limitations are
provenance-hygiene and architectural, not scientific-correctness.

## §1 Evidence bundle (byte-exact, external)

`artifacts/coin_recovery_baseline/`: `evidence_manifest.json` (73 coin files: path/tracked/sha256/bytes/mtime/encoding/
line-ending — 72 untracked, 1 CRLF), `working_tree.patch`, `untracked_source.tar` (71 files, extract-verified hash-
equal), `git_meta.txt`, and `planar_grasp_env.py.CRLF-original.bytes` (the byte-exact CRLF original). No formatting, no
CRLF normalization, no auto-fixes applied to the working tree during capture.

## §2 The one committed recovery baseline

Branch **`recovery/coin-delivery-dirty-snapshot`**, commit **`c0f62ab`** — "recovery: preserve uncommitted coin delivery
snapshot" (marked *not a validated scientific baseline*). **73 coin production files, 9805 insertions.** Staged the coin
set explicitly (no `add -A`) → zero of the 2131 unrelated dirty files included. All coin production `.py` now tracked
(`cooperative_objective.py` is a 2026-07-10 orphan not imported by the arc → correctly excluded).

- **Limitation (documented)**: the repo `.gitattributes` enforces `*.py text eol=lf`, so `planar_grasp_env.py`'s 765
  CRLF working-tree lines were normalized to LF on commit (functionally identical Python). The byte-exact CRLF original
  is preserved in the evidence bundle. So the commit is byte-equivalent to the dirty tree **for every file except that
  one line-ending normalization**.

## §3 Corpus provenance (see [corpus-provenance report](2026-07-20-coin-delivery-corpus-provenance.md))

New isolated module `hymeko_rl/coin_delivery/provenance/state_identity.py` (`CorpusId`, `StateId`, `snapshot_hash`,
`legacy_seed_to_index` — no production-rollout change). `corpus_manifest.json` (376 items, sha `9262f6dc…`, qpos 7,
**from the prior `2026_07_18_arcrl` campaign — stale/reused**). `state_mapping.json`: the 90 eval seeds
(64000–64089) select **82 unique bank indices + 8 duplicates (max 4×)** — "n=90 independent states" is false.

## §4–§6 Golden harness, restore determinism, model fingerprints

- **Golden** (`golden_results.json`, schema v1, tol 1e-6): 8 explicit unique StateIds × {K0, K1-neutral, K1-aware,
  K1-scramble} with a fixed A0 command sequence; **restore determinism (run-twice equal) = True**. Test package
  `hymeko_rl/tests/golden_coin_delivery/` (8 provenance tests pass).
- **Model fingerprints** (`model_fingerprints.json`): K0 nu=4 / K1 nu=6, and **K1-neutral == K1-aware == K1-scramble**
  (byte-identical compiled model — only control trajectories differ). Independently confirmed by the physics auditor.
- **Restore history-independence**: confirmed here (test: restore→perturb→restore → identical qpos) and independently
  (max\|Δqpos\|=max\|Δqvel\|=0.0 across differing histories, both embodiments).

## §7 Headline reproduction (see [golden-reproduction report](2026-07-20-coin-delivery-golden-reproduction.md))

Through the **real** `rollout_delivery` path (an inline copy *drifted* — evidence of the duplicated-rollout-loop risk):
K0_A4 strict 5/60, K1-neutral/aware B_LR 0.0 — **reproduces the PAD-AWARE-CONTROL-0 manifest**. Historical (60, 3 dup)
vs unique (57): **duplication shifts a continuous metric** (K0_A4 B_LR 0.505→0.439) but flips **no** qualitative verdict.

## §11 Final answers

1. **Every coin production source committed?** Yes — 73 files at c0f62ab (`cooperative_objective.py` is a non-arc orphan).
2. **Snapshot byte-equivalent to the dirty tree?** Yes for all files **except** `planar_grasp_env.py` (CRLF→LF per the
   repo LF policy; byte-exact CRLF original preserved in the evidence bundle).
3. **Reproduces headline results?** Yes (real-path K0_A4 strict 5, K1 B_LR 0.0 = manifest).
4. **Unique StateIds among historical episodes?** 90 → **82 unique** (60 → 57 in the pad-aware subset).
5. **Duplicate weighting affects a conclusion?** A continuous metric shifts (K0_A4 B_LR 0.505→0.439); no qualitative
   verdict flips.
6. **Restore history-independent?** Yes (both embodiments, bit-exact).
7. **K1 neutral/aware/scramble physically identical except commands?** Yes (byte-identical compiled models).
8. **Actual collision geom controlled by the distal actuator?** Yes (`a_pad_left → pad_hinge_left → pad_left` body
   carrying the `fingertip_left` *collision* geom; independently confirmed).
9. **Reproducible from the committed recovery branch?** Yes.
10. **Which results remain invalid/ambiguous?** None invalidated. **Pending your review** (findings held): the PAD-AWARE
    `K1-INCONCLUSIVE` should be **relabeled a floor-limited bounded-negative under an unavoidable embodiment change** (the
    physics auditor showed K0-vs-K1-neutral capacity-matching is *structurally impossible* — you cannot add a distal DOF
    and keep the fingertip rigid; K1-neutral already has both_frac=0, so orientation has no bilateral load to rebalance).
    Provenance caveats: the corpus is stale (arcrl); the seed-as-state-selector; the framework is a parallel unused layer.
11. **Golden harness strong enough to begin refactoring?** **Partially.** It captures behavioral outputs + provenance
    gates, but the architecture auditor found **3 uncaught silent-wiring mutations** — L/R fingertip-force swap, the
    `[2,4]` snapshot-padding layout, and the absent K1-neutral≈K0 test. **Add those 3 wiring-regression tests before the
    refactor.**
12. **Invariants the refactor must preserve:** restore history-independence (bit-exact); K1 model equality across
    variants; the `a_pad_{side}→pad_hinge_{side}→fingertip_{side}` collision wiring + L/R map; the strict monitor
    predicate (dwell/settle/fingertip-attribution/body-shove/mechanism); `golden_results.json` within 1e-6; the 7→9 qpos
    padding layout (hinges at indices 2,5); explicit `StateId` (never seed-as-identity); correct-beats-scramble.

## Files added this stage (all new/isolated — no behavior refactor)

`hymeko_rl/coin_delivery/provenance/{__init__,state_identity}.py`; `hymeko_rl/tests/golden_coin_delivery/{__init__,
test_provenance_baseline}.py`; the `artifacts/coin_recovery_baseline/` bundle; 3 reports. A **second commit** on the
recovery branch tracks these baseline modules (the dirty-snapshot commit c0f62ab stays pure).

## Stop

Per §9/§16 — no architecture refactoring performed (no `planar_grasp_env` split, no private-field replacement, no
ports-and-adapters, no legacy quarantine, no bulk ruff fixes). **Waiting for explicit user approval before the
architecture-refactoring campaign.**
