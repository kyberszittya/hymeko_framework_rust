---
campaign: COIN-DELIVERY-RECOVERY-BASELINE-0
title: Golden behavioral harness + headline reproduction
date: 2026-07-20
parent: 2026-07-20-coin-delivery-recovery-baseline.md
---

# Golden harness + reproduction

**Created-at:** 2026-07-20 17:40 JST.

## Golden harness (§4)

`hymeko_rl/tests/golden_coin_delivery/` + `artifacts/coin_recovery_baseline/golden_results.json` (schema v1, tol 1e-6).
8 **explicit unique** StateIds (bank indices spanning the `dist` metadata, no duplicates) × {K0 canonical, K1-neutral,
K1-geometry-aware, K1-scramble}, driven by a fixed deterministic A0 command sequence. Each config records the state hash,
final/min coin-to-zone distance and progress. The harness depends on no report file or historical scalar constant.

- **Restore determinism (§5): PASS.** Every (StateId, config) run **twice** returns identical results; and
  restore→perturb→restore→run equals a fresh restore (test `test_restore_is_history_independent`). Independently
  confirmed bit-exact (`max|Δqpos|=max|Δqvel|=0.0`) for K0 and all K1 variants.
- **Model fingerprints (§6): K1-neutral == K1-aware == K1-scramble** (byte-identical compiled `mjModel`; only the distal
  control command differs). K0 differs structurally (nu 4 vs 6) — the full diff is in the model-diff report; all
  differences are enumerated. `model_fingerprints.json`.

## Reproduction (§7) — through the real `rollout_delivery` path

| metric | historical (60 seeds, 3 dup) | unique (57) | classification |
|---|---|---|---|
| K0 A4 strict delivery | 5/60 | 5/57 | exact (count) |
| K0 A4 median B_LR | 0.505 | 0.439 | **materially different** (Δ 0.066) |
| K1-neutral A0 B_LR | 0.0 | 0.0 | exact (reproduces manifest) |
| K1-aware A0 B_LR | 0.0 | 0.0 | exact (reproduces manifest) |

- **Historical-seed reproduction** reproduces the PAD-AWARE-CONTROL-0 headline (K1-neutral/aware B_LR 0.0, K0_A4 high
  balance) — confirming the committed snapshot reproduces its own results.
- **Unique-State reproduction** (one seed per unique bank index) shifts only the *continuous* K0_A4 B_LR (0.505→0.439);
  the strict-delivery counts and all qualitative verdicts are unchanged. So **duplicate weighting does not change any
  conclusion**, but it does bias a continuous mean — the two are reported separately and not merged.
- **Caveat (a real limitation):** the current `rollout_delivery` API is **seed-driven only** — it cannot be handed an
  explicit `StateId`. Evaluating "unique states" required either dedup-by-index (used here, faithful) or an inline
  restore loop (which **drifted** from the real path — direct evidence of the duplicated-rollout-loop risk the
  architecture audit flagged). Wiring `rollout` to accept an explicit `StateId` is a refactor precondition, not done
  here.

## Silent-wiring gaps the golden must close before refactor (§14 of the audit)

Three mutations survive the current tests and would corrupt attribution/pairing without crashing: (1) swapping the L/R
fingertip normal forces (no orientation-pinning attribution test); (2) the `[2,4]→[0,1]` snapshot-padding layout (no
qpos-layout assertion — the K0/K1 byte-pairing is unguarded); (3) no K1-neutral≈K0 capacity test. These three
regression tests are the minimum to add before the architecture refactor is safe.
