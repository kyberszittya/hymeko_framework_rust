---
campaign: COIN-DELIVERY-LAST-KNOWN-GOOD-RECOVERY
title: Forensic recovery of historical Coin Delivery code states
date: 2026-07-20
classification: CURRENT_IMPLEMENTATION_BEST_AVAILABLE
modification_free: true (no production file modified/committed)
---

# COIN-DELIVERY-LAST-KNOWN-GOOD-RECOVERY

**Created-at:** 2026-07-20 17:30 JST. Modification-free forensic recovery: no production file was modified, formatted,
cleaned, or committed; no recovery commit created; MEMORY/findings untouched.

## Final classification: **CURRENT_IMPLEMENTATION_BEST_AVAILABLE**

No earlier implementation delivered the coin more reliably; no coin file has a recoverable overwritten earlier version;
the apparent "0.30 → 8.9 %" regression is a **loose→strict monitor tightening**, not lost code.

## §1–2 Search — exhaustive, and the coin arc was never in version control

- **Git**: current branch `exp/demo-seed-replay` @ 291bb4b (commits are SAC / Coffee-Push / Humanoid / demo-seed —
  unrelated to coin). No stashes. Reflog: no coin commits. **git fsck**: 1109 dangling blobs + 6 dangling commits —
  searched **all** of them for coin fragments (`rollout_delivery`, `coop_push_step_reward`, `DeliveryActor`, `p_push`,
  `bilateral_balance`, `v_plow`, `distal_targets`) → **ZERO hits**. The 6 dangling commits are unrelated WIP
  (CIP/humanoid/MetaWorld). **The coin arc was never staged or committed.**
- **Editor / filesystem**: no VS Code `History` dir, no JetBrains local history, no `*.py.bak` / `*.swp` / `*~` / `.orig`
  files. **So no coin file has a recoverable earlier on-disk version — each module has exactly one state.**
- Corpus: the delivery bank `c1_heldseed_bank.pkl` is a **stale artifact from the prior `2026_07_18_arcrl` campaign**
  (untracked, seed universe 62000–62312); the eval "seed" is an RNG→bank-index selector (90 seeds → 82 distinct states
  + 8 duplicates). Provenance recorded in `artifacts/coin_recovery/source_hashes.json` (72 files hashed).

## §3 Candidate inventory (5 candidates — all EXACT on-disk, none overwritten)

| id | what | campaign | mtime | claim |
|---|---|---|---|---|
| **C0** current push/plow | `coin_delivery_actor.py` + `pad_aware_control.py` | ACTOR-1 / PAD-AWARE (this session) | 15:43–16:54 | strict best 8/90 (A4) |
| **C1** earlier primitives | `coin_delivery1.py` (grasp_carry/carry_pulse/push/settle) | COIN-DELIVERY-1 (overnight) | 01:21 | **0.30 LOOSE** ceiling; STAGE2_FAIL |
| **C2** earlier RL | `coin_delivery_rl.py` + rl1/rl2 | RL-1/2 (overnight) | 03:03–03:55 | CASE_D "matches scripted"; CASE_C "residual exhausted" |
| **C3** earlier acquisition | `coin_delivery_acquisition.py` + primitives/hardstate | acquisition (overnight) | 03:42–04:27 | stable_acq 5/5 but **chained zone_entry 0** |
| **C4** arcrl RL checkpoints | `2026_07_18_arcrl/*.pt` (9) | coin-TOSS genuine-RL R14–R60 | 2026-07-18 | **different task** (delivery-PRESERVATION, not to-zone) |

Full inventory: `artifacts/coin_recovery/candidate_inventory.json`.

## §5–6 Behavioral test (explicit StateIds, not RNG seeds) — the decisive comparison

Explicit unique bank indices `[0,47,94,141,188,235,282,329,375]` (9 states), same MuJoCo 3.10.0, same horizon, both
monitors:

| candidate | LOOSE (in-zone-ever) | STRICT (current monitor) |
|---|---|---|
| **C1 earlier `carry_pulse`** | **6/9** | 2/9 |
| **C0 current `A4_recovery`** | **6/9** | 1/9 |

- **C1 and C0 are byte-for-byte the same behavior lineage**: identical LOOSE delivery (6/9) on identical states, and
  both ~1–2/9 STRICT (statistically equal on 9 states). The current `DeliveryActor.A4` is a clean re-implementation of
  the earlier `p_carry_pulse` (pulse-release recovery) — same mechanism.
- So the earlier "0.30 loose ceiling" (`coin_delivery1`) and the current "8.9 % strict" (`ACTOR-1`) are **the same
  behavior under two monitors**. The drop is the STRICT monitor (dwell + settle + fingertip-attribution + body-shove +
  clean-mechanism) correctly rejecting the momentary/coasting/bulldoze "deliveries" the loose in-zone-ever counted.
- **C2/C3 (earlier RL + acquisition)**: their own manifests claim NO working delivery-to-zone (RL "matches scripted";
  acquisition "chained zone_entry 0"). Not rerun to beat this — the manifests already report no delivery advantage, and
  the strict monitor can only lower a loose number.
- **C4 (arcrl `.pt` checkpoints)** are trained for the coin-**toss delivery-preservation** task (the v0–v28 / R14–R60
  deploy line, best {v16f 0.875/0.750}) — a **different success definition** than delivery-to-zone. Their 0.875 is not
  comparable to the current task's 8.9 %; treating it as "better delivery" would conflate two tasks.

## §7 Overwritten-logic check

Nothing was overwritten. C1/C2/C3 are **separate modules that still exist** (different files, not earlier versions of
C0). No deleted controller branch, no changed qpos slicing that removed a working path — the current strict monitor is
an *addition* over the earlier loose one, not a regression of the actor code.

## §8 Scientific-truth vs architecture

The earlier flat modules (C1–C3) are procedurally written but scientifically **not superior**; there is no working
historical behavior to freeze as a golden regression baseline that beats the current strict result. The one thing worth
freezing as a golden reference is the **current** push/plow behavior itself (C0 ≡ C1 under loose) — see the companion
[behavior-comparison report](2026-07-20-coin-historical-behavior-comparison.md).

## §10 Answers

1. **Earlier impl delivered more reliably?** No — earlier `carry_pulse` ≡ current `A4` (6/9 loose, ~1–2/9 strict).
2. **Earlier RL performed better?** No for delivery-to-zone (RL-1/2 "matches scripted / residual exhausted"). The arcrl
   `.pt` are a different task (toss preservation, 0.875), not comparable.
3. **Which files/functions produced the best delivery?** The pulse-release recovery mechanism — `coin_delivery1.p_carry_pulse`
   (earlier) ≡ `coin_delivery_actor.actor_action(A4_RECOVERY)` (current). Best strict ≈ A4 recovery.
4. **Real under the current strict monitor?** No advantage: earlier 2/9, current 1/9 (equal within noise on 9 states).
5. **Which later change degraded/removed it?** None removed. The strict monitor (dwell/settle/attribution/mechanism)
   *tightened* the success predicate; the underlying push behavior is unchanged.
6. **Freeze older behavior as golden regression baseline?** There is no *superior* older behavior to freeze; the current
   push/plow behavior (deterministic on explicit StateIds) is the freezable golden reference.
7. **Which code should be the canonical recovery commit?** The current push/plow (`coin_delivery_actor` + the framework)
   — it subsumes the earlier lineage. C2/C3/C4 add no delivery-to-zone value.
8. **Discard rather than refactor?** The earlier duplicate delivery-primitive scripts (`coin_delivery1`, `delivery_env0`,
   `delivery_rescore`, `coin_delivery_primitives`, `coin_delivery_rl1/rl2`) are superseded by `coin_delivery_actor` +
   the framework; quarantine, don't refactor. The arcrl `.pt` belong to the toss task, not this one.

## Deliverables

`artifacts/coin_recovery/{source_hashes.json, candidate_inventory.json, per_candidate_results.json}` + this report +
[behavior-comparison](2026-07-20-coin-historical-behavior-comparison.md). No recovery commit created; no rewrite.
