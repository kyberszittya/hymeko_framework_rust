---
title: BALLTIP_INTERARM_FILTERED_V1 — spherical-tip robot variant + inter-arm collision filtering (matched-panel regression + exploit audit)
date: 2026-07-24
branch: feat/balltip-interarm-filtered-v1
baseline: executable-hymeko-option-rl-v1 @ 772a11a4
status: FILTERING_ADDS_ONLY_EXPLOIT_DELIVERIES / DO_NOT_DEFAULT_FILTERED_BALLTIP (provisional, first-pass)
correction: 2026-07-24 — gate-contamination bug fixed (see "Correction" below); ball-variant numbers superseded
---

# BALLTIP_INTERARM_FILTERED_V1 (2026-07-24)

A NEW HyMeKo-declared robot/env variant — **spherical fingertips + simulator-level inter-arm collision filtering** —
introduced WITHOUT modifying the frozen canonical baseline, then measured against it on a matched fixed panel with an
adversarial exploit audit. The frozen deploy stack (`executable-hymeko-option-rl-v1` @ `772a11a4`) is untouched.

## Correction (2026-07-24)
The first pass of this regression passed the reconstructed engagement `gate` **by reference** into the per-variant
committed rollout (`_clearance_trace`), and `structured_carry_rollout` **mutates** the gate (`s,_coin,_ltip,_rtip,
disarm_count,steps`). So every variant after the first ran on a gate contaminated by the preceding variant's rollout —
the searches were clean (they deep-copy internally) but the committed metrics were not. The bug surfaced when the Stage-B1
support decomposition showed a 192-shot search finding *fewer* handoffs than its own 64-shot prefix (impossible). Fixed by
deep-copying the gate per committed rollout. The **canonical clamp (variant 1) is unchanged** (it always ran first on a
clean gate) — evidence the fix is correct — while the ball-variant numbers moved. All numbers below are POST-fix.

## Verdicts (provisional — first-pass, 24 states, 1 search seed, frozen clamp-tuned controller)
- **`FILTERING_ADDS_ONLY_EXPLOIT_DELIVERIES`** — the honest collision-on ball solves {3,4,10,11}; the filtered ball solves
  {0,3,4,10,11,14,15}. The **legitimate** filtered solved set (those that survive a collision-on replay) is *identical* to
  collision-on {3,4,10,11}; the three extra K6 ({0,14,15}) are ALL pass-through exploits. Filtering's apparent +3 K6 is
  entirely arms-through-arms.
- **`FILTERING_CAUSES_LOAD_BEARING_OVERLAP_EXPLOIT`** — 4 filtered states (0,1,14,15) are blocked when the same committed
  option is replayed with collision re-enabled (state 1 = handoff-only; 0,14,15 = full K6 via pass-through).
- **`DO_NOT_DEFAULT_FILTERED_BALLTIP`** — §8 decision: do NOT promote the filtered ball. The honest **collision-on** ball
  (4/24) is the right variant and is close to the clamp (5/24); filtering only inflates the score with cheats.
- **NOT** `NO_MEASURABLE_MECHANICAL_EFFECT` and **NOT** `FILTERING_REMOVES_SIMULATOR_ARTIFACTS`.

**Confound stated in writing (§3 evaluation-metric integrity).** The controller is **clamp-tuned** (proposal + b=8 search
fit on the E0 clamp). Applying it unchanged to the ball measures *the frozen controller's robustness to a geometry
change*, not the ball's **intrinsic** capability — so 4/24 (collision-on ball) is a LOWER BOUND. Stage B1
(`coin_balltip_b1_capability.py`) decomposes the ball's capability under stronger controllers to separate transfer
failure from an embodiment wall; this report does not claim "spherical tips cannot deliver."

## What is HyMeKo-declared (§1+§2)
`data/robotics/galambos_planar_balltip_v1.hymeko` selects the whole variant by choosing a robot spec:
- **Spherical fingertips** r 0.014 → **0.020** (collision = visual, one geom). The golden `galambos_inertia` block is
  **preserved unchanged** — only the tip contact/visual radius changes; mass/inertia are identical to the frozen arm.
- **Inter-arm filtering via MuJoCo collision groups** (not event-ignoring): left `contype 4`, right `contype 8`, both
  `conaffinity 3`; coin `2/2`, floor `1/3`. `(4&3)|(8&3)=0` ⇒ left↔right **disabled**; arm↔coin, arm↔floor, coin↔floor
  all remain **enabled**. `collision_manifest()` records the derived field, the exact **9 filtered body pairs**, and masks.

## §5 — 4-way matched-panel regression (`reports/2026-07-24-balltip-regression/regression.{json,png}`)
Deployed controller `θ_center (proposal) → fixed b=8 search_select → committed push/brake/release → frozen settling
pi_0`, **identical** across variants (initial state via faithful transplant, env seed, search seed 9000+i, θ_center,
b=8, certificate, horizon 160 — only the robot differs). 24 held-out states (seeds 14000–15200).

| variant | fingertip | inter-arm | K6 | handoff | solved set | §6 exploit |
|---|---|---|---:|---:|---|---:|
| canonical_clamp | concave clamp r0.012 | collide | **5/24** | 7 | {0,11,15,18,23} | 0 |
| point_sphere | sphere r0.014 | collide | **2/24** | 3 | {3,17} | 0 |
| **balltip_nofilter** (**COLLISION_ON**) | ball r0.020 | collide | **4/24** | 4 | {3,4,10,11} | 0 |
| balltip_filtered | ball r0.020 | **filtered** | **7/24** | 8 | {0,3,4,10,11,14,15} | **4** |

- The honest **collision-on ball** delivers **4/24**, near the clamp's 5/24 — the clamp is *not* dramatically superior to
  a ball under the (clamp-tuned) frozen controller. The tiny sphere (r0.014) is the weak one at 2/24.
- **Filtering inflates K6 to 7/24, but the gain is all exploit.** Its legitimate solved set = collision-on's {3,4,10,11};
  the extra {0,14,15} are pass-through.

## §6 — exploit audit (discriminating test)
Raw `mj_geomDistance < 0` is artefact-prone (every variant, incl. collision-enabled, shows it — the query catches the two
tips hugging the coin). The clean test: **replay the filtered committed θ with collision RE-ENABLED** (same ball, same θ —
isolates only the masks). Blocked ⇒ the win relied on pass-through. Result: exploit at states **0,1,14,15** (nofilter-replay
K6/handoff → 0). So 4 of the filtered variant's 8 handoffs (and 3 of its 7 K6) vanish under honest collision.

## §7 — deterministic 4-way video (`reports/2026-07-24-balltip-regression/video/`)
Side-by-side rollout of all four robots from an identical start, live min-inter-arm-clearance HUD (physics/timestep
unchanged). MP4 + GIF + provenance JSON per scene. Scenes chosen post-correction:
- a **legitimate** clamp delivery + a **filtered pass-through exploit** in the same state (only the filtered panel shows
  "DELIVERED" beside "OVERLAP!", while the collision-on ball — same geometry — does not).
- a state where the **honest collision-on ball** delivers legitimately (the ball embodiment working without cheats).

## Files touched
- **NEW** `data/robotics/galambos_planar_balltip_v1.hymeko` — *committed f4b2af97*
- **NEW** `hymeko_rl/coin_delivery/coin_robot_variant.py` (PANEL_VARIANTS, build_variant_rl, transplant_handoff, manifest,
  clearance, interarm_contact_count)
- **NEW** `hymeko_rl/tests/test_coin_robot_variant.py` (8 tests)
- **NEW** `experiments/…/coin_balltip_regression.py` (§5+§6, gate-deepcopy fixed), `coin_balltip_video.py` (§7),
  `coin_balltip_b1_capability.py` (Stage B1)
- **EDIT (additive, default None ⇒ canonical unchanged)** `env_factory.py`, `coin_neutral_start.py`, `coin_rl_env.py`
  (thread `arm_mjcf_transform` + `geom`); `coin_carry_structured.py` (+`structured_random_best_with_support`, delegating).
- **CORE.YAML items touched:** none. Frozen baseline untouched.

## Test results
- `test_coin_robot_variant.py` 8 passed; `test_coin_carry_option{,_rl}.py` 19 passed (threading unregressed);
  `test_coin_carry_option.py` 11 passed after the `structured_random_best` delegation refactor.

## §8 — decision & follow-up
**Adopt the physically-honest collision-on ball (BALLTIP_COLLISION_ON_V1) as the ball embodiment; do NOT filter.** Filtering
only adds pass-through exploits. The honest ball is 4/24 under the frozen clamp controller — but that controller is
clamp-tuned, so Stage B1 measures the ball's capability under stronger controllers before deciding whether the ball needs a
refit proposal (Case A), a ball settling skill (Case B), an action-language change (Case C), or is already solvable (Case D).
