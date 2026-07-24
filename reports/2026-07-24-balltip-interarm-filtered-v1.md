---
title: BALLTIP_INTERARM_FILTERED_V1 — spherical-tip robot variant + inter-arm collision filtering (matched-panel regression + exploit audit)
date: 2026-07-24
branch: feat/balltip-interarm-filtered-v1
baseline: executable-hymeko-option-rl-v1 @ 772a11a4
status: FILTERING_CAUSES_LOAD_BEARING_OVERLAP_EXPLOIT / DO_NOT_DEFAULT_FILTERED_BALLTIP (provisional, first-pass)
---

# BALLTIP_INTERARM_FILTERED_V1 (2026-07-24)

A NEW HyMeKo-declared robot/env variant — **spherical fingertips + simulator-level inter-arm collision filtering** —
introduced WITHOUT modifying the frozen canonical baseline, then measured against it on a matched fixed panel with an
adversarial exploit audit. The frozen deploy stack (`executable-hymeko-option-rl-v1` @ `772a11a4`) is untouched.

## Verdicts (provisional — first-pass, 24 states, 1 search seed, frozen clamp-tuned controller)
- **`FILTERING_CAUSES_LOAD_BEARING_OVERLAP_EXPLOIT`** — the filtered variant's *single* K6 delivery (state 12) is
  **blocked** when the same option is replayed with collision re-enabled: its arms pass 0.49 cm through each other. The
  discriminating collision-on replay confirms the win relied on the removed contact, not on mechanics.
- **`FILTERING_DOES_NOT_IMPROVE_DELIVERY`** — filtered **1/24** < ball-nofilter **3/24** under the identical controller.
  Removing inter-arm collision *reduces* delivery and its only success is the pass-through above.
- **`DO_NOT_DEFAULT_FILTERED_BALLTIP`** — the §8 decision: do **not** promote the filtered ball to the default robot for
  `OBJECT_TO_TARGET_VARIANTS_V1`. The E0 concave clamp stays the deployed robot.
- **NOT** `NO_MEASURABLE_MECHANICAL_EFFECT` (there is an effect: filtering lowers delivery + enables an exploit) and
  **NOT** `FILTERING_REMOVES_SIMULATOR_ARTIFACTS` (the collision-on ball was not snagging at the decisive states — it
  simply failed to deliver by legitimate means; there was no real artifact to remove).

**Confound stated in writing (§3 evaluation-metric integrity).** The controller is **clamp-tuned** (the proposal +
b=8 search were fit on the E0 clamp). Applying it unchanged to spheres/balls measures *the frozen controller's
robustness to a geometry change*, not the ball robot's **intrinsic** capability. So the sphere 0/24 and ball 3/24 are
**lower bounds**, not ceilings — a ball-tuned controller could do better. This report does **not** claim "spherical tips
cannot deliver"; it reports what the *deployed* controller does on each robot, plus the filtering-specific exploit,
which is controller-independent (it is a physics fact about the removed contact).

## What is HyMeKo-declared (§1+§2)
`data/robotics/galambos_planar_balltip_v1.hymeko` selects the whole variant by choosing a robot spec:
- **Spherical fingertips** r 0.014 → **0.020** (collision = visual, one geom). The golden `galambos_inertia` block is
  **preserved unchanged** — only the tip contact/visual radius changes; mass/inertia are identical to the frozen arm.
- **Inter-arm filtering via MuJoCo collision groups** (not event-ignoring): left `contype 4`, right `contype 8`, both
  `conaffinity 3`; coin `2/2`, floor `1/3`. `(4&3)|(8&3)=0` ⇒ left↔right **disabled**; arm↔coin, arm↔floor, coin↔floor
  all remain **enabled**. The `collision_manifest()` query records the derived `inter_arm_collision` field, the exact **9
  filtered body pairs** (3 left links × 3 right links), and the per-side group masks.

## §5 — 4-way matched-panel regression (`reports/2026-07-24-balltip-regression/regression.{json,png}`)
Deployed controller `θ_center (proposal) → fixed b=8 search_select → committed push/brake/release → frozen settling
pi_0`, **identical** across variants: initial state (canonical E0 handoff, transplanted by the shared nq=7 layout —
verified faithful, dtz Δ≈6e-5, obs Δ≈8e-4), env seed, search seed (9000+i), θ_center (from the canonical obs), b=8,
certificate, horizon 160. The only difference is the robot. 24 held-out states (seeds 14000–15200).

| variant | fingertip | inter-arm | K6 | handoff | mean dwell | mean min clr (m) | §6 exploit |
|---|---|---|---:|---:|---:|---:|---:|
| canonical_clamp | concave clamp r0.012 | collide | **5/24** | 7 | 1.375 | −0.024 | 0 |
| point_sphere | sphere r0.014 | collide | **0/24** | 1 | 0.125 | −0.025 | 0 |
| balltip_nofilter | ball r0.020 | collide | **3/24** | 3 | 0.750 | −0.037 | 0 |
| balltip_filtered | ball r0.020 | **filtered** | **1/24** | 1 | 0.250 | −0.032 | **1** |

- **Fingertip type dominates**: clamp 5 > ball 3 > sphere 0. The concave clamp cups the coin; a small sphere fails
  outright; a bigger ball partially recovers (more contact area). Solved sets are **disjoint** — clamp {0,11,15,18,23},
  ball-nofilter {4,13,16}, filtered {12} — each geometry solves *different* states (consistent with the clamp-tuned
  confound and with genuine geometry differences), not a nested subset.
- **Filtering slightly hurts**: 3/24 → 1/24. Removing the inter-arm contact removes a stabilising brace, and the one
  remaining "success" is the exploit.

## §6 — exploit audit (discriminating test)
Raw `mj_geomDistance < 0` (min inter-arm clearance) is **artefact-prone**: every variant, *including collision-enabled
ones*, shows negative clearance (nofilter −0.037 is the deepest) because the query catches the two fingertips hugging
the coin from both sides and ignores the coin between them. A negative *number* is therefore **not** an exploit.

The clean test: **replay the filtered variant's committed θ with collision RE-ENABLED** (same ball geometry, same θ —
isolates only the inter-arm masks). If the outcome is blocked, the win relied on pass-through.
- **State 12 (seed 14005, contact_retention)**: filtered K6 1 / handoff 1; collision-on replay → **K6 0 / handoff 0**;
  clr_gap +0.00373 (the filtered arms went 3.7 mm deeper than collision allows). **Exploit confirmed.**
- No other variant can exploit (they already collide). So the filtered variant's entire delivery advantage is one
  physically-implausible pass-through.

## §7 — deterministic 4-way video (`reports/2026-07-24-balltip-regression/video/`)
Side-by-side rollout of all four robots from the identical start, live min-inter-arm-clearance HUD (physics/timestep
unchanged; same non-behavioral `frame_hook` as the regression). MP4 + GIF + provenance JSON per scene.
- `balltip_4way_state0_seed14000` — a **legitimate** clamp delivery: clamp K6, all clearances positive.
- `balltip_4way_state12_seed14005` — the **exploit**: only `balltip_filtered` shows "DELIVERED (K6)" and simultaneously
  "inter-arm clr: −0.5 cm OVERLAP!", while the collision-on ball (same geometry) fails at +1.3 cm. The cheat is watchable.

## Files touched
- **NEW** `data/robotics/galambos_planar_balltip_v1.hymeko` (§1+§2, robot variant spec) — *committed f4b2af97*
- **NEW** `hymeko_rl/coin_delivery/coin_robot_variant.py` (+PANEL_VARIANTS, build_variant_rl, transplant_handoff, manifest, clearance)
- **NEW** `hymeko_rl/tests/test_coin_robot_variant.py` (8 tests: manifest/masks/radii/clearance + panel-build + transplant)
- **NEW** `experiments/2026_07_22_coin_v3_learning/rl_entry/coin_balltip_regression.py` (§5+§6)
- **NEW** `experiments/2026_07_22_coin_v3_learning/rl_entry/coin_balltip_video.py` (§7)
- **EDIT (additive, default None ⇒ canonical unchanged)** `hymeko_rl/coin_delivery/env_factory.py`,
  `hymeko_rl/experiments/coin_neutral_start.py`, `hymeko_rl/coin_delivery/coin_rl_env.py` — thread `arm_mjcf_transform`
  (+ `geom`) so a robot variant supplies its whole arm MJCF.
- **CORE.YAML items touched:** none. Frozen baseline (`772a11a4`) untouched (matched-panel canonical = the exact E0 robot).

## Test results
- `test_coin_robot_variant.py` — 8 passed (0.83 s). `test_coin_carry_option{,_rl}.py` — 19 passed (2.73 s), confirming
  the additive threading does not regress the canonical `geom=None` E0 path.
- §5 regression + §6 exploit determinism: each committed θ re-run bit-reproduces its K6/handoff (asserted in-loop).

## §8 — decision & follow-up
**Keep the E0 clamp as the deployed robot; do not adopt inter-arm filtering.** Filtering yields no legitimate delivery
gain and one pass-through exploit. If a future task *requires* filtering (e.g. a genuinely self-colliding embodiment),
a **proximity constraint** (min-clearance penalty or a hard qacc/penetration guard inside the certificate) would be
required to bar the pass-through — but that is out of scope here because filtering is not beneficial.

Open (deferred): the ball robot's **intrinsic** capability is unmeasured — a ball-tuned controller (re-fit proposal +
search on the ball) would separate "clamp is load-bearing" from "clamp-controller does not transfer." Recommend that as
the first step *if* a ball tip is ever wanted for `OBJECT_TO_TARGET_VARIANTS_V1`; the clamp remains the default until then.
