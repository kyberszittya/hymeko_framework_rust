# FANUC pick-place — versioned clean expert (v2): baseline + attempt + reach-envelope blocker

**Date:** 2026-07-06 22:49 JST · **Plan:** `docs/plans/2026-07-06-pick-place-clean-expert/` (tex/pdf/tikz/mmd) ·
**Complements** the forensics report `reports/2026-07-06-pick-place-clearance-forensics.md` (v1-dirty diagnosis).
**Status: v2 does NOT pass the clearance gate — do NOT regenerate demos / start BC/DAgger.**

## Summary

Built the clearance measurement harness and a **versioned** expert (`PickPlaceEnv(expert_version=1|2)`, scene /
object / reward unchanged; v1 kept byte-identical for the dirty baseline). Measured the old (v1) expert as the
contact-exploiting baseline, then implemented and iterated a v2 clearance-oriented trajectory. **v2 improves
clearance but breaks the grasp**, and the measurements pin *why*: the object radius (0.28–0.40 m) is near the
arm's reach edge, so a clean top-down approach is geometrically blocked. This is a modeling/scene constraint,
not a tuning miss — reporting it and halting per §11 rather than shipping a non-grasping expert.

## Old expert (v1) — contact-exploiting baseline (measured, N=8, seed0=0)

| lift | place | approach finger↔table | approach gripper↔table | approach min clearance |
|---|---|---|---|---|
| **0.75** | **0.75** | **47.2%** | 47.2% | **−0.0256 m** (fingers 2.6 cm *through* the table) |

Corroborates the frozen forensics (32/32 negative clearance ~2.6 cm, first table strike far before the grasp).
The gripper↔table contact **is** the fingers (tool/palm add nothing). The transit is clean (0% table contact,
min clearance ≈ 0). **The 0.75/0.875 ceiling belongs to this dirty expert and does NOT carry over.**

## Corrected expert (v2) — attempted, blocked (measured, N=8)

| variant | lift | place | approach finger↔table | approach min clearance | why it fails |
|---|---|---|---|---|---|
| v2a high hover+2-stage | 0.0 | 0.0 | 0.0% | 0.0 | high hover unreachable over far object → never centres |
| v2b reachable hover + rise-first | 0.0 | 0.0 | 65.7% | −0.004 | rise-first commands an unreachable straight-up at the far xy → tool pinned at the table |
| v2c low hover + damped descent | 0.0 | 0.0 | 64.1% | −0.005 | reachable altitude over the far object is *low* → gripper grazes the table while centring |

**Root cause (decisive measurement).** At the object radius (this seed 0.356 m), the tool's max reachable
top-down height is ≈ grasp_z (0.255 m); a hover 0.14 m above (0.395 m) is UNREACHABLE — the DLS IK folds the
arm low and the tool sinks to the table (z 0.12). v1 "works" precisely by *pushing through* the table contact on
the diagonal reach (it dips to z 0.22 then recovers and grasps at ~step 325). Every clean-approach v2 either
cannot centre (unreachable high hover) or centres low and grazes the table. The finger tips at the grasp are
inherently ~1 cm above the table (the box is only 0.04 m tall), so the grasp region itself is near the surface.

## Update — lateral/staged attempt (user chose "different approach strategy", 2026-07-06 22:55)

Tried an **envelope-following lateral descend-while-extending** approach (target the reachable endpoint
`(obj, grasp_z)` directly, slow rate, no high hover). Result (N=8): approach **finger↔table 10.3%** (down from
47%) but **gripper↔table 89%** (the tool/palm assembly rides low near the table at full extension), min
clearance −0.006, **lift 0.0** (still no grasp). Better on the finger axis, still fails the gate.

**Why it still fails + concrete next step (needs a focused session, not blind iteration):** at the far radius
the *whole* gripper is low, so per-geom clearance is negative somewhere along the extended reach, and the slow
approach doesn't reliably centre+close. The right method is to **compute the reach envelope explicitly** (sample
the top-down reachable height vs radius from the actual IK), then parametrise the approach waypoints to ride the
envelope top (height = f(radius), always ≥ tool-clearance above the table), ending over the object — rather than
guessing hover heights. This is a bounded, measurable design task for a dedicated session; the harness + gate
+ versioned scaffold are all in place to drive it. **Not continued blindly here to avoid a wild-goose chase.**

## Reach-envelope measurement (2026-07-06, next-session task done) + IK-fold root cause

Measured the IK reach envelope (`scratchpad/reach_envelope.py`; grid over radius × top-down target height at
azimuth 0, solving the down-orientation DLS IK from home, recording pose error / achieved height / fingertip
height / finger↔table & gripper↔table clearance / collision). Result (`reports/figures/pick_place_clean_expert/
reach_envelope.{json,png}`):

- **`z_max(r)` is ≈ 0.30–0.35 m across the whole object-radius band (0.28–0.40)** — *above* grasp_z (0.255) and
  well above the table (0.12). At r=0.35, a top-down hover at **z=0.30 is IK-reachable with 5.9 cm finger
  clearance**. **This REVERSES the earlier "geometrically blocked" conclusion:** a clean top-down hover over the
  far object IS reachable; v1's dip came from targeting hover_z=0.395 — *above* z_max(0.33), so the IK folded.
- Proposed `approach_z(r) = z_max(r) − margin ≈ 0.30`, and the v2 hover was set to grasp_z+0.06 (z≈0.315,
  IK-valid; +0.10/+0.14 self-collide).

**But the envelope-based v2 STILL fails the harness** (N=8): lift/place **0.0**, approach finger↔table **65%**,
gripper↔table **90%**, min clearance −0.005. **Discriminating instrument (confirmed root cause):** the tool
starts high (tool_z 0.387) then **folds to the table (0.137) and sticks** (horiz frozen at 0.22, never centres).
So the static envelope reachability (IK from *home* reaches (obj, 0.315)) does **not** translate to the
closed-loop trajectory: the expert's IK solves from the **drifting `q_now`** (120 iters + rate-limit 0.22) and
descends into a **folded local minimum** it cannot climb out of. **The blocker is IK SEEDING / the closed-loop
path — not the reach envelope and not the object radius.**

**Concrete next fix (path/IK level, not another height):** seed the IK from a GOOD configuration each step
(re-solve from the home / a high-elbow posture, or blend toward it) instead of the drifting `q_now`; or plan a
short Cartesian path that stays on the envelope (home-altitude → over-object at approach_z → straight down). The
harness + envelope + `z_max(r)` are all in place to validate it. Do NOT change heights again — the height is
solved; the seed/path is the open variable.

## Recorded result (FROZEN 2026-07-06, user-confirmed) — v2 does NOT pass the clearance gate

- The lateral/staged approach was **directionally useful**.
- **finger↔table contact dropped from 47% to 10%.**
- However, **gripper↔table contact remains too high at 89%** (the tool/palm rides low at full extension).
- **The grasp still fails** (lift/place 0.0).
- **Therefore the v2 clean expert does NOT pass the clearance gate.**

**Do not regenerate demos. Do not run BC, DAgger, or RL.** The old **0.75 / 0.875 dirty-expert ceiling stays
versioned separately (`v1_dirty`) and must NOT be used as the clean-v2 baseline.**

## Next session — concrete task (reach-envelope FIRST, not another guessed hover height)

1. **Sample the IK reach envelope** across the workspace, recording per sample: radius / horizontal distance
   from the base; **maximum feasible top-down gripper height**; fingertip height; gripper/palm clearance to the
   table; IK success/failure.
2. **Build `z_max(r)`** (or an equivalent lookup/table) from the samples.
3. **Redesign the approach waypoints from the measured envelope:** approach height `= z_max(r) − margin`;
   enforce finger↔table clearance; enforce gripper↔table clearance; end **directly above the object**.
4. **Re-run only the expert clearance harness** (`scratchpad/pick_clearance.py`): success rate, finger↔table
   contact rate, gripper↔table contact rate, minimum clearance, clearance-over-time plot.

**Acceptance:** the corrected expert must pass the clearance gate BEFORE any new demonstrations or learning.

## Artifacts

- Harness: `scratchpad/pick_clearance.py` (finger/gripper↔table contacts + signed clearance via
  `mj_geomDistance` + lift/place success, per phase; representative per-step series).
- Plots: `reports/figures/pick_place_clean_expert/clearance_v1.png` (v1 penetrates the table),
  `clearance_v2.png` (v2 stays ≥ 0 but does not grasp); `results.json` (both, N=8).
- Code: `hymeko_rl/env/pick_place_env.py` — `expert_version` param, `_ik_step` (shared), `_expert_action_v1`
  (unchanged), `_expert_action_v2` (WIP, marked in-code as not-gate-passing). CORE.YAML: none.

## Provenance

MuJoCo 3.9, single env, no training; seeds 0–7; `fanuc_pick_env()` config (robot fanuc, table_top 0.12,
obj_radius 0.28–0.40, target 0.34/0.0). v1 tests pass (9); ruff clean. No persistent-state mutation.
