# Report — CORE limits extraction: the emitted arm now carries its declared joint limits

**Date:** 2026-06-19
**Author:** Aiko (agent), for Dr. Csaba Hajdu
**Status:** ✅ **CORE limits fix landed** (faithful: emitted MJCF/URDF/SDF carry the declared
joint range + effort + velocity). ⏸ **The torque-BC architecture comparison is NOT delivered** —
it additionally needs realistic masses (ripples to 6 fixtures) and **action normalization** (a
broad action-interface change), a larger effort than this plan scoped. Those were reverted to
keep the limits fix clean; flagged below for a decision.

**CORE edit token:** `APPROVED-CORE-EDIT: extract-joint-limits-ref` (granted).

## What landed (the faithful limits fix)
- **CORE — `hymeko_query/src/kinematics/kinematic.rs`:** `extract_joint_limits` now follows the
  joint's `limit -> …` reference (declared *or inherited*) and reads `lower/upper/effort/velocity`
  off the target node. Previously it only read direct `limit_*` children, so the arm's
  `limit -> joint0_limit` (and the inherited `limit -> joint_rev_limit`) were never resolved →
  joints emitted unlimited, actuators unbounded. New helper `find_inherited_ref_target` walks own
  children then bases (cycle-bounded), mirroring the axis-resolution pattern. **Additive** — the
  direct-child path is unchanged; the ref-follow is a fallback. clippy clean; my edit fmt-clean.
- **Non-CORE — `hymeko_formats/src/transforms.rs`:** `emit_mjcf` motors now emit
  `ctrlrange="-effort effort"` when the joint has a limit (joint `range` was already emitted from
  the limits, now populated).
- **Result (model-path emit):** `emit -f mjcf` → `<motor ctrlrange="-500 500">` (j0/j1) and
  `"-50 50"` (j2–jtool, inherited); `<joint range="-3.1416 3.1416">`. `emit -f urdf`/`sdf` →
  `<limit lower="-180" upper="180" effort="500|50" velocity="4|1">`. The arm faithfully carries
  the limits its `.hymeko` declares. (The `compile` *template* path still shows the static
  `effort=100` placeholder — pre-existing, separate from the model emitters.)

## Tests
- **`hymeko_query` integration: 212 passed, 0 failed** — including the new regression
  `per_joint_limits_resolved_through_the_limit_ref` (j0/j1 effort 500, j2–jtool effort 50, all
  ±180°, via the ref). The limits CORE edit rippled to **zero** existing tests.
- **`hymeko_rl`:** the emitted-arm tests now assert the declared ctrlrange (±500/±50) instead of
  the env fallback; `test_arm_reach_from_hymeko` + `test_reach_arch_compare` green. ruff clean.

## Why the comparison is not delivered (the honest finding)
The plan's goal was a learnable torque-control arm for the HSiKAN-vs-MLP comparison. **Four**
measured layers stood in the way. The limits fix clears one; the user approved the full effort
("A"), so #2 and #3 were *implemented and measured*, which uncovered #4 — a fundamental one.
1. **Limits/ctrlrange** — *fixed* (above). With ±500/±50 bounds (vs the ±25 fallback), expert
   saturation dropped from 80 % toward manageable.
2. **Saturation / masses** — *measured fix.* The placeholder link masses (2–25 kg) make the
   computed-torque demand ~2600 N·m. Realistic Moveo-scale masses (~0.5–2 kg) + a tuned expert
   `kp` (sweep: kp=1500 → 7 % saturation, still reaching) fix it. Ripples to a sibling fixture
   (`anthropomorphic_arm_using.hymeko`) + 5 value-asserting tests.
3. **Action scale** — *measured fix.* Implemented **action normalization** (Box(±1) action space;
   the expert returns physical torque ÷ ctrlrange, `step` rescales). This **worked**: BC loss
   collapsed from **~18 000 to 0.003** — the network now fits the demonstrations exactly. This is
   the standard, correct fix and a real env improvement.
4. **BC compounding error** — *the wall.* Even with a near-perfect fit (loss 0.003), the cloned
   policy does **not reach**: hsikan 0.703 / mlp 0.714 m vs the expert's 0.058 m (floor ≈ 0.75).
   Per-step torque errors of a few % compound over 80 steps on the redundant 6-DOF arm. This is a
   **fundamental limitation of behaviour cloning** on hard control — not a bug. The remedy is an
   on-policy learner (PPO, robust to compounding via the reward) or DAgger, **not** more BC.

So #2 + #3 were done and validated (saturation + fitting both solved). #4 is the real blocker: a
clean architecture comparison on the canonical **6-DOF torque** arm needs PPO/DAgger — a research
effort, not a bounded task (and PPO on torque control is itself known-hard; the existing PPO test
deliberately uses position control). The #2/#3 changes were **reverted** to keep the tree clean
(the limits fix stands); they are documented here to re-apply when the PPO effort is scoped.

## Files touched
| File | Change |
|---|---|
| `hymeko_query/src/kinematics/kinematic.rs` | **CORE** — `extract_joint_limits` ref-follow + `find_inherited_ref_target` (+61/−4) |
| `hymeko_formats/src/transforms.rs` | `emit_mjcf` motor `ctrlrange` from effort |
| `hymeko_query/tests/test_anthropomorphic_generation.rs` | new `per_joint_limits…` regression |
| `hymeko_rl/tests/test_arm_reach_from_hymeko.py` | emitted-arm asserts declared ±500/±50 |
| `hymeko_rl/reach_arch_compare.py`, `bc.py` | the HSiKAN-vs-MLP harness (from the prior task) |

**CORE.YAML:** one item (`extract_joint_limits`), under the approved token. The change is additive
and backward-compatible; the legacy direct-child path is preserved.

## Decision for the user
The limits fix is a clean, faithful framework win on its own (URDF/SDF/MJCF now carry declared
joint limits), and the "A" attempt produced a clear, useful negative result: **BC cannot deliver
the comparison on the canonical 6-DOF torque arm** (it compounds). The comparison **result** needs
a different learner or an easier task:
1. **PPO on the canonical arm** — the on-policy path (robust to compounding). Requires re-applying
   the #2/#3 infra (masses + normalization + fixtures) and a `run_ppo` env-factory + tuning. PPO on
   *torque* is itself known-hard (the existing PPO test uses position) → a multi-step research
   effort, not a session task.
2. **arm_world comparison now** *(recommended for a result)* — the 4-DOF arm BC works (HSiKAN reach
   0.369 < floor 0.433); run the 5-seed HSiKAN-vs-MLP comparison there. Real architecture data
   point this session; canonical-torque is the follow-up research.
3. **Position-control canonical arm** — emit `<position>` actuators (BC compounds far less under
   position control); gives a canonical-arm result without the torque difficulty. Needs emitter
   actuator-mode support.
4. **Stop here** — bank the limits fix; defer the comparison.

## Provenance
- Plan: `docs/plans/2026-06-19-emitted-arm-physics/` (4 artifacts, pdf built).
- Platform: Windows 11, MuJoCo 3.9.0, Python 3.12. CLI built from `7d16ad0` + non-CORE emit fixes.
- Measured: expert raw torque median 2608 / max 15154 N·m (heavy masses); kp sweep 3000→200
  (saturation 24 %→0 %, reach degrades below kp≈800); BC loss ~18 000 at ±500 action scale.
