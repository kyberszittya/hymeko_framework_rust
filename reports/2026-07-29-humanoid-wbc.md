# Humanoid whole-body controller — the walking action-space change

**Date:** 2026-07-29
**Worktree:** `hymeko_humanoid` (branch `research/humanoid-com-lyapunov`)
**Context:** the walking-feasibility study (`reports/2026-07-29-humanoid-walking-feasibility.md`) showed the
balance-residual action space cannot walk (the lateral CoM is pinned in double support). User approved the
action-space change. This report delivers a **whole-body controller (WBC)** and validates the two
capabilities a gait is built from; the full gait itself is an in-progress prototype.

## Summary

New module `scenarios/humanoid/wbc.py` — a **contact-consistent task-space inverse-dynamics** controller.
Each tick it solves, via a KKT linear system (**no QP-solver dependency**), for the joint accelerations +
stance-foot contact wrenches that best track weighted task accelerations (CoM, a body's Cartesian pose,
posture), subject to the floating-base dynamics and the stance-foot no-acceleration constraint, then
recovers the actuated torques by inverse dynamics. A cost on a contact's wrench **unloads** that foot — the
load-transfer primitive walking needs.

**Validated (tested):**
- **Stable standing** — both-foot contact, CoM + pelvis-orientation + posture tasks: holds upright with
  steady pelvis height (`std < 0.01 m`) over 600 steps.
- **Load transfer** — a wrench cost on the swing foot drives its contact load from ~1240 N (no cost) to
  **< 120 N** (min ~20 N) while balancing — the double-support unloading that every quasi-static controller
  in the feasibility study could not achieve.

**In progress (prototype, NOT shipped as a working feature):** a walking FSM on top of the WBC
(DS load-transfer → SS swing/lift → alternate) completes ~4–6 steps but is **not yet a stable indefinite
gait** — it loses balance after several steps. Remaining work: a DCM/capture-point CoM reference (instead
of a static stance-foot CoM target), capture-point swing-foot placement to arrest the drift, and smoother
DS↔SS contact-schedule transitions. Kept as a scratch prototype, not in the shipped module.

## Formulation

    variables  x = [q̈ (nv); λ (6 per stance contact)]
    minimise   Σ wᵢ ‖Jᵢ q̈ − aᵢ‖²  +  Σ wf ‖λ_c‖²                  (soft tasks + force cost)
    s.t.       M[base] q̈ − Jc[base]ᵀ λ = −h[base]                 (unactuated floating-base dynamics)
               Jc q̈ = 0                                           (stance contact no-acceleration)
    then       τ = M[act] q̈ + h[act] − Jc[act]ᵀ λ

`M` via `mj_fullM`, `h = qfrc_bias`, Jacobians via `mj_jacSubtreeCom`/`mj_jacBody`. Friction cones and
torque limits are handled by clipping (adequate on flat ground); lifting them into inequality constraints
would need a QP solver (a dependency = core change, not taken).

## Files touched

- `scenarios/humanoid/wbc.py` — NEW, ~110 lines. `Task` dataclass + `WholeBodyController` (`solve` KKT,
  `com_jacobian`, `body_jacobian`, `orientation_error`, `posture_task`).
- `tests/test_humanoid_wbc.py` — NEW, 4 tests (finite torques, stable stand, orientation-error zero,
  force-cost load transfer with a regression assertion vs the no-cost baseline).

**CORE.YAML items touched:** none. **New dependencies:** none (numpy + mujoco only).

## Test results

- `pytest tests/test_humanoid_wbc.py` → **4 passed**. Full humanoid suite → **43 passed** (39 prior + 4).
- `ruff check scenarios/humanoid/wbc.py tests/test_humanoid_wbc.py` → clean.

## §6.5 anti-patterns

None. The WBC is a single class with a clean task abstraction (no Cartesian-product API, no globals, no
string-typed config).

## DCM walking pattern generator on the WBC (multi-step dynamic gait)

Built a **Divergent-Component-of-Motion (DCM) walking controller** on the WBC (Englsberger lineage):
ω = √(g/z); ξ = CoM + ĊoM/ω; an offline footstep plan (alternating stance ZMP for marching-in-place); a
**backward-recursion DCM reference**; and the DCM tracking law
``r_zmp_cmd = ξ_ref − ξ̇_ref/ω + (1 + k/ω)(ξ − ξ_ref)`` clipped to the stance foot, converted to the CoM
task ``ẍ_com = ω²(CoM − r_zmp_cmd)`` fed to the WBC, with a DS load-transfer phase + a swing-foot Cartesian
lift per step.

**Result — genuine multi-step dynamic walking, marginally stable:**
- The DCM feedback **solves the balance instability** the hand-tuned controllers had (fall-mode was sagittal
  pitch): the gait now stays upright through **16–57 steps** depending on tuning (vs falling in ~4–9 before).
- **Not yet indefinitely stable.** The binding limit is measured and honest: with the robot's **small feet**
  (support half 0.05 m) the ZMP saturates at the foot edge, so it cannot fully regulate the DCM; the CoM
  amplitude slowly grows (resonant) and it eventually falls (~40–57 steps at the best tuning). Larger,
  more visible foot lifts destabilise sooner (~15 steps). A **true indefinite gait needs capture-point
  footstep adaptation** — adjusting *where/when* the next foot lands from the measured DCM — which
  fixed-footstep marching lacks. That adaptation layer is the remaining work.
- Video: `humanoid_dcm_walk.mp4` — 12 steps of the dynamic gait (rock + single-support swing), upright
  throughout the clip.

This walking controller is kept as a **documented prototype** (the scratch driver), **not shipped as a
"stable walk" module** — it walks many steps but is not a certified indefinite limit cycle, and shipping it
as stable would over-claim. The WBC core + load-transfer primitive (this report's tested deliverable) are
what walking is built on and are validated.

## Open items / follow-ups

- **Capture-point footstep adaptation** (adapt the next foothold to the measured DCM) — the last layer for
  an indefinitely-stable gait; the robot's small feet make fixed-footstep marching only marginally stable.
- The WBC core + the load-transfer primitive it needs are in place and tested; the DCM layer above it
  already gives multi-step dynamic walking.

## Provenance

- Git: working tree adds `scenarios/humanoid/wbc.py`, `tests/test_humanoid_wbc.py`, this report. Seed 0,
  deterministic. Shared venv `hymeko_framework_rust/.venv`; mujoco 3.10.0; macOS 25.5.0.
