# R12.2-A — orientation-aware geometric model: yaw-varied handoff adapter + feasibility gate G_A

**Date:** 2026-08-08 · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher` · **Plan:**
`docs/plans/2026-08-08-r12-2-orientation-aware-geometric/` (4-format, compiles)
**Verdict:** `R12_2A_YAW_VARIED_HANDOFFS_ACHIEVABLE_GA_PASS` — a certified straddle grasp **preserves** a commanded object
yaw (slope 0.99) over a usable range ~[0°, 30°]; R12.2 is well-posed. A2/B are unblocked (user-gated per the plan).

## Why R12.2 exists (from R12.1 / T2)

R12.1 scoped-closed the static ranker. Its T2 closure check found object orientation pinned to ≤1.9° on the
certified-grasp panel — and the cause was concrete: `_home_with_coin` sets only `qpos[4:6]=(x,y)` and never the planar
yaw `qpos[6]` (the `disk_rz` hinge). The benchmark never varies orientation, so a structured model has no orientation
signal to exploit and R12.2 (the rotor/quaternion rung) cannot even be posed. R12.2-A builds the missing capability:
place the object at a *commanded* yaw and ask whether a certified straddle grasp *preserves* it.

## What was built (all new / non-core; frozen pipeline untouched)

- **`hymeko_rl/coin_delivery/r12_orientation.py`** — the non-invasive adapter:
  - `home_with_coin_yaw(rig, coin_xy, yaw)` — mirrors the frozen `_home_with_coin` but also sets `qpos[6]=yaw`; at
    `yaw=0` it is **bit-identical** to the frozen builder (regression-tested), so the yaw=0 pipeline that R11.6C/R11.7
    depend on cannot move.
  - `reach_capture_at_yaw(...)` — the same reach→certified-capture→descriptor chain as `reach_capture_descriptor`,
    reusing every acquisition primitive (`ir_adapter`, `pipeline._do_reach_and_capture`,
    `moving_precapture.is_certified_grasp`, `descriptor`) UNCHANGED — only the home carries the yaw.
  - `object_yaw(snap)` — joint-agnostic world-frame geom yaw (promoted from the T2 probe; same measure, comparable).
- **`hymeko_rl/experiments/r12_2a_yaw_feasibility.py`** — the G_A probe (yaw grid × scenario × seed → certification
  rate + post-grasp yaw).
- **`hymeko_rl/tests/test_r12_2_orientation.py`** — 4 tests: qpos[6] set to yaw; **yaw=0 bit-identical to frozen home**;
  `object_yaw` tracks placement 1:1; non-finite yaw rejected.

Model fact verified against the live rig (not the misleading "6-DoF freejoint" docstring in `fixed_position.py`): the
object is a **3-DOF planar joint** — `disk_x`(slide, qpos4), `disk_y`(slide, qpos5), **`disk_rz`(hinge, qpos6 = yaw)**.
Had the docstring been trusted, `qpos[6]` would have set Z-height, silently corrupting the probe.

## CORE.YAML items touched

**None.** `hymeko_rl/**` is non-core (`on_unknown_path: treat_as_non_core`); acquisition primitives reused unmodified;
no dependency change.

## Test results

Unit/integration (`test_r12_2_orientation.py`): **4 passed** (22.7 s; the mujoco rig build dominates). Includes the
bit-identity regression proving the frozen path is unperturbed. `ruff` clean; `mypy --strict` clean on the adapter.

## G_A feasibility gate

Box (O4-S), yaw grid {0,15,30,45}° × 2 scenarios × 2 seeds (16 handoffs, 8.2 min). `ga_yaw_feasibility.json`,
`ga_probe.log`.

| placement yaw | certification | mean post-grasp yaw |
|---|---|---|
| 0° | 4/4 | 0.25° |
| 15° | 4/4 | 14.84° |
| 30° | 2/4 | 29.85° |
| 45° | 0/4 | — |

Post-grasp yaw spread **29.61°** (gate ≥15°); **slope 0.99** (post-grasp yaw tracks placement ~1:1). **G_A PASS.**

**Interpretation.** Two facts, both useful:
1. **Where the grasp certifies, it preserves orientation** (slope 0.99, post-grasp ≈ placement to <1°). So placing the
   object at a commanded yaw genuinely produces an orientation-varied handoff — the >15° of variation R12.2 needs, vs
   the ≤1.9° the benchmark had.
2. **Certification degrades off-axis** — 4/4 at 0–15°, 2/4 at 30°, 0/4 at 45°. The axis-aligned straddle targets
   (computed from `coin_xy`, blind to yaw) cannot grip the box past ~30°. The usable orientation range is **~[0°,30°]**;
   A2 must stay inside it and log the drop-off (no silent caps). Widening it later is a straddle-target-vs-yaw change,
   a separate lever — not needed to pose R12.2.

## Next (user-gated per the plan)

- **R12.2-A2** — orientation-varying handoff dataset: box, yaw grid in the certified range (e.g. {0,10,20,30}°) ×
  scenarios × seeds, apply the pooled-θ bank → (K6, dtz, safe), **recording yaw per handoff**. This is the
  orientation-varying supervision R12.2 needs; cost ~ the R12.1 dataset scale.
- **R12.2-B** — add `(R,ω)` to the descriptor + an orientation node/edges in the HSiKAN; retrain MLP and task-HSiKAN
  *with* vs *without* orientation; report the interaction `Δ_HSiKAN − Δ_MLP` (CI). Gate on the interaction, not
  absolute score. Now that orientation genuinely varies, a "Δ≈0" would be real evidence, not the underpowered null T2
  would have produced.

## Provenance

Env: Python 3.11.15, mujoco 3.10.0, torch 2.12.0, numpy 2.4.6, macOS (Apple Silicon), `OMP_NUM_THREADS=1`.
Deterministic (seeds 0–1). Adapter `hymeko_rl/coin_delivery/r12_orientation.py`, probe
`hymeko_rl/experiments/r12_2a_yaw_feasibility.py`, tests `hymeko_rl/tests/test_r12_2_orientation.py`. No dependency
change (torch/numpy pinned).
