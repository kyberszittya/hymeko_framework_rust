# Humanoid kinematics → Vukobratović model (added rotations), Lyapunov control kept

**Date:** 2026-07-27 (JST)
**Branch:** `research/humanoid-com-lyapunov` (worktree `hymeko_humanoid`)
**SIMULATION.** · **Verdict: `KINEMATICS_UPGRADED_TO_VUKOBRATOVIC_LYAPUNOV_PRESERVED`.**

---

## Why

Measured finding: the sagittal-only humanoid (all AXIS_Y joints) **cannot take a protective
step** — a foot-lift probe gave +0.00–0.02 m clearance, because unloading a foot needs
lateral weight transfer and there is no frontal-plane DOF. This is the same kinematic
limitation that made the base roll/yaw uncontrollable (LQR audit).

User direction: **keep the Lyapunov criterion** (Tedrake-style regulation — LQR-trees/Drake
control the same way), and make the **kinematics follow the Vukobratović anthropomorphic
model by adding the rotations** — editing the `.hymeko` source.

## CORE + emitter constraint

- **CORE.YAML check (§1):** `data/robotics/humanoid.hymeko` is **not** protected (crates,
  `docs/spec`, `rtl`, pinned deps are; `policies.on_unknown_path: treat_as_non_core`). No
  approval token needed to edit the model.
- **Emitter constraint:** HyMeKo the DSL accepts arbitrarily many axes per joint, but the
  **MJCF emitter maps one `<body>` per joint**, so two joints sharing a child emit a
  duplicate `<body name=…>` (MuJoCo error). Fixing the emitter map lives in the (CORE)
  crate → per §1 the **non-core workaround** is used: express each added rotation as a
  chained 1-DOF joint through a tiny **massless coupler body** at the .hymeko level.

## The change (`humanoid.hymeko`, +26 / −8)

Added the **frontal-plane rotations** of the Vukobratović leg:

- **hip → 2-DOF**: abduction (AXIS_X, via `hip_ab_{l,r}` coupler) + flexion (AXIS_Y, existing)
- **ankle → 2-DOF**: roll/eversion (AXIS_X, via `ank_rl_{l,r}` coupler) + pitch (AXIS_Y, existing)

Result: **13 → 17 revolute joints** (13 sagittal AXIS_Y + 4 frontal AXIS_X), env action dim
**nu 12 → 16**, obs 35 → 43. The couplers are massless (0.1) 2 cm spheres; parent/child
collision auto-exclusion means they add no spurious contacts. (Hip yaw AXIS_Z + a full
3-DOF hip is the next increment; the frontal pair is what unblocks lateral weight transfer.)

## Verification

- `hymeko validate` passes (only the pre-existing benign `world`-parent warning).
- Env loads: **nu=16**, standing COM height **0.665 preserved**, feet in contact.
- **Lyapunov certificate STILL holds — PD-hold-`q0` (a=0) certifies 6/6** on the nominal
  envelope. The kinematic upgrade preserves the certified balance; Lyapunov control kept.
- **Kinematic unit tests (`test_humanoid_kinematics.py`, user-requested):** every one of the
  16 actuated joints, when commanded 0.3 rad, rotates its child body by **0.3 rad about the
  declared axis** (Y sagittal / X frontal, within 0.02 rad); all 17 kinematic elements
  present; 16 distinct actuators. This is FK-only (deterministic), and would catch a
  mis-wired joint, wrong axis, or dropped body.
- **Abduction demo** (`hip_abduction_neutral_vs_spread.png`): commanding hip abduction moves
  `foot_l` **y 0.09 → 0.42 m** (33 cm lateral spread) + lifts z 0.07 → 0.15 — the lateral
  weight-transfer capability that was kinematically absent is now real.

## 16-DOF residual re-verification (kept the certified-envelope result consistent)

Retrained the bounded residual over the certified PD-hold scaffold on the upgraded model
(same harness, ANNEAL + best-checkpoint):

| policy | certified (held-out 12 seeds), envelope 0.4–0.8 |
|---|---|
| PD-hold scaffold (a=0) | **0 / 12** |
| SAC residual (best-val ckpt) | **9 / 12** |
| SAC residual (last ckpt) | **12 / 12** |

`RESIDUAL_EXTENDS_CERTIFIED_ENVELOPE` holds on the 16-DOF Vukobratović model (val curve 1.0
on 11/15 evals). Videos re-rendered on the upgraded model: `compare` (CERTIFIED | SURVIVES |
CERTIFIED) and `frontier` (CERTIFIED 0.8 → SURVIVES 2.6 → FELL 4.5).

## Files touched

```
data/robotics/humanoid.hymeko                    +26/-8  (hip abduction + ankle roll, AXIS_X couplers)
tests/test_humanoid_kinematics.py                NEW  4 FK tests (per-joint axis + elements)
tests/test_humanoid_balance_env.py               +1 test (nu=16 / frontal DOF present)
scenarios/humanoid/render_balance_video.py       nu-dynamic zero-action; frontier kicks retuned
reports/2026-07-27-humanoid-sac-residual/*        retrained 16-DOF ckpts + re-rendered videos + gates
reports/2026-07-27-humanoid-vukobratovic/hip_abduction_neutral_vs_spread.png  NEW
```

## Tests / lint

`ruff` clean. **22/22 humanoid tests pass** (1.33 s): 4 kinematics + 6 balance-env + 5
Lyapunov + others. The certificate was **not modified**; only the scenario's physical model
was extended (measured, validated).

## Provenance

- Parent SHA `a5d2387e`. MuJoCo model emitted by `target/release/hymeko` from the edited
  `.hymeko`. Seeds: train 0, val 2000–2007, test 3000–3011. Peak RSS ≈ 0.3 GB, wall ≈ 9 min.

## CORE.YAML / protocol notes

- **CORE.YAML items touched: none** (humanoid.hymeko is non-core; verified against the manifest).
- **§6.5:** no anti-patterns; the coupler pattern is the standard multi-DOF-via-1-DOF-joints
  idiom, documented inline. `render_balance_video` refactor removed the hardcoded action dim.
- **§2 plan artifacts:** measurement-driven change within the arc; design documented above
  (probe → CORE check → emitter constraint → coupler workaround), not a back-dated 4-format plan.

## Bottom line

The humanoid **kinematics now follow the Vukobratović anthropomorphic model** (hip + ankle
frontal rotations added in the `.hymeko`), giving genuine lateral weight transfer — verified
by per-joint FK tests and a 33 cm abduction spread. The **Lyapunov certificate is preserved**
(PD-hold 6/6; residual still extends the certified envelope on the 16-DOF model), so the
Tedrake-style regulation carries over unchanged. A protective step is now kinematically
*possible*; learning it is the open next step.
