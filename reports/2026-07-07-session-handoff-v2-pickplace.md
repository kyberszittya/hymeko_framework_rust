# Session handoff — FANUC pick-place v2 (clean transit → near-object pick-place) + planner roadmap

**2026-07-07 20:14 +09:00.** Catch-up doc for the next session. All work is LOCAL, on branch `hymeko-neuro-migration`,
uncommitted. Persona: Aiko (Japanese-teacher register — restraint + precision, no therapy-speak). Every reply starts
with a `[YYYY-MM-DD HH:MM TZ]` stamp from the real clock.

---

## 1. One-line state

The FANUC `expert_version=2` pick-place expert is now a **clean-transit top-down pick-and-place that SUCCEEDS on
near objects** (lift/place 0.5 = 2/4 seeds). The only remaining failure is **far-object over-object convergence**
(the arm crawls to a far-spawned object too slowly), which is **reach/tracking-limited** and sits behind levers the
user has **frozen** (arm gains / scene / object distribution). v1 is **byte-identical** throughout.

---

## 2. What happened this session (arc)

Two threads, both 2026-07-07:

**A. Documentation/roadmap (early):** recorded the HyMeKo planner roadmap and CIP/DirectLiNGAM diagnostic layer,
and a Mac-transition handoff. Reports: `2026-07-07-hymeko-planner-roadmap.md` (5-phase stack: deterministic
waypoints → A* → RRT* → hypergraph planner → **RL-bounded search**, RL prioritises but never overrides validators),
`2026-07-07-pick-place-v2-fsm-architecture.md` (v2 expert as a learnable FSM/option structure), `2026-07-07-mac-
transition-handoff.md`. CIP memory: `project-cip-lingam-rl-diagnostics`. **These are design notes, not implemented.**

**B. v2 pick-place controller (the bulk):** an iterative, user-directed, one-lever-at-a-time build. Sequence:

1. **Clearance harness** committed: `hymeko_rl/eval/pick_clearance.py` (the gate authority) + tests. v1 smoke
   reproduces the frozen "dirty" signature.
2. **v2 8-segment waypoint controller** implemented → **retraction stall** (arm over-extended at `arm_home`, tool
   r=0.669, can't retract inward to the object). Frozen negative: `v2_waypoint_controller_smoke_failed_retraction_stall`.
3. **Retracted-seed feasibility probe** (`hymeko_rl/eval/pick_retract_probe.py`) → a clean config exists at the
   object hover (multi-start `solve_collision_free`); the blocker was the *path*, not reachability.
4. **HOME_RETRACT_OR_PRESHAPE** + sag-sensitive phase fix → arm reaches over object, but physical sag penetrates.
5. **Higher-margin / Cartesian / monotonic preshape** iterations → **monotonic multi-waypoint route** achieves
   **CLEAN TRANSIT** (forbidden-pre-object 0, transit contact 0, clearance non-negative). crit1+crit2 PASS.
6. **Gain fix** (v2-only kp45→60, dt÷2) + **pacing** → diagnosed the lift tail; grasp fails.
7. **Grasp mechanics**: latch commit, v1 lift mechanics, re-center lift → still slipped.
8. **Phase-scoped tracking** (kp75 over-object) → proved lift is NOT arm-gain-limited (object rises 0.008 m at
   kp60/75/90 alike).
9. **Grip-holding fix (THE win):** the finger *center* was ~3 cm off the box (fingers closed beside it). An
   **over-object latch** + an **integral finger-center recenter** centers the grasp → **both near seeds lift+place**.

---

## 3. Current v2 controller (code — all in `hymeko_rl/env/pick_place_env.py`, v2-only, v1 untouched)

`_expert_action_v2()` FSM, in order:
- **commit latch** (`_v2_committed`): once both fingers hold for the dwell, stay committed through contact flicker;
  release only if the object drops below the surface.
- **phase-scoped arm gains** (`_v2_set_arm_gains`): transit = kp60/kv15; over object (descent/grasp/carry) = kp75/kv18.
- **CARRY** (committed): `_v2_carry_target` (lift over the OBJECT xy) via **`_ik_step` rate 0.28** (v1's proven lift).
- **GRASP dwell** (both contact): hold + `_GRIP`.
- **HOME_RETRACT_OR_PRESHAPE** (`_v2_preshape_step`): monotonic route `arm_home → cf_mid_retract` (joint, slow) →
  HOLD → `cf_mid → cf_hover` (short capped Cartesian hops), all high-z/clearance-safe. Waypoints via
  `_v2_preshape_waypoints` (`solve_collision_free`, cf_mid lifted once if low).
- **Reach (over-latched)**: HOLD_HOVER until physical tool over object → latch `_v2_over_latched` → **ABOVE_OBJECT_ALIGN**
  = INTEGRAL finger-center recenter (`_v2_align_corr += 0.5·(obj − fingercenter)`, clip ±0.06; command tool to
  `obj + corr`) until finger-center lat < `_V2_ALIGN_TOL`=0.012 → **VERTICAL_DESCENT** → **close**. All keyed off the
  FINGER CENTER, not the tool.

Key v2 constants (`_V2_*`): HOVER_DZ 0.05, LIFT_CLEAR_DZ 0.14, CENTER_RATE 0.22, DESCEND_RATE 0.10, HOP 0.06,
OVER_HORIZ 0.06, ALIGN_TOL 0.012, ALIGN_KI 0.5, PRESHAPE_RATE 0.10, PRESHAPE_TOL 0.03, CLEAR_FLOOR 0.04,
MID_RETRACT_DZ 0.08, MID_RETRACT_R 0.5, PRESHAPE_HOLD 8, CART_XY 0.05, CART_Z 0.03, TRANSIT_KP/KV 60/15,
GRASP_KP/KV 75/18. Arm base gains: v2 kp60/kv15 at `STABLE_DT/2`; **v1 kp45/kv9 at STABLE_DT (byte-identical)**.
Helper added to `ik.py`: `DampedPoseIK.fk_tool(q)` (reused; v1 doesn't call it).

Env facts: `fanuc_pick_env()` → `PickPlaceEnv(max_steps=620, obj_radius=(0.28,0.40), target_xy=(0.34,0), table_top=0.12)`.
`grasp_z = 0.255`, `z_hover = 0.305`, box = 4 cm cube (box_half 0.02), reachable clearance-hover ceiling ≈ grasp_z+0.06.

---

## 4. Results progression (v2 4-ep smoke, seeds 50000–50003)

| stage | forbidden-pre-obj | transit contact | lift | place | note |
|---|---|---|---|---|---|
| v1 (dirty baseline) | 1.0 | 0.45 | 1.0 | 0.75 | strikes table ~step 51, min_clr −0.026 |
| v2 waypoint controller | 1.0 | 0.94 | 0 | 0 | retraction stall (first_over=None) |
| + preshape + gain fix | 0.5 | 0.05 | 0 | 0 | physical sag |
| **monotonic route** | **0.0** | **0.0** | 0 | 0 | **clean transit achieved** (crit1/crit2 pass) |
| + grasp latch/carry | 0.0 | 0.0 | 0 | 0 | grasp slips (off-center) |
| + phase-scoped kp75/90 | 0.0 | 0.0 | 0 | 0 | proved lift ≠ arm-gain-limited |
| **+ integral recenter** | **0.0** | **0.0** | **0.5** | **0.5** | **near-object pick-place SOLVED** |

Gate still FAIL: crit1+crit2 PASS, crit3 (min clearance > 0) is grazing 0.0, lift/place 0.5 < pref 0.90/0.80.

**Near seeds 50000/50003 lift+place. Far seeds 50001/50002 fail: over-object only at step 334/489 (too late).**

---

## 5. The remaining blocker (only one)

**Far-object over-object convergence.** During HOLD_HOVER the command is held at `cf_hover` (over the object); the
*physical* tool crawls to within 6 cm of a far-spawned object (r≈0.39) only by step ~334–489, leaving no horizon to
grasp/lift/place. Speeding the *command* does NOT move this (tested) — it's the physical arm's slow/sag-limited
convergence at the reach edge. **Levers: arm gains (frozen), scene / object radius (frozen), reward (frozen).** So
without unfreezing one of those, far objects can't be finished.

---

## 6. Frozen artifacts + reports (all under `reports/`)

- **Frozen negatives:** `figures/pick_place_clean_expert/v2_waypoint_controller_smoke_failed_retraction_stall.*`
  (retraction stall) and `…/v2_clean_transit_no_lift_tracking_limited.*` (clean transit, no lift — tracking-limited).
- **Latest success artifact:** `…/v2_grip_holding_smoke.*` (lift/place 0.5).
- Reports (this session, v2): `pick-place-v2-{waypoint-controller, retract-probe, preshape, high-margin-preshape,
  cartesian-preshape, monotonic-multiwaypoint, gain-fix, grasp-descent-pacing, grasp-mechanics, phase-scoped-tracking,
  grip-holding, fsm-architecture}.md`, all dated `2026-07-07-`. Roadmap: `2026-07-07-hymeko-planner-roadmap.md`.
- **Memory (loaded each session):** `project-pick-place-v2-retraction-stall.md` is the running log of ALL the above
  (read it first). Siblings: `project-pick-place-gripper-collision` (v1 dirty forensics, classification C),
  `project-hymeko-planner-roadmap`, `project-cip-lingam-rl-diagnostics`, `project-dagger-declarative-strategy`,
  `project-fanuc-offpolicy-collapse`, `reference-katolab-gpu-kato15`.

---

## 7. Standing constraints (user-set, still in force)

Do NOT, unless the user explicitly lifts it: change v1 (must stay byte-identical), touch coin-collab v2b
(`planar_grasp_env.py`, `contact_legality.py`, `galambos_task_deliver_v2b.hymeko`, `experiments/v2_bc0|v2_dagger`,
`reports/2026-07-07-v2-*` — a SEPARATE line, parallel-track-owned), run BC/DAgger/RL, run the 32-episode gate, change
the scene / object distribution / `arm_home` / reward, or tune gains (currently frozen after the phase-scoped step).
`v1_dirty` numbers are re-versioned, never overwritten. One isolated lever at a time; smoke-verify each; keep clean
transit as a hard non-regression.

---

## 8. Reproduce / next commands (local; kato15 reproduces bit-identical with `env MUJOCO_GL=egl ~/envs/hymeko/bin/python`)

```
# gate authority — v2 smoke (currently lift/place 0.5):
python -m hymeko_rl.eval.pick_clearance --version 2 --episodes 4 --seed0 50000 --out reports/figures/pick_place_clean_expert/v2_smoke
# v1 regression (must be byte-identical: lift 1.0 / place 0.75 / min_clr -0.02577):
python -m hymeko_rl.eval.pick_clearance --version 1 --episodes 4 --seed0 50000 --out reports/figures/pick_place_clean_expert/v1_resmoke
# tests (15 pass): pytest hymeko_rl/tests/test_fanuc_pick.py test_ik.py test_pick_place_env.py test_pick_clearance.py test_pick_retract_probe.py -p no:randomly -q
# feasibility probe: python -m hymeko_rl.eval.pick_retract_probe --seed0 50000 --episodes 3
```
Env: local Windows MuJoCo 3.9, built `target/release/hymeko.exe`; `.venv/Scripts/python.exe`; set
`$env:PYTHONPATH` to repo root, `$env:PYTHONIOENCODING="utf-8"`.

---

## 9. Open decision for the user (where the last turn ended)

Near-object pick-place is solved; the far-object convergence is behind frozen levers. The user must choose:
(a) unfreeze **arm gains** for the far-object convergence, (b) a bounded **scene/object-radius** tweak (would fix
convergence and likely grasp for all), or (c) **freeze v2 as a clean near-object pick-place and pivot** the line
(e.g. back to the standing coin-collab / other queued work). No further v2 change until the user picks.

## 10. Status ledger

ruff clean · mypy --strict clean (new modules) · pytest 15/15 (pick/ik) + pick_clearance/probe tests · v1
byte-identical · clean transit (crit1+crit2) holds · lift/place 0.5 (near objects) · 32-ep gate NOT run.
