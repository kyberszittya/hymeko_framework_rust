# Push controller demonstrator: 0.205 → 0.80 delivery, declared as a .hymeko hybrid FSM

**Date:** 2026-07-05 (night session, ~00:50–02:30 JST) · **Branch:** `hymeko-neuro-migration` · **Base commit:**
`4320202` (working tree carries this change, uncommitted at report time) · **Plan:**
`docs/plans/2026-07-05-galambos-fingertip-demonstrator/` (tex/pdf/tikz/mmd, with the FSM addendum).

## Cleanup Note (Codex, 2026-07-05)

This report describes a **scenario-local scripted controller**. It does not prove that the HyMeKo framework itself is controlled by a general dataflow event + FSM + monitor substrate.

Preserve the push/plow controller as a useful Galambos reference scenario and teacher. Do not treat this scenario FSM as satisfying the framework-level architecture requirement.

## Summary

The Galambos coin-toss teacher was rebuilt for fingertip-only physics and the delivery ceiling moved from
**0.205 to 0.80** (median over 3×50 episodes, dwell rule). The controller is no longer a hand-coded phase
script: it is a **hybrid dynamical system declared in `.hymeko`** — discrete FSM modes + continuous target
laws + guard events — walked by a generic `ControllerSpec`, with arms/slots handled by **rotation-group
indexing** (k-arm general, no left/right naming). Three user directives shaped the session: (1) use real
constructs — FSM, dataflow observations, events; (2) replace left/right with rotation groups and indices;
(3) move toward framework-level HyMeKo control. This implementation only satisfies that third point as a
scenario-local prototype, not as the framework substrate.

## Delivery measurements (dwell rule, difficulty 0.3, 300-step horizon, 50 eps/seed-set)

| controller | delivery per seed-set (9000/0/3000) | median |
|---|---|---|
| pinch-carry `GalambosDemonstrator` (old teacher) | 0.16 / 0.20 / 0.20 (+0.26 @20000) | **0.205 pooled** |
| push controller, press 0.006 | 0.70 / 0.54 / 0.50 | 0.54 |
| **push controller, press 0.012 (declared default)** | 0.84 / 0.80 / 0.78 | **0.80** |
| push controller, press 0.018 | 0.84 / 0.82 / 0.78 | 0.82 (≈0.012; noise) |

BC re-clone from the push-controller teacher (collab `sa_hsikan` h=64, dwell-filtered demos, 200 epochs, 3 seeds,
50-ep eval @ seed 9000): see `experiments/2026_07_05_*_galambos_bc_only/results.json` (appended below).

## The diagnosis chain (each step measured before acting)

1. **Why the old teacher failed (100-ep probe):** the pinch always formed (100/100) but the hard-clamp carry
   dropped the free cylinder within ~3 steps ("marble squirt": two point fingertips squeezing a curved side),
   loop pinch↔carry 5–11×, 68–80% of episodes lost, many by knocking the coin out of the workspace. Old
   physics masked this: arm *bodies* used to cage the coin; the 2026-07-03 fingertip-only collision default
   removed the cage and the old teacher was never re-tuned.
2. **Push-controller design:** don't grasp the unstable object — **push controller** it. k fingertips form a slot fan
   *behind* the coin along the coin→zone ray (slot i = back direction rotated by the i-th fan angle) and plow
   it, re-aiming from the current coin every step; press scales down linearly inside `brake_dist` (the coin
   settles in the zone; dwell accrues).
3. **First implementation bug (measured, fixed):** swing waypoints stayed on the orbit circle and never
   descended onto the slots — 0/24 delivery, tips hovering 2.4–3.6 cm off. Fix: descend once angularly
   aligned. → 0.833 smoke.
4. **Plow-stall (measured, fixed):** the residual failures froze mid-plow at full press — the position servo
   at its target exerts ~zero force, and table friction won. Reachability was checked and **refuted** first.
   Causal sweep: press 0.006→0.012 lifted delivery 0.54→0.80 (deeper press = persistent servo error = force).
5. **FSM refactor regression (measured, fixed):** the first FSM version changed swing/gate semantics
   (unpressed slots, tighter gate) and delivery fell 0.80→0.38. Event-log probe: 33/33 failures never left
   SWING (gate starved). Restored the measured-better semantics (pressed slots throughout; gate tolerance
   `slot_tol + press_max`); delivery back to 0.80 — **bit-for-bit the same medians as pre-refactor**, so the
   declarative rewrite is behavior-preserving.
6. **A real bug the refactor caught:** deep press exercised the V-re-formation path, where the old code
   cleared the slot assignment mid-step (`KeyError: 'left'`). The FSM structure (pure guards, transition →
   re-assign → target law pipeline) makes that class of bug impossible; a pure-layer regression test pins it.

## The declarative controller layer (scenario-local prototype, not framework substrate)

- **`data/robotics/meta_controller.hymeko`** — vocabulary: `controller.phase` / `controller.param` /
  `controller_spec` (mirrors meta_reward / meta_strategy).
- **`data/robotics/galambos_push.hymeko`** — the push controller itself: three `fsm_phase` nodes (`swing`,
  `plow`, `hold`) each binding a target `law` by name and declaring transitions `on <event> to <phase>;`
  the geometry/gait scalars (incl. the measured `press_max 0.012`). The bundle arc orders the phases; the
  first is initial. **Rewiring the FSM or retuning the gait is now a `.hymeko` edit** — the reward-in-hymeko
  discipline extended to controllers.
- **`hymeko_rl/control/controller_spec.py`** — `ControllerSpec.from_hymeko` (same `read_bundle` bridge as
  `StrategySpec`), validating laws/transition targets at load; `spec.step(phase, fires)` is the generic FSM
  walk (first declared guard that fires wins — declaration order is priority).
- **`hymeko_rl/experiments/galambos_demo.py`** — `PushObs` (a frozen `(k,2)`-tips snapshot; the single
  env→controller dataflow boundary), `GUARDS`/`LAWS` immutable registries binding the declared names,
  `PushEvent` transition log on the instance, `fan_offsets`/`push_slots` (rotation-group slots),
  `assign_slots` (min-cost permutation, k ≤ 6), `PushConfig.from_params` (strict: a `.hymeko` typo fails
  loud). `_ik_action` generalized to (arm, target) pairs and shared with the legacy pinch-carry class.

## The hybrid-dynamical-systems view (user directive #4)

The declared controller **is** a hybrid automaton: FSM phases = discrete modes, target laws = continuous
flows, guard events = jump conditions. The natural next step is **RL inside the declared structure** —
replace hand-coded laws with learned per-mode policies (BC warm-start → TD3+BC per mode) while the mode
graph, guards, and parameters stay declarative. That is precisely the reward-machine/options loop of
`docs/plans/2026-06-23-fsm-structured-rl/plan.tex`; tonight's flat BC clone is the baseline a mode-wise
hybrid learner must beat.

## Framework fixes riding along

- **Metric-consistency defect removed:** `collect_galambos_demos` filtered demos by *momentary* `in_zone`
  while everything is graded by *dwell* — the exact §3 filter≡grading violation on record from 2026-07-01,
  still live until tonight. Now dwell-filtered, teacher injectable (strategy), push controller default (§6.5 #19:
  measured winner becomes the default; all 6 call sites inherit via the single entrypoint).
- Observability: `demo.events` replaces ad-hoc phase re-derivation in probes.

## Tests (all green: 13/13 in `test_galambos_demo.py`; ruff + mypy --strict clean on changed files)

- Unit (pure, no MuJoCo): slot geometry behind-coin/symmetry/press; fan offsets k=1/2/3; swing orbit +
  descend; FSM walk on synthetic obs (all four guards); spec parse of the real profile; strict param typo.
- Regression: (a) push controller > 0.3 delivery on a fixed 12-ep set with zero deaths (old teacher ~0.2 — fails
  pre-change); (b) plow→swing re-assignment before targets (fails as `KeyError` pre-fix); (c) dwell-filter +
  idle-teacher RuntimeError (fails pre-fix).
- Performance: median `action()` latency asserted < 1 ms (5×200-call iterations, after warm-up).

## Complexity / waivers

- `_extract_arms` cyclomatic D(26): **pre-existing, untouched** — declared, not introduced.
- `ControllerSpec.from_hymeko` C(12): above warn 10, below fail 15 — parse+validate in one place; accepted.
- No new `#[allow]`/`# type: ignore`/broad excepts. No §6.5 anti-patterns introduced; the `left/right`
  Cartesian naming was *removed*. CORE.YAML items touched: none (`hymeko_rl` + `data/robotics` are non-core).

## Files touched

- `hymeko_rl/experiments/galambos_demo.py` (+~330/−~110: push controller + FSM + registries + obs; legacy class kept)
- `hymeko_rl/experiments/galambos_bc.py` (dwell filter + injectable teacher, ~+15/−8)
- `hymeko_rl/train/bc.py` (`device=` lever + on-device loss accumulation, ~+15/−6; defaults unchanged)
- `hymeko_rl/control/controller_spec.py` (new, ~110 LOC)
- `data/robotics/meta_controller.hymeko`, `data/robotics/galambos_push.hymeko` (new)
- `hymeko_rl/tests/test_galambos_demo.py` (+~150: 10 new tests), `hymeko_rl/tests/test_bc.py` (+1 device test)
- Plan: `docs/plans/2026-07-05-galambos-fingertip-demonstrator/` (4 artifacts + addendum, compiles)

## Provenance

Seeds: eval seed-sets 9000/0/3000 (+20000 for the old-teacher pool); BC seeds 0/1/2. Host: Windows 11,
CPU-only MuJoCo (same as all 2026-07 galambos runs). Wall: probes ≈3 min/100 eps; press sweep ≈13 min;
BC pipeline run log in the experiment dir. Peak RSS far under the 16 GB cap (planar scene + 64-hidden nets).
No new dependencies. In-flight artifacts: `experiments/2026_07_05_*_galambos_bc_only/` (run.log grows live).

## BC re-clone results (run `experiments/2026_07_05_02_12_galambos_bc_only/`, wall 825 s)

| policy | delivery (50 eps, dwell, seed 9000) | both_contact | BC loss |
|---|---|---|---|
| push-controller teacher (anchor, same protocol) | **0.840** | ~0.10–0.13 | — |
| BC clone seed 0 | 0.340 | 0.105 | 0.00107 |
| BC clone seed 1 | 0.440 | 0.016 | 0.00105 |
| BC clone seed 2 | 0.280 | 0.040 | 0.00110 |
| **BC clone median** | **0.34** | — | — |
| *(old regime: clone 0.12 / teacher 0.205)* | | *clone ~0.01* | *0.0021* |

The clone nearly **tripled** (0.12 → 0.34 median) and — structurally — now *inherits the teacher's contact
behaviour* (both_contact up to 0.105 vs the old ~0.01). The remaining 0.34 vs 0.84 gap is classic BC
compounding error (a reactive clone drifts off the closed-loop teacher's corridor over ~140 steps); the
levers, in order: TD3+BC refine on the new corpus (the campaign's existing next stage), DAgger-style
re-collection, or the hybrid per-mode learner below. Demos: 159/200 episodes delivered (0.795), 27,475
dwell-filtered transitions (8.6× the old corpus).

## Performance measurements (2026-07-05, this host; BC bench uncontended, step probe contended by BC)

- **Simulation:** 286 control steps/s end-to-end (push controller + IK + MuJoCo, frame_skip 20 → ≈5,700 physics
  steps/s); episode median 150 control steps (successful ≈141) on the 300-step horizon.
- **Controller:** `action()` median 0.259 ms (IQR 0.257–0.260, worst 0.299; 5×200 warm calls) — ~7% of the
  3.5 ms control step; physics owns the rest.
- **BC training (27,475×6×8 corpus, collab sa_hsikan h=64):** cpu b128 0.18 ep/s; cuda b128 0.18 ep/s
  (launch-bound — the B-small HSiKAN dispatch story again); **cuda b512 0.70 ep/s (3.9×)**. `behaviour_clone`
  gained a `device=` lever (ddpg convention; corpus moved once; epoch loss accumulated on-device; model
  returned to CPU). Library defaults unchanged — batch/device alter optimization, so the accelerated config
  (cuda, b512, 100 epochs) is explicit in the driver and recorded in `results.json` (§6.5 #19).
  Pipeline wall: ~55 min (projected) → **825 s measured**.

## Addendum (02:35–02:55): user-ordered follow-ups 2 → 3 → 1, executed

**Item 2 — monitor extraction (DONE).** The four guards are now STL-robustness monitors: each returns a
margin ρ (satisfied iff ρ > 0; conjunction = min, disjunction = max of per-arm atoms — the
`hymeko_monitor` semantics). `PushEvent.margin` records the firing robustness; `last_margins` exposes
the per-step readings (graded distance-to-jump — future PBRS/curriculum signals). Verified: 200-case parity
property test (ρ>0 ≡ the old booleans), gradedness test, and a full 9-cell press-sweep re-run with **every
delivery number identical** to pre-extraction. Plan: `docs/plans/2026-07-05-guard-monitors-robustness/`.
The Rust `hymeko_monitor` crate (STL AST + robustness done, `observe()` todo, no Python bindings yet) is
the documented future seam; the guard formulas and thresholds already live in the controller profile.

**Item 3 — learned per-mode laws under the declared FSM (measured: TIED with flat BC).** Library seam:
per-instance law injection (`PushDemonstrator(env, laws={...})`) + `last_targets` exposure (both
tested). Experiment (`experiments/2026_07_05_02_48_galambos_hybrid_modewise/`): mode-labeled corpus from
159 held episodes (swing 14,189 / plow 13,286 samples), tiny 11→64→64→4 `TargetNet` per mode, BC on GPU
(seconds), executed under the declared FSM with scripted robustness guards.
**Result: delivery median 0.28 [0.20, 0.28, 0.44] — tied with the flat clone 0.34 [0.28, 0.34, 0.44].**
- *Measured:* mode separation with near-perfect target fits (plow MSE 5×10⁻⁶ ≈ 2 mm RMS) does NOT close
  the clone→teacher gap. The gap is not mode-mixing.
- *Hypothesis (next):* trajectory-level compounding — mm-scale target errors amplify through contact
  dynamics; and the swing orbit's discrete cw/ccw branch is multi-modal, which plain regression averages
  over. Remedies: corrective data (DAgger-lite on perturbed states) or value-driven refinement (item 1).

**Item 1 — TD3+BC refine on the push-controller corpus (LAUNCHED overnight).** §3 production smoke passed
(`--variant baseline --smoke`, 65 s, full path incl. push controller demos). Full run in flight: 3 seeds × 200k,
best-checkpoint on delivery, `experiments/2026_07_05_02_50_galambos_coord_ab_baseline/` (log grows live;
persistent monitor attached). The question it answers: does off-policy value-refinement lift the 0.34 BC
floor toward the 0.84 teacher — the flat-vs-hybrid comparison above says refinement, not better fitting,
is the lever.

## Next step (user direction, 02:25): extract the MONITORS

The push controller's guards are STL atoms over hypergraph entities (`v_broken` ≡ ∃i ⟨tip_i−coin, û⟩ > d/2, …),
the declared FSM is a monitor automaton, and `DwellMetric(in_zone, K)` is □₍t,t+K₎ in_zone — grading,
control, and explanation are one temporal object. The `hymeko_monitor` crate already holds the STL
AST/combinators + robustness (implemented) and the sliding-window `observe()` (todo); the extraction =
declare the guard formulas in the controller profile, evaluate them as robustness margins (graded
close-to-failure signals — free PBRS potentials/curriculum), and unify `RolloutMetric` with the same layer.
This is the live case study the `hymeko_monitor/paper/` RV skeleton needs, and the reward-machine/options
loop of `docs/plans/2026-06-23-fsm-structured-rl/` supplies the RL-in-modes half (hybrid dynamical system:
declared modes + learned flows). Needs its own plan dir before implementation.
