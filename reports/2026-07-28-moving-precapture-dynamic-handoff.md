# R10 Stage 1B — phase-shaped moving-precursor capture: HOME_V1 → strict K6 through the frozen downstream (no TD3)

**2026-07-28 · branch `recovery/coin-r9-causal-residual-delivery` · parent `eb09a16a` · dev s1 (14250) · downstream + analytic transit FROZEN · s4/s7 untouched · f1–f4 SEALED · NO RL · no tag moved**

## Summary

Stage 1A solved `HOME_V1 → READY` analytically. Stage 1B closes the last, contact-rich transition into the frozen
downstream and completes the **end-to-end home-start positive control**. The decisive reframing (correct): the cradle
handoff `(q★, qvel★, τ★)` is **not** a global capture target but a **phase-point of a moving orbit** — `q` is the
position and `qvel` the *tangent* of the same trajectory, `τ★` the control-history memory. A terminal-state objective
that treats them as independent is ill-conditioned: I measured that position and velocity are individually reachable
(dq 0.048 / dqv 0.049) but **not jointly** (arriving *at* q★ needs a fast slam; matching the 0.66 rad/s velocity needs a
slow approach that falls short). That is the signature of hitting an on-orbit point off-orbit — a mis-posed positive
control, not a physical wall.

The fix arrives **moving along the tangent**: a quintic terminal segment drives the tips from READY to a backward-tangent
precursor `q_pre(Δt) = q★ − Δt·qvel★` with terminal velocity ≈ qvel★; the `prev_tau` preload ramps **during** that moving
segment (applying τ★ *afterward* accelerates the arm to ~2× the deliverable band — τ★ is the accelerating torque,
`qdot` jumps 0.66→1.07 in one downstream step); a short residual under a low-dimensional structured CEM closes the last
margin. The frozen APPROACH → HANDOFF_RESET → R2 → coast then generates the full coupling itself.

**Verdict: `HOME_V1_TO_DYNAMIC_HANDOFF_REACHABILITY_PASS`.** From `HOME_V1_GENERIC`, teacher-free, no state edit, no
snapshot injection, the full chain delivers **strict K6 in 3/3 independent planner seeds** (min_dtz 0.12 / 5.86 / 2.11
mm, all safe), each with **exactly one HANDOFF_RESET** and a **distinct** parameter solution (3 distinct `(s, preload_start)`),
and all three READY→CAPTURE boundary tests pass.

## The chain

```
HOME_V1_GENERIC → frozen analytic transit → READY (55 mm) → quintic phase-shaped moving precursor
(during-segment preload ramp + short residual) → frozen APPROACH → HANDOFF_RESET → R2 → coast → strict K6
```

## Localization + tube diagnostics (why this design)

| finding | measurement |
|---|---|
| downstream delivers from **just** (q,qvel,τ,contact) | reconstructed state, **fresh history** → K6 1.0mm (history irrelevant) |
| qvel and τ **jointly** load-bearing | qvel=0 → 34mm; τ=0 → KINETIC engages 2 steps → 35mm |
| backward-tangent tube deliverable | `(q★−n·dt·qvel★, qvel★, τ★)` → K6 for **n=0…4-5** |
| direction cosine tolerance **wide** | cos 0.85–1.0 all deliver K6 |
| preload tolerance **tight** | dτ 0.0→K6, 0.13→marginal, **0.27→fail** |
| post-hoc τ★ ramp accelerates out of band | scale → 1.6–2.8× (τ★ is the accelerating torque) |

The winning captures deliver via *compatible continuation states on the orbit* (e.g. seed 11 at cos 0.91, scale 1.58,
dτ 0.88), not the exact cradle point — confirming the continuation-basin reclassification.

## Gates

| gate | result |
|---|---|
| `H_DYN_RECLASSIFIED_AS_LOCAL_CONTINUATION_BASIN` (diagnostic) | ✅ |
| `MOVING_PRECAPTURE_TO_HANDOFF_PASS` (≥1 seed strict K6, safe, no state edit) | ✅ |
| `READY_TO_CAPTURE_ONE_STEP_IDENTITY_PASS` (capture continues from READY's exact state) | ✅ |
| `TRANSIT_TO_CAPTURE_CONTINUATION_PASS` (deterministic replay) | ✅ |
| `BOUNDARY_OFF_BY_ONE_REJECTED_PASS` (degenerate zero-shaping capture does *not* deliver) | ✅ |
| `HOME_V1_TO_DYNAMIC_HANDOFF_REACHABILITY_PASS` (3 seeds K6, exactly-one HANDOFF_RESET, boundary) | ✅ |

Saved traces + winning parameters + causal provenance: `reports/2026-07-28-moving-precapture-dynamic-handoff/moving_precapture.json`.

## Files touched

| file | LOC | role |
|---|---|---|
| `hymeko_rl/coin_delivery/theta_option/moving_precapture.py` | +304 (new) | `HandoffReference` (moving phase-point + backward-tangent precursor), `QuinticSegment` helpers, `CaptureParams`/`CaptureSearchSpec`, `PhaseShapeCapture` (quintic track + during-segment preload + residual, governed servo), `FrozenDownstream` adapter, structured `plan_capture` CEM |
| `hymeko_rl/experiments/coin_kinetic_moving_precapture.py` | +115 (new) | 3-seed reachability gate + boundary tests + saved traces + provenance JSON |
| `hymeko_rl/tests/test_moving_precapture.py` | +117 (new) | 8 tests (unit + integration + boundary + perf) |

**CORE.YAML items touched: none.** Frozen downstream (`kinetic_clone`, `kinetic_handoff_reset`, `kinetic_residual[2]`,
`velocity_transport`, `kinetic_contract`) and the Stage-1A transit (`planar_geometric_approach`, `planar_arm_2r`,
`home_states`) — `git status` confirms **untouched**. The capture adds no new torque authority (same
`step_ablation`→`govern_torque` clip; residual bounded to ±slew inside it).

## Test results

`pytest -p no:randomly` — **8 passed in 34 s**.

- **Unit (analytic):** quintic BVP boundary conditions; backward-tangent precursor (recedes along the tangent);
  reference determinism; `_theta_to_params` clamping; `_cost` (K6→0, else min_dtz, unsafe→1e3).
- **Physics:** `HandoffReference.from_cradle` determinism; `PhaseShapeCapture` continues from READY's exact state +
  deterministic; `FrozenDownstream` delivers from the cradle (1 HANDOFF_RESET); the **degenerate zero-shaping capture
  does not deliver** (proves the phase-shaping is load-bearing); reduced-budget `plan_capture` reaches strict K6 with
  deterministic replay.

**Coverage:** every new public/private function exercised by ≥1 new test; the degenerate-capture and preload-coupling
negatives would fail a naive implementation.

## Performance results

| metric | measured | budget |
|---|---|---|
| capture + downstream (1 eval) | ≈ 0.1 s | — |
| 3-seed reachability run wall | ≈ 3 min (checkpointed per seed) | ≤ ~3 min |
| reduced-budget solve (test) | < 120 s (asserted) | < 120 s |
| peak RSS | 0.23 GB | < 1 GB (hard cap 16 GB) |

Static analysis: `ruff check` clean; `radon cc` no block at C or worse (avg A / 1.73), after extracting `_solve_seeds`
/ `_summarize` from `run`.

## §6.5 anti-patterns

None introduced. The CEM searches **bounded structural parameters only** (Δt, s, preload-start, bmax, 2 residual knots)
— never a raw 56-D per-step torque dump. The frozen downstream is a **Facade** (`FrozenDownstream`), imported as-is.
Config is typed dataclasses (`CaptureParams`/`CaptureSearchSpec`), no string-typed modes. No new globals; the governed
callback is set/reset within the roll. No `unwrap`/broad `except` in non-test code.

## Provenance

- Parent SHA `eb09a16a` (clean tree apart from the new files).
- Env: Python 3.11.15 / mujoco 3.10.0 / numpy 2.4.6 / torch 2.12.0 / macOS-arm64 (CPU). Planner seeds 11/23/42.
  Cradle seed `S1_SEED` = 14250. R2 checkpoint `reports/2026-07-28-coin-r9-r2-h1-multiseed/seed_01/checkpoint.json`.
- Handoff reference: q★ `[-0.5836,-0.6513,-0.4457,2.1245]`, qvel★ `[-0.454,-0.428,0.010,0.213]` (‖·‖=0.659), τ★ `[-1.1712,-0.9291,-0.3088,2.1943]` (‖·‖=2.67).
- Plan: `docs/plans/2026-07-28-moving-precapture-dynamic-handoff/` (tex/pdf/tikz/mmd, tectonic).

## Open issues / follow-ups

1. **Next (deferred for green): TD3.** The positive control is proven and frozen; a learned capture policy
   (`HOME_V1_START_STRICT_K6`) can now amortize/robustify the structured CEM — trained against a *validated* target.
   **STOP before TD3** per the stop logic.
2. Fork B (reachable-deliverable-basin, direct downstream score) is **not needed** — fork A delivered.
3. Deferred (unchanged): full-workspace cross-host on kato14/kato15; C1 dwell-refinement paired panel.

## Status

`HOME_V1_TO_DYNAMIC_HANDOFF_REACHABILITY_PASS`. The end-to-end home-start dynamic delivery is a reproducible positive
control (3/3 seeds strict K6), teacher-free, no state edit, frozen downstream. **STOP** — awaiting green before TD3.
