# R11.2 — HyMeKo hybrid-delivery IR + initial-condition / energy / provenance certificates

**2026-07-29 · branch `recovery/coin-r9-causal-residual-delivery` · parent `bf5e542e` (R11 v2 plan) · additive, non-RL · no CORE.YAML items · no new dependencies**

## Summary

R11.2 delivers the first implementation boundary of the R11 v2 program: a **domain-generic HyMeKo IR** for the exact-zero-home hybrid delivery task, plus the **coin adapter** that reads a live MuJoCo rollout into it. This is the verifiable, additive foundation the demonstration bank (R11.3) and the conditioned policy (R11.4+) build on. **No RL, no BC, no demonstration generation, no energy shaping, no new planner** — and, per the two review corrections locked into the plan, **RRT stays a deployed component (not a teacher)** and **energy is *measured*, never asserted conserved**.

The IR is simulator-free: it reasons purely over a `RolloutState` struct, so the initial-condition certificate is verified without a physics engine in the loop. The coin adapter (`hymeko_rl/coin_delivery/ir_adapter.py`) is the *only* file that touches MuJoCo.

## What was built

**`hymeko_rl/ir/` (generic, numpy + stdlib only, mypy --strict clean):**

| module | contents |
|---|---|
| `rollout.py` | `RolloutState` — the read-only dynamical snapshot (q, q̇, prev_τ, object velocity, task-contact count, memory/step/snapshot/teacher flags) |
| `initial_condition.py` | `InitialCondition` (spec) + `certify → InitialConditionCertificate`; `InitialDistribution` (certificate-filtered, not a raw box), `AdmissibilityResult`, `RejectionLedger` |
| `hybrid_mode.py` | `HybridMode` (IntEnum M0..M7), `StateInvariant`, `zero_home_invariant`, `ModeTrace` (rejects skips/regressions) |
| `transition.py` | `TransitionGuard`, `HandoffDescriptor` (`is_complete`), `HybridTransition` (`fire` is guard-consistent) |
| `energy.py` | `MeasuredEnergyLedger` (W₊/W₋ separate), `EnergyTransitionCertificate` — asserts `ENERGY_LEDGER_COMPLETE`/`ENERGY_BALANCE_RESIDUAL_RECORDED`, **refuses** a conservation verdict (raises; that is R11.8) |
| `provenance.py` | `RolloutProvenance` (deterministic SHA-256 content hash), `SuccessCertificate` (bound to that hash) |

**`hymeko_rl/coin_delivery/ir_adapter.py` (the MuJoCo↔IR boundary):** `EXACT_ZERO_HOME_V1`, `read_rollout_state`, `certify_zero_home`, `coin_admissibility` (start-collision-free + non-empty goal set), `make_coin_distribution`, measured energy (`_robot_ke` via `mj_fullM`, `_object_ke`, `EnergyProbe` work integral, `measured_reach_ledger`), `zero_home_reach_trace`, `k6_success_certificate`, and `instrument_reach_rrt` — which emits the full IR bundle (IC cert + energy cert + mode trace + provenance + K6 cert) for a reach rollout.

## The two review corrections, in code

1. **RRT is deployed, not a teacher.** The IR never treats the geometric planner as a training oracle. `instrument_reach_rrt` runs the *deployed* RRT reach; the CEM capture is the only teacher (used at R11.3 for demonstrations). Teacher-free is scoped to the learned capture/delivery.
2. **Energy is measured, not conserved.** `EnergyTransitionCertificate.conservation_verdict()` **raises `NotImplementedError`** naming R11.8 — the certificate can only assert measurement completeness + a recorded residual. A test pins this refusal so the contract can't silently drift.

## The 10 R11.2 gates (all PASS)

| gate | how it is proven |
|---|---|
| `EXACT_ZERO_INITIAL_CONDITION_CERTIFICATE_PASS` | `certify` on a fresh at-rest zero state → valid, no violations (fast); a real fresh zero home certifies valid, no violations (slow) |
| `INITIAL_DISTRIBUTION_ADMISSIBILITY_PASS` | in-box admissible vs. in-box `start_in_collision` vs. `out_of_bounds`; `RejectionLedger` keeps rejects out of the denominator; canonical coin admissible + `right +5cm` = `start_in_collision` on the rig |
| `ROLLOUT_STATE_CONTINUITY_PASS` | each of q̇≠0 / object-moving / prev_τ≠0 / step≠0 / contacts≠0 / memory-nonempty fails its clause |
| `NO_SNAPSHOT_INJECTION_PASS` | snapshot-parent and teacher-state flags each fail the certificate |
| `HYBRID_MODE_TRACE_VALID_PASS` | canonical M0..M7 valid; skip / regression / wrong-start invalid; `zero_home_invariant` holds/fails |
| `HYBRID_TRANSITION_GUARDS_PASS` | guard fires on the admitted state (reset map applied) and raises `TransitionGuardError` when blocked; guards must connect consecutive modes |
| `HANDOFF_DESCRIPTOR_COMPLETE_PASS` | full descriptor complete; NaN / empty / non-consecutive-modes incomplete |
| `MEASURED_ENERGY_LEDGER_COMPLETE_PASS` | complete ledger → `is_complete` + `ENERGY_LEDGER_COMPLETE`; a NaN field → incomplete; measured ledger complete on the instrumented reach |
| `ENERGY_BALANCE_RESIDUAL_RECORDED_PASS` | finite `balance_residual` → `ENERGY_BALANCE_RESIDUAL_RECORDED`; `conservation_verdict()` raises (two-level contract) |
| `K6_CERTIFICATE_PROVENANCE_LINK_PASS` | provenance hash deterministic + field-sensitive; `SuccessCertificate` requires a 64-char digest; **on the instrumented reach, `k6_certificate.provenance_hash == provenance.content_hash()`** |

## Production-scale smoke caught two real reader bugs (before any queue)

Per CLAUDE.md §3, the rig-backed tests were run at production scale before finalizing. They failed first — and the failures were **real bugs in the adapter's live-state reader**, exactly the class of bug toy unit tests cannot surface:

1. **`_planar_metrics.disk_vel` is a noisy derived metric, not the true velocity.** At the fresh, genuinely-at-rest home, `disk_vel ≈ [2e-4, −6e-4]` m/s while the true generalized coin velocity `qvel[4:]` is exactly `0`. Reading `disk_vel` made `object_at_rest` spuriously fail. **Fix:** read the coin velocity from `qvel[4:]`.
2. **`data.time` is construction-settling time, not rollout progress.** The home is built by settling the coin for ~1.72 s of sim time, so `data.time/dt ≈ 3440` at a *fresh* home, spuriously failing `step_zero`. **Fix:** the rollout `step` is passed **explicitly** (0 for a fresh home; §6.5 #11 — rollout state passed, not read from ambient sim time), never derived from absolute `data.time`.

A third, mechanical bug: `mj_fullM` in mujoco ≥ 3 takes `(model, data, dst)`, not `(model, dst, sparse_M)`. Fixed. With all three fixed, `certify_zero_home` on the real rig returns **valid, no violations**, and `_robot_ke`/`_object_ke` are exactly `0.0` at rest — the correct measured KE.

## Files touched

| file | LOC | role |
|---|---|---|
| `hymeko_rl/ir/rollout.py` | +55 | generic dynamical snapshot |
| `hymeko_rl/ir/initial_condition.py` | +141 | IC spec + certificate + certificate-filtered distribution |
| `hymeko_rl/ir/hybrid_mode.py` | +93 | modes M0..M7 + invariants + trace validation |
| `hymeko_rl/ir/transition.py` | +87 | guards + handoff + guarded transition |
| `hymeko_rl/ir/energy.py` | +73 | measured ledger + energy certificate (no conservation) |
| `hymeko_rl/ir/provenance.py` | +73 | provenance hash + success certificate |
| `hymeko_rl/ir/__init__.py` | +64 | public IR surface (22 exports) |
| `hymeko_rl/coin_delivery/ir_adapter.py` | +191 | MuJoCo↔IR boundary (coin adapter) |
| `hymeko_rl/tests/test_ir_r11_2.py` | +268 | the 10 gates (fast pure-IR + slow physics) |

No existing file was modified — R11.0/R11.1 code (`coin_zero_home_reach.py`, `coin_zero_home_rrt.py`) is reused read-only via the adapter (the `EnergyProbe` plugs into the existing read-only `frame_hook`). **Behavior of committed code is bit-unchanged.**

## Tests & static gates

- **`ruff check`: clean.** **`radon cc -a -nc`: no C+ block** (`InitialCondition.certify` refactored to extract `_clause_results`, dropping the class from C→B). **`mypy --strict` on `hymeko_rl/ir/`: clean.**
- **Fast (pure IR): 21 passed** in ~0.7 s (`pytest -p no:randomly -m "not slow"`).
- **Slow (physics-backed): 2 passed in 227 s** — `test_certify_zero_home_and_admissibility_against_rig` and `test_instrument_reach_rrt_links_k6_certificate_to_provenance` (real rig, RRT reach + CEM re-solve; the provenance link holds end-to-end). Runtime dominated by the pre-existing CEM capture, not the IR.
- **Coverage rule:** every new public/private function/method is exercised (pure helpers incl. `EnergyProbe.__call__`, `zero_home_reach_trace`, `k6_success_certificate`, `_git_sha` fast; the rig-backed `read_rollout_state`/`certify_zero_home`/`coin_admissibility`/`make_coin_distribution`/`_robot_ke`/`_object_ke`/`measured_reach_ledger`/`instrument_reach_rrt` via the slow tests).

## Performance / resources

R11.2 is non-RL and additive. Fast suite ~0.7 s; the slow suite is dominated by the pre-existing rig build + CEM capture (unchanged from R11.0/R11.1). No RSS-sensitive path introduced (well under the 16 GB cap). No performance budget assertion applies at this boundary (no new hot loop); the reach/capture cost is inherited, not created.

## §6.5 anti-patterns

No anti-patterns introduced. The IR is trait-like (dataclasses + injected predicates), modes are an **enum** (not string-typed), the distribution admissibility is a single injected predicate (no per-axis Cartesian functions), `EXACT_ZERO_HOME_V1` is an immutable program constant (allowed under §6.5 #11), and all state is passed explicitly (no globals; the `disk_vel`/`data.time` fixes above deliberately replaced ambient-sim reads with explicit rollout state). The energy two-level split is *class-per-structural-concern* (measured vs. modelled), matching §6.5 #8.

## Honest limitations / open items

- **Energy measurement is honest but coarse.** `object_ke` uses a unit-mass proxy; `EnergyProbe` integrates work at **waypoint resolution**, not substep; PE = 0 for the planar top-down task (constant height). These are *documented measured proxies* — R11.2 asserts ledger **completeness**, not accuracy. R11.8 (`ModelledHamiltonianCertificate`) calibrates the true masses and the acceptable residual band.
- **`coin_admissibility` uses the cheap geometric predicate** (start collision-free + non-empty goal set), not the full RRT-reach-within-budget + ≤1 mm precontact-motion check. The full certificate filter is the R11.3 demonstration-bank gate.
- The 2–4 mm precontact nudge (R11.1) is unchanged here; closing it with larger obstacle inflation is an R11.3 precondition, not an R11.2 task.

## Provenance

- Parent `bf5e542e` (R11 v2 plan commit). Working tree: only the R11.2 additions listed above. CORE.YAML items touched: **none**. New/removed dependencies: **none** (stdlib + numpy, already pinned).
- Env: Python 3.11 / mujoco 3.10.0 / numpy 2.x / macOS-arm64 (CPU). Deterministic (fixed seeds; RRT seed 0; provenance hash rounds floats to 1e-9). Tooling from the main-tree `.venv` (ruff 0.6.x, radon 6.x, mypy 1.x, pytest 8.x — all within the `tools.yaml` major locks).

## Verdict

`R11_2_HYMEKO_IR_AND_INITIAL_CONDITION_CERTIFICATE_PASS` — the IR + certificates are in place, all 10 gates pass, static gates clean, additive with no behavior change; the production-scale smoke found and fixed two real live-state reader bugs before anything was queued. **Stop at the R11.2 boundary for review before R11.3 (demonstration bank).**
