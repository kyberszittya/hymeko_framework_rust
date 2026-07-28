# R10 Stage 1A — object-generic Geometric Approach layer: HOME_V1 → READY by explicit analytic planar IK (no RL, no BC)

**2026-07-28 · branch `recovery/coin-r9-causal-residual-delivery` · parent `8a7caf6c` · dev s1 (14250) · downstream FROZEN · s4/s7 untouched · f1–f4 SEALED · NO RL · no tag moved**

## Summary

The HOME-start composition audit (`8a7caf6c` line) proved the frozen stack needs exactly one upstream skill — HOME →
pre-capture straddle. This change delivers that skill for the planar gripper **without any learning**, on the insight
(Siciliano) that a planar 2R arm's inverse kinematics is **exact and closed-form**, with the two elbow branches selected
by the sign of the elbow angle. The transit is built as an **object-generic Geometric Approach layer**: object geometry
→ assigned sides → pre-contact tip targets → analytic branch-continuous IK along a validated collision-free tip-space
path → time-parameterised reference tracked through the *same* frozen governed servo.

The retained generic home **`HOME_STATE_V1_GENERIC`** (`[-0.9,-1.4,0.0,2.7]`, both tips on the coin's lower side) is not
overwritten — it is preserved as the topology-changing benchmark and now *solved*. A sibling contract
**`HOME_STATE_V2_READY`** is added as the wide-open reference of the READY set (documented, not a start we transit
through — its left tip is at the arm singularity). The transit hands off at the **~55 mm pre-capture shell** (tips on
their assigned sides, 15 mm surface margin, non-singular), the closest clean staging point for the downstream dynamic
capture.

**Verdict: `HOME_V1_TO_READY_COLLISION_FREE_TRANSIT_PASS`.** From `V1`, teacher-free and with no state edit, the analytic
transit reaches READY with **coin perturbation 0.0 mm, zero contact, a single elbow branch, and slew-feasibility (max
joint step 0.069 ≪ 0.300)**. The geometry that makes the left arm hard is explicit and load-bearing: the coin's far edge
sits at ≈0.316 m from the left base while the arm reaches only 0.30 m, so the far-side route is *unreachable* and the
planner correctly routes the near-base way; a singular start (V2) and the over-the-top route are both *rejected*.

## Gates

| gate | result |
|---|---|
| `SINGLE_BRANCH_MAINTAINED_PASS` (one elbow sign per arm across the path) | ✅ |
| `COIN_UNPERTURBED_DURING_TRANSIT_PASS` (coin_pert ≤ 1 mm, 0 contact) | ✅ — 0.0 mm, 0 contacts |
| `TRANSIT_REACHES_READY_PASS` (both tips in the READY band) | ✅ — L 61.8 mm @ 111.5°, R 51.5 mm @ −111.5° |
| `READY_HANDOFF_DETERMINISTIC_PASS` (READY snapshot branch bit-identity) | ✅ |
| `HOME_V1_TO_READY_COLLISION_FREE_PASS` (all of the above from V1) | ✅ (mode simultaneous) |
| negative control: singular V2 start rejected | ✅ `TransitInfeasible` |
| negative control: over-the-top left route tip-infeasible | ✅ (far edge > reach) |

## What this is / isn't

- **Is:** an explicit, deterministic, auditable kinematic transit. No RNG, no learned policy, no BC. The analytic IK
  round-trips FK to **0.00 mm**; `link_points` matches FK to **0.00 mm**.
- **Isn't:** the dynamic capture. The transit ends with 15 mm surface margin; closing the last gap and building the
  velocity-matched contact into `H_dyn` is the *next* milestone (short-horizon CEM/MPC from READY, then TD3). The hard
  part is unchanged and downstream stays frozen.
- **Object-generic seam:** pre-contact targets come from `ApproachTargets` (`CoinStraddleTargets` is the coin instance);
  a pyramid/prism replaces only the seam — the IK, routing, validation, and servo are shared.

## Files touched

| file | LOC | role |
|---|---|---|
| `hymeko_rl/coin_delivery/theta_option/planar_arm_2r.py` | +172 (new) | exact closed-form 2R IK, calibration, analytic link geometry |
| `hymeko_rl/coin_delivery/theta_option/planar_geometric_approach.py` | +399 (new) | object seam + collision-free planner (tip/link/annulus/slew validation) + sequential fallback + governed-servo controller |
| `hymeko_rl/coin_delivery/theta_option/home_states.py` | +85 (new) | `HOME_STATE_V1_GENERIC` + `HOME_STATE_V2_READY` contracts + `build_home_snapshot` (single source) |
| `hymeko_rl/experiments/coin_kinetic_geometric_transit.py` | +123 (new) | transit gate ladder + negative controls + provenance JSON |
| `hymeko_rl/tests/test_planar_geometric_approach.py` | +217 (new) | 20 tests (unit + integration + perf) |
| `hymeko_rl/experiments/coin_kinetic_home_composition.py` | +5 / −16 | **pure-rename refactor** — re-exports `HOME_Q`/`build_home_state` from `home_states` (single source; verified byte-identical verdict + min_dtz 76.44 mm) |

**CORE.YAML items touched: none** (all work under `hymeko_rl/`, `tests/`, `reports/`, `docs/plans/`; no pinned
dependency). Frozen downstream modules (`kinetic_clone`, `kinetic_handoff_reset`, `kinetic_residual[2]`,
`velocity_transport`, `kinetic_contract`, `hybrid_approach`, `forward_displacement`, `motion_contract`,
`coin_strict_markov_ablation`) — `git status` confirms **untouched**.

## Test results

`pytest -p no:randomly` — **20 passed in 24.8 s** (one module).

- **Unit (analytic, no physics):** calibration cleanliness; IK round-trip (< 1e-3 m); two-branch elbow split;
  `ik_cont` continuity; `link_points` == FK (< 1e-9); coin-straddle opposite sides; tip-feasibility accept near-base /
  **reject far side**; link-clearance accept / **reject coin-crossing**; segment distance; arc direction; pad + branch
  consistency; densify ≤ slew; home-state shapes + snapshot invariants; V2 == analytic IK of the 75 mm targets;
  plan rejects singular V2 start; plan qref slew-feasible + single-branch.
- **Integration (rolled through the governed servo):** V1 transit collision-free (0 contacts, coin_pert ≤ 1 mm,
  min clearance ≥ 40 mm, single branch); execute-transit determinism; READY-handoff snapshot determinism.
- **Performance:** transit wall time median **< 3 s** over 5 iterations (asserted).

**Coverage:** every new public/private function is exercised by ≥ 1 new test. Regression-style negatives (far-side
route, coin-crossing links, singular start) would fail against a naive implementation.

## Performance results

| metric | measured | budget (plan) |
|---|---|---|
| transit roll wall (median of 5) | < 3 s (test-asserted) | < 3 s |
| full gate experiment wall | ≈ 3 s | — |
| peak RSS (experiment) | **0.23 GB** | < 1 GB (hard cap 16 GB) |
| analytic IK round-trip | 0.00 mm | exact |
| coin perturbation during transit | 0.0 mm | ≈ 0 |

No regression comparison applies (new capability). Static analysis: `ruff check` clean on all new/modified files;
`radon cc` shows no block at C or worse (average A / 1.93) — well under the fail-15 gate.

## §6.5 anti-patterns

None introduced. The analytic arm is separated from the transit orchestration (kept each module under the 400-LOC/
≥2-concern heuristic: 172 + 399). The object axis is a **Strategy seam** (`ApproachTargets` protocol), not a per-object
function family. String-typed config avoided (`mode` is `{"simultaneous","sequential"}` at one call site only). The
duplicated home scaffold was **unified** into `home_states` (single source; §6.1). No new globals, no `unwrap`/broad
`except` in non-test code; `TransitInfeasible` carries the failing reason (no silent truncation).

## Provenance

- Parent SHA `8a7caf6c` (clean working tree apart from the files above).
- Env: Python 3.11.15 / mujoco 3.10.0 / numpy 2.4.6 / torch 2.12.0 / macOS-arm64 (CPU). Seed: `S1_SEED` = 14250.
- Coin geometry: disk radius 0.02 m; fingertip sphere radius 0.02 m (⇒ 0.040 m centre-distance contact threshold);
  link capsules radius 0.010–0.012 m. Arm: base (∓0.14, −0.02), l1 = 0.160, l2 = 0.140, reach 0.300, annulus (0.02, 0.30).
- Plan: `docs/plans/2026-07-28-planar-geometric-approach-transit/` (plan.tex/pdf/tikz/mmd, built with tectonic).
- Result JSON: `reports/2026-07-28-planar-geometric-approach-transit/geometric_transit.json`.

## Open issues / follow-ups

1. **Next milestone (not this change):** short-horizon `DYNAMIC_CAPTURE` CEM/MPC from the READY shell → land in `H_dyn`
   → frozen APPROACH → HANDOFF_RESET → R2 → strict K6. Gate `HOME_V1_TO_DYNAMIC_HANDOFF_REACHABILITY_PASS` (≥ 3 planner
   seeds, no state edit); then `HOME_V1_START_STRICT_K6_PASS` (TD3 finishes). If capture reaches `H_dyn` but K6 fails,
   **STOP** and correct the certificate — do not tune the frozen downstream.
2. `HOME_STATE_V1_GENERIC` remains a standalone *global* motion-planning benchmark (arbitrary safe home → READY);
   the analytic layer already solves the fixed V1, a distribution of safe homes is future work (still no RL needed).
3. Deferred (unchanged): full-workspace cross-host on kato14/kato15; C1 dwell-refinement paired panel.

## Status

`HOME_V1_TO_READY_COLLISION_FREE_TRANSIT_PASS`. The kinematic reach is solved deterministically and auditably from the
retained generic home; the learned effort is reserved for the contact-rich dynamic capture. **STOP** — awaiting review
before building the capture CEM (no RL started; downstream frozen; tags untouched).
