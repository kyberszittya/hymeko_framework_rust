# Launch-feasible acquisition — the corrected option contract, and G2 (null preload wrench) is the wall

**Date:** 2026-07-25 23:10 JST
**Physics:** frozen `RUBBER_TIP_LOW_DRAG_COIN_V2` + `V4`. Deterministic, no RL. O3 stays paused.
**Why:** E2B's semantic audit showed `balanced contact ≠ launch-capable contact` — E1's Fn-balance was the wrong ACQUIRE
option-postcondition. This replaces it with the correct lexicographic contract and measures where it holds.
**One-line outcome:** with the corrected contract **G1 clean contact → G2 null realized preload wrench → G3 launch-wrench
feasible**, the current servo acquisition clears **G1 on 7/72 candidates but G2 on 0/72** — it can make clean *contact* but
never a net-wrench-null *preload* (the uncancelled rubber-μ tangential force). So **no state has a launch-feasible
acquisition under the current controller**, which establishes that an **active net-wrench-nulling controller (E3) is
necessary, not an optional later refinement**.

---

## The corrected option contract (lexicographic, replacing Fn-balance)

`done = True` only when all three gates pass — so ACQUIRE terminating means the next option (LAUNCH) is executable:

- **G1 clean contact** — dual-contact dwell, bounded penetration, settled qdot, no torque saturation (`acquire_clean_preload`).
- **G2 null preload wrench** — the realized NET fingertip wrench on the coin `‖(Fx,Fy)‖ ≤ 0.30 N ∧ |τ| ≤ 0.010 N·m`
  (`realized_coin_wrench`). This is the *physical* preload target; Fn-balance is only a proxy — asymmetric contact points
  can be Fn-balanced yet carry a large net tangential force (here the rubber tip μ = 2.0 makes tangential forces up to
  μ·Fn per tip, which do not cancel).
- **G3 launch feasibility** — forward direction inside the friction cone (`forward_feasibility` coeff > 0) *and* a directed
  grasp solve with F∥ ≥ 0.10 and low cross (`launch_feasibility_certificate`).

Fn-balance and the far-side sign are demoted to **diagnostics / candidate-ordering priors**; the decision is G1∧G2∧G3.

## G2 is validated — the net wrench predicts the release drift

Before trusting G2 as the binding gate, `realized_coin_wrench` was checked against the physical effect it should cause.
Summing **only the fingertip–disk contacts** (excluding floor / boundary / arm-link reactions, which are static and do
not drive drift), the net wrench direction matches the passive-release drift direction:

| state | net wrench ‖F‖ (N) | net-force dir | passive drift dir | alignment (·) |
|---|---|---|---|---|
| s1 | 0.43 | (0.74, −0.68) | (0.68, −0.73) | **0.997** |
| s7 | 4.25 | (−0.99, −0.17) | (−0.99, 0.17) | **0.943** |

(An earlier version summed *all* disk contacts and was dominated by three horizontal boundary contacts (~1.65 N each),
giving a net that mis-aligned with the drift (s1 · = −0.07) — the fingertip-only restriction is the fix.)

## Result — the funnel

| gate | candidates passing (of 72) | states with ≥1 |
|---|---|---|
| G1 clean contact | **7** (s1:2, s5:2, s7:3) | 3/8 |
| G2 null preload wrench | **0** | 0/8 |
| G3 launch feasible | 0 | 0/8 |
| **done (G1∧G2∧G3)** | **0** | **0/8** |

Verdict: `NO_LAUNCH_FEASIBLE_ACQUISITION_UNDER_CURRENT_SEARCH`. **G2 is the wall.** Every clean-contact candidate carries a
non-null net wrench: s1's best is 0.43 N (just over the 0.30 threshold → 2.6 cm passive drift), and s7's far-side frame
is 4.25 N **forward-directed** — it is already *launching* the coin at rest, not holding a null preload (which is exactly
why it drifts forward and why A2 launches strongly from it). The servo balances *penetration*, not the *net wrench*, and
the high tip friction leaves a large uncancelled tangential component.

## What this establishes

- The Fn-balance acquisition postcondition is replaced by a physically-grounded lexicographic gate; the E2B mis-abstraction
  is fixed in code, not just in prose.
- **E3 (active net-wrench-nulling) is necessary**: no passive/servo acquisition in the ±4 cm sweep produces a G2-clean
  preload, so a controller that explicitly drives the realized net wrench (common-mode preload + differential Fn +
  contact-frame offset) to zero is the required next step — not optional.
- The far-side frame (s7) is launch-*capable* but not preload-*null*; E3 must null the net wrench **while keeping the
  frame launch-feasible** (G3), i.e. reduce the resting forward force without losing the forward authority.

## Claims / non-claims

**Claimed (measured + validated):** the corrected contract holds; `realized_coin_wrench` predicts the release drift
(alignment 0.94–0.997); under the current servo, G1 passes on 7/72 candidates but G2 on 0/72, so no launch-feasible
acquisition exists in the search.

**NOT claimed:** that no acquisition controller can reach G2 (E3 untested); that the G2 thresholds (0.30 N, 0.010 N·m) are
final (calibrated to a ~1 cm release-drift budget); that a wider / 2-D search would not find a G1∧G2∧G3 frame. The
`forward_feasibility` coefficient is a directional sign gate, **not** a physical force magnitude (pass `fn_max` for the
bounded magnitude).

## Exact next rung

- **E3 — active net-wrench-null acquisition controller.** Three channels (common-mode total preload, differential Fn_L−Fn_R,
  slow contact-frame offset toward the launch-feasible authority centre); release-ready gate over several frames: small
  ‖w_net‖, small d‖w_net‖/dt, quiet coin, settled qdot, no saturation, valid penetration, and G3 still feasible. Then the
  paired A0-vs-A2 launch from a G1∧G2∧G3 preload is finally a clean comparison. Only then demos / proposal / RL. O3 paused.

---

### Files touched
- `hymeko_rl/coin_delivery/cooperative_launch.py` — `forward_feasibility` (renamed from `forward_authority`; directional
  coefficient + optional `fn_max` bounded magnitude), `realized_coin_wrench` (G2, fingertip-only, validated),
  `launch_feasibility_certificate` (G3), `launch_feasible_acquisition_search` (lexicographic G1∧G2∧G3); config gate
  thresholds `g2_*`, `g3_*`.
- `hymeko_rl/experiments/bimanual_launch_feasible_acquisition_benchmark.py` — the funnel benchmark.
- `hymeko_rl/tests/test_cooperative_grasp.py` — feasibility-oracle tests updated to `forward_feasibility` (+ bounded form).

### Test results
- Unit: `test_cooperative_grasp` **8/8** pass; ruff clean.
- Benchmark: 8 states × 9-candidate sweep, ~14 min wall, single-thread, seeds 14000+250·i.
- Artifact: `reports/2026-07-25-coin-dynamics-contract-v2/launch_feasible_acquisition.json`.

### Preserved unchanged
`SINGLE_TIP_LOW_FRICTION_COIN_V1`, `RUBBER_TIP_LOW_DRAG_COIN_V2`, V2/V3/V4 contracts, the B1 barrier, all prior results.
CORE.YAML items touched: none.
