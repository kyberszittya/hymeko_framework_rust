# E3a — active wrench-nulling: mechanism demonstrated, proportional controller not yet convergent

**Date:** 2026-07-25 23:58 JST
**Physics:** frozen `RUBBER_TIP_LOW_DRAG_COIN_V2` + `V4`. Deterministic, no RL. O3 stays paused.
**Why:** LFA established that the passive servo reaches G1 but never G2 (null preload wrench) — so an active net-wrench-
nulling controller is required. This is the first E3a controller.
**One-line outcome:** the object-level wrench-nulling **mechanism is demonstrated** — the feedback drives the realized coin
wrench from 0.43–4.25 N toward zero on all three balanced states — but the current **proportional tangential-slide
controller does not converge to a certified state**: on the small round coin it nulls ‖w‖ by *sliding the tips off the
coin* (losing dual contact), not by a balanced squeeze, so `done` (G1∧G2∧G3 with dual-contact dwell) never latches. A
constrained controller (wrench-Jacobian / QP with dual-contact and G3 as hard constraints) is the next step. E3a also
surfaced a structural tension on s7.

---

## The controller (three channels, per the design)

On an already-G1 soft-pinned preload, `null_coin_wrench` runs object-level feedback on the realized coin wrench
`w = [Fx, Fy, τ]` (`realized_coin_wrench`, fingertip-only, validated). Channels separated so they do not fight: the
**radial** target stays the δ penetration servo (common-mode → holds G1 depth); a **tangential** slide on each tip
(`slide = −k_F·(F·t̂) − k_τ·τ`, clamped) cancels the net force's tangential projection and the net torque.

## What happened

| state | ‖w‖ before | ‖w‖ after | dual contact held? | G3 after | `done` |
|---|---|---|---|---|---|
| s1 | 0.43 N | → 0.0 | **no** (tips slid off) | — | no |
| s5 | 2.89 N | → 0.0 | **no** | — | no |
| s7 | 4.25 N | → 0.0 | **no** | infeasible | no |

Two facts, both informative:

1. **The mechanism works but the controller is unstable on this geometry.** An earlier un-clamped run reduced s1 to
   **0.025 N while G3 stayed feasible** (coeff 1.52, F∥ 0.46) — a genuine near-null, launch-capable state existed
   transiently. But the proportional law does not *hold* it: the tangential slide, on a 2 cm round coin, walks the tips off
   the surface, so ‖w‖ reaches zero by **contact loss** (`left_contact ∧ right_contact` fails), which the `done` gate
   correctly refuses. Clamping the slide and gentling the gain traded contact-loss for non-convergence — the proportional
   feedback is simply insufficient for this coupled, contact-fragile system.
2. **s7 exposes a null-vs-G3 tension.** s7's launch feasibility (G3) *comes from* the forward-directed resting force
   (4.25 N — the far-side frame is already a launch at rest). Nulling that resting wrench moves the contact geometry to one
   where the directed grasp solve no longer produces a low-cross forward wrench, so **G3 becomes infeasible**. Null-preload
   and launch-feasibility are not automatically co-achievable at a fixed far-side frame; the controller must null while
   *constraining* G3, not null freely.

## Honest ledger

```
ACTIVE_WRENCH_NULLING_MECHANISM_DEMONSTRATED           PASS (‖w‖ reducible; s1 → 0.025 N with G3 held, transiently)
PROPORTIONAL_TANGENTIAL_CONTROLLER_CONVERGES           FAIL (nulls by contact loss, not a balanced squeeze)
NULL_PRELOAD_AND_G3_CO-ACHIEVABLE_AT_A_FIXED_FRAME     STATE-DEPENDENT (s1 yes transiently; s7 in tension)
CERTIFIED_G1∧G2∧G3_ACQUISITION                          OPEN
```

## Claims / non-claims

**Claimed (measured):** the wrench-null feedback reduces the realized coin wrench toward zero on all three states; a
near-null (0.025 N), still-G3-feasible state for s1 exists transiently; the proportional controller does not converge to a
dual-contact-held certified state (it nulls by sliding off the coin); nulling s7's resting wrench breaks its G3.

**NOT claimed:** that no controller can reach a certified G1∧G2∧G3 acquisition (only that this proportional one does not);
that null-preload and G3 are fundamentally exclusive (s1 shows they coexist transiently) — only that they are in tension at
a strongly launch-biased frame and must be co-optimised. The gains here are tuned by hand, not against a metric.

## Exact next rung

- **E3a v2 — constrained wrench-nulling.** Replace the proportional tangential slide with a **wrench-Jacobian** step
  (`Δq = −J_w⁺ w` via FD of `dw/dq` on the pinned coin) that accounts for the coupling, with **dual-contact and G3 kept as
  hard constraints** (penetration band + `launch_feasibility_certificate` as a barrier / null-space objective), or a small
  per-step QP `min ‖w‖² s.t. Fn ≥ F_min, cone-feasible, F∥ ≥ F∥_min`. The infrastructure (`realized_coin_wrench`,
  `launch_feasibility_certificate`, `null_coin_wrench` scaffold, the G1→G2→G3 gate) is in place; only the inner update law
  changes.
- Then E3b release-only sanity from the certified null state, E3c A2 launch from it, E3d full composition. O3 paused.

---

### Files touched
- `hymeko_rl/coin_delivery/cooperative_launch.py` — `null_coin_wrench` (E3a nulling phase on a G1 env),
  `active_wrench_null_acquire` (acquire + null); config gains `wrench_null_*`.
- (no benchmark committed for E3a — the controller does not yet produce a certifiable result; validated via focused traces.)

### Test results
- Unit: `test_cooperative_grasp` 8/8 pass; ruff clean.
- E3a traces: 3 states (s1/s5/s7), single-thread, seeds 14000+250·i; `realized_coin_wrench` validated against release drift
  (s1 dot 0.997) in the LFA report.

### Preserved unchanged
`SINGLE_TIP_LOW_FRICTION_COIN_V1`, `RUBBER_TIP_LOW_DRAG_COIN_V2`, V2/V3/V4 contracts, the B1 barrier, all prior results.
CORE.YAML items touched: none.
