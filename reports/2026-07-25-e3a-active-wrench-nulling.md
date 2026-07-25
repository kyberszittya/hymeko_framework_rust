# E3a — active wrench-nulling: G3 corrected to incremental; soft-constraint control insufficient, hard QP is next

**Date:** 2026-07-26 00:30 JST
**Physics:** frozen `RUBBER_TIP_LOW_DRAG_COIN_V2` + `V4`. Deterministic, no RL. O3 stays paused.
**Why:** LFA established the passive servo reaches G1 but never G2 (null preload wrench), so an active net-wrench-nulling
controller is required. This builds it, corrects the G3 semantics, and localises the remaining gap.
**One-line outcome:** two results. (1) **G3 is corrected to INCREMENTAL feasibility** (`∃ Δf ∈ C : G·Δf ≈ [F∥*;0;0]` — at
rest don't move the coin, on command push it forward), which dissolves the apparent G2-vs-G3 tension: **s7 is now
G3-feasible after its wrench is nulled** (coeff 1.48, F∥ 0.56). (2) The wrench-null **mechanism works** but **soft-constraint
control cannot converge** — with a *soft* penetration weight, a strong weight holds contact but barely nulls (s1
0.43→0.41 N) and a weak weight nulls but slides the tips off the coin (→0 by contact loss). Enforcing `Fn > 0` needs a
**hard-constraint QP**, exactly as specified. Decision flagged: scipy is available but is a §1 dependency choice.

---

## G3 corrected — incremental, not absolute (the key semantic fix)

The old G3 asked "does the resting contact already push forward", which conflates with G2 and makes a null preload look
launch-infeasible. The correct condition (per the analysis) is **incremental**: from the preload `f_preload` (with `w ≈ 0`),
can a positive forward wrench *change* be commanded — `∃ Δf ∈ C : G(f_preload + Δf) − G·f_preload ≈ [F∥*; 0; 0]`. Since G is
linear the increment is the directed grasp solve, so `launch_feasibility_certificate` now gates on: forward direction inside
the cone (coeff > 0) ∧ the directed solve reaches F∥ ≥ min ∧ a small wrench residual. **Effect: s7, whose resting 4.25 N
forward force previously "broke G3" when nulled, is now correctly G3-feasible at the null state** — null-preload and
launch-feasibility are no longer in false conflict.

## The controller — from proportional to constrained, and the wall

- **v1 proportional tangential slide** (`slide = −k_F·(F·t̂) − k_τ·τ`): reduced ‖w‖ (s1 0.43→0.025 transiently with G3
  held), but on the 2 cm round coin it nulls by *sliding the tips off* → dual contact lost, `done` never latches.
- **v2 constrained least-squares** (`null_coin_wrench`): each step solves `Δq = argmin ‖J_w Δq − (w*−w0)‖² +
  μ‖J_p Δq − (p*−p0)‖² + λ‖Δq‖²` with the FD wrench/penetration Jacobians (`_wrench_penetration_jacobian`), so nulling and
  holding the δ penetration are traded in one solve. This is principled and cannot slide off *if* penetration is held — but
  the penetration term is **soft**:

| penetration weight | s1 ‖w‖ 0.43 → | contact held? | s7 ‖w‖ 4.25 → | s7 G3 |
|---|---|---|---|---|
| 40 | 0.0 | no (slid off) | 0.0 | feasible |
| 200 | 0.0 | no | 0.0 | feasible |
| 400 | **0.41** | yes | 0.0 | feasible |

The trade-off is intrinsic to a soft constraint: heavy penetration weight preserves contact but forbids the nulling motion;
light weight nulls but drops `Fn` through zero. **A soft objective cannot keep `Fn > 0` while minimising ‖w‖** — the two
are opposed near the contact boundary. (s7 nulls to 0 at every weight because its large forward force gives the solver an
easy contact-preserving direction; s1's small wrench does not.)

## Honest ledger

```
G3_SEMANTICS_CORRECTED_TO_INCREMENTAL                  PASS (s7 G3-feasible at the nulled state)
ACTIVE_WRENCH_NULLING_MECHANISM_DEMONSTRATED           PASS (‖w‖ reducible; s1 → 0.025 N with G3 held, transiently)
SOFT_CONSTRAINT_CONTROL_CONVERGES                      FAIL (contact-hold vs null is an intrinsic soft trade-off)
HARD_CONSTRAINT_QP_REQUIRED                            ESTABLISHED
CERTIFIED_G1∧G2∧G3_CRADLE                              OPEN (needs the QP)
```

## Claims / non-claims

**Claimed (measured):** the incremental G3 makes s7 launch-feasible at the nulled state; the wrench-null mechanism reduces
‖w‖ on all three states; a soft-penetration constrained least-squares cannot simultaneously hold `Fn > 0` and null the
wrench (demonstrated weight sweep: 400 holds contact but ‖w‖ stays 0.41; ≤200 nulls but loses contact).

**NOT claimed:** that no controller can reach a certified G1∧G2∧G3 cradle (a hard-constraint QP is untested); that the QP
will succeed (it is the principled next attempt, not a proven result). Gains are hand-set, not metric-optimised.

## Exact next rung (decision required)

- **E3a v3 — hard-constraint QP.** `min_Δq ‖w_coin(Δq)‖² + λ‖Δq‖²` s.t. **`Fn_i + J_{Fn}·Δq ≥ F_min`** (dual contact as a
  hard inequality), `|Ft_i| ≤ μ Fn_i`, Δq bounded, incremental forward feasibility > ε. Then the INSERT mode is the *same*
  controller with `w* = [F∥*·e_par ; 0]` — a synchronised cooperative push — giving the canonical skill
  `ACQUIRE → NULL PRELOAD → SYNCHRONISED INSERT → RELEASE → B1 → SETTLE` (the `w_target` argument of `null_coin_wrench` is
  already wired for this). Then E3b drift-free unpin, E3c cooperative insertion on the mobile coin, E3d composition.
- **§1 dependency decision:** the QP wants a solver. `scipy` 1.17.1 is installed (transitively) but is not an explicit
  project dependency, so adopting `scipy.optimize` in the core algorithm path is a §1 call — **or** I implement a small
  dependency-free active-set QP in pure numpy. Awaiting your preference before building E3a v3.

---

### Files touched
- `hymeko_rl/coin_delivery/cooperative_launch.py` — `launch_feasibility_certificate` (incremental G3 + `g3_residual_max`);
  `_wrench_penetration_jacobian` (FD `dw/dq`, `dpen/dq`); `null_coin_wrench` rewritten as the constrained least-squares
  controller with a `w_target` (null / insert) argument; `active_wrench_null_acquire`; config `wrench_null_*` / `jac_*`.

### Test results
- Unit: `test_cooperative_grasp` 8/8 pass; ruff clean.
- E3a traces: 3 states (s1/s5/s7), penetration-weight sweep {40, 200, 400}; single-thread, seeds 14000+250·i.
  `realized_coin_wrench` validated against release drift (s1 dot 0.997) in the LFA report.

### Preserved unchanged
`SINGLE_TIP_LOW_FRICTION_COIN_V1`, `RUBBER_TIP_LOW_DRAG_COIN_V2`, V2/V3/V4 contracts, the B1 barrier, all prior results.
CORE.YAML items touched: none.
