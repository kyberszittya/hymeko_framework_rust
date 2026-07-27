# R7 — velocity-regulated transport: V0 correct, V1 hits the SOFT-FRICTIONAL-GRIP wall (RL still blocked)

**Date:** 2026-07-27 (JST) · **Branch:** `recovery/coin-r7-velocity-regulated-transport` (from R6 tip `90142eea`) ·
Contract: `reports/2026-07-27-coin-r7-velocity-regulated-transport-contract.md`.

**Result in one line:** the R7 velocity-servo primitive is built and **V0-correct** (the pure servo + saturating
non-reversing stop are unit-verified), but the **V1 mechanism gate fails 0/4** for a *physical* reason: a Δτ-**increment**,
slew-limited servo cannot velocity-regulate a coin held by a **soft frictional grip** — the coin is dragged by the gripped
tips whose accumulated torque + momentum the increment servo cannot counteract in time, so it **blows up** (vpar 1.09 m/s,
far over `v_max=0.35`) and **reverses** (vpar → −1.19, coin flies backward to 386 mm) *despite* the non-reversing command
clamp. The clamp guards the **command** (`a_cmd·Δt` prediction); it does not guard the **physics** (contact force +
momentum). Verdict **`VELOCITY_SERVO_THROUGH_FRICTIONAL_GRIP_DOES_NOT_REGULATE`**. **SAC/TD3 remain BLOCKED** (V4 not
reached).

## 1. What was built (committed `261ad506`)

- `theta_option/velocity_transport.py` — a **new per-step control law inside the frozen physics** (the frozen
  `_schedule_increment` is NOT used; the governed step + K6 + the frozen force *directions* `arm_jac_dir` / `arm_inward_geom`
  are reused). Pure servo `velocity_ref` (decays to 0 at the zone) / `accel_cmd` (saturated) / `non_reversing_accel`
  (positive motion → rest, never → escape); the `VelocityTransportController` (cradle-agnostic — no intent/decoder) + the
  `velocity_rollout` driver (rollout_primitive-shaped metrics; `_accumulate` factored out; all fns < CC 15).
- **V0 unit tests pass** (bounded `v_ref` ∈ [0,v_max]; `accel_cmd` saturation; the non-reversing stop brings any positive
  motion to *rest*, never negative; determinism). 15 fast tests total; ruff clean.

## 2. V1 — mechanism (cradle-agnostic servo, all 4 states)

| state | dtz start→end | peak coin speed | max\|vpar\| | reversed? | delivered |
|---|---|---|---|---|---|
| s1 | 76→176 mm | 0.72 | 0.52 | **yes** | ✗ |
| s3 | 100→386 mm | 1.37 | 1.35 | **yes** | ✗ |
| s4 | 96→264 mm | 1.18 | 0.95 | **yes** | ✗ |
| s7 | 138→231 mm | 1.05 | 1.00 | **yes** | ✗ |

**0/4; every state blows up over `v_max` and reverses.** Trace (s3): the coin builds to vpar 0.65 by t11 (already 2.8× the
`v_ref=0.23` there), reaches the zone at t16 (dtz 18) at vpar **1.09**, then reverses to **−1.19** and flies out.

## 3. Why — the soft-frictional-grip wall (the load-bearing finding)

The servo commands a torque **increment** `Δτ` (slew-limited to ±`τ̇·dt`), but the coin's velocity is set by the
**accumulated absolute torque + momentum**, transmitted through a **compliant frictional grip** (the coin is *dragged by the
gripped tips*, not a rigid velocity source). Consequences the V0-correct algebra cannot prevent:

1. **Windup / overshoot.** During transport the servo builds forward torque; as `v→v_ref` the command → 0, but the built-up
   torque + tip/coin momentum overshoot, and the slew-limited increment cannot bleed the torque off fast enough ⇒ the coin
   blows past `v_max`.
2. **Physics-driven reversal.** The non-reversing clamp only guarantees `v_par + a_cmd·Δt ≥ 0` for the *commanded* accel; the
   *realised* coin acceleration is the contact-force + momentum response, which is not `a_cmd`. When the servo brakes hard
   (v_ref = 0 in-zone), the reversing tips drag the coin backward through the grip ⇒ vpar → −1.

So a proportional velocity servo on the coin, through a Δτ-increment interface and a compliant frictional grip, is not a
valid velocity regulator — the tip→coin coupling is soft contact, not a commandable velocity source.

## 4. Where this points

The V0 primitive and the certificate remain sound; the failure is the *coin-velocity-through-grip* control assumption. Three
redirections (fresh contract), most-specific first:

1. **Regulate the TIP velocity, not the coin velocity**, with a torque-**target** (not increment) servo + anti-windup: the
   joint→tip map is direct and stiff; the coin then follows through the grip. This keeps R7's `v_ref`/non-reversing law but
   moves the regulated variable to the observable, directly-actuated one.
2. **Stiffer grip + a computed hold torque** so the coin is closer to a rigid follower before regulating its velocity.
3. **Learned control** (RL over the servo residual) that captures the nonlinear soft-contact tip→coin map — the axis the
   whole arc keeps pointing to, still gated behind the same 4/4 held-out-2/2 delivery gate.

## 5. The cumulative arc (R1→R7) — a clean convergent negative

| axis | ruled out | obstacle revealed |
|---|---|---|
| R1/R2/R3 | flat / relational / physical-intent prediction | held-out decision not inferable from 6 dev cradles |
| R4 | fixed-coast closed-loop correction | cradle-specific settle timing |
| R5 | online coast estimation | `R_coast` is a post-release (unobservable) quantity |
| R6 | brake-to-stop over the frozen option | frozen accelerating-push/opposing-brake can't hold speed nor stop at zero |
| R7 | velocity-regulated transport primitive | coin-velocity-through-a-soft-frictional-grip is not a valid deterministic control target |

Each deterministic approach fails for a **distinct, measured** reason, and every reason traces to the same root: the
**soft-frictional, cradle-specific contact dynamics** are nonlinear in ways no fixed law has captured. The release
**certificate** (R6) and the **non-reversing stop / velocity-reference** algebra (R7) are the durable, reusable pieces; the
missing element is a **learned or tip-referenced** contact controller. **SAC/TD3 remain BLOCKED** until a controller clears
the delivery gate.

## 6. Status

`velocity_transport.py` built + V0 unit-tested; 15 fast tests + the slow GOLDEN pass; ruff clean; all fns < CC 15. **Verdict:**
`VELOCITY_SERVO_THROUGH_FRICTIONAL_GRIP_DOES_NOT_REGULATE` (V1 fail). **SAC/TD3 BLOCKED** (V2–V4 not reached). **Exact next
action:** a fresh contract for a **tip-referenced, torque-target velocity regulator with anti-windup** (§4.1), keeping the
R7 `v_ref` + non-reversing law and the R6 certificate; V1→V4 then RL.
