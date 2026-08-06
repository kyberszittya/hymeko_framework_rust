# Spring-legged AIBO — series-elastic knees store & return launch energy

**Date:** 2026-07-28 (JST) · **Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION.** · **Verdict: `SERIES_ELASTIC_KNEE_LAUNCHES_WHERE_RIGID_CANNOT_BUT_LOADING_IS_MOTOR_BOUND`.**

---

## Why

The rigid-leg full-body hop (running report) rose only **+2 cm and never left the ground**: the
geared knee, capped at a realistic joint speed, cannot extend fast enough to launch the 5.7 kg body.
Real jumping robots and animals use **series elasticity** — a leg spring loaded *slowly* (within the
motor) and released *fast* (a passive catapult) — which decouples launch velocity from motor
velocity. This builds a spring-legged AIBO variant and tests whether the elastic knees enable the
launch the rigid legs could not.

## Model variant (`scenarios/aibo/spring_leg.py`)

`add_knee_springs(mjcf, SpringLegSpec)` injects a MuJoCo joint spring (`stiffness` toward `springref`,
low `damping` = energy return) into each of the four knees of the emitted quadruped. `springref = 0`
sits at the near-straight standing knee, so the spring is neutral at stance — **the spring-legged
model still stands steadily** (settled torso 0.186 m, no spontaneous launch).

## Result — elastic energy return (`ElasticLaunch`, deterministic)

Load the knees to a crouch (−1.0 rad), release the actuator (zero torque → **pure passive spring**):

| leg | torso rise | airborne? | flight steps | peak knee speed at release |
|---|---|---|---|---|
| **rigid geared** | +0.005 m | ❌ | 0 | 4.1 rad/s |
| **spring K=150** | **+0.347 m** | ✅ | 190 | 77.9 rad/s |

`spring_leg.png` (left): the spring leg lifts the torso past the paw-clearance line (all four paws
off the ground); the rigid leg never clears it. The spring returns stored PE as an airborne launch
the rigid knee cannot produce.

**Honesty — this is NOT the retracted capture-step exploit.** That exploit was a *fabricated* 27 rad/s
**motor**. Here the fast (78 rad/s) knee motion is a **passive spring** — physically legitimate series
elasticity, and it is exactly *why* the launch works (the spring is not bound by the motor's speed
cap). A regression test asserts the release speed exceeds the 8 rad/s motor cap *because* it is a
spring, not a motor.

## The honest constraint — loading is motor-torque-bound (your "50 N·m is large" point)

The emitted MJCF declares a ±50 N·m knee `ctrlrange`, but **50 N·m is a placeholder, far too large
for a 5.7 kg robot** (standing needs only ~1.5–3 N·m per knee). A realistic AIBO-class knee servo is
**~5 N·m**. The motor can only *statically* load the spring until its torque equals the spring's
(`θ* = τ_max/K`), storing `½Kθ*²` per knee:

| K (N·m/rad) | statically-loadable θ* | motor-loadable hop ceiling |
|---|---|---|
| 30 | 0.167 rad | ~3.0 cm |
| 150 | 0.033 rad | ~0.6 cm |
| 300 | 0.017 rad | ~0.3 cm |

`spring_leg.png` (right): under a realistic 5 N·m motor the static-load hop ceiling is only a few cm —
and the launch demo stores ~300 J, orders of magnitude more than the ≤2 J the motor can statically
load. **Conclusion: a launch-capable spring cannot be loaded by the small motor statically; the
energy must come dynamically — body weight / landing momentum — the SLIP regime.** That the loading
is body-weight-driven, not motor-driven, is the whole point of elasticity.

## Files

```
scenarios/aibo/spring_leg.py         NEW  (SpringLegSpec, add_knee_springs, build_spring_legged, ElasticLaunch, stands, static_load_limit)
scenarios/aibo/render_spring_leg.py  NEW  (spring vs rigid launch + static-load hop ceiling)
tests/test_aibo_spring_leg.py        NEW  11 tests
reports/2026-07-27-aibo-hop/spring_leg.png  NEW
```

## Tests / lint

`ruff` clean. **11/11 spring-leg tests pass**; full AIBO suite **49/49**. Locks: the spec rejects
non-positive stiffness / negative damping; `add_knee_springs` sets all four knees; the spring model
stands (doesn't self-launch); the spring launches airborne where the rigid leg does not; the release
speed exceeds the motor cap (passive spring); the launch is deterministic; the static-load limit
shrinks with stiffness; and the realistic 5 N·m motor cannot statically load a launch spring.

## Bottom line

The spring-legged AIBO exists, stands, and its **series-elastic knees do what the rigid legs could
not: return stored energy as an airborne launch** (+35 cm vs +0.5 cm). The remaining honesty is the
**loading**: a realistic small (~5 N·m) knee motor cannot statically compress a launch-capable spring
— so a real hop must load the spring dynamically (body weight / landing), a passive SLIP limit cycle.
That closed-loop hopping controller on the articulated model is the honest next step; the elastic
mechanism and its energy return are certified here.

---

## Follow-up — spring-hop TO A GOAL (`scenarios/aibo/spring_hop_gait.py`)

User: "designate a goal it must reach" — then: *"almost good, but it stops to stabilise itself
instead of continuing forward with the momentum."* A green reach disk is placed ahead; the
spring-legged AIBO reaches it with a **continuous, momentum-preserving** spring bound.
`SpringHopGait.run()` loops a stride: **LOAD** (crouch the knee springs) → **LAUNCH** (release =
passive-spring vertical lift + a motor-limited hip-flex push, ∝ remaining distance) → **CATCH** (a
*short* catch that PD-holds only the **upright-critical joints** — knees for height, hip-abduction
for roll — while the hip-flex joints carry a small **sustained forward drive**, so the body does
**not** brake to a stand and the forward momentum passes through into the next stride). Once inside
the reach zone it **ARRIVES** (comes to an upright rest at the goal).

**The fix (momentum, not braking):** the old version PD-held the *full* standing posture back to q0
each hop, which drove the forward velocity to ~0 — it stopped and re-stood every hop. Replacing that
with the momentum-preserving catch keeps the forward velocity alive **through** each stride
(mid-run catch-phase vx mean **+0.9 m/s**, never braked to 0), so the gait *flows* forward.

**Result:** reaches **x = start + 0.8 m** in **6 strides** (was 9 hops), **1.2 m in 7 strides**,
**upright throughout** (min uprightness 0.95–0.97, never flips), carrying its momentum between
strides, then settles upright at the goal.

**Honest torque budget (your "50 N·m" point applied consistently):** *every* leg actuator torque is
clamped to a realistic motor — the forward hip push ≤ **5 N·m**, the catch/arrive drive & posture
hold ≤ **8 N·m** (a regression test asserts both). Only the **vertical lift is the fast passive
spring**; the horizontal drive is a real ≤8 N·m actuator, not an injected velocity.

Video `aibo_spring_hop_goal.mp4` (+ `.gif`): the AIBO bounds forward continuously, phase-labelled
(LOAD/LAUNCH/CATCH/ARRIVE), remaining-distance + uprightness overlaid, and finishes standing **on the
green goal disk**.

```
scenarios/aibo/spring_hop_gait.py            NEW  (SpringHopGait: LOAD->LAUNCH->CATCH->ARRIVE, momentum-preserving)
scenarios/aibo/render_spring_hop_goal_video.py NEW  (goal-disk hop-to-goal video)
tests/test_aibo_spring_hop_gait.py           NEW  8 tests
reports/2026-07-27-aibo-hop/aibo_spring_hop_goal.{mp4,gif}  NEW
```

**Tests:** 8/8 gait tests (reaches goal, stays upright, monotone forward progress, farther goal =
more strides, **catch preserves forward momentum**, push ≤5 N·m + all torque ≤8 N·m, goal>0
validated, deterministic); full AIBO suite 57/57.

**Bottom line:** with a designated goal, the spring-legged AIBO **reaches it by a continuous forward
bound** — passive elastic lift, motor-limited forward drive, upright throughout, **carrying its
momentum between strides** instead of stopping to re-stabilise. This is the goal-directed hopping
locomotion the rigid legs could not do, inside the honest torque budget.
