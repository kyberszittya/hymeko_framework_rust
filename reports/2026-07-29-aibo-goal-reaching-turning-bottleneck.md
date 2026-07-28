# The AIBO goal-reaching bottleneck is TURNING — the scripted skid-steer barely rotates

**Date:** 2026-07-29 (JST) · **Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION. Diagnostic + one scaffold knob.** · Refocus (user): the main goal is **AIBO reaches the
target**, not a symmetric crab. **Verdict: `SCRIPTED_TURNING_IS_THE_GOAL_REACHING_WALL`.**

---

## Refocus

The crab-symmetry axis was rigorously closed (multi-seed: all recipes tied at 0.40 goal-reach — symmetry
barely moves the objective). Back to the real goal: **does the AIBO reach a designated target at any
bearing?** The crab was a workaround for weak turning; a quadruped's natural goal-reaching is
**turn-to-face + walk**.

## Baseline — the AIBO only reaches straight goals

Pure scaffold (trot + heading-pursuit, `a=0`) over a bearing distribution (0, ±20, ±40, ±90, ±135°) at two
distances: **2/18 = 0.11 reached — only bearing 0**, at both distances. Uprightness ≈ 1.0 everywhere (the
robot is stable), and for the wide bearings the min-distance **stays at the start distance** (0.5→0.5,
0.7→0.7): the robot **does not turn toward the goal**, it walks straight past.

## Root cause — the turning mechanism produces almost no yaw

Tracing heading error toward a 90° goal over 2400 steps: **90° → 75°** — only ~15° of turning when 90° is
needed. Measuring the `AgileTurnGait` yaw authority directly (the "agile" reverse-inner skid-steer):

| turn cmd | 0.2 | 0.3 | 0.4 | 0.5 (pivot) | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|---|
| yaw °/1000 steps | 19 | 15 | 14 | 10 | **−10** | **−14** | 5 |
| min upright | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.99 | 0.97 |

**Max ~19°/1000 steps, and the sign FLIPS at turn ≥ 0.6** (reverse-inner turns the *wrong* way). Nothing
tips. This is **budget-robust and not the governor**: identical with and without the motion-contract
velocity governor. The memory's "~168°/1000" was from an earlier setup and does **not** reproduce here.
So the **scripted skid-steer turning is fundamentally too weak** for this AIBO's dynamics — differential
stride does not generate enough turning torque, and reversing the inner legs is counter-productive.

`turn_first` knob (cut forward drive to turn in place when the heading error is wide) was added and tested
— it caps the drive as specified, but does **not** help (2/18 → 2/18): the skid-steer needs *stride* to
yaw, so cutting the drive removes the (already tiny) turning too. The mechanism, not the drive schedule,
is the wall.

## Why this is the right problem for LEARNING + HSiKAN (user's insight)

Turning is **whole-body coordination**: the heading error must be routed through the body's kinematic
structure into a coordinated, differential per-leg stepping pattern. The scripted skid-steer is one weak
hand-authored strategy; a **learned** policy over the same leg actuators may discover an effective turning
gait the script cannot. And — the user's point — this is exactly where **HSiKAN's structural propagation
(obs → structure → per-leg action)** could beat a flat MLP, unlike the crab (a simple lateral push that
needed no structure, hence HSiKAN ≈ MLP there). Turning is the structured coordination problem HSiKAN was
meant for.

## Action-space check → FIX: the rotational-couple turn (0.11 → 0.50 goal-reach)

Before RL, the cheap check (does the *action space* allow strong turning?). Hand-crafted turn patterns,
yaw authority:

| pattern | yaw °/1000 | upright |
|---|---|---|
| skid-steer (scripted) | 19 | 1.0 |
| aggressive spin / abduction-row | −11 … −8 | tips |
| **rotational hip couple** (diagonal legs OPPOSITE: fl+,fr−,bl−,br+) `g=1.0` | **47** | **0.98** |
| rotational hip couple `g=1.5` | 169 | −0.34 (tips) |

**The action space turns fine — the script used the wrong pattern.** The rotational couple (diagonal
pairs stride in opposite directions → a real yaw couple) is 2.5× the skid-steer and stable at `g=1.0`,
with headroom to ~169°/1000 (tipping) — a rate-vs-stability frontier. Wiring a **turn-then-walk** scaffold
(rotational couple to face the goal, then trot in) lifts goal-reach **0.11 → 0.50** (9/18; near-to-mid
bearings 0/±20/±40 and +90° now reach). Delivered as `RotationalTurnGait` + `heading_mode="turn_then_walk"`
(default `"arc"` unchanged).

## Next (proposed)

The turn-then-walk scaffold (0.50) is now a **working base to residual over** (as the trot was for the
crab). The wide bearings (±135°) and the occasional tip at the turn→walk transition are the headroom. A
bounded-residual RL over this scaffold — reward = reduce heading error + reach — is the refinement, and
the natural **HSiKAN vs MLP** test: routing the heading error through the body structure into the
coordinated turn-couple + walk transition is the whole-body-coordination problem where structure may help.

## Files / tests

```
scenarios/aibo/residual_trot.py     MOD  turn_first_deg/turn_drive config + _base_drive() helper (turn-in-place drive schedule)
tests/test_aibo_turn_first.py       NEW  4 tests: default off / within-reach stop / wide-bearing drive cap / narrow full drive
```

`ruff` clean; **125/125** AIBO tests green. CPU, seed 0, MuJoCo. CORE.YAML: none. SIMULATION.
