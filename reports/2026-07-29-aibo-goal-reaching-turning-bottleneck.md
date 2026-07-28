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

## Next (proposed)

A **turning / goal-reaching RL problem**: reward = reduce heading error + reach the goal; action = a
residual on the gait's turn/steer (or raw leg targets); compare **HSiKAN (structural) vs MLP (flat)** on
turn-rate and goal-reach across bearings. First establish that *any* learned policy can turn effectively
(fixing the mechanism wall), then test whether structure helps.

## Files / tests

```
scenarios/aibo/residual_trot.py     MOD  turn_first_deg/turn_drive config + _base_drive() helper (turn-in-place drive schedule)
tests/test_aibo_turn_first.py       NEW  4 tests: default off / within-reach stop / wide-bearing drive cap / narrow full drive
```

`ruff` clean; **125/125** AIBO tests green. CPU, seed 0, MuJoCo. CORE.YAML: none. SIMULATION.
