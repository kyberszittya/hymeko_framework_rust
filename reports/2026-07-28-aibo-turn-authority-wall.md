# The AIBO turn/stability wall — why off-axis goals are unreachable (the richer-primitive verdict)

**Date:** 2026-07-28 (JST) · **Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION. Model-based diagnostic (no RL).** · **Verdict: `TURN_AUTHORITY_STABILITY_WALL_BOTTLENECKS_OFF_AXIS_REACH`.**

---

## Why

The residual-RL arc concluded that no bounded residual over the trot gait extends multi-position
reach (leg REGRESS / phase REGRESS-less / steer MATCH), and pointed at a **richer turning primitive**
as the real lever. This builds and *measures* that primitive directly — and it hits a wall.

## The primitive + the measurement (`turn_authority.py`)

`AgileTurnGait` generalises the trot's weak "reduce-inner" skid-steer to a **reverse-inner in-place
spin** (inner legs stride backward: `side = 1 − 2·|turn|`, so |turn|=1 = full spin) — the strongest
skid-steer turn available. `measure_turn` / `stable_turn_ceiling` sweep the turn magnitude:

| turn | yaw rate (deg / 1000 steps) | min uprightness | tipped? |
|---|---|---|---|
| 0.3 | **15.4** | 0.997 | no |
| 0.5 | 10.1 | 0.996 | no |
| 0.7 | −13.6 | 0.99 | no (but sign-flips — asymmetric) |
| 0.9 | −11.9 | **−0.48** | **YES** |
| 1.0 | −0.6 | −0.31 | **YES** |

**The wall:** the fastest *stable* (upright) turn is only **~15°/1000 steps**, and beyond
**|turn| = 0.9 the robot tips over** (uprightness → negative). Facing a ±40° goal needs ~2600 steps
of pure *stable* turning — longer than a reasonable episode — and turning any faster **falls over**.

## Two approaches, one wall (turn-then-walk pursuit, measured)

A hand-built turn-then-walk pursuit (spin to face the goal, then walk) confirmed both horns:
- with a long horizon the slow stable turn *does* align the heading (min |heading err| → ~0.1°) — but
  aligning + walking oscillates and rarely closes distance in time;
- a max-spin pursuit "reached" 3/10 off-axis goals **only by tipping over** (uprightness −0.27 to
  −0.45 on every off-axis reach — the robot flips and tumbles near the goal; invalid). Only the two
  *straight* goals reach **upright**.

So off-axis reaching is bottlenecked by the **turn/stability tradeoff**: stable turning is too slow,
and fast turning tips — exactly the instability the campaign found earlier with abduction turning
(`reports/2026-07-27-aibo-simple-scenarios`). It is a **kinematic/dynamic property of the skid-steer
on this AIBO model**, not a controller or learning deficit.

## The full answer to "can we train the AIBO to reach multiple positions?"

Convergent evidence across **four** designs (3 residual RL + 1 hand-built primitive): **no** — on
this model, multi-position reach is capped by yaw authority. The residual can't help (a raw residual
breaks the gait; a param residual can't exceed its kinematic reach), and the richer primitive can't
either (stable turn too slow; fast turn tips). The real lever is a **more agile embodiment** (or a
genuinely different locomotion — a proper stepping-turn that repositions feet without skid-steer
slip), not more control/learning over the current trot. Honest, well-characterised close.

## Files

```
scenarios/aibo/turn_authority.py       NEW  (AgileTurnGait reverse-inner spin + measure_turn / stable_turn_ceiling diagnostic)
tests/test_aibo_turn_authority.py      NEW  4 tests
```

## Tests / provenance

`ruff` clean. **4/4** turn-authority tests (valid action; stable turn is slow; strong turn tips; the
stability wall has a small stable rate + a tipping onset); full AIBO suite green. Diagnostic only —
`AgileTurnGait`'s fast-turn regime is a *characterised unstable exploit*, measured to locate the
tipping onset, **not** shipped as a locomotion controller. CPU, seed 0, MuJoCo, base=free.

## Bottom line

The richer turning primitive was built and measured: the AIBO's **stable turn is ~15°/1000 steps and
it tips above |turn| = 0.9**. Off-axis multi-position reach is walled by this turn/stability tradeoff
— a model-level kinematic/dynamic limit. Combined with the residual-RL negatives, the honest verdict
is that reaching arbitrary positions needs a **more agile embodiment**, not more control over the
current gait.
