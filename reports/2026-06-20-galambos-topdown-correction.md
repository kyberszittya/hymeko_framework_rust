# Report — Galambos env corrected to the intended top-down planar table

**Date:** 2026-06-20 · **Status:** ✅ Env rebuilt per the corrected spec; scoreboard re-run (honest).

## The correction
My first build misread the task: I had the coin **falling under gravity** in a **vertical** plane.
The user corrected it:
1. MuJoCo + **planar robots** — keep, but the arms must operate *in* the manipulation plane.
2. The coin is **placed in reach on the 2D plane, not dropped from above**.
Chosen setup: **top-down horizontal table**.

## What changed
- **`data/robotics/galambos_planar.hymeko`** — redesigned: revolute axes now **Z** (the chain sweeps
  in the XY plane — verified: all four joint world-axes are Z, the fingertip moves in XY at constant
  z), and **box** links (horizontal rods; a cylinder would point out of plane). Two arms at x=±0.28,
  each yawed +90° to reach forward into the workspace.
- **`hymeko_rl/env/planar_grasp_env.py`** — rebuilt: the coin is a **planar table body**
  (slide-x / slide-y / hinge-z, confined to the arms' plane, light damping ≈ table friction) and is
  **PLACED at a random reachable spot** at reset (no fall); the zone is a marker site on the table.
  Metrics, obs (`disk_x, disk_y, disk-zone offsets`), success (coin in zone), and death (coin
  knocked out of the workspace) are all in-plane. The disk no longer has a vertical/falling DOF.
- Tests updated: the coin stays in the plane (no z motion), is placed in-reach outside the zone, and
  a coin planted at the zone centre terminates as success. 5 planar tests pass; ruff/mypy clean.

## Scoreboard (corrected env, honest)
PPO (HSiKAN over the two-arm hypergraph), 100 iters × 512 steps, return **−15.16 → −13.29**. Eval,
20 episodes (goal = coin settled in zone; death = coin knocked out; points = reward):

| source | goals | deaths | timeouts | mean return |
|---|---|---|---|---|
| PPO (100 it) | 0 | 0 | 20 | **−15.4** |
| random | 0 | 0 | 20 | −30.9 |

Artifacts: `reports/2026-06-20-galambos-scoreboard.png`, `reports/2026-06-20-galambos-ppo.gif`
(top-down view: two arms, the placed coin, the centre zone).

**Honest read (unchanged conclusion, now on the right task):** PPO beats random on points (it keeps
the coin closer and doesn't knock it out) but achieves **0 goals** at 100 iters — it does not yet
complete the pull. Contact-rich planar manipulation with a limited reach envelope; needs much more
training and likely reward shaping / curriculum. The env and harness are correct; the policy is not
trained to competence.

## CORE.YAML / dependencies
None (hymeko_rl + data/robotics, non-core). No new dependency.

## Follow-up
- Longer training + reward/curriculum tuning to actually achieve goals (spawn nearer the zone first,
  widen the success dwell, tune the contact bonus); report the return curve honestly.
- Reach-envelope tuning: the coin is reachable by both arms only centrally; the spawn box is set to
  that region, but link lengths / base spacing could be tuned for a larger feasible workspace.

## Provenance
- Git SHA `73ee5a6` (working tree dirty; uncommitted increment). MuJoCo 3.9.0, torch per CORE pins,
  matplotlib 3.11. Windows 11, CPU. Eval seeds 0–19; PPO seed 0; render seed 2.
