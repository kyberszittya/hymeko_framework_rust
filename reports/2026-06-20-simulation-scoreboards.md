# Report — simulation scoreboards: points / goals / deaths, plots + GIFs

**Date:** 2026-06-20 · **Status:** ✅ Eval harness built; **reach scoreboard meaningful**, **Galambos
scoreboard honest (learning, 0 goals at 100 iters)**.

## Harness
`hymeko_rl/evaluate.py` (env-agnostic): runs N episodes, classifies each as **goal** (terminated,
not death), **death** (terminated + `info["death"]`), or **timeout** (truncated); tallies points
(summed reward), success rate, mean return. `plot_scoreboard` → grouped outcome bars + per-episode
points PNG; `render_episode_gif` → a rendered episode GIF (offscreen `mujoco.Renderer`).

## Reach arm + safety (4-DOF) — meaningful scoreboard
20 episodes each, position control, safety on (goal = reach within tol; death = ground-contact /
self-collision; points = reward):

| source | goals | deaths | timeouts | success | mean return |
|---|---|---|---|---|---|
| expert (DLS-IK) | 17 | 3 | 0 | **85%** | −16.05 |
| random | 0 | 0 | 20 | 0% | −56.16 |

Artifacts: `reports/2026-06-20-reach-scoreboard.png`, `reports/2026-06-20-reach-expert.gif`.

**Finding (honest):** on the **6-DOF** arm the expert **dies every episode by self-collision** during
the reaching sweep (diagnosed: `self_collision=True`, dist 1.03 — far from target). The slim/exclusion
+ floor-seating fixes cleaned the *home* pose, but the 6-DOF arm's reaching *motion* still folds
non-adjacent links into contact, so the safety reward is too strict for it as configured. The
scoreboard therefore uses the clean **4-DOF** arm; fixing the 6-DOF motion-collision is follow-up.

## Galambos planar grasp — honest scoreboard (learning, not solved)
PPO (HSiKAN over the two-arm hypergraph), 100 iters × 512 steps, return **−21.46 → −16.62** during
training. Eval, 20 episodes (goal = disk settles in zone; death = disk knocked out of bounds):

| source | goals | deaths | timeouts | mean return |
|---|---|---|---|---|
| PPO (100 it) | 0 | 0 | 20 | **−20.9** |
| random | 0 | 0 | 20 | −37.7 |

Artifacts: `reports/2026-06-20-galambos-scoreboard.png`, `reports/2026-06-20-galambos-ppo.gif`.

**Honest read (no over-claiming):** the trained policy **beats random on points** (−20.9 vs −37.7 —
it keeps the disk closer to the zone and doesn't knock it out) but achieves **0 goals**: it does not
yet complete the catch-and-pull. The render shows the arms folding flat rather than trapping the disk.
This is the expected state for 100 iters on a contact-rich, underactuated two-finger pull — it needs
**much more training and likely reward shaping / a curriculum** (e.g. spawn nearer first, contact
bonus tuning). The machinery is correct; the policy is not trained to competence.

## Files
- `hymeko_rl/evaluate.py` (NEW — harness + plot + GIF), `hymeko_rl/train_planar_grasp.py` (NEW — PPO
  runner, sizes the policy from `observation_space`), `planar_grasp_env.py` (+ death = disk out of
  bounds), `checkpoints/galambos/ppo.{pt,json}` (trained policy + return curve).
- Dep: `matplotlib 3.11` installed (already declared in the `demo` group — not a new dependency).

## Tests / static
`hymeko_rl` suite green earlier (102); planar tests still pass; ruff clean; mypy only the `mujoco`
baseline (one scoped `# type: ignore[arg-type]` on the duck-typed `train_ppo(env)` call).

## Next
1. **Longer Galambos training + reward/curriculum tuning** to actually achieve goals (report the
   return curve + scoreboard, honestly).
2. **6-DOF self-collision-during-motion** — slimmer collision geoms / contact-penetration threshold,
   so the 6-DOF safety scoreboard becomes meaningful (currently all deaths).
3. Generalise `bc._make_policy` to size from `observation_space` (so the planar env uses it directly).

## Provenance
- Git SHA `73ee5a6` (working tree dirty; uncommitted increment). MuJoCo 3.9.0, torch per CORE pins,
  matplotlib 3.11. Windows 11, CPU. Eval seeds 0–19; PPO seed 0; render seeds 0 (reach) / 2 (galambos).
