# Galambos hyperedge integration — coin/zone in the graph + the discriminating A/B

2026-06-24 · hymeko_rl · follows the Galambos HSiKAN==MLP tie root-cause (memory
`project-galambos-hsikan-tie-rootcause`)

## Why
On Galambos, HSiKAN tied a params-matched MLP under PPO (curve-max −224.6±11.2 vs −223.4±13.3; **all 10 cells at
failure-level, no coin delivered**). Root cause (code-grounded): the coin and zone are **not vertices** in the
graph HSiKAN reasons over — they enter as flat per-vertex features, and the coin→zone vector is broadcast
identically to every vertex. So HSiKAN's message-passing (its only edge) has nothing task-relevant to reason
over. Fix: put the task objects **in the graph** as hyperedges — a grasp hyperedge {fingertips, coin} and a goal
hyperedge {coin, zone} — and test whether the structural prior then beats the MLP.

## What shipped (all non-core)
- **`HypergraphState.with_task_hyperedges()`** (built earlier) — adds entity vertices + hyperedge hub nodes.
- **`PlanarGraspEnv(task_graph=True)`** — opt-in variant: 6 robot vertices + {coin, zone} + {grasp_hub,
  goal_hub} = 10 vertices. `node_features` emits the coin's xy on the coin vertex, the zone's xy on the zone
  vertex, zeros on the hubs (structural routing). **Baseline (`task_graph=False`) is byte-identical** — the 6
  robot rows match between variants, so HSiKAN(augmented) vs HSiKAN(baseline) isolates the coin/zone hyperedges.
  `from_hymeko(task_graph=...)` threads it through.
- **Harness:** `offpolicy_eval` task `galambos_taskgraph` (the augmented env) beside `galambos` (robot-only).
- **Tests:** `test_galambos_task_graph.py` (baseline unchanged; +coin/zone/hubs; robot-rows identical;
  HSiKAN-feed + step). 4/4 pass; with `test_task_hyperedges` (6) + star/H★ (15): **25 graph tests green.**
  `ruff` + `mypy --strict` clean.

## The discriminating A/B (in flight)
- **Command:** `python -m hymeko_rl.offpolicy_eval --task galambos_taskgraph --algo ppo --mode full
  --journal reports/2026-06-24-galambos-taskgraph-ppo.jsonl --out reports/2026-06-24-galambos-taskgraph-ppo.json`
  (CPU-only, resumable; ~3.5 min/cell × 10 cells ≈ 35 min). Smoke confirmed end-to-end
  (params-match hsikan@64≈15368 ~ mlp@86≈15144).
- **Baseline to compare against** (robot-only, from `reports/2026-06-23-stage2-arch-algo.jsonl`):
  HSiKAN −224.6, MLP −223.4 (tie at failure).
- **Reading the result:**
  - HSiKAN(augmented) **beats** MLP(augmented) and/or baseline ⇒ *structure pays off once the task entities are
    in the graph* — the central hypothesis confirmed.
  - Still a tie at failure ⇒ the representation is not the bottleneck; the **long-horizon credit**
    (reach→grasp→transport→place) is — points to the FSM / reward-machine direction
    (`project-fsm-structured-rl`), not more representation work.

## Caveat
PPO is the fast probe; it tied at failure on the baseline, so it may tie at failure here too even if structure
helps exploration — the augmented graph could still fail to *deliver* under PPO's weak long-horizon credit. SAC
(stronger, but ~30–60 min/cell) is the follow-up if PPO is inconclusive. Either outcome is informative: it tells
us whether to invest next in representation (more hyperedges) or in task structure (FSM).

## Verdict (2026-06-24) — coin-placement ruled out; the wall is hard-exploration
The PPO A/B is **uninformative because PPO never learns the task at any difficulty**, so I stopped it rather than
burn compute. Measured PPO-from-scratch curve-max (HSiKAN, task_graph):
| coin spawn (difficulty) | coin–zone dist | curve-max | curve |
|---|---|---|---|
| 1.0 (whole table) | up to ~0.32 m | −224 | flat at failure |
| 0.3 (near zone) | shell ~0.06–0.17 m | −214 | flat at failure |
| **0.0 (at the zone edge)** | **0.071 m (zone_half 0.04)** | **−198** | **`-200.8…-198.0`, never positive** |

Even with the coin essentially touching the zone, the policy never delivers it and the curve is **flat** — it
isn't learning to approach/pinch/pull at all. So the binding constraint is **not reach, backbone, or
representation**; it's the two-arm grasp **hard-exploration** (pure PPO-from-scratch, no demos, no task
structure) — the same wall as the FANUC pick-place (0% place).

**Control-interface check (user-prompted) — a real bug, but NOT the cause.** The hand-authored fallback
(`make_planar_arms_mjcf`, `robot=None`) had `<position>` actuators with **no `ctrlrange`** → `ctrllimited=False`
→ action space `[0,0,0,0]`: uncontrollable (a full command moved the arm 0.027 rad). Fixed (`ctrlrange="-2.8 2.8"`,
gains matched to the emitted `kp=40/kv=4`; full command now swings 2.65 rad) + a regression test
(`test_control_interface_is_actuated`). **But the emitted env (`from_hymeko`) — which the A/B and all difficulty
checks used — has working control** (2.33 rad swing), and the **fixed** hand-authored env at difficulty 0 (coin
0.071 m from the zone) *also* fails flat (`-195.3 … -191.7`). So across three configs — emitted, fixed
hand-authored, every difficulty — the curve is flat and nothing delivers. Control was a latent bug worth fixing;
it is not why the goal is never achieved.

**Implications:**
- The HSiKAN-vs-MLP A/B can't be read on a task neither learner solves; both tie at failure regardless of the
  coin being in the graph or where it spawns.
- The hyperedge representation (`with_task_hyperedges`, `PlanarGraspEnv(task_graph=True)`) is **built, tested,
  and correct** — it's the right representation *for when a learner engages the task*; necessary, not sufficient.
- **Productive next move = declared STRUCTURE (FSM / reward machine, `project-fsm-structured-rl`) or
  DEMONSTRATIONS (BC warm-start, as the FANUC env uses), not more coin-placement / backbone / hyperedge work.**
- Caveat: `project-galambos-reward-shaping` reached 5/8 goals at some earlier config, so the task IS achievable
  with the right setup/budget; these 18–30k-step PPO-from-scratch runs are not it.
