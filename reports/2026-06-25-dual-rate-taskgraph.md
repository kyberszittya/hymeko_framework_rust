# Task-graph A/B: does coin/zone-in-graph + BC delivery vindicate HSiKAN? (2026-06-25)

## Summary

The earlier dual-rate experiment tied on the **arm-only** Galambos graph (`reports/2026-06-25-dual-rate-galambos.md`)
— expected, since the coin/zone are not in the graph (`project-galambos-hsikan-tie-rootcause`). This A/B tests the
*not-yet-run* regime: **BC warm-start (which delivers) × `task_graph=True` (coin/zone as grasp/goal hyperedges)**.
The hypothesis: with the task objects in the graph *and* a learner that engages the task, HSiKAN's structural
prior should finally beat the MLP.

**Result: the hypothesis is falsified (under BC).** Putting coin/zone in the graph **helped the MLP and hurt
HSiKAN** — the opposite of the prediction.

## Results (delivery rate, 24 eval episodes/seed, 3 seeds)

| policy     | baseline (arm-only, N=6) mean±std | taskgraph (N=10) mean±std |
|------------|-----------------------------------|---------------------------|
| MLP-alone  | 0.139 ± 0.109                     | **0.250 ± 0.090**         |
| HSiKAN-alone | 0.111 ± 0.052                   | **0.042 ± 0.000**         |
| dual (N=1) | 0.139 ± 0.039                     | 0.125 ± 0.034             |
| dual (N=4) | 0.125 ± 0.090                     | 0.167 ± 0.090             |
| dual (N=8) | 0.069 ± 0.071                     | 0.153 ± 0.086             |

(`task_graph` adds 4 vertices: coin, zone, grasp_hub, goal_hub → N=6→10.)

## Interpretation

- **Measured.** With `task_graph`, MLP rises (0.139→0.250) and HSiKAN falls (0.111→0.042, identical across all 3
  seeds). The MLP, not HSiKAN, benefits from the task objects being observable.
- **Inferred (not yet isolated).** The MLP simply reads the 4 extra vertices as richer flat features and gains
  signal. HSiKAN *degrades* — hypothesis: the coin/zone/hub vertices, folded into the row-normalized signed
  adjacency, **dilute** the arm message-passing (the hub rows mix unrelated nodes; the per-arm signal BC needs is
  averaged away). The dead-flat 0.042 (= 1/24) across seeds suggests HSiKAN collapses to a near-constant policy in
  the augmented graph.
- **Conclusion.** "Put the task objects in the graph" is **not sufficient** to make the structural prior pay off,
  and under BC it is actively **harmful** to HSiKAN as currently wired. This strengthens
  `project-galambos-hsikan-tie-rootcause`: the hyperedge representation is necessary-not-sufficient, and the
  current signed-adjacency aggregation does not exploit it — it dilutes.

## Caveats

3 seeds, n_eval=24 (noisy at the 1/24 grid); CPU, `robot=None` hand-authored arms, difficulty 0.3, BC 200 epochs.
The HSiKAN collapse (0.042) is consistent enough to be real, but *why* it collapses (hub dilution vs a
representational bug in how `task_graph` vertices feed HSiKAN) is **not isolated** — it is the discriminating
follow-up. This does not test PPO/off-policy (the campaign does).

## Follow-ups

1. **Isolate the HSiKAN collapse**: ablate the hub rows / try `incidence="weighted"` so the arm arcs are not
   diluted by the coin/zone/hub rows; inspect the learned adjacency. This is the real question the result raises.
2. The **structural critic** (`docs/plans/2026-06-25-structural-critic/`) attacks the same gap from the value
   side — per-cycle value over the arm–coin–zone cycle — and may exploit the structure the flat aggregation
   dilutes. The arm–coin–zone cycle exists only with `task_graph`, so this result motivates that plan.

## Files / provenance

Driver: `hymeko_rl/exp_dual_rate.py` (`--task-graph`). Log: `reports/2026-06-25-dual-rate-taskgraph.log`.
Git: branch `fix-hsikan`, dirty. Seeds 0–2; eval seed 9000. CPU run; no OOM. §6.5: none (one `--task-graph`
flag through the existing harness, env reused).
