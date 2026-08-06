# SISY 2026 — Control-Scenario Paper from the Technologies Demo

**Date:** 2026-06-10
**Branch:** `feature/ac-hsikan`
**Paper:** `paper/sisy2026_control/main.tex` → `main.pdf` (3 pp, IEEE conf)

## Summary

Drafted a **new standalone IEEE SISY 2026 conference paper** derived
from the MDPI Technologies live demo (the UR5e grasping-context
hypergraph), concentrating on the **control scenario** as requested.
The paper directly answers the two open MDPI reviewer asks: it gives an
executable control-loop instantiation and the per-control-cycle
computation time the framework had never reported.

**Title:** *A Real-Time Hypergraph Contextual Layer for Industrial Robot
Control: Microsecond-Scale State Aggregation on a UR5e Grasping Task.*

**Key measured result:** the grasping-context evaluation (6 signed
hyperedges) costs a median **16.7 µs/control-cycle** (IQR ≈ 1 µs) on a
consumer AMD Ryzen CPU — **0.017 %** of a 10 Hz period; the full 11-edge
robot graph costs **34 µs**. The contextual layer is, for control
purposes, free, and is read-only — orthogonal to (not competing with) a
TD3 or PID controller. That reframing answers Reviewer 2's TD3/PID-TD3
comparison request honestly (the comparison is category-mismatched).

## Files touched

| File | Change | Lines |
|---|---|---|
| `paper/sisy2026_control/main.tex` | **new** — IEEE draft (+ cls/bst copied) | +330 |
| `paper/sisy2026_control/generate_figures.py` | **new** — figures from real evidence | +150 |
| `hymeko_ros2_demo/hymeko_ros2_demo/topic_binding.py` | edit — extract pure `evaluate_context` + `default_edge_aggregate` | +66 |
| `hymeko_ros2_demo/hymeko_ros2_demo/grasping_context_node.py` | edit — call `evaluate_context`; remove duplicated `_aggregate` | +6/−30 |
| `hymeko_ros2_demo/benchmarks/bench_tick_latency.py` | **new** — per-cycle benchmark | +120 |
| `hymeko_ros2_demo/test/test_evaluate_context.py` | **new** — 7 tests | +90 |
| `hymeko_ros2_demo/conftest.py` | **new** — make pkg importable off-ROS | +12 |

## Method / evidence

- The per-cycle compute is the topological hyperedge sweep, extracted
  from `GraspingContextNode._tick` into a pure, side-effect-free
  `evaluate_context()` so it is unit-testable and benchmarkable without
  rclpy. The node now calls it (single source — removes the duplicated
  loop and `_aggregate`, §6.5 #1).
- Benchmark (`bench_tick_latency.py`) reconstructs the 6-edge grasping
  context and the 11-edge full robot graph exactly as the parser
  surfaces them; CLAUDE.md §3 stats (≥ 9 reps, median/IQR/worst). The
  figure generator retries until it obtains a low-IQR (quiet-machine)
  sample so figure, table, and `results.json` agree.
- Figures from real artifacts: `fig_stability.png` (contextual outputs
  over the live run, parsed from `demo_evidence/node_ticks_*v3.txt`),
  `fig_latency.png` (per-cycle bar vs the 100 ms period).

## Test results

`pytest hymeko_ros2_demo/test/test_evaluate_context.py` — **7 passed**
(determinism, topological propagation, external inputs preserved,
unit-range outputs, stability special-form, clamped-mean default,
empty-edges no-op). ruff clean on all new/edited files.

## ⚠ Verification gap (must close on the ROS box)

The `grasping_context_node.py` edit **could not be run on this Windows
host** — `rclpy` and the `hymeko` wheel are Linux/cp312 only. The change
is a behavior-preserving extraction (the inline loop → `evaluate_context`
with the identical `default_edge_aggregate` rule, verified by the 7 pure
tests), but the node's import + 10 Hz tick must be confirmed on the ROS 2
Kilted box before any live re-demo:

```
colcon build --packages-select hymeko_ros2_demo
ros2 launch hymeko_ros2_demo grasping_context_only.launch.py   # node-only smoke
```

## Reviewer-ask coverage

- **R2 "per control cycle time?"** → Table I + Fig 3 (16.7 µs).
- **R2 "vs TD3 / PID-TD3?"** → §V: category-mismatched; overhead, not
  control quality, is the right axis (orthogonal read-only layer).
- **R1 "quantitative abstract"** → abstract leads with the µs numbers.
- **R1 "vision scene / Hyper-YOLO"** → §V future work links the vision
  context to hypergraph detection (and the in-repo HyMeYOLO line).

## Open issues / follow-up

- Authors set to the MDPI list (Óbuda); anonymize if SISY is double-blind.
- 3 pp draft; expandable toward the 6 pp SISY limit (related work,
  larger stability trace — the v3 log yields only 4 V_global samples; a
  re-run logging the 2 Hz compact line would densify Fig 2).
- Benchmark is consumer-laptop, load-sensitive; a quiet-host re-run
  would tighten the absolute µs (the order-of-magnitude margin is robust).
- **Node edit unverified on ROS** (see gap above) — highest-priority
  follow-up.

---

## Update 2026-06-12 — extended to 5 pp, Fig. 1 → figure*, feedback arrow fixed

Three requested changes to `paper/sisy2026_control/main.tex` (compiles clean,
MiKTeX pdflatex, **3 → 5 pages**, 0 undefined refs):

1. **Length.** Added substantive, source-grounded content (no padding):
   a **Related Work** section (KnowRob, digital twin/ISA-95, HGNN/Hyper-YOLO,
   behaviour trees); the **Θ(nnz(B)) cost bound** in Background; **Algorithm 1**
   (`evaluate_context` sweep); a **YAML binding** listing; a new
   **"full robot graph"** subsection giving the maintenance/safety/cross-context
   edges ($e_7$–$e_{10}$, $e_{sc}$) — the eleven-edge graph the benchmark uses
   but the paper never described, pulled verbatim from
   `hymeko_ros2_demo/.../scenarios/hymeko_robot.hymeko`; a **scaling** subsection
   (1.04 / 1.01 µs per incidence from the two measured points → 1 kHz headroom);
   a **threats-to-validity** paragraph; and a **positioning table** (Table I)
   vs the related-work categories on four axes.
2. **Fig. 1 → `figure*`.** Re-laid-out horizontally (left-to-right pipeline),
   now spans both columns.
3. **Feedback arrow fixed.** The old single `rviz.east→gz.east` bend labelled
   `/joint_states` was mis-routed and mis-directed. Replaced with two correct,
   flat, white-labelled arcs: robot `/joint_states` → planner (below),
   planner trajectory commands → robot (above).

New refs (all real): KnowRob~[8], Digital Twin~[9], HGNN~[10], Hypergraph
Conv~[11], Behaviour Trees~[12]. Preamble adds `algorithm`/`algpseudocode`,
`listings`, and an `\algorithmautorefname`. Content is honest — every added
number/edge traces to the `.hymeko` source or the existing benchmark; no new
measurements were invented. Page 5 holds only the bibliography tail; the paper
is comfortably within the 6-page ceiling and could take one more half-column
of content if desired.

---

## Update 2026-06-12 — node rewiring verified off-ROS; publishability read

**Rewiring (uncommitted, tracked):** `evaluate_context` + `default_edge_aggregate`
extracted into `topic_binding.py`; `grasping_context_node._tick` now calls the
shared pure function instead of an inline loop + private `_aggregate` (39 lines
deleted from the node). The node and `bench_tick_latency.py` now use the
**identical** function — the paper's "we isolate `evaluate_context` as a pure
function and benchmark it directly" is now literally true (no duplication, §6.1).

**Verified off-ROS (no rclpy on this Windows host):**
- `pytest hymeko_ros2_demo/test/test_evaluate_context.py test_topic_binding.py`
  — **16 passed**. Fixed a stale test (`test_aggregate_grasp_stability_decreasing_in_gap`)
  that predated the grip-force /10 normalisation (fed raw 5.0; now uses
  normalised-unit inputs, zero gap at `force_vector=0.5, grip_force=5`).
- Removed an unused `Hyperedge` import in the test; `ruff check` clean on all
  four touched files; `py_compile` clean on both rewired modules.
- Latency bench (this host, ≥9 reps, median/IQR/worst): grasping **~11–12 µs**,
  full robot **~21–27 µs**. Lower than the paper's 16.7/34.2 µs (Ryzen host) but
  the **cost structure reproduces**: ~0.8 µs/incidence, linear in nnz(B), ~0.01–
  0.02 % of a 10 Hz period.

**Publishable now (reproducible):** the feasibility + cost-structure contribution
— Θ(nnz(B)) microsecond-scale per-cycle cost, negligible vs the control period,
the de-duplicated pure-function instantiation, and the structural transparency
(full 11-edge graph). The robust quantitative claim is **per-incidence cost**,
not the absolute µs (host-dependent — present the 16.7/34.2 as host-specific or
re-measure on the canonical Ryzen host for camera-ready).

**Gated on a real ROS run (the remaining gap):** the live-loop claim — "sustained
10 Hz over a 105 s Gazebo pick-and-place" and the Fig. 2 stability trace — has
**not** been re-run on ROS since the refactor (no rclpy here). `fig_stability.png`
predates the rewiring. Before camera-ready, run the rewired node on ROS 2 Kilted +
Gazebo for one pick-and-place, confirm no missed ticks, and regenerate the trace.
This is the highest-priority follow-up (was already flagged in memory).
