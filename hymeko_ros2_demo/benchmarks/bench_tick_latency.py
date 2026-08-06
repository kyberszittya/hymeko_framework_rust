"""Per-control-cycle latency of the hypergraph contextual evaluation.

Answers the SISY reviewer question (Reviewer 2, MDPI Technologies):
*"what is the total computation time for a single control cycle?"*

The per-cycle cost is the topological evaluation of the context's signed
hyperedges (``topic_binding.evaluate_context``) — the same pure function
the live ``GraspingContextNode`` calls every tick. The ``.hymeko`` parse
is a one-time startup cost (outside the control cycle) and the ROS
publish/subscribe is I/O; the *contextual computation* measured here is
the novel per-cycle work.

This is a standalone harness (no rclpy / no hymeko wheel needed), so it
runs on any host. It reconstructs the grasping context's 6 hyperedges
(and the full 11-edge robot graph) exactly as the parser surfaces them
and times ``evaluate_context`` with CLAUDE.md §3 statistics: 1000 timed
batches after warmup, reporting median / IQR / p95 / p99 / max over the
per-batch per-cycle costs. Run on a quiet host (background CPU contention,
e.g. a live Gazebo sim, inflates the numbers and invalidates the run).

    python hymeko_ros2_demo/benchmarks/bench_tick_latency.py
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

# Make the package importable when run as a plain script.
_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from hymeko_ros2_demo.topic_binding import Hyperedge, evaluate_context  # noqa: E402


# ── Context definitions (as the parser surfaces them; see
#    scenarios/hymeko_robot.hymeko) ───────────────────────────────────

GRASPING_EDGES = [
    Hyperedge("derive_tool", ("active_tool",), ("tool_params",)),
    Hyperedge("derive_payload", ("active_payload",), ("payload_params",)),
    Hyperedge("loading_state", ("tool_params", "payload_params"), ("loaded_state",)),
    Hyperedge("grasp_config", ("mode_parallel", "payload_params"), ("configuration",)),
    Hyperedge("load_force", ("configuration", "robot_pose"), ("force_vector",)),
    Hyperedge("grasp_stability", ("force_vector", "grip_force"), ("stability_margin",)),
]

# Maintenance + safety + cross-context constraint (full 11-edge graph).
_EXTRA_EDGES = [
    Hyperedge("wear_indicator",
              ("joint_temp", "vibration", "motor_current", "cycle_count"),
              ("degradation_level",)),
    Hyperedge("health_state", ("degradation_level", "brake_response"),
              ("component_health",)),
    Hyperedge("braking_capability", ("component_health", "brake_response"),
              ("braking_estimate",)),
    Hyperedge("safety_state",
              ("robot_speed", "human_distance", "operating_mode", "braking_estimate"),
              ("risk_assessment",)),
    Hyperedge("speed_constraint", ("component_health", "operating_mode"),
              ("max_permissible_speed",)),
]
FULL_EDGES = GRASPING_EDGES + _EXTRA_EDGES

# Realistic bound inputs (the demo's mid-run sample, hymeko_robot run v3).
_INPUTS = {
    "robot_pose": 0.3, "active_tool": 1.0, "active_payload": 3.0,
    "mode_parallel": 0.0, "grip_force": 5.001,
    "joint_temp": 0.4, "vibration": 0.2, "motor_current": 0.5,
    "cycle_count": 1000.0, "brake_response": 0.9,
    "robot_speed": 0.5, "human_distance": 1.2, "operating_mode": 1.0,
}


def _bench_one(edges, iters: int) -> float:
    """Wall time (seconds) for `iters` evaluations of one context."""
    t0 = time.perf_counter()
    for _ in range(iters):
        v = dict(_INPUTS)
        evaluate_context(edges, v)
    return time.perf_counter() - t0


def _percentile(sorted_us: list, p: float) -> float:
    """`p`-th percentile of an already-sorted list (nearest-rank)."""
    return sorted_us[min(len(sorted_us) - 1, int(p / 100.0 * len(sorted_us)))]


def bench(edges, label: str, reps: int = 1_000, iters: int = 1_000,
          warmup: int = 5) -> dict:
    """Per-cycle latency over `reps` batches of `iters` evaluations.

    Each batch times `iters` sweeps and divides by `iters`, so the
    ``perf_counter`` overhead is amortised to a few ns per cycle — accurate
    at µs scale, unlike timing single calls (which is dominated by timer
    overhead and OS-scheduling jitter). Reports median / IQR / p95 / p99 /
    max over the `reps` per-batch per-cycle costs: a 1000-sample
    distribution, not a handful of batch means.
    """
    for _ in range(warmup):
        _bench_one(edges, iters)
    per_cycle_us = []
    for _ in range(reps):
        wall = _bench_one(edges, iters)
        per_cycle_us.append(wall / iters * 1e6)
    per_cycle_us.sort()
    q = statistics.quantiles(per_cycle_us, n=4)
    median = statistics.median(per_cycle_us)
    return dict(
        label=label, n_edges=len(edges), reps=reps, iters=iters,
        median_us=median, iqr_us=q[2] - q[0],
        p95_us=_percentile(per_cycle_us, 95.0),
        p99_us=_percentile(per_cycle_us, 99.0),
        worst_us=per_cycle_us[-1], max_us=per_cycle_us[-1],
        cycles_per_s=1e6 / median,
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="per-control-cycle latency")
    ap.add_argument("--reps", type=int, default=1_000,
                    help="number of timed batches per context (default 1000).")
    ap.add_argument("--iters", type=int, default=1_000,
                    help="evaluations per batch (timer amortisation; default 1000).")
    args = ap.parse_args()

    print(f"{'context':<16s} {'edges':>5s} {'reps':>6s} {'median_us':>10s} "
          f"{'IQR_us':>8s} {'p95_us':>8s} {'p99_us':>8s} {'max_us':>8s} "
          f"{'% of 100ms':>11s}")
    print("-" * 92)
    for edges, label in ((GRASPING_EDGES, "grasping"), (FULL_EDGES, "full_robot")):
        r = bench(edges, label, reps=args.reps, iters=args.iters)
        pct_100ms = r["median_us"] / 1e5 * 100.0  # 100 ms = 10 Hz period
        print(f"{r['label']:<16s} {r['n_edges']:>5d} {r['reps']:>6d} "
              f"{r['median_us']:>10.3f} {r['iqr_us']:>8.3f} "
              f"{r['p95_us']:>8.3f} {r['p99_us']:>8.3f} {r['max_us']:>8.3f} "
              f"{pct_100ms:>10.4f}%")


if __name__ == "__main__":
    main()
