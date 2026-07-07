---
name: project-sisy2026-control-paper
description: SISY 2026 control-scenario paper from the Technologies demo — status and the unverified node edit
metadata: 
  node_type: memory
  type: project
  originSessionId: 7d38ee1f-9710-4c95-87d5-425cc349680a
---

A new standalone IEEE SISY 2026 paper was drafted on 2026-06-10 from the
MDPI Technologies live demo (UR5e grasping-context hypergraph),
concentrating on the **control scenario**. It lives at
`paper/sisy2026_control/main.tex` (compiles, 3 pp) — distinct from the
existing `paper/sisy2026/` "Cycles vs. Walks" KAN paper.

Core result: the grasping-context evaluation costs ~16.7 µs/control-cycle
(0.017% of a 10 Hz period), measured by
`hymeko_ros2_demo/benchmarks/bench_tick_latency.py` over the pure
`evaluate_context()` extracted into `topic_binding.py`. This answers the
MDPI reviewers' per-control-cycle-time and TD3/PID-TD3 asks (the layer is
read-only, orthogonal to any controller).

**Verification gap (highest-priority follow-up):** the
`grasping_context_node.py` edit (inline tick loop → `evaluate_context`,
`_aggregate` removed) was NOT runnable on the Windows dev host (rclpy +
hymeko wheel are Linux/cp312 only). The 7 pure tests in
`test_evaluate_context.py` pass, but the node import + 10 Hz tick must be
confirmed on the ROS 2 Kilted box (`colcon build` +
`grasping_context_only.launch.py`) before any live re-demo.

Report: `reports/2026-06-10-sisy-control-paper.md`. Authors set to the
Óbuda MDPI list; anonymize if SISY is double-blind. Related:
[[project-voc-hymeyolo-baseline]].
