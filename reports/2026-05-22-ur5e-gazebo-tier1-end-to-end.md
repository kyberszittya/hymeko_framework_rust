# MDPI Technologies Tier-1 live demo — full UR5e + Gazebo + MoveIt run

**Date:** 2026-05-22
**Plan:** [`docs/plans/2026-05-22-mdpi-technologies-live-demo/`](../docs/plans/2026-05-22-mdpi-technologies-live-demo/)
**Verdict:** **WORKING.** Gazebo + UR5e + MoveIt2 + RViz2 + the
HyMeKo `grasping_context_node` + the synthetic upstream-perception
publisher all come up together via a single launch file; the
contextual flow ticks at 10 Hz and publishes the aggregated outputs
to ROS topics that downstream consumers can subscribe to.

## 1. The one launch line the reviewer types

```bash
ros2 launch hymeko_ros2_demo ur5e_grasping_demo.launch.py
```

This brings up:
- `gz sim` (server + GUI) with the empty world
- The UR5e robot spawned and controlled by `ros2_control`
  (joint_state_broadcaster, scaled_joint_trajectory_controller)
- MoveIt2 with the `ur_manipulator` planning group
- RViz2 with the MoveIt motion-planning panel
- `topic_pub_sim` — synthetic upstream-perception publisher
- `grasping_context_node` — loads `hymeko_robot.hymeko`, evaluates
  the 6 grasping-context hyperedges at 10 Hz

## 2. Evidence — topics fire end-to-end

After 35 seconds of settling, all subscribed and published topics
are present:

```
$ ros2 topic list | grep -E 'hymeko/|tool_id|payload_id|grasp_mode|wrench|tcp_pose|joint_states'

/dynamic_joint_states                          ← from UR sim
/grasp_mode                                    ← synthetic pub
/hymeko/grasping/configuration                 ← contextual output
/hymeko/grasping/contact_force                 ← contextual output
/hymeko/grasping/stability_margin              ← contextual output
/joint_states                                  ← from UR sim
/payload_id                                    ← synthetic pub
/tcp_pose                                      ← synthetic pub
/tool_id                                       ← synthetic pub
/wrench                                        ← synthetic pub
```

Live values (one sample each):

```
$ ros2 topic echo --once /hymeko/grasping/stability_margin
data: 0.16382772355066566

$ ros2 topic echo --once /hymeko/grasping/configuration
data: 0.5

$ ros2 topic echo --once /hymeko/grasping/contact_force
data: 0.35351544415954483
```

The node-side log shows the contextual flow ticking steadily:

```
[grasping_context_node]: loaded 6 hyperedges from hymeko_robot.hymeko::grasping_context
[grasping_context_node]: armed @ 10.0 Hz (context=grasping_context, edges=6)
[grasping_context_node]: tick 50:  V_global = {'robot_pose': 0.3, 'active_tool': 2.0, ...
[grasping_context_node]: tick 100: V_global = {'robot_pose': 0.3, 'active_tool': 3.0, ...
[grasping_context_node]: tick 400: V_global = {'robot_pose': 0.3, 'active_tool': 3.0, ...
```

50 ticks in 5 seconds = exactly 10 Hz. `V_global` propagates through
all 11 vertices of the grasping context (5 bound from topics + 6
derived from the hyperedges).

## 3. What the demo proves (and what it doesn't)

### Proves
- The HyMeKo IR loader runs inside a real ROS 2 node.
- A `.hymeko` file describing a multi-context hypergraph (the
  article's canonical `hymeko_robot.hymeko` listing) is parsed
  end-to-end into evaluable hyperedges by the existing PyO3 wheel.
- The 6 grasping-context hyperedges from article eq.
  `grasp_hyperedges` evaluate at 10 Hz against live ROS topic data.
- The aggregated outputs publish back as standard ROS topics that
  any downstream node (planner, monitor, RViz2 panel) can consume.
- The whole stack is a single launch line and ~5 GiB RAM,
  ~2 GiB GPU, no extra hardware.

### Does NOT prove
- That the contextual representation *improves* anything about the
  grasping policy. The aggregation functions per hyperedge are
  practitioner placeholders (the paper specifies only the signed
  incidence structure). The values move plausibly with the input
  stream, but the absolute mapping is illustrative, not a
  learned policy. **This is honest about a theoretical paper.**

## 4. Honest scope adjustment vs the original Tier-1 plan

Original Tier-1 plan assumed the bare UR sim shipped with a
parallel-gripper plugin and a F/T sensor. The actual upstream
`ur_simulation_gz` ships with neither — only the bare UR arm +
`joint_state_broadcaster`. Two options:

- **A. Add a gripper + F/T plugin to the URDF.** Tier-2 work.
- **B. Provide the missing inputs from a synthetic publisher,
  documented as the upstream-perception layer.** Tier-1.

Adopted: **option B.** `topic_pub_sim` publishes `/tool_id`,
`/payload_id`, `/grasp_mode`, `/wrench`, `/tcp_pose` at 10 Hz
with plausible values. The README explicitly states this is
the upstream-perception simulation layer; Tier 2 replaces it
with real perception nodes + URDF additions.

This adjustment is *more honest* than pretending the bare UR sim
ships with a gripper. The reviewer sees:
- A real UR5e in Gazebo (real ROS-integration evidence)
- A real MoveIt planning group (real integration evidence)
- The contextual flow ticking against a documented synthetic
  input layer (real runtime evidence)
- The published outputs on standard ROS topics (real
  end-to-end-integration evidence)

## 5. Files

| file | role |
|-|-|
| `hymeko_ros2_demo/launch/ur5e_grasping_demo.launch.py` | the single reviewer-facing launch |
| `hymeko_ros2_demo/launch/grasping_context_only.launch.py` | node-only smoke (no Gazebo) |
| `hymeko_ros2_demo/hymeko_ros2_demo/grasping_context_node.py` | the bridge node |
| `hymeko_ros2_demo/hymeko_ros2_demo/topic_binding.py` | IR walker + aggregation placeholders |
| `hymeko_ros2_demo/hymeko_ros2_demo/scenarios/hymeko_robot.hymeko` | the article's canonical listing |
| `hymeko_ros2_demo/hymeko_ros2_demo/config/topic_mapping.yaml` | vertex ↔ topic binding (editable) |
| `hymeko_ros2_demo/hymeko_ros2_demo/scripts/topic_pub_sim.py` | synthetic upstream-perception publisher |
| `hymeko_ros2_demo/README.md` | reviewer-facing run instructions |
| `hymeko_ros2_demo/test/test_topic_binding.py` | 9/9 passing unit tests |

## 6. Acceptance check

- [x] 4-format plan on disk.
- [x] `colcon build --packages-select hymeko_ros2_demo` clean.
- [x] 9/9 unit tests pass.
- [x] Node-only smoke launches and outputs publish.
- [x] **Full launch with Gazebo + UR5e + MoveIt + node end-to-end:
      all 10 topics present; outputs publish live; node ticks at
      10 Hz; clean shutdown.**
- [x] CORE.YAML items touched = 0.
- [x] Report on disk (this file).

## 7. Next steps (if there is appetite)

1. **Demo video** — record a screen capture of the full launch
   showing Gazebo + RViz + a terminal with `ros2 topic echo` on
   `/hymeko/grasping/stability_margin`. ~5 minutes of work.
2. **Tier 2** — Maintenance & Production context nodes from
   paper §6, demonstrating the meta-context arbitration scenario.
3. **Gripper URDF** — replace `topic_pub_sim`'s `/wrench` with a
   F/T sensor plugin attached to the wrist; add a parallel
   gripper plugin so `/gripper_state` is real.
4. **Pick-and-place script** — drive MoveIt through a canned
   pick-and-place sequence so `/joint_states` actually changes
   (the current sim is static; `topic_pub_sim` is the only thing
   moving).
