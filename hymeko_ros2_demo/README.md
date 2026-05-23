# hymeko_ros2_demo — MDPI Technologies live demo (Tier 1)

Live demonstration of the HyMeKo Multi-Contextual State Representation
on a UR5e + MoveIt2 + Gazebo stack. The grasping context from the
MDPI Technologies article runs in real time against the robot's
joint/wrench/gripper topics and publishes the aggregated outputs
(stability margin, configuration, contact force) at 10 Hz.

**Scope.** Tier 1 demonstrates *runtime existence* — the contextual
representation runs against a real robot's data stream. It does
**not** demonstrate that the contextual representation makes the
robot work better; that would be a separate experimental
contribution. The paper is theoretical.

## Prerequisites

ROS 2 **Kilted** (verified) on Ubuntu 24.04. All UR / MoveIt / Gazebo
dependencies are official packages:

```bash
sudo apt install \
  ros-kilted-ur-description ros-kilted-ur-robot-driver \
  ros-kilted-ur-moveit-config ros-kilted-ur-simulation-gz \
  ros-kilted-moveit-* ros-kilted-gz-* ros-kilted-ros2-controllers
```

The HyMeKo Python wheel is built locally from this repo:

```bash
pip install --user --break-system-packages \
  target/wheels/hymeko-0.1.0-cp312-cp312-linux_x86_64.whl
```

(System Python on Ubuntu 24.04 is 3.12, matching the wheel ABI.)

## Build

From a ROS 2 workspace `src/` containing this package:

```bash
colcon build --packages-select hymeko_ros2_demo
source install/setup.bash
```

## Run

### A. Node-only smoke (no UR, no Gazebo)

Quickest way to confirm the contextual flow runs against any data
stream. A synthetic publisher feeds the 5 input topics at 10 Hz.

```bash
ros2 launch hymeko_ros2_demo grasping_context_only.launch.py
```

You should see, every ~5 seconds, the node print the current
`V_global` state:

```
[grasping_context_node] tick 50: V_global = {'robot_pose': 0.398, ...
  'stability_margin': 0.124, 'configuration': 0.55, 'force_vector': 0.30}
```

And the outputs publish on `/hymeko/grasping/{stability_margin, configuration, contact_force}`:

```bash
ros2 topic echo /hymeko/grasping/stability_margin
```

### B. Full Tier-1 (UR5e in Gazebo + MoveIt + grasping context)

```bash
ros2 launch hymeko_ros2_demo ur5e_grasping_demo.launch.py
```

Defers to the upstream `ur_sim_moveit.launch.py` (Gazebo + UR5e +
ros2_control + MoveIt + RViz2), then alongside:

- `topic_pub_sim` — provides the synthetic upstream-perception
  topics (`/tool_id`, `/payload_id`, `/grasp_mode`, `/wrench`,
  `/tcp_pose`) that an industrial deployment would normally get
  from a vision stack and a F/T sensor. The bare UR sim does
  **not** ship with a gripper or F/T plugin; these inputs are
  app-level metadata, not part of the controller surface.
- `grasping_context_node` — loads `hymeko_robot.hymeko`,
  evaluates the 6 signed hyperedges, publishes
  `/hymeko/grasping/*` at 10 Hz.

**What this proves at Tier 1:** the contextual flow runs in real
time inside a ROS 2 + Gazebo + MoveIt2 simulation, ticking
against (synthetic) upstream-perception inputs and publishing
back to ROS topics that RViz2 / a downstream MoveIt planner could
consume.

**Tier 2 (deferred):** drop the synthetic publisher; add real
parallel-gripper + F/T plugin to the URDF; add a Maintenance
context node (joint temperature / vibration topics) and the
arbitration meta-context from paper §6.

**Real UR5e:** not handled by this launch — use upstream
`ur_robot_driver`'s `ur_control.launch.py` separately and start the
grasping_context_node with `grasping_context_only.launch.py` for
the synthetic inputs.

## What the demo proves

The contextual representation runs in real time at 10 Hz on a real
robot's data stream, applying the 6-hyperedge grasping context flow
(`derive_tool`, `derive_payload`, `loading_state`, `grasp_config`,
`load_force`, `grasp_stability`) to live ROS topic values and
publishing the aggregated outputs back.

## What the demo does NOT prove

The aggregation functions for each hyperedge use practitioner
placeholders (see `hymeko_ros2_demo/topic_binding.py::aggregate_*`).
The paper specifies the *signed incidence structure*, not the
closed-form aggregations — so the value of `S_l` moves plausibly
with the input stream, but the absolute mapping is illustrative,
not a learned policy.

## Reviewer instructions (one-page summary)

1. `pip install --user --break-system-packages ./target/wheels/hymeko-0.1.0-cp312-cp312-linux_x86_64.whl`
2. `colcon build --packages-select hymeko_ros2_demo && source install/setup.bash`
3. **Smoke (60 s):** `ros2 launch hymeko_ros2_demo grasping_context_only.launch.py`
4. **Full demo:** `ros2 launch hymeko_ros2_demo ur5e_grasping_demo.launch.py`
5. Watch `ros2 topic echo /hymeko/grasping/stability_margin` for the
   live aggregation output.

## Tests

```bash
colcon test --packages-select hymeko_ros2_demo
colcon test-result --verbose
```

## File map

```
hymeko_ros2_demo/
├── package.xml, setup.py, setup.cfg
├── resource/hymeko_ros2_demo
├── hymeko_ros2_demo/
│   ├── __init__.py
│   ├── grasping_context_node.py     # the bridge node
│   ├── topic_binding.py             # IR walker + aggregation
│   ├── scenarios/hymeko_robot.hymeko
│   ├── config/topic_mapping.yaml
│   └── scripts/topic_pub_sim.py     # synthetic input pub
├── launch/
│   ├── grasping_context_only.launch.py
│   └── ur5e_grasping_demo.launch.py
├── test/test_topic_binding.py
└── README.md
```

## Tier 2 / 3 hooks

The scenario file already declares Maintenance and Safety contexts (the
paper §6 cross-context arbitration scenario). Tier 1 only binds the
grasping context; Tier 2 would add nodes for the other two and the
arbitration meta-context.

The XR / digital-twin layer (article §11) is the Tier-3 target — see
`docs/plans/2026-05-21-hymeko-portable-surfaces/design-sketch.md` for
the Three.js viewer direction.
