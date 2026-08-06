# HyMeKo manipulation models — file-by-file reference

Item-by-item detail for every `.hymeko` file in this folder. See `README.md` for the summary tables and the
two composition diagrams.

---

## Scenarios (the top-level "tests")

A **scenario** is the single model that names a whole task — it composes a robot + scene + reward (+ strategy)
by `@"…"` import, so one file is the entry point for that test.

### `pick_place_scenario.hymeko` — FANUC top-down pick-and-place
Composes: `meta_scenario` (vocab) · `arm_gripper_fanuc_import` (robot) · `pick_place_task` (reward).
Declares one `@scene` with the verified env geometry:
- `mount_height 0.12`, `table_top 0.12` — pedestal + table heights (m).
- `box_mass 0.15` — the object's mass (kg).
- `lift_thresh 0.035` — "lifted clear of the table" threshold (m).
- `place_radius 0.075` — success radius around the target (m).
- `max_steps 620` — episode horizon.
- `obj_radius [0.28, 0.40]` — object spawn annulus (collision-free top-down reach).
- `target_xy [0.34, 0.0]` — place-target location on the table.
- `arm_home [0.0, 1.0, 0.8, 0.0, 0.8, 0.0]` — bent, non-singular ready posture (6 joints).

### `galambos_scenario.hymeko` — two-arm planar coin-grasp
Composes: `meta_scenario` (vocab) · `galambos_planar` (robot) · `galambos_env` (scene) · `galambos_task`
(reward) · `galambos_strategy` (RL strategy). Declares one `@scene` with `max_steps 160`; the rest of the scene
lives in `galambos_env`. This is the Galambos analogue of `pick_place_scenario` (it also carries the strategy).

---

## Vocabularies (`meta_*` — the type roots that concrete models instantiate)

### `meta_kinematics.hymeko`
- **`units`**: `length "m"`, `angle "degree"`, `mass "kg"`, `time "s"`.
- **`elements`**: `meta_element` root; `link`, `frame`, `control`, `sensor` (each `<isa> meta_element`);
  `@joint` (with an embedded `@control`), `@control_definition`.
- **`geometry`**: shape kinds (`box`, …) used by `link_geometry`.
- **`axes`**: named axes `AXIS_X`, `AXIS_Y`, `AXIS_Z` for joint directions.
- **joint kinds**: `fixed_joint`, `rev_joint` (revolute), `prismatic_joint`, `conti_joint` (continuous).
- **`sensors`**: `joint_state_broadcaster`, `rgb_camera` (type `camera`), `laser_scanner` (`gpu_lidar`),
  `@sensor_connection`.
- **`controllers`**: `meta_controller` (+ `@state_interface`, `@command_interface`),
  `joint_trajectory_controller`, `diff_drive_controller`, …

### `meta_reward.hymeko`
- Roots: `@reward_term` (one weighted signal), `@reward_spec` (a bundle = the scalar reward Σ weight·term).
- **`terms`** (28 kinds, each `<isa> reward_term` with a default `weight 0.0`): `reach_distance`, `success_bonus`,
  `action_cost`, `ground_penalty`, `self_collision_penalty`, `joint_limit_penalty`, `below_ground_penalty`,
  `grasp_approach`, `both_contact`, `in_zone`, `settle`, `arm_motion`, `center_bonus`, `arm_collision`,
  `out_of_bounds`, `goal_progress`, `time_penalty`, `joint_velocity`, `joint_acceleration`, and the pick terms
  `pick_approach`, `pick_contact`, `pick_lift`, `pick_place_distance`, `pick_place_bonus`,
  `pick_approach_penalty`, `pick_disturbance`.

### `meta_observation.hymeko`
- Roots: `@feature` (per-vertex channel), `@global` (broadcast/task-level channel), `@space` (a bundle).
- **`features`**: `joint_position` (dim 1), `joint_velocity` (1), `joint_effort` (1), `link_pose` (3),
  `link_twist` (6).
- **`globals`**: `target_position` (3), `ee_error` (3), `command` (1).
- `@observation_space` — a channel bundle.

### `meta_task.hymeko`
- Roots: `@action`, `@composite`, `@condition`, `@coordination_primitive`, `@scene_element`.
- **`actions`**: `move_to`, `joint_move`, `grip_open`, `grip_close`, `wait` (`duration_s`), `apply_force`, `noop`.
- **`composites`** (behaviour-tree/FSM operators): `sequence`, `parallel` (`policy "all"`), `fallback`,
  `loop` (`max_iterations 100`), `invert`, `entry`.
- **`conditions`**: `precondition`, `postcondition`, `at_pose` (`tolerance_m 0.005`, `tolerance_rad 0.01`),
  `holding`, `at_config`, …

### `meta_env.hymeko`
- Roots: `@param`, `@env_spec`. **`params`**: `target_zone`, `coin_spawn`, `workspace`, `success`, `disk`.

### `meta_strategy.hymeko`
- Roots: `@param`, `@strategy_spec`. **`params`**: `exploration`, `exploitation`.

### `meta_scenario.hymeko`
- `scenario` namespace with one kind `@scene` (a scene-parameter bundle that builds an RL environment).

---

## Robots (kinematic structure)

### `fanuc_lrmate.hymeko` — FANUC LR Mate-config 6-DOF arm
- Links: `base_link`, `link_0 … link_4`, `tool` (slim collision cylinders).
- Joints: `@j0 @j1 @j2 @j3 @j4 @jtool` — axis pattern **Z·Y·Y·Z·Y·Z** (base yaw + Y shoulder/elbow + Z-Y-Z
  spherical wrist), the config that lets the tool point straight down collision-free. `@j_fix` mounts the base;
  `@arm_joint_control` is the shared control definition.

### `arm_gripper_fanuc_import.hymeko` — gripper attached to the arm
- Imports `fanuc_lrmate` (as `arm`) + `meta_kinematics`. Adds links `finger_l`, `finger_r` (mass 0.1, box geom)
  and prismatic joints `@grip_l`, `@grip_r` on `arm.tool` at `[±0.035, 0, 0.06]` along `AXIS_X` (the parallel-jaw
  gripper, no robot duplication).

### `galambos_planar.hymeko` — two 2-link planar arms
- Links: `base_left`, `base_right` (mass 0.4, 44×44×24 mm boxes); `upper_left`, `upper_right` (mass 0.25,
  160 mm); `lower_left`, `lower_right` (mass 0.2, 140 mm); a `world` frame.
- Joints: `@jl1, @jl2` (left shoulder+elbow), `@jr1, @jr2` (right) — Z-hinges sweeping the table; `@fix_left`,
  `@fix_right` mount the bases.

---

## Rewards (Σ weight·term)

### `pick_place_task.hymeko` — FANUC reward (7 terms)
`@approach` (`pick_approach`, w 1.0) · `@contact` (`pick_contact`, 0.5) · `@lift` (`pick_lift`, 5.0) ·
`@place` (`pick_place_distance`, 1.0) · `@placed` (`pick_place_bonus`, 20.0) · `@noground`
(`pick_approach_penalty`, 2.0) · `@nonudge` (`pick_disturbance`, 3.0). Bundle `@pick_reward` sums all seven.

### `galambos_task.hymeko` — Galambos reward
`@approach` (`grasp_approach`, w 4.0) · `@pull` (`reach_distance`, 1.0, edge `(+disk, -target_zone)`) ·
`@both` (`both_contact`, 3.0) · `@zone` (`in_zone`, 10.0) · `@center` (`center_bonus`, 5.0) · `@explore`
(`arm_motion`, 0.5) · `@noclash` (`arm_collision`, 1.0) · `@oob` (`out_of_bounds`, 5.0) · `@timecost`
(`time_penalty`, 0.10) · `@smoothv` (`joint_velocity`, 0.005) · `@smootha` (`joint_acceleration`, 0.01).
Bundle `@grasp_reward` sums them.

---

## Scenes & strategies

### `galambos_env.hymeko` — Galambos scene
- `@zone` (`target_zone`): `half 0.04`, region `rx[-0.05,0.05] ry[0.10,0.18]`, `randomize 1.0`.
- `@spawn` (`coin_spawn`): region `rx[-0.20,0.20] ry[0.05,0.23]`, `clearance 0.03`.
- `@bounds` (`workspace`): `x_bound 0.40`, `y_min -0.08`, `y_max 0.45`.
- `@succ` (`success`): `steps 5.0` (in-zone steps to count a delivery).
- `@dsk` (`disk`): `radius 0.02` (the coin).
- `@env_spec` bundles all five.

### `galambos_strategy.hymeko` — Galambos RL strategy
- `@explore` (`exploration`): `ent_coef 0.01`, `log_std_init -0.5`, `curriculum_iters 200`.
- `@exploit` (`exploitation`): `gamma 0.99`, `lam 0.95`, `clip 0.2`, `lr 0.0003`, `update_epochs 8`,
  `value_warmup 0`, `n_steps 512`, `n_iters 300`.
- `@strategy_spec` bundles both.
