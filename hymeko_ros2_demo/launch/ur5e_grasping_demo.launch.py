"""Full Tier-1 demo: UR5e in Gazebo + MoveIt2 + the HyMeKo grasping_context_node.

Sim layout
----------
* ``ur_sim_moveit.launch.py`` (upstream, from ``ur_simulation_gz``)
  brings up Gazebo + the UR5e + ros2_control + MoveIt2 + RViz2.
* ``topic_pub_sim`` provides the synthetic *upstream-perception*
  topics (``/tool_id``, ``/payload_id``, ``/grasp_mode``,
  ``/wrench``, ``/tcp_pose``) that an industrial deployment would
  get from a vision stack / wrench sensor.  These are app-level
  inputs to the grasping context, not part of the bare driver
  surface.
* ``grasping_context_node`` loads ``hymeko_robot.hymeko``,
  evaluates the 6 signed hyperedges at 10 Hz, and publishes the
  aggregated outputs on ``/hymeko/grasping/*``.

Tier-2 (not in this launch): replace ``topic_pub_sim`` with real
perception nodes; add a parallel-gripper plugin in the URDF and a
F/T sensor for true wrench data.

Example (sim, default):
    ros2 launch hymeko_ros2_demo ur5e_grasping_demo.launch.py

Real-robot mode is *not* this launch file --- use
``ur_robot_driver``'s ``ur_control.launch.py`` separately and run
``grasping_context_only.launch.py``-style nodes against it; the
sim launch deliberately ships only the sim path because that's
what the Tier-1 reviewer demo targets.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare("hymeko_ros2_demo")

    # ─── Args ─────────────────────────────────────────────────────
    ur_type_arg = DeclareLaunchArgument(
        "ur_type", default_value="ur5e",
        description="UR variant. Default ur5e per the demo plan.",
    )
    scenario_arg = DeclareLaunchArgument(
        "scenario_file",
        default_value=PathJoinSubstitution(
            [pkg_share, "scenarios", "hymeko_robot.hymeko"]
        ),
        description="Path to the .hymeko scenario file.",
    )
    mapping_arg = DeclareLaunchArgument(
        "topic_mapping_file",
        default_value=PathJoinSubstitution(
            [pkg_share, "config", "topic_mapping.yaml"]
        ),
        description="Path to topic_mapping.yaml.",
    )
    tick_rate_arg = DeclareLaunchArgument(
        "tick_rate_hz", default_value="10.0",
    )
    enable_synth_arg = DeclareLaunchArgument(
        "enable_synthetic_inputs", default_value="true",
        description=(
            "Run topic_pub_sim alongside the UR sim to provide "
            "tool_id/payload_id/grasp_mode/wrench/tcp_pose. "
            "Disable when real perception nodes are present."
        ),
    )
    enable_motion_arg = DeclareLaunchArgument(
        "enable_motion", default_value="true",
        description=(
            "Start pick_and_place after the controllers spawn so "
            "the UR5e visibly moves in Gazebo. Disable to keep the "
            "arm static (e.g. when MoveIt is driven from RViz)."
        ),
    )
    motion_delay_arg = DeclareLaunchArgument(
        "motion_delay_s", default_value="20.0",
        description=(
            "Seconds to wait after launch start before pick_and_place "
            "begins. Needs >= ~15 s for ros2_control to spawn the "
            "scaled_joint_trajectory_controller action server."
        ),
    )
    launch_rviz_arg = DeclareLaunchArgument(
        "launch_rviz", default_value="true",
        description=(
            "Bring up RViz2 + MoveIt2 via ur_sim_moveit.launch.py. "
            "Pass launch_rviz:=false to use ur_sim_control.launch.py "
            "instead (sim + controllers only — no MoveIt, no RViz). "
            "Recommended workaround if RViz2 crashes with SIGSEGV on "
            "NVIDIA + OGRE-Next."
        ),
    )
    launch_dashboard_arg = DeclareLaunchArgument(
        "launch_dashboard", default_value="true",
        description=(
            "Bring up the PyQt5 dashboard showing V_global, edge "
            "diagram, time-series, joint motion, and HTL monitor."
        ),
    )
    htl_formula_arg = DeclareLaunchArgument(
        "htl_formula", default_value="G(stability_margin > 0.05)",
        description=(
            "HTL (Hypergraph Temporal Logic) formula to evaluate "
            "against the live V_global stream in the dashboard's "
            "HTL panel. Empty = no monitor."
        ),
    )
    motion_backend_arg = DeclareLaunchArgument(
        "motion_backend", default_value="direct",
        choices=["direct", "moveit"],
        description=(
            "How to drive the UR5e:\n"
            "  direct  - send joint trajectories to scaled_joint_trajectory_controller\n"
            "            directly. Simple, deterministic, doesn't need MoveIt.\n"
            "  moveit  - call MoveIt's /move_action with joint goals. Plans, smooths,\n"
            "            and collision-checks. Also gates each waypoint on the live\n"
            "            /hymeko/grasping/stability_margin (closed-loop demo).\n"
            "            REQUIRES launch_rviz:=true so move_group is running."
        ),
    )
    stability_gate_arg = DeclareLaunchArgument(
        "stability_gate", default_value="0.05",
        description=(
            "When motion_backend:=moveit, the planner waits until "
            "/hymeko/grasping/stability_margin > stability_gate before "
            "executing the next waypoint. Set to 0 to disable the gate."
        ),
    )

    # ─── UR + (optional) MoveIt + RViz ──────────────────────────
    # The upstream ur_sim_moveit.launch.py hard-codes launch_rviz=true,
    # which crashes on some NVIDIA + OGRE-Next configurations (SIGSEGV).
    # When launch_rviz:=false, fall back to ur_sim_control.launch.py
    # (sim + controllers only — no MoveIt, no RViz).  The HyMeKo
    # contextual flow and the pick_and_place motion both work without
    # MoveIt; only the visual planning panel is missed.
    #
    # IMPORTANT: we pass a RELAXED controllers YAML so the
    # scaled_joint_trajectory_controller's goal_time / per-joint
    # tolerance / stopped_velocity_tolerance work for Gazebo
    # simulation.  The upstream defaults (goal_time=0, goal=0.1 rad)
    # are tight for sim physics and cause MoveIt's /move_action to
    # report GOAL_TOLERANCE_VIOLATED even though the trajectory
    # executed.  See config/ur_controllers_relaxed.yaml.
    relaxed_controllers = PathJoinSubstitution(
        [pkg_share, "config", "ur_controllers_relaxed.yaml"]
    )

    ur_sim_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("ur_simulation_gz"),
                "launch", "ur_sim_moveit.launch.py",
            ])
        ),
        launch_arguments={
            "ur_type": LaunchConfiguration("ur_type"),
            "controllers_file": relaxed_controllers,
        }.items(),
        condition=IfCondition(LaunchConfiguration("launch_rviz")),
    )
    ur_sim_control_only = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("ur_simulation_gz"),
                "launch", "ur_sim_control.launch.py",
            ])
        ),
        launch_arguments={
            "ur_type": LaunchConfiguration("ur_type"),
            "launch_rviz": "false",
            "controllers_file": relaxed_controllers,
        }.items(),
        condition=__import__("launch.conditions", fromlist=["UnlessCondition"]).UnlessCondition(
            LaunchConfiguration("launch_rviz")
        ),
    )

    # ─── Synthetic upstream-perception publisher ─────────────────
    # Sim-time parameter: Gazebo publishes /clock, so every node that
    # stamps messages must use sim time, otherwise TF mixes wall-clock
    # and sim-clock stamps → tf2_buffer "jump back in time" warnings.
    sim_time_params = [{"use_sim_time": True}]

    pub_sim = Node(
        package="hymeko_ros2_demo",
        executable="topic_pub_sim",
        name="topic_pub_sim",
        output="screen",
        parameters=sim_time_params,
        condition=IfCondition(LaunchConfiguration("enable_synthetic_inputs")),
    )

    # pick_and_place backends — pick one based on motion_backend arg.
    # Both delayed so the controller action server (direct) or MoveIt
    # move_group (moveit) is fully up.
    motion_is_direct = __import__(
        "launch.substitutions", fromlist=["PythonExpression"]
    ).PythonExpression([
        "'", LaunchConfiguration("motion_backend"), "' == 'direct'"
    ])
    motion_is_moveit = __import__(
        "launch.substitutions", fromlist=["PythonExpression"]
    ).PythonExpression([
        "'", LaunchConfiguration("motion_backend"), "' == 'moveit'"
    ])

    pick_and_place_direct = Node(
        package="hymeko_ros2_demo",
        executable="pick_and_place",
        name="hymeko_pick_and_place",
        output="screen",
        parameters=sim_time_params,
        condition=__import__("launch.conditions", fromlist=["IfCondition"]).IfCondition(
            __import__("launch.substitutions", fromlist=["PythonExpression"]).PythonExpression([
                "'", LaunchConfiguration("enable_motion"), "' == 'true' and ",
                "'", LaunchConfiguration("motion_backend"), "' == 'direct'"
            ])
        ),
    )
    pick_and_place_moveit = Node(
        package="hymeko_ros2_demo",
        executable="pick_and_place_moveit",
        name="hymeko_pick_and_place_moveit",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "stability_gate": LaunchConfiguration("stability_gate"),
        }],
        condition=__import__("launch.conditions", fromlist=["IfCondition"]).IfCondition(
            __import__("launch.substitutions", fromlist=["PythonExpression"]).PythonExpression([
                "'", LaunchConfiguration("enable_motion"), "' == 'true' and ",
                "'", LaunchConfiguration("motion_backend"), "' == 'moveit'"
            ])
        ),
    )
    pick_and_place_delayed = TimerAction(
        period=20.0,
        actions=[pick_and_place_direct, pick_and_place_moveit],
    )

    # MoveIt simulation tolerance relaxation.  After move_group is up,
    # set trajectory_execution.allowed_start_tolerance high enough that
    # Gazebo's joint-position drift doesn't reject a planning request.
    # Upstream default is 0.01 rad (real-robot tight); we set 0.2 rad.
    # (Only fires when launch_rviz:=true → move_group exists.)
    relax_moveit_tolerance = ExecuteProcess(
        cmd=[
            "bash", "-c",
            "ros2 param set /move_group trajectory_execution.allowed_start_tolerance 0.2 "
            "&& ros2 param set /move_group trajectory_execution.allowed_goal_duration_margin 5.0 "
            "&& echo '[relax] MoveIt tolerances relaxed for sim'",
        ],
        shell=False,
        output="screen",
        condition=IfCondition(LaunchConfiguration("launch_rviz")),
    )
    relax_moveit_delayed = TimerAction(period=18.0, actions=[relax_moveit_tolerance])

    # PyQt5 dashboard — visualization + HTL monitor.
    dashboard = Node(
        package="hymeko_ros2_demo",
        executable="dashboard_node",
        name="hymeko_dashboard",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "htl_formula": LaunchConfiguration("htl_formula"),
        }],
        condition=IfCondition(LaunchConfiguration("launch_dashboard")),
    )
    # Delay so grasping_context_node has registered its publisher.
    dashboard_delayed = TimerAction(period=8.0, actions=[dashboard])

    # ─── The HyMeKo contextual-flow node ─────────────────────────
    grasping_node = Node(
        package="hymeko_ros2_demo",
        executable="grasping_context_node",
        name="grasping_context_node",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "scenario_file": LaunchConfiguration("scenario_file"),
            "topic_mapping_file": LaunchConfiguration("topic_mapping_file"),
            "tick_rate_hz": LaunchConfiguration("tick_rate_hz"),
        }],
    )

    return LaunchDescription([
        ur_type_arg, scenario_arg, mapping_arg, tick_rate_arg,
        enable_synth_arg, enable_motion_arg, motion_delay_arg,
        launch_rviz_arg, launch_dashboard_arg, htl_formula_arg,
        motion_backend_arg, stability_gate_arg,
        ur_sim_moveit, ur_sim_control_only,
        pub_sim,
        grasping_node,
        pick_and_place_delayed,
        dashboard_delayed,
        relax_moveit_delayed,
    ])
