"""Niitsuma Tier-2 nodes-only smoke launch (no Gazebo).

Brings up the Day-1 stack:
- 2× grasping_context_node — one per robot, each bound to its own
  topic_mapping YAML (per-robot binding).
- 1× scene_context_node — computes inter-robot distance, shared
  payload, and arbitration commit signal.
- 1× arbitration_meta_node — standalone variant (redundant with the
  inline arbitration in scene_context_node; kept for the Niitsuma
  audience's "separate process" framing).
- 1× topic_pub_sim_dual — synthetic upstream-perception publisher
  that drives both robots' inputs with periodic close-approach and
  shared-payload cycles.

This is the Day-1 integration test launch.  Day-2 adds Gazebo with
two real UR5e robots; this launch validates the contextual flow +
arbitration logic before that Gazebo work begins.

Usage::

    ros2 launch hymeko_ros2_demo niitsuma_nodes_only.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare("hymeko_ros2_demo")

    scenario_arg = DeclareLaunchArgument(
        "scenario_file",
        default_value=PathJoinSubstitution(
            [pkg_share, "scenarios", "hymeko_robot_dual.hymeko"]
        ),
    )
    tick_rate_arg = DeclareLaunchArgument(
        "tick_rate_hz", default_value="10.0",
    )
    safety_threshold_arg = DeclareLaunchArgument(
        "safety_threshold", default_value="0.30",
        description="Inter-robot distance below which arbitration commits 0 (safety pause).",
    )
    commit_min_margin_arg = DeclareLaunchArgument(
        "commit_min_margin", default_value="0.50",
        description="Stability margin threshold above which a robot is committable.",
    )
    launch_arb_meta_arg = DeclareLaunchArgument(
        "launch_arbitration_meta", default_value="false",
        description=(
            "Launch arbitration_meta_node as a separate process. "
            "scene_context_node already computes the arbitration commit "
            "inline; the standalone node is for §6 demonstration only."
        ),
    )

    # Robot A grasping context
    grasp_a = Node(
        package="hymeko_ros2_demo",
        executable="grasping_context_node",
        name="grasping_context_robot_a",
        output="screen",
        parameters=[{
            "scenario_file": LaunchConfiguration("scenario_file"),
            "topic_mapping_file": PathJoinSubstitution(
                [pkg_share, "config", "topic_mapping_robot_a.yaml"]
            ),
            "tick_rate_hz": LaunchConfiguration("tick_rate_hz"),
        }],
    )

    # Robot B grasping context
    grasp_b = Node(
        package="hymeko_ros2_demo",
        executable="grasping_context_node",
        name="grasping_context_robot_b",
        output="screen",
        parameters=[{
            "scenario_file": LaunchConfiguration("scenario_file"),
            "topic_mapping_file": PathJoinSubstitution(
                [pkg_share, "config", "topic_mapping_robot_b.yaml"]
            ),
            "tick_rate_hz": LaunchConfiguration("tick_rate_hz"),
        }],
    )

    # Scene context — computes inter-robot distance, shared payload,
    # and arbitration commit inline.
    scene = Node(
        package="hymeko_ros2_demo",
        executable="scene_context_node",
        name="scene_context_node",
        output="screen",
        parameters=[{
            "tick_rate_hz":      LaunchConfiguration("tick_rate_hz"),
            "safety_threshold":  LaunchConfiguration("safety_threshold"),
            "commit_min_margin": LaunchConfiguration("commit_min_margin"),
        }],
    )

    # Optional: separate-process arbitration meta-context.
    arb_meta = Node(
        package="hymeko_ros2_demo",
        executable="arbitration_meta_node",
        name="arbitration_meta_node",
        output="screen",
        parameters=[{
            "tick_rate_hz":      LaunchConfiguration("tick_rate_hz"),
            "safety_threshold":  LaunchConfiguration("safety_threshold"),
            "commit_min_margin": LaunchConfiguration("commit_min_margin"),
        }],
        condition=__import__("launch.conditions", fromlist=["IfCondition"]).IfCondition(
            LaunchConfiguration("launch_arbitration_meta")
        ),
    )

    # Synthetic dual publisher
    pub_dual = Node(
        package="hymeko_ros2_demo",
        executable="topic_pub_sim_dual",
        name="topic_pub_sim_dual",
        output="screen",
    )

    return LaunchDescription([
        scenario_arg, tick_rate_arg,
        safety_threshold_arg, commit_min_margin_arg,
        launch_arb_meta_arg,
        pub_dual, grasp_a, grasp_b, scene, arb_meta,
    ])
