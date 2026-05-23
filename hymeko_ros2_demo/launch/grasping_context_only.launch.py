"""Node-only smoke launch: grasping_context_node + synthetic publisher.

Use this when the UR + Gazebo stack is not available (CI, headless
review, quick demonstration). The synthetic publisher feeds plausible
ROS topic values at 10 Hz so the contextual flow has something to
chew on.

Example:
    ros2 launch hymeko_ros2_demo grasping_context_only.launch.py
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
            [pkg_share, "scenarios", "hymeko_robot.hymeko"]
        ),
        description="Path to the .hymeko scenario file.",
    )
    mapping_arg = DeclareLaunchArgument(
        "topic_mapping_file",
        default_value=PathJoinSubstitution(
            [pkg_share, "config", "topic_mapping.yaml"]
        ),
        description="Path to the topic_mapping.yaml config.",
    )
    rate_arg = DeclareLaunchArgument(
        "tick_rate_hz",
        default_value="10.0",
        description="Context evaluation rate.",
    )

    grasping_node = Node(
        package="hymeko_ros2_demo",
        executable="grasping_context_node",
        name="grasping_context_node",
        output="screen",
        parameters=[{
            "scenario_file": LaunchConfiguration("scenario_file"),
            "topic_mapping_file": LaunchConfiguration("topic_mapping_file"),
            "tick_rate_hz": LaunchConfiguration("tick_rate_hz"),
        }],
    )

    pub_sim = Node(
        package="hymeko_ros2_demo",
        executable="topic_pub_sim",
        name="topic_pub_sim",
        output="screen",
    )

    return LaunchDescription([scenario_arg, mapping_arg, rate_arg,
                                grasping_node, pub_sim])
