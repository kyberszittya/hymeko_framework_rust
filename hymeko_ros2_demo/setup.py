from glob import glob
from setuptools import setup

package_name = "hymeko_ros2_demo"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob(package_name + "/config/*.yaml")),
        ("share/" + package_name + "/scenarios", glob(package_name + "/scenarios/*.hymeko")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="HyMeKo",
    maintainer_email="csaba.hajdu@bme.hu",
    description=(
        "Live demonstration of the HyMeKo Multi-Contextual State "
        "Representation on a UR5e + MoveIt2 + Gazebo stack."
    ),
    license="MIT OR Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "grasping_context_node = "
            "hymeko_ros2_demo.grasping_context_node:main",
            "topic_pub_sim = "
            "hymeko_ros2_demo.scripts.topic_pub_sim:main",
            "pick_and_place = "
            "hymeko_ros2_demo.scripts.pick_and_place:main",
            "pick_and_place_moveit = "
            "hymeko_ros2_demo.scripts.pick_and_place_moveit:main",
            "dashboard_node = "
            "hymeko_ros2_demo.scripts.dashboard_node:main",
            "scene_context_node = "
            "hymeko_ros2_demo.scripts.scene_context_node:main",
            "arbitration_meta_node = "
            "hymeko_ros2_demo.scripts.arbitration_meta_node:main",
            "topic_pub_sim_dual = "
            "hymeko_ros2_demo.scripts.topic_pub_sim_dual:main",
        ],
    },
)
