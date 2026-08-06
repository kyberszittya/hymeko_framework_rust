"""Make the ament-python package importable under a plain ``pytest`` run.

On the ROS 2 box the package is colcon-installed and on ``PYTHONPATH``;
off it (e.g. running the pure topic_binding / evaluate_context tests on a
dev machine without ROS) the inner ``hymeko_ros2_demo`` package dir must
be added to ``sys.path`` so ``import hymeko_ros2_demo.topic_binding``
resolves. This affects only test collection.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
