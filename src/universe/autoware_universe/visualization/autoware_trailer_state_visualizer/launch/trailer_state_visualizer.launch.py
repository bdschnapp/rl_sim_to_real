"""Launch the trailer state visualizer node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("autoware_trailer_state_visualizer")
    config = os.path.join(pkg_dir, "config", "trailer_state_visualizer.param.yaml")

    return LaunchDescription([
        Node(
            package="autoware_trailer_state_visualizer",
            executable="autoware_trailer_state_visualizer_node",
            name="trailer_state_visualizer",
            parameters=[config],
            remappings=[
                # Truck odometry from the simple_planning_simulator
                ("~/input/kinematic_state", "/localization/kinematic_state"),
                # TrailerState published by the simple_planning_simulator
                ("~/input/trailer_state", "/vehicle/trailer_state"),
                # MarkerArray visible in RViz via the MarkerArray display
                ("~/output/markers", "/visualization/trailer_state"),
            ],
        )
    ])
