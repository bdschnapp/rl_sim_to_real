"""Launch the trailer state visualizer node.

Loads two parameter files:
  1. trailer_state_visualizer.param.yaml — visualization-only knobs (mesh
     resource/scale/offset, marker toggles).
  2. The vehicle-specific trailer geometry yaml (hitch_offset, trailer_wheelbase,
     trailer_width, etc.) — same file the simple_planning_simulator loads, so
     the two nodes can't disagree about trailer shape. Override via the
     trailer_geometry_param_file launch arg for non-default vehicles.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("autoware_trailer_state_visualizer")
    vis_config = os.path.join(pkg_dir, "config", "trailer_state_visualizer.param.yaml")

    default_geometry = os.path.join(
        get_package_share_directory("electrans_robot_vehicle_description"),
        "config",
        "simulator_model.param.yaml",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "trailer_geometry_param_file",
            default_value=default_geometry,
            description=(
                "Path to the yaml that owns trailer geometry (hitch_offset, "
                "trailer_wheelbase, trailer_width, etc.). Defaults to the "
                "electrans_robot_vehicle_description simulator_model file so "
                "the visualizer matches the sim. Override for other vehicles."
            ),
        ),
        Node(
            package="autoware_trailer_state_visualizer",
            executable="autoware_trailer_state_visualizer_node",
            name="trailer_state_visualizer",
            parameters=[vis_config, LaunchConfiguration("trailer_geometry_param_file")],
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
