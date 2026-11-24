"""Launch the full roundabout stack (expects /scan and camera already up)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("roundabout_pkg")
    default_params = os.path.join(pkg_share, "config", "roundabout.yaml")

    params_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to roundabout YAML params",
    )
    params = LaunchConfiguration("params_file")

    common = {"parameters": [params]}

    return LaunchDescription(
        [
            params_arg,
            Node(
                package="roundabout_pkg",
                executable="lane_follower_node",
                name="lane_follower_node",
                output="screen",
                **common,
            ),
            Node(
                package="roundabout_pkg",
                executable="yield_detector_node",
                name="yield_detector_node",
                output="screen",
                **common,
            ),
            Node(
                package="roundabout_pkg",
                executable="car_tracker_node",
                name="car_tracker_node",
                output="screen",
                **common,
            ),
            Node(
                package="roundabout_pkg",
                executable="right_of_way_node",
                name="right_of_way_node",
                output="screen",
                **common,
            ),
            Node(
                package="roundabout_pkg",
                executable="cmd_vel_mux_node",
                name="cmd_vel_mux_node",
                output="screen",
                **common,
            ),
        ]
    )
