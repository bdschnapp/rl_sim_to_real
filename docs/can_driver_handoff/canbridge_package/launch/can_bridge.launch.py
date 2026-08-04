from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    can_interface_arg = DeclareLaunchArgument(
        'can_interface',
        default_value='can1',
        description='SocketCAN interface name (e.g. can0 or can1)'
    )
    can_interface = LaunchConfiguration('can_interface')

    socketcan_node = Node(
        package='canbridge',
        executable='ros2can_bridge',
        name='ros2socketcan_bridge',
        parameters=[{'can_interface': can_interface}],
        output='screen',
    )

    canbridge_node = Node(
        package='canbridge',
        executable='canbridge',
        name='controller_canbridge',
        parameters=[{'can_interface': can_interface}],
        output='screen',
    )

    return LaunchDescription([
        can_interface_arg,
        socketcan_node,
        canbridge_node,
    ])
