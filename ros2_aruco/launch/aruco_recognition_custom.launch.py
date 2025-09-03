import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    config_path = LaunchConfiguration('aruco_custom_config')

    return LaunchDescription([
        DeclareLaunchArgument(
            'aruco_custom_config',
            default_value='/absolute/path/to/aruco_custom.yaml',  # Replace with actual path
            description='Path to custom aruco YAML config file'
        ),

        Node(
            package='ros2_aruco',
            executable='aruco_node',
            name='aruco_node',
            output='screen',
            parameters=[config_path]  # This works only when launched with ros2 launch
        )
    ])
