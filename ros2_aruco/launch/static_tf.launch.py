# import os
# from ament_index_python.packages import get_package_share_directory
# from launch import LaunchDescription
# from launch_ros.actions import Node
# 
# def generate_launch_description():
#     # Define the static transform publisher node
#     static_tf_camera_to_baselink = Node(
#         package='tf2_ros',
#         executable='static_transform_publisher',
#         name='camera_to_baselink_publisher', # Unique name for this node instance
#         output='screen',
#         arguments=[
#             '--x', '0.5',                     # x translation (in meters)
#             '--y', '0.0',                     # y translation
#             '--z', '0.0',                     # z translation
#             '--qx', '0.0',                   # qx quaternion rotation
#             '--qy', '0.0',                    # qy quaternion rotation
#             '--qz', '0.0',                   # qz quaternion rotation
#             '--qw', '1.0',                    # qw quaternion rotation
#             '--frame-id', 'base_link',        # Parent frame_id
#             '--child-frame-id', 'front_camera_link' # Child frame_id
#             # The period argument (e.g., '100') is NOT used here as it's not a CLI arg in ROS2
#             # The node will default to a high publication rate (e.g., 100 Hz)
#         ]
#     )
# 
#     # Return the LaunchDescription object with all nodes
#     return LaunchDescription([
#         static_tf_camera_to_baselink
#     ])

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_front_camera_broadcaster',
            arguments=['0.473', '0.013', '0.061', '0', '0', '0', '1', 'base_link', 'front_camera_link']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_back_camera_broadcaster',
            arguments=['-0.473', '-0.013', '0.061', '0', '0', '1', '0', 'base_link', 'back_camera_link']
        )
    ])
