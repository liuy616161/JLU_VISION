import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    node_params = os.path.join(
        get_package_share_directory('v4l2_camera'), 'config', 'camera_params.yaml')

    return LaunchDescription([
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            output='screen',
            emulate_tty=True,
            parameters=[node_params],
        )
    ])
