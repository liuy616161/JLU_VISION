import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command
from launch_ros.actions import Node

launch_params = yaml.safe_load(open(os.path.join(
    get_package_share_directory('rm_vision_bringup'), 'config', 'launch_params.yaml')))

robot_description = Command(['xacro ', os.path.join(
    get_package_share_directory('rm_gimbal_description'), 'urdf', 'rm_gimbal.urdf.xacro'),
    ' xyz:=', launch_params['odom2camera']['xyz'], ' rpy:=', launch_params['odom2camera']['rpy']])

robot_state_publisher = Node(
    package='robot_state_publisher',
    executable='robot_state_publisher',
    parameters=[{'robot_description': robot_description,
                 'publish_frequency': 1000.0}]
)

front_params = os.path.join(
    get_package_share_directory('rm_vision_bringup'), 'config', 'front_params.yaml')

back_params = os.path.join(
    get_package_share_directory('rm_vision_bringup'), 'config', 'back_params.yaml')

serial_params = os.path.join(
    get_package_share_directory('rm_vision_bringup'), 'config', 'serial_params.yaml')

front_tracker_node = Node(
    package='armor_tracker',
    executable='armor_tracker_node',
    namespace='front',
    output='both',
    emulate_tty=True,
    parameters=[front_params],
    ros_arguments=['--log-level', 'armor_tracker:='+launch_params['tracker_log_level']],
)

back_targeter_node = Node(
    package='armor_targeter',
    executable='armor_targeter_node',
    namespace='back',
    output='both',
    emulate_tty=True,
    parameters=[back_params],
)
