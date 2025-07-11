import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    # config = os.path.join(
    #     get_package_share_directory('rm_serial_driver'), 'config', 'serial_driver.yaml')
    rm_serial_driver_dir = get_package_share_directory('rm_serial_driver')
    rm_serial_driver_yaml_path = os.path.join(rm_serial_driver_dir,'config','serial_driver.yaml')
    declare_config_path_cmd = DeclareLaunchArgument(
        'rm_serial_driver_yaml_path', default_value=rm_serial_driver_yaml_path,
        description='Yaml config file path'
    )
    rm_serial_driver_node = Node(
        package='rm_serial_driver',
        executable='rm_serial_driver_node',
        namespace='',
        output='screen',
        emulate_tty=True,
        parameters=[rm_serial_driver_yaml_path],
    )

    return LaunchDescription([declare_config_path_cmd,rm_serial_driver_node])
