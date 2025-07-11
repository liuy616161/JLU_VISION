import os
import sys
from ament_index_python.packages import get_package_share_directory
sys.path.append(os.path.join(get_package_share_directory('rm_vision_bringup'), 'launch'))

def generate_launch_description():

    from common import front_params, back_params, serial_params, launch_params, robot_state_publisher, front_tracker_node, back_targeter_node
    from launch_ros.descriptions import ComposableNode
    from launch_ros.actions import ComposableNodeContainer, Node
    from launch.actions import TimerAction, Shutdown
    from launch import LaunchDescription
    from launch.actions import ExecuteProcess

    def get_camera_detector_container(namespace, camera_package, camera_plugin, node_params):
        return ComposableNodeContainer(
            name='camera_detector_container',
            namespace=namespace,
            package='rclcpp_components',
            executable='component_container', 
            composable_node_descriptions=[
                ComposableNode(
                    package=camera_package,
                    plugin=camera_plugin,
                    name='camera_node',
                    namespace=namespace,
                    parameters=[node_params],
                    extra_arguments=[{'use_intra_process_comms': True}]
                ),
                ComposableNode(
                    package='armor_detector',
                    plugin='rm_auto_aim::ArmorDetectorNode',
                    name='armor_detector',
                    namespace=namespace,
                    parameters=[node_params],
                    extra_arguments=[{'use_intra_process_comms': True}]
                )
            ],
            output='both',
            emulate_tty=True,
            ros_arguments=['--ros-args', '--log-level',
                           'armor_detector:='+launch_params['detector_log_level']],
            on_exit=Shutdown(),
        )

    if (launch_params['camera'] == 'hik'):
        front_detector = get_camera_detector_container('front', 'hik_camera', 'hik_camera::HikCameraNode', front_params)
    elif (launch_params['camera'] == 'mv'):
        front_detector = get_camera_detector_container('front', 'mindvision_camera', 'mindvision_camera::MVCameraNode', front_params)
    elif (launch_params['camera'] == 'galaxy'):
        front_detector = get_camera_detector_container('front', 'galaxy_camera', 'galaxy_camera::GalaxyCameraNode', front_params)    
    
    back_detector = get_camera_detector_container('back', 'v4l2_camera', 'rm_auto_aim::V4L2CameraNode', back_params)
    
    revivatory_node = Node(
        package='armor_revivatory',
        executable='armor_revivatory_node',
        output='both'
    )
    
    serial_driver_node = Node(
        package='rm_serial_driver',
        executable='rm_serial_driver_node',
        name='serial_driver',
        output='both',
        emulate_tty=True,
        parameters=[serial_params],
        on_exit=Shutdown(),
        respawn=True,
        respawn_delay=1,
        ros_arguments=['--ros-args', '--log-level',
                       'serial_driver:='+launch_params['serial_log_level']],
        #prefix=['xterm -e gdb -ex run --args']
    )

    delay_serial_node = TimerAction(
        period=1.5,
        actions=[serial_driver_node],
    )

    delay_front_tracker_node = TimerAction(
        period=2.0,
        actions=[front_tracker_node],
    )

    delay_back_targeter_node = TimerAction(
        period=2.0,
        actions=[back_targeter_node],
    )

    delay_revivatory_node= TimerAction(
        period=2.0,
        actions=[revivatory_node],
    )
    
    decision_bring_together = ExecuteProcess(
        cmd = ['ros2','launch','rm_behavior_tree','rm_behavior_tree.launch.py'],
        output = 'screen'
    )
    
    return LaunchDescription([
         robot_state_publisher,
         front_detector,
         back_detector,
         delay_serial_node,
         delay_front_tracker_node,
         delay_back_targeter_node,
         delay_revivatory_node,
         decision_bring_together,
    ])
