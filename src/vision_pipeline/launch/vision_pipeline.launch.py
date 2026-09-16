from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vision_pipeline',
            executable='camera_node',
            name='camera_node',
            output='screen'
        ),
        Node(
            package='vision_pipeline',
            executable='face_detector_node',
            name='face_detector_node',
            output='screen'
        ),
        Node(
            package='vision_pipeline',
            executable='tracker_node',
            name='tracker_node',
            output='screen'
        ),
        Node(
            package='vision_pipeline',
            executable='visualizer_node',
            name='visualizer_node',
            output='screen'
        ),
    ])
