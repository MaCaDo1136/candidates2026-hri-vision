from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='audio_pipeline',
            executable='vad_node',
            name='vad_node',
            output='screen'
        ),
        Node(
            package='audio_pipeline',
            executable='transcribe_node',
            name='transcribe_node',
            output='screen'
        ),
        Node(
            package='audio_pipeline',
            executable='voice_id_node',
            name='voice_id_node',
            output='screen'
        ),
    ])
