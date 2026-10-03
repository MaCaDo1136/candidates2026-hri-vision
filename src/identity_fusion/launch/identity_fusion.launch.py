import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('vision_pipeline'),
            'launch', 'vision_pipeline.launch.py'))
    )

    audio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('audio_pipeline'),
            'launch', 'audio_pipeline.launch.py'))
    )

    return LaunchDescription([
        vision_launch,
        audio_launch,
        Node(
            package='rag_service',
            executable='rag_service_node',
            name='rag_service_node',
            output='screen'
        ),
        Node(
            package='identity_fusion',
            executable='identity_fusion_node',
            name='identity_fusion_node',
            output='screen'
        ),
    ])
