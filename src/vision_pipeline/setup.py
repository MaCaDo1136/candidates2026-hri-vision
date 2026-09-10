import os

from setuptools import find_packages, setup
from glob import glob

package_name = 'vision_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'),
         glob('vision_pipeline/models/*.pt')),
    ],
    install_requires=['setuptools', 'ultralytics'],
    zip_safe=True,
    maintainer='Mario Casas',
    maintainer_email='contact@mariocasas.dev',
    description='A ROS 2 package for processing vision data for face recognition',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_node = vision_pipeline.camera_node:main',
            'face_detector_node = vision_pipeline.face_detector_node:main',
        ],
    },
)
