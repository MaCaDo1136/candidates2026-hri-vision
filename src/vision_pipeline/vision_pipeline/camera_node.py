#! /usr/bin/env python3

"""
Description:
        Camera node for the vision pipeline. This node is responsible for capturing images
    from the camera and publishing them to a specific topic.

------------------------------
Publishing topics:
    Camera node publishes images to the following topic:
        /camera/image_raw - Image

------------------------------
Subscription Topics:
        None

------------------------------
Author: Mario Casas Donjuan
Date: September 9, 2026
"""

from typing import override

import rclpy
import numpy as np
import cv2

from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge


class CameraNode(Node):
    """Camera node for the vision pipeline.

    Args:
            Node: Clase base de rclpy para nodos ROS2.
    """

    def __init__(self):
        """Node initialization."""
        super().__init__('camera_node')
        self.get_logger().info('Camera node has been started.')
        self.publisher = self.create_publisher(
            # Publisher for the camera images
            Image, '/camera/image_raw', qos_profile_sensor_data)
        self.bridge = CvBridge()
        self.camera = cv2.VideoCapture(0)  # Initialize the camera

        periodo = 0.1  # Periodo de publicación en segundos
        self.timer = self.create_timer(periodo, self.publish_image)

    def publish_image(self):
        """Image publishing function. This function captures an image from the camera and publishes it to the topic."""
        ret, frame = self.camera.read()
        if not ret:
            self.get_logger().error('Failed to capture image from camera')
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.publisher.publish(msg)  # Published in the topic /camera/image_raw

    @override
    def destroy_node(self):
        """Override the destroy_node method to release the camera resource."""
        self.camera.release()  # Release the camera resource
        super().destroy_node()


def main(args=None):
    """Entry point for the camera node."""
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
