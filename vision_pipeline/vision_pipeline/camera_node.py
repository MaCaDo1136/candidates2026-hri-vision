#! /usr/bin/env python3

"""
Description:
    Camera node for the vision pipeline. Captures images from the camera
    (local device index or a stream URL set in HRI_CAMERA_SOURCE) and publishes
    them to a topic. A reader thread drains the source continuously so the
    published frame is always the latest one (no buffered delay).

------------------------------
Publishing topics:
    /camera/image_raw - Image

------------------------------
Subscription Topics:
    None

------------------------------
Author: Mario Casas Donjuan
Date: September 9, 2026
"""

import os
import threading
import time
from typing import override

import cv2
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def camera_source():
    """Camera index (int) or URL (str) from HRI_CAMERA_SOURCE; default 0."""
    src = os.environ.get('HRI_CAMERA_SOURCE', '0')
    return int(src) if src.isdigit() else src


class CameraNode(Node):
    """Camera node for the vision pipeline."""

    def __init__(self):
        """Node initialization."""
        super().__init__('camera_node')
        self.get_logger().info('Camera node has been started.')
        self.publisher = self.create_publisher(
            Image, '/camera/image_raw', qos_profile_sensor_data)
        self.bridge = CvBridge()

        self.source = camera_source()
        self.camera = cv2.VideoCapture(self.source)
        try:
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.lock = threading.Lock()
        self.latest = None
        self.running = True
        self.reader = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader.start()

        periodo = 0.1  # Periodo de publicación en segundos
        self.timer = self.create_timer(periodo, self.publish_image)

    def _reader_loop(self):
        """Read the source as fast as it delivers, keeping only the newest frame."""
        fallos = 0
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                fallos = 0
                with self.lock:
                    self.latest = frame
            else:
                fallos += 1
                time.sleep(0.05)
                if fallos >= 60:  # ~3 s sin frames: reabrir la fuente
                    self.camera.release()
                    self.camera = cv2.VideoCapture(self.source)
                    fallos = 0

    def publish_image(self):
        """Publish the latest captured frame to /camera/image_raw."""
        with self.lock:
            frame = self.latest
            self.latest = None
        if frame is None:
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher.publish(msg)

    @override
    def destroy_node(self):
        """Stop the reader thread and release the camera resource."""
        self.running = False
        self.reader.join(timeout=1.0)
        self.camera.release()
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
