#! /usr/bin/env python3

"""
Description:
    This node visualizes the identified faces by drawing a bounding box,
    tracker ID and identity label over the live camera feed, using
    OpenCV's imshow window. Drawing runs at the tracker's frame rate,
    while identity labels are updated whenever face_id_node publishes
    a new result, so the video stays smooth even though identification
    runs at a lower rate.

------------------------------
Publishing topics:
    None

------------------------------
Subscription Topics:
    /camera/image_raw - Image
    /faces/tracked - Detection2DArray
    /faces/identified - Detection2DArray

------------------------------
Author: Mario Casas Donjuan
Date: September 17, 2026
"""

import cv2
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge


class VisualizerNode(Node):
    """Draws face detections and their identity on top of the live
    camera feed and displays them in a window, for demo purposes.
    """

    def __init__(self):
        """Initializes the node and its three subscriptions."""
        super().__init__('visualizer_node')
        self.get_logger().info('Visualizer node has been started.')

        self.bridge = CvBridge()
        self.latest_frame = None
        self.last_identities = {}  # tracker id (str) -> (identity, score)

        self.camera_subscriber = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, qos_profile_sensor_data)
        self.tracked_subscriber = self.create_subscription(
            Detection2DArray, '/faces/tracked', self.tracked_callback, 10)
        self.identified_subscriber = self.create_subscription(
            Detection2DArray, '/faces/identified', self.identified_callback, 10)

    def camera_callback(self, msg):
        """Saves the latest camera frame for visualization."""
        self.latest_frame = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='bgr8')

    def identified_callback(self, msg):
        """Stores the latest known identity per tracked ID, without drawing."""
        for detection in msg.detections:
            if len(detection.results) > 1:
                identity = detection.results[-1].hypothesis.class_id
                score = detection.results[-1].hypothesis.score
                self.last_identities[detection.id] = (identity, score)

    def tracked_callback(self, msg):
        """Draws each tracked face at the tracker's frame rate, using the
        latest known identity for that ID if one is available."""
        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()

        for detection in msg.detections:
            top_left = (
                int(detection.bbox.center.position.x - detection.bbox.size_x / 2),
                int(detection.bbox.center.position.y - detection.bbox.size_y / 2))
            bottom_right = (
                int(detection.bbox.center.position.x + detection.bbox.size_x / 2),
                int(detection.bbox.center.position.y + detection.bbox.size_y / 2))

            if detection.id in self.last_identities:
                identity, score = self.last_identities[detection.id]
                label = f'ID {detection.id}: {identity} ({score:.2f})'
            else:
                label = f'ID {detection.id}: detecting...'

            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, label,
                        (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Face Tracking', frame)
        cv2.waitKey(1)


def main(args=None):
    """Node entry point."""
    rclpy.init(args=args)
    node = VisualizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
