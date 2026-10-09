#! /usr/bin/env python3

"""
Description:
    This node visualizes the identified faces by drawing a bounding box,
    tracker ID, identity label and lip-movement state (speaking / silent)
    over the live camera feed, using OpenCV's imshow window. Drawing runs
    at the tracker's frame rate, while identity labels and lip state are
    updated whenever face_id_node / mouth_movement_node publish a new
    result, so the video stays smooth even though they run at a lower rate.

------------------------------
Publishing topics:
    None

------------------------------
Subscription Topics:
    /camera/image_raw - Image
    /faces/tracked - Detection2DArray
    /faces/identified - Detection2DArray
    /faces/speaking - Detection2DArray

------------------------------
Author: Mario Casas Donjuan
Date: September 17, 2026
"""

import time

import cv2
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

SPEAKING_TTL = 1.5


class VisualizerNode(Node):
    """Draws face detections, their identity and lip state on top of the
    live camera feed and displays them in a window, for demo purposes.
    """

    def __init__(self):
        """Initializes the node and its four subscriptions."""
        super().__init__('visualizer_node')
        self.get_logger().info('Visualizer node has been started.')

        self.bridge = CvBridge()
        self.latest_frame = None
        self.last_identities = {}
        self.last_speaking = {}

        self.camera_subscriber = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, qos_profile_sensor_data)
        self.tracked_subscriber = self.create_subscription(
            Detection2DArray, '/faces/tracked', self.tracked_callback, 10)
        self.identified_subscriber = self.create_subscription(
            Detection2DArray, '/faces/identified', self.identified_callback, 10)
        self.speaking_subscriber = self.create_subscription(
            Detection2DArray, '/faces/speaking', self.speaking_callback, 10)

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

    def speaking_callback(self, msg):
        """Stores the latest lip state per tracked ID, without drawing."""
        now = time.monotonic()
        for detection in msg.detections:
            if detection.results:
                hypothesis = detection.results[-1].hypothesis
                self.last_speaking[detection.id] = (
                    hypothesis.class_id == 'speaking',
                    float(hypothesis.score), now)

    def tracked_callback(self, msg):
        """Draws each tracked face at the tracker's frame rate, using the
        latest known identity and lip state for that ID if available."""
        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()
        now = time.monotonic()

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

            lips = self.last_speaking.get(detection.id)
            if lips and now - lips[2] <= SPEAKING_TTL:
                speaking, activity = lips[0], lips[1]
            else:
                speaking, activity = None, 0.0

            color = (0, 255, 0) if speaking else (255, 170, 60)
            thickness = 3 if speaking else 2

            cv2.rectangle(frame, top_left, bottom_right, color, thickness)
            cv2.putText(frame, label,
                        (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if speaking is not None:
                state = (f'HABLANDO (labios {activity:.2f})' if speaking
                         else f'callado (labios {activity:.2f})')
                cv2.putText(frame, state,
                            (top_left[0], bottom_right[1] + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
