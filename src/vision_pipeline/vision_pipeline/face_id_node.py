#! /usr/bin/env python3

"""
Description:
    This node identifies tracked faces by comparing their ArcFace
    embedding (InsightFace) against a pre-enrolled reference embedding,
    using cosine similarity, and publishes the identity guess for each
    tracked detection.

------------------------------
Publishing topics:
    /faces/identified - Detection2DArray

------------------------------
Subscription Topics:
    /camera/image_raw - Image
    /faces/tracked - Detection2DArray

------------------------------
Author: Mario Casas Donjuan
Date: September 16, 2026
"""

import rclpy
import numpy as np
import os

from insightface.app import FaceAnalysis
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from ament_index_python.packages import get_package_share_directory

embedding_path = os.path.join(get_package_share_directory(
    'vision_pipeline'), 'mario_embedding.npy')  # Path to the known embedding


class FaceIdNode(Node):
    """Identifies tracked faces against a known embedding using
    InsightFace (ArcFace) and cosine similarity.
    """

    def __init__(self):
        """Initializes the node and the two subscriptions."""
        super().__init__('face_id_node')
        self.get_logger().info('Face ID node has been started.')

        self.known_embedding = np.load(
            embedding_path)  # Load the known embedding
        self.frame_counter = 0

        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

        self.bridge = CvBridge()
        self.latest_frame = None

        self.identified_publisher = self.create_publisher(
            Detection2DArray, '/faces/identified', 10)

        self.camera_subscriber = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, qos_profile_sensor_data)
        self.tracked_subscriber = self.create_subscription(
            Detection2DArray, '/faces/tracked', self.tracked_callback, 10)

    def camera_callback(self, msg):
        """Saves the latest camera frame for visualization."""
        self.latest_frame = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='bgr8')

    def tracked_callback(self, msg):
        """Identifies tracked faces by comparing their embeddings against the known embedding."""
        self.frame_counter += 1
        if self.frame_counter % 5 != 0:  # Process every 5th frame to reduce computation
            return

        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()
        identified_msg = Detection2DArray()
        identified_msg.header = msg.header

        for detection in msg.detections:
            x1 = int(detection.bbox.center.position.x -
                     detection.bbox.size_x / 2)
            y1 = int(detection.bbox.center.position.y -
                     detection.bbox.size_y / 2)
            x2 = int(detection.bbox.center.position.x +
                     detection.bbox.size_x / 2)
            y2 = int(detection.bbox.center.position.y +
                     detection.bbox.size_y / 2)

            pad_x = int(detection.bbox.size_x * 0.2)
            pad_y = int(detection.bbox.size_y * 0.2)

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(frame.shape[1], x2 + pad_x)
            y2 = min(frame.shape[0], y2 + pad_y)

            faces = self.app.get(frame[y1:y2, x1:x2])

            if faces:
                detected_embedding = faces[0].embedding
                similarity = np.dot(self.known_embedding, detected_embedding) / (
                    np.linalg.norm(self.known_embedding) * np.linalg.norm(detected_embedding))

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = 'Mario' if similarity > 0.5 else 'Unknown'
                hypothesis.hypothesis.score = float(similarity)
                detection.results.append(hypothesis)

            identified_msg.detections.append(detection)

        self.identified_publisher.publish(identified_msg)


def main(args=None):
    """Node entry point."""
    rclpy.init(args=args)
    node = FaceIdNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
