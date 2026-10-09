#! /usr/bin/env python3

"""
Description:
    Detects which tracked faces are speaking by measuring how much each
    face's mouth opening changes over a short time window. The mouth opening
    comes from MediaPipe Face Mesh (inner-lip landmarks); if MediaPipe is not
    available it falls back to the 68-point landmarks of InsightFace.

------------------------------
Publishing topics:
    /faces/speaking - Detection2DArray
        (results[0].hypothesis.class_id is 'speaking' or 'silent',
         score = lip activity; detection.id = tracker id)

------------------------------
Subscription Topics:
    /camera/image_raw - Image
    /faces/tracked - Detection2DArray

------------------------------
Parameters:
    rate_hz (float)            - max landmark runs per second
    window_sec (float)         - window used to measure lip movement
    speaking_threshold (float) - minimum lip activity to count as speaking
    hold_sec (float)           - keep 'speaking' this long after the last movement
    debug (bool)               - log mouth ratio and activity once per second
    backend (str)              - 'mediapipe' (default) or 'insightface'
    landmark_model (str)       - InsightFace 1k3d68.onnx (fallback backend)

------------------------------
Author: Mario Casas Donjuan
Date: October 8, 2026
"""

import copy
import os

import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose

from vision_pipeline.lip_activity import (
    LipActivityTracker, bbox_xyxy, crop_with_padding, mediapipe_mouth_ratio,
    mouth_aspect_ratio)


class MouthMovementNode(Node):
    """Publishes which tracked faces are speaking."""

    def __init__(self):
        """Loads the landmark model and creates subscriptions/publisher."""
        super().__init__('mouth_movement_node')

        self.declare_parameter('rate_hz', 8.0)
        self.declare_parameter('window_sec', 2.0)
        self.declare_parameter('speaking_threshold', 0.02)
        self.declare_parameter('hold_sec', 0.8)
        self.declare_parameter('debug', False)
        self.declare_parameter('backend', 'mediapipe')
        self.declare_parameter(
            'landmark_model', '~/.insightface/models/buffalo_l/1k3d68.onnx')

        self.backend = self.get_parameter('backend').value
        if self.backend == 'mediapipe':
            try:
                import mediapipe as mp
                self.mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1,
                    refine_landmarks=False, min_detection_confidence=0.3)
            except Exception as e:
                self.get_logger().warn(
                    f'MediaPipe no disponible ({e}); uso InsightFace.')
                self.backend = 'insightface'
        if self.backend == 'insightface':
            self._load_insightface()
        self.get_logger().info(f'Backend de labios: {self.backend}')

        self.tracker = self._build_tracker()
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_seq = 0
        self.last_seq = -1
        self.last_run = 0.0

        self.publisher = self.create_publisher(
            Detection2DArray, '/faces/speaking', 10)
        self.create_subscription(
            Image, '/camera/image_raw', self.image_callback,
            qos_profile_sensor_data)
        self.create_subscription(
            Detection2DArray, '/faces/tracked', self.tracked_callback, 10)
        self.get_logger().info('Mouth movement node has been started.')

    def _load_insightface(self):
        """Fallback backend: 68-point landmarks that ship with buffalo_l."""
        from insightface.model_zoo import get_model
        model_path = os.path.expanduser(
            self.get_parameter('landmark_model').value)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f'No encuentro {model_path}. Corre una vez face_id_node para '
                'que InsightFace descargue buffalo_l.')
        self.landmarks = get_model(
            model_path, providers=['CPUExecutionProvider'])
        if getattr(self.landmarks, 'taskname', '') != 'landmark_3d_68':
            raise RuntimeError(f'{model_path} no es el modelo de 68 puntos.')
        self.landmarks.prepare(ctx_id=-1)

    def _mouth_ratio(self, frame, bbox):
        """Returns the mouth aspect ratio of the face in bbox, or None."""
        if self.backend == 'mediapipe':
            return mediapipe_mouth_ratio(
                self.mesh, crop_with_padding(frame, bbox))
        from insightface.app.common import Face
        face = Face(bbox=np.array(bbox, dtype=np.float32))
        self.landmarks.get(frame, face)
        return mouth_aspect_ratio(face['landmark_3d_68'][:, :2])

    def _build_tracker(self):
        return LipActivityTracker(
            window_sec=self.get_parameter('window_sec').value,
            threshold=self.get_parameter('speaking_threshold').value,
            hold_sec=self.get_parameter('hold_sec').value)

    def image_callback(self, msg):
        """Keeps the most recent frame."""
        self.latest_image = msg
        self.image_seq += 1

    def tracked_callback(self, msg):
        """Measures mouth movement for every tracked face."""
        now = self.get_clock().now().nanoseconds / 1e9
        period = 1.0 / self.get_parameter('rate_hz').value
        # one run per new frame, no faster than rate_hz
        if (self.latest_image is None or self.image_seq == self.last_seq
                or now - self.last_run < period):
            return
        self.last_seq, self.last_run = self.image_seq, now

        # parameters can change at runtime (ros2 param set ...)
        self.tracker.threshold = self.get_parameter('speaking_threshold').value
        self.tracker.hold_sec = self.get_parameter('hold_sec').value
        self.tracker.window_sec = self.get_parameter('window_sec').value
        debug = self.get_parameter('debug').value

        frame = self.bridge.imgmsg_to_cv2(self.latest_image, 'bgr8')
        out = Detection2DArray()
        out.header = msg.header
        for index, detection in enumerate(msg.detections):
            track_id = detection.id or str(index)
            try:
                mar = self._mouth_ratio(frame, bbox_xyxy(detection))
            except Exception as e:
                self.get_logger().warn(f'Landmarks fallaron: {e}')
                continue
            if mar is None:
                continue
            speaking, activity = self.tracker.update(track_id, mar, now)
            if debug:
                self.get_logger().info(
                    f'track {track_id}: mar={mar:.3f} '
                    f'activity={activity:.3f} speaking={speaking}',
                    throttle_duration_sec=1.0)

            result = copy.deepcopy(detection)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = 'speaking' if speaking else 'silent'
            hypothesis.hypothesis.score = float(activity)
            result.results = [hypothesis]
            result.id = track_id
            out.detections.append(result)
        self.tracker.prune(now)
        if out.detections:
            self.publisher.publish(out)


def main(args=None):
    """Entry point of the node."""
    rclpy.init(args=args)
    node = MouthMovementNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
