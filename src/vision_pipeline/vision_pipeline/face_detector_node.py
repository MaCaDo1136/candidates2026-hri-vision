#! /usr/bin/env python3

"""
Description:
	<Qué hace este nodo, en 1-3 líneas>

------------------------------
Publishing topics:
	<Descripción del topic>
	<topic_name> - <msg_type>

------------------------------
Subscription Topics:
	/camera/image_raw - Image

------------------------------
Author: Mario Casas Donjuan
Date: September 9, 2026
"""

import os

import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory

model_path = os.path.join(get_package_share_directory(
    'vision_pipeline'), 'models', 'yolov8n-face-lindevs.pt')


class FaceDetector(Node):
    """<Qué representa este nodo publisher>.

    Args:
            Node: Clase base de rclpy para nodos ROS2.
    """

    def __init__(self):
        """Inicializa el nodo, el publisher y el timer."""
        super().__init__('face_detector_node')
        self.get_logger().info('Face detector node has been started.')
        self.publisher = self.create_publisher(
            Detection2DArray, '/faces/detections', 10)
        self.bridge = CvBridge()
        self.model = YOLO(
            # Load the YOLOv8 face detection model
            f'{model_path}')
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, qos_profile_sensor_data)

    def image_callback(self, msg):
        """Callback function for the image subscription.
        Args:
                msg (Image): Mensaje de imagen recibido del topic /camera/image_raw.
        """
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(cv_image)
        detection_array = Detection2DArray()
        detection_array.header = msg.header

        for result in results:
            for box in result.boxes:
                detection = Detection2D()
                detection.bbox.center.position.x = float(box.xywh[0][0])
                detection.bbox.center.position.y = float(box.xywh[0][1])
                detection.bbox.size_x = float(box.xywh[0][2])
                detection.bbox.size_y = float(box.xywh[0][3])

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = 'face'
                hypothesis.hypothesis.score = float(box.conf[0])
                detection.results.append(hypothesis)

                detection_array.detections.append(detection)

        self.publisher.publish(detection_array)


def main(args=None):
    """Punto de entrada del nodo."""
    rclpy.init(args=args)
    node = FaceDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
