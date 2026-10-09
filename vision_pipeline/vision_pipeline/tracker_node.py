#! /usr/bin/env python3

"""
Description:
    This node tracks detected faces across frames by comparing each new
    detection against the previous frame's tracked faces (using IoU
    overlap), assigning the same stable ID to a face that persists between
    frames, and a new ID to a face that appears for the first time.

------------------------------
Publishing topics:
    /faces/tracked - Detection2DArray

------------------------------
Subscription Topics:
    /faces/detections - Detection2DArray

------------------------------
Author: Mario Casas Donjuan
Date: September 16, 2026
"""

import rclpy

from rclpy.node import Node
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose


class TrackerNode(Node):
    """This node tracks detected faces over time, 
    assigning unique IDs to each face and publishing the tracked results.
    """

    def __init__(self):
        """Initializes the tracker node, sets up the publisher and 
        subscriber, and initializes tracking data structures."""

        super().__init__('tracker_node')
        self.get_logger().info('Tracker node has been started.')

        self.tracked_faces = {}
        self.next_id = 0

        self.subscription = self.create_subscription(
            Detection2DArray, '/faces/detections', self.detections_callback, 10)
        self.publisher = self.create_publisher(
            Detection2DArray, '/faces/tracked', 10)

    def detections_callback(self, msg):
        """Callback function for the detection subscription."""
        tracked_msg = Detection2DArray()
        tracked_msg.header = msg.header
        user_ids = set()

        for detection in msg.detections:
            best_iou = 0.0
            best_id = None

            for user_id, tracked_detection in self.tracked_faces.items():
                iou = self.iou_calculator(
                    detection.bbox, tracked_detection.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_id = user_id

            if best_iou > 0.3:
                assigned_id = best_id
            else:
                assigned_id = self.next_id
                self.next_id += 1

            detection.id = str(assigned_id)
            self.tracked_faces[assigned_id] = detection
            user_ids.add(assigned_id)
            tracked_msg.detections.append(detection)

        self.tracked_faces = {
            id: box for id, box in self.tracked_faces.items() if id in user_ids}
        self.publisher.publish(tracked_msg)

    def iou_calculator(self, box1, box2):
        """Calculate the Intersection over Union (IoU) between two bounding boxes."""
        x1_min = box1.center.position.x - box1.size_x / 2
        x1_max = box1.center.position.x + box1.size_x / 2
        y1_min = box1.center.position.y - box1.size_y / 2
        y1_max = box1.center.position.y + box1.size_y / 2

        x2_min = box2.center.position.x - box2.size_x / 2
        x2_max = box2.center.position.x + box2.size_x / 2
        y2_min = box2.center.position.y - box2.size_y / 2
        y2_max = box2.center.position.y + box2.size_y / 2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_width = max(0.0, inter_x_max - inter_x_min)
        inter_height = max(0.0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height

        area1 = box1.size_x * box1.size_y
        area2 = box2.size_x * box2.size_y
        union_area = area1 + area2 - inter_area

        if union_area == 0.0:
            return 0.0
        return inter_area / union_area


def main(args=None):
    """Punto de entrada del nodo."""
    rclpy.init(args=args)
    node = TrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
