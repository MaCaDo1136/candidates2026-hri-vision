#! /usr/bin/env python3

"""
Description:
	<Qué hace este nodo, en 1-3 líneas>

------------------------------
Publishing topics:
	None

------------------------------
Subscription Topics:
	<Descripción del topic>
	<topic_name> - <msg_type>

------------------------------
Author: Mario Casas Donjuan
Date: <Month Day, Year>
"""

import rclpy
from rclpy.node import Node


class TrackerNode(Node):
    """<Qué representa este nodo subscriber>.

    Args:
            Node: Clase base de rclpy para nodos ROS2.
    """

    def __init__(self):
        """Inicializa el nodo y la suscripción."""
        super().__init__('<node_name>')


def main(args=None):
    """Punto de entrada del nodo."""
    rclpy.init(args=args)
    node = TrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
