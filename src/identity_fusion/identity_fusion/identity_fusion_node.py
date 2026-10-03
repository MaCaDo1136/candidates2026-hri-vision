#! /usr/bin/env python3

"""
Description:
    Fuses face identity and voice identity within a time window, applies
    a three-state policy (respond / ask / deny), and queries rag_service
    only when identity is confirmed with enough confidence.

------------------------------
Publishing topics:
    /system/response - std_msgs/String

------------------------------
Subscription Topics:
    /faces/identified - vision_msgs/Detection2DArray
    /audio/identity - hri_vision_interfaces/VoiceIdentity
    /audio/transcripts - hri_vision_interfaces/Transcript

------------------------------
Author: Mario Casas Donjuan
Date: October 1, 2026
"""

import rclpy

from rclpy.node import Node
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
from hri_vision_interfaces.msg import VoiceIdentity, Transcript
from hri_vision_interfaces.srv import RagQuery


class IdentityFusionNode(Node):
    """Combines face + voice identity and gates access to rag_service."""

    def __init__(self):
        """Initializes state, parameters, subscriptions and the RAG client."""
        super().__init__('identity_fusion_node')
        self.get_logger().info('Identity fusion node has been started.')

        self.declare_parameter('fusion_window_sec', 2.5)
        self.declare_parameter('threshold_respond', 0.55)
        self.declare_parameter('threshold_ask', 0.35)

        self.last_faces = []  # {'person': str, 'score': float, 'time': Time}
        self.last_voice = None

        self.response_pub = self.create_publisher(
            String, '/system/response', 10)

        self.create_subscription(
            Detection2DArray, '/faces/identified', self.faces_callback, 10)
        self.create_subscription(
            VoiceIdentity, '/audio/identity', self.voice_callback, 10)
        self.create_subscription(
            Transcript, '/audio/transcripts', self.transcript_callback, 10)

        self.rag_client = self.create_client(RagQuery, '/rag/query')

    def faces_callback(self, msg):
        """Stores identities of ALL currently detected faces (not just one),
        so evaluate_identity can match by name instead of by array index."""
        self.last_faces = []
        now = self.get_clock().now()
        for detection in msg.detections:
            if len(detection.results) < 2:
                continue
            identity = detection.results[-1].hypothesis
            self.last_faces.append({
                'person': identity.class_id,
                'score': identity.score,
                'time': now,
            })

    def voice_callback(self, msg):
        """Stores the most recent voice identity."""
        self.last_voice = {
            'speaker': msg.speaker,
            'score': msg.score,
            'time': self.get_clock().now(),
        }

    def evaluate_identity(self):
        """Returns (person_id, confidence, state). Cross-matches the voice
        identity against ALL faces currently in frame, picking the face
        that claims to be the same person as the voice — not just the
        first detection."""
        window = self.get_parameter('fusion_window_sec').value
        now = self.get_clock().now()

        voice_valid = (self.last_voice is not None and
                       (now - self.last_voice['time']).nanoseconds / 1e9 <= window)
        if not voice_valid or self.last_voice['speaker'] == 'Unknown':
            return None, 0.0, 'negar'

        voice_person = self.last_voice['speaker']

        matching_faces = [
            f for f in getattr(self, 'last_faces', [])
            if f['person'] == voice_person
            and (now - f['time']).nanoseconds / 1e9 <= window
        ]

        if not matching_faces:
            return None, 0.0, 'negar'  # nadie en cuadro coincide con la voz

        best_face = max(matching_faces, key=lambda f: f['score'])
        confidence = min(best_face['score'], self.last_voice['score'])

        respond_th = self.get_parameter('threshold_respond').value
        ask_th = self.get_parameter('threshold_ask').value

        if confidence >= respond_th:
            return voice_person, confidence, 'responder'
        elif confidence >= ask_th:
            return voice_person, confidence, 'preguntar'
        else:
            return None, confidence, 'negar'

    def transcript_callback(self, msg):
        """Triggered when someone finishes speaking: decides whether to
        answer, ask for confirmation, or deny, based on fused identity."""
        person_id, confidence, state = self.evaluate_identity()
        self.get_logger().info(
            f'Fusion state: {state} (person={person_id}, confidence={confidence:.2f})')

        if state == 'negar':
            self.publish_response(
                'No puedo compartir esa información: no logré confirmar tu identidad.')
            return

        if state == 'preguntar':
            self.publish_response(
                'No estoy del todo seguro de quién eres. ¿Puedes acercarte y hablar de nuevo?')
            return

        # state == 'responder'
        if not self.rag_client.service_is_ready():
            self.publish_response(
                'El servicio de consulta no está disponible.')
            return

        request = RagQuery.Request()
        request.person_id = person_id
        request.question = msg.text
        future = self.rag_client.call_async(request)
        future.add_done_callback(self.handle_rag_response)

    def handle_rag_response(self, future):
        """Callback for the async RAG service call."""
        try:
            result = future.result()
            self.publish_response(result.answer)
        except Exception as e:
            self.get_logger().error(f'RAG call failed: {e}')
            self.publish_response('Hubo un error al consultar la información.')

    def publish_response(self, text):
        """Publishes the final response and logs it."""
        msg = String()
        msg.data = text
        self.response_pub.publish(msg)
        self.get_logger().info(f'Response: {text}')


def main(args=None):
    """Entry point of the node."""
    rclpy.init(args=args)
    node = IdentityFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
