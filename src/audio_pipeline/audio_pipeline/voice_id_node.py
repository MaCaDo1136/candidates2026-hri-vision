#! /usr/bin/env python3

"""
Description:
    Identifies the speaker of each audio segment by comparing its
    voice embedding (Resemblyzer) against a pre-enrolled reference
    embedding, using cosine similarity, and publishes the identity
    guess independently from transcription.

------------------------------
Publishing topics:
    /audio/identity - VoiceIdentity

------------------------------
Subscription Topics:
    /audio/segments - AudioSegment

------------------------------
Author: Mario Casas Donjuan
Date: September 25, 2026
"""

import os

import rclpy
import numpy as np

from rclpy.node import Node
from resemblyzer import VoiceEncoder, preprocess_wav
from hri_vision_interfaces.msg import AudioSegment, VoiceIdentity
from ament_index_python.packages import get_package_share_directory

embedding_path = os.path.join(get_package_share_directory(
    'audio_pipeline'), 'mario_voice_embedding.npy')


class VoiceIdNode(Node):
    """Identifies the speaker of each audio segment against a known
    voice embedding using Resemblyzer and cosine similarity.
    """

    def __init__(self):
        """Initializes the node, the subscriber, and the publisher."""
        super().__init__('voice_id_node')
        self.get_logger().info('Voice ID node has been started.')

        self.encoder = VoiceEncoder()
        self.reference_embed = np.load(embedding_path)

        self.publisher = self.create_publisher(
            VoiceIdentity, '/audio/identity', 10)
        self.subscription = self.create_subscription(
            AudioSegment, '/audio/segments', self.audio_segment_callback, 10)

    def audio_segment_callback(self, msg):
        """Callback function for the audio segment subscription.

        Args:
            msg (AudioSegment): The received audio segment message.
        """
        msg_samples = np.array(msg.samples, dtype=np.float32)
        wav = preprocess_wav(msg_samples)
        embed = self.encoder.embed_utterance(wav)

        similarity = np.dot(embed, self.reference_embed) / \
            (np.linalg.norm(embed) * np.linalg.norm(self.reference_embed))

        identity_msg = VoiceIdentity()
        identity_msg.header = msg.header
        identity_msg.speaker = 'Mario' if similarity > 0.70 else 'Unknown'
        identity_msg.score = float(similarity)
        self.publisher.publish(identity_msg)
        self.get_logger().info(
            f'Speaker: {identity_msg.speaker} (score: {similarity:.4f})')


def main(args=None):
    """Point of entry for the VoiceIdNode."""
    rclpy.init(args=args)
    node = VoiceIdNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
