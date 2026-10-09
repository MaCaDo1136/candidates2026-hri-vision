#! /usr/bin/env python3

"""
Description:
    Identifies who is speaking in each audio segment by comparing its voice
    embedding (Resemblyzer) against every enrolled person's voice
    (~/hri_people/<name>/voice.npy) with cosine similarity, and publishes
    the best match independently from transcription.

------------------------------
Publishing topics:
    /audio/identity - VoiceIdentity

------------------------------
Subscription Topics:
    /audio/segments - AudioSegment

------------------------------
Parameters:
    threshold (float) - minimum similarity to accept a match

------------------------------
Author: Mario Casas Donjuan
Date: October 9, 2026
"""

import numpy as np
import rclpy

from rclpy.node import Node
from resemblyzer import VoiceEncoder, preprocess_wav
from hri_vision_interfaces.msg import AudioSegment, VoiceIdentity
from hri_common.people import VOICE_FILE, best_match, load_embeddings


class VoiceIdNode(Node):
    """Identifies the speaker of each audio segment among enrolled people."""

    def __init__(self):
        """Initializes the node, the subscriber, and the publisher."""
        super().__init__('voice_id_node')

        self.declare_parameter('threshold', 0.60)

        self.encoder = VoiceEncoder()
        self.references = load_embeddings(VOICE_FILE)
        self.get_logger().info(
            f'Voice ID node started. Enrolled voices: {list(self.references)}')

        self.publisher = self.create_publisher(
            VoiceIdentity, '/audio/identity', 10)
        self.subscription = self.create_subscription(
            AudioSegment, '/audio/segments', self.audio_segment_callback, 10)

    def audio_segment_callback(self, msg):
        """Identifies the speaker of the received audio segment."""
        wav = preprocess_wav(np.array(msg.samples, dtype=np.float32))
        embed = self.encoder.embed_utterance(wav)

        speaker, score = best_match(
            embed, self.references, self.get_parameter('threshold').value)

        identity_msg = VoiceIdentity()
        identity_msg.header = msg.header
        identity_msg.speaker = speaker
        identity_msg.score = score
        self.publisher.publish(identity_msg)
        self.get_logger().info(f'Speaker: {speaker} (score: {score:.4f})')


def main(args=None):
    """Point of entry for the VoiceIdNode."""
    rclpy.init(args=args)
    node = VoiceIdNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
