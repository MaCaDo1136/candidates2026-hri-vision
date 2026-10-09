#! /usr/bin/env Python3

"""
Description:
	<description of the node>

------------------------------
Publishing topics:
	/audio/transcripts - Transcript

------------------------------
Subscription Topics:
	/audio/segments - AudioSegment

------------------------------
Author: Mario Casas Donjuan
Date: September 24, 2026
"""

import rclpy
import numpy as np

from rclpy.node import Node
from hri_vision_interfaces.msg import Transcript, AudioSegment
from faster_whisper import WhisperModel


class TranscribeNode(Node):
    """Transcribes audio segments into text."""

    def __init__(self):
        super().__init__('transcribe_node')
        self.get_logger().info('Transcribe node has been started.')

        self.model = WhisperModel("small", device="cpu", compute_type="int8")

        self.publisher = self.create_publisher(
            Transcript, '/audio/transcripts', 10)

        self.audioSegment_subscription = self.create_subscription(
            AudioSegment,
            '/audio/segments',
            self.audio_segment_callback,
            10
        )

    def audio_segment_callback(self, msg):
        """Callback function for audio segment messages.

        Args:
            msg (AudioSegment): The received audio segment message.
        """
        audio_data = np.array(msg.samples, dtype=np.float32)

        segments, info = self.model.transcribe(audio_data, language='es')

        transcript_text = ' '.join(segment.text for segment in segments)

        if not transcript_text.strip():
            self.get_logger().info('No speech detected in the audio segment.')
            return

        transcript_msg = Transcript()
        transcript_msg.header = msg.header
        transcript_msg.text = transcript_text
        transcript_msg.language = info.language
        self.publisher.publish(transcript_msg)
        self.get_logger().info(f'Transcript: {transcript_msg.text}')


def main(args=None):
    rclpy.init(args=args)

    transcribe_node = TranscribeNode()

    rclpy.spin(transcribe_node)

    transcribe_node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
