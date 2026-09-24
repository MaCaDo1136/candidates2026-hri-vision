#! /usr/bin/env python3

"""
Description:
	Captures live audio from the microphone, detects speech using
	Silero VAD, buffers audio while the person is speaking, and
	publishes the complete segment once enough silence follows.

------------------------------
Publishing topics:
	/audio/segments - AudioSegment

------------------------------
Subscription Topics:
	None

------------------------------
Author: Mario Casas Donjuan
Date: September 17, 2026
"""

import torch
import numpy as np
import rclpy
import sounddevice as sd

from hri_vision_interfaces.msg import AudioSegment
from silero_vad import load_silero_vad
from rclpy.node import Node


class VADNode(Node):
    """Detects speech segments from the microphone using Silero VAD.

    Args:
        Node: Clase base de rclpy para nodos ROS2.
    """

    MIN_SPEECH_BLOCKS = 16

    def __init__(self):
        """Initializes the VADNode."""
        super().__init__('vad_node')
        self.get_logger().info('VAD node has been started.')

        self.publisher = self.create_publisher(
            AudioSegment, '/audio/segments', 10)

        self.sample_rate = 16000
        self.buffer = []

        self.audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=512,
            callback=self.audio_callback
        )

        self.silence_counter = 0
        self.vad_model = load_silero_vad()
        self.audio_stream.start()

    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio input."""
        if status:
            self.get_logger().warn(f'Audio input status: {status}')

        GAIN = 2.5
        indata = torch.from_numpy(indata).float().squeeze() * GAIN
        indata = torch.clamp(indata, -1.0, 1.0)

        speech_prob = self.vad_model(indata, self.sample_rate)

        if (speech_prob > 0.65):
            self.buffer.append(indata.numpy())
            self.silence_counter = 0
        else:
            if self.buffer:
                self.silence_counter += 1
                if self.silence_counter >= 16:
                    if len(self.buffer) >= self.MIN_SPEECH_BLOCKS:
                        audio_segment_msg = AudioSegment()
                        full_audio = np.concatenate(self.buffer)
                        self.get_logger().info(
                            f'Amplitud promedio del segmento: {np.abs(full_audio).mean():.4f}')

                        audio_segment_msg.samples = full_audio.tolist()
                        audio_segment_msg.sample_rate = self.sample_rate
                        audio_segment_msg.header.stamp = self.get_clock().now().to_msg()
                        self.publisher.publish(audio_segment_msg)
                        self.get_logger().info(
                            f'Published audio segment of length {len(self.buffer)} blocks.')
                    self.buffer.clear()
                    self.silence_counter = 0


def main(args=None):
    """Entry point of the node."""
    rclpy.init(args=args)
    node = VADNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
