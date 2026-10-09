#! /usr/bin/env python3

"""
Description:
    Enrolls a person's voice. Records a few seconds from the microphone and
    saves the Resemblyzer embedding in ~/hri_people/<name>/voice.npy.

Usage:
    python3 -m audio_pipeline.voice_embedding_generator --name ana

Author: Mario Casas Donjuan
Date: October 8, 2026
"""

import argparse
import sys
import time
import wave

import numpy as np
import sounddevice as sd

from resemblyzer import VoiceEncoder, preprocess_wav
from hri_common.people import (
    VOICE_FILE, ensure_person, load_people, person_path, remove_person)

SAMPLE_RATE = 16000  # same as vad_node
SECONDS = 8
MIN_SPEECH_SECONDS = 2.0


def record(seconds=SECONDS):
    """Records mono audio from the default microphone."""
    print('Preparando micrófono...')
    time.sleep(1)
    print(f'¡Habla ahora! Grabando {seconds} segundos...')
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1)
    sd.wait()
    print('Grabación terminada.')
    return audio.flatten()


def save_wav(path, audio):
    """Saves the recording so you can listen to what was captured."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def main():
    """Entry point of the voice enrollment script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True, help='person id, e.g. ana')
    parser.add_argument('--display', help='display name, e.g. Ana')
    args = parser.parse_args()

    existed = args.name.lower() in load_people()
    person_id = ensure_person(args.name, args.display)
    done = False
    try:
        print('Di algo natural, por ejemplo: "Hola, soy '
              f'{args.display or person_id}. Esta es una grabación de mi voz '
              'para el sistema de reconocimiento."')
        audio = record()
        save_wav(person_path(person_id, 'enroll_voice.wav'), audio)

        wav = preprocess_wav(audio)  # same preprocessing as voice_id_node
        if len(wav) / SAMPLE_RATE < MIN_SPEECH_SECONDS:
            sys.exit('Casi no detecté voz (nivel muy bajo o mucho silencio). '
                     'Acércate al micrófono y vuelve a intentar.')

        embedding = VoiceEncoder().embed_utterance(wav)
        path = person_path(person_id, VOICE_FILE)
        np.save(path, embedding)
        done = True
        print(f'Voz de {person_id} guardada en {path}')
    finally:
        if not done and not existed:
            remove_person(person_id)  # do not leave empty registrations


if __name__ == '__main__':
    main()
