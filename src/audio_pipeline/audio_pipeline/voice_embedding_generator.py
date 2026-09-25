import os
import sounddevice as sd
import time
import wave
import numpy as np

from resemblyzer import VoiceEncoder, preprocess_wav

SAMPLE_RATE = 16000
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print('Preparando micrófono...')
time.sleep(1)
print('Grabando audio durante 5 segundos.')
audio = sd.rec(int(5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
sd.wait()
print('Grabación terminada.')

with wave.open(os.path.join(SCRIPT_DIR, 'audio.wav'), 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes((audio * 32767).astype(np.int16).tobytes())

encoder = VoiceEncoder()

wav = preprocess_wav(audio.flatten())
embed = encoder.embed_utterance(wav)

embed_path = os.path.join(SCRIPT_DIR, 'mario_voice_embedding.npy')
np.save(embed_path, embed)
