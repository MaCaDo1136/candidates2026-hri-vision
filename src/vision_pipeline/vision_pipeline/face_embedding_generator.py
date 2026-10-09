#! /usr/bin/env python3

"""
Description:
    Enrolls a person's face. Captures several frames from the camera (or
    reads a photo with --image), computes InsightFace embeddings and saves
    their normalized mean in ~/hri_people/<name>/face.npy.

Usage:
    python3 -m vision_pipeline.face_embedding_generator --name ana
    python3 -m vision_pipeline.face_embedding_generator --name ana --image foto.jpg

Author: Mario Casas Donjuan
Date: October 8, 2026
"""

import argparse
import sys
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from hri_common.people import (
    FACE_FILE, ensure_person, load_people, person_path, remove_person)

NUM_SAMPLES = 8
MIN_SAMPLES = 3
MAX_SECONDS = 15


def load_app():
    """Loads InsightFace (same model as face_id_node)."""
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def embed_frame(app, frame):
    """Returns the normalized embedding of the largest face, or None."""
    faces = app.get(frame)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0])
               * (f.bbox[3] - f.bbox[1]))
    embedding = np.asarray(face.embedding, dtype=np.float32)
    return embedding / np.linalg.norm(embedding)


def samples_from_camera(app, camera_index=0):
    """Collects embeddings from the camera over a few seconds."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        sys.exit('No pude abrir la cámara. Si el launch de visión está '
                 'corriendo (camera_node la usa), ciérralo, o usa '
                 '--image foto.jpg')
    for _ in range(10):  # warm-up: auto exposure
        cap.read()
    print('Mira a la cámara, de frente y con buena luz...')
    samples, start = [], time.time()
    while len(samples) < NUM_SAMPLES and time.time() - start < MAX_SECONDS:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        embedding = embed_frame(app, frame)
        if embedding is not None:
            samples.append(embedding)
            print(f'  muestra {len(samples)}/{NUM_SAMPLES}')
        time.sleep(0.4)
    cap.release()
    return samples


def samples_from_image(app, path):
    """Returns the embedding of the largest face in a photo (as a list)."""
    image = cv2.imread(path)
    if image is None:
        sys.exit(f'No pude leer la imagen: {path}')
    embedding = embed_frame(app, image)
    return [embedding] if embedding is not None else []


def main():
    """Entry point of the face enrollment script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True, help='person id, e.g. ana')
    parser.add_argument('--display', help='display name, e.g. Ana')
    parser.add_argument('--image', help='photo instead of the camera')
    args = parser.parse_args()

    existed = args.name.lower() in load_people()
    person_id = ensure_person(args.name, args.display)
    done = False
    try:
        app = load_app()
        if args.image:
            samples = samples_from_image(app, args.image)
            minimum = 1
        else:
            samples = samples_from_camera(app)
            minimum = MIN_SAMPLES
        if len(samples) < minimum:
            sys.exit(f'Solo detecté {len(samples)} muestra(s) con cara '
                     f'(mínimo {minimum}). Mejora la luz o acércate.')

        mean = np.mean(samples, axis=0)
        mean = mean / np.linalg.norm(mean)
        path = person_path(person_id, FACE_FILE)
        np.save(path, mean)
        done = True
        print(f'Cara de {person_id} guardada ({len(samples)} muestras) '
              f'en {path}')
    finally:
        if not done and not existed:
            remove_person(person_id)  # do not leave empty registrations


if __name__ == '__main__':
    main()
