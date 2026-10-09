"""Mouth-movement features from 68-point facial landmarks (iBUG layout)."""

from collections import deque

import numpy as np


def mouth_aspect_ratio(points):
    """Returns the inner-mouth aspect ratio from 68 (x, y) landmarks.

    iBUG layout: inner lip is 60..67, corners are 60 and 64. Dividing by the
    mouth width makes it independent of how far the person is.
    """
    p = np.asarray(points, dtype=np.float32)
    width = float(np.linalg.norm(p[60] - p[64]))
    if width < 1e-3:
        return None
    vertical = (np.linalg.norm(p[61] - p[67]) + np.linalg.norm(p[62] - p[66])
                + np.linalg.norm(p[63] - p[65]))
    return float(vertical / (2.0 * width))


def bbox_xyxy(detection):
    """Returns (x1, y1, x2, y2) of a vision_msgs Detection2D bounding box."""
    center = detection.bbox.center
    if hasattr(center, 'position'):  # vision_msgs 4.x (Humble/Jazzy)
        cx, cy = center.position.x, center.position.y
    else:                            # older vision_msgs
        cx, cy = center.x, center.y
    half_w, half_h = detection.bbox.size_x / 2.0, detection.bbox.size_y / 2.0
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


# MediaPipe Face Mesh (468 points): inner-lip landmarks.
MESH_CORNERS = (78, 308)
MESH_VERTICAL_PAIRS = ((81, 178), (13, 14), (311, 402))


def mesh_mouth_ratio(points):
    """Returns the mouth aspect ratio from MediaPipe Face Mesh (x, y) points
    given in pixels: mean inner-lip opening divided by the mouth width."""
    p = np.asarray(points, dtype=np.float32)
    width = float(np.linalg.norm(p[MESH_CORNERS[0]] - p[MESH_CORNERS[1]]))
    if width < 1e-3:
        return None
    opening = np.mean([np.linalg.norm(p[a] - p[b])
                       for a, b in MESH_VERTICAL_PAIRS])
    return float(opening / width)


def crop_with_padding(frame, bbox, pad=0.3):
    """Crops a frame around a bbox with relative padding, clamped to bounds."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x, pad_y = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    x2, y2 = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
    return frame[y1:y2, x1:x2]


def mediapipe_mouth_ratio(mesh, crop_bgr):
    """Runs MediaPipe Face Mesh on a face crop; returns the mouth ratio or None."""
    if crop_bgr.size == 0:
        return None
    rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
    result = mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    height, width = crop_bgr.shape[:2]
    landmarks = result.multi_face_landmarks[0].landmark
    points = np.array([[l.x * width, l.y * height] for l in landmarks],
                      dtype=np.float32)
    return mesh_mouth_ratio(points)


class LipActivityTracker:
    """Per-track sliding window of mouth openness; decides speaking/silent.

    A face is 'speaking' when the standard deviation of its mouth aspect
    ratio over `window_sec` reaches `threshold`. The state is held for
    `hold_sec` after the last active moment to bridge pauses between words
    and the low frame rate of the pipeline.
    """

    def __init__(self, window_sec=1.5, threshold=0.03, hold_sec=0.8,
                 min_samples=3, stale_sec=3.0):
        self.window_sec = window_sec
        self.threshold = threshold
        self.hold_sec = hold_sec
        self.min_samples = min_samples
        self.stale_sec = stale_sec
        self.samples = {}
        self.last_active = {}
        self.last_seen = {}

    def update(self, track_id, mar, now):
        """Adds a sample for a track; returns (speaking, activity)."""
        window = self.samples.setdefault(track_id, deque())
        window.append((now, mar))
        while window and now - window[0][0] > self.window_sec:
            window.popleft()
        self.last_seen[track_id] = now
        activity = 0.0
        if len(window) >= self.min_samples:
            activity = float(np.std([m for _, m in window]))
            if activity >= self.threshold:
                self.last_active[track_id] = now
        speaking = now - self.last_active.get(track_id, -1e9) <= self.hold_sec
        return speaking, activity

    def prune(self, now):
        """Forgets tracks that have not been seen for `stale_sec`."""
        for track_id in [t for t, seen in self.last_seen.items()
                         if now - seen > self.stale_sec]:
            for table in (self.samples, self.last_active, self.last_seen):
                table.pop(track_id, None)
