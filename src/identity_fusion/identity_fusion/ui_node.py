#! /usr/bin/env python3

"""
Description:
    Web UI for the demo. Serves a page (default http://localhost:8080) with
    the camera feed (boxes, names, speaking flag), who spoke with their
    confidence, and the clean question / answer of each turn.

------------------------------
Subscription Topics:
    /camera/image_raw - Image
    /faces/tracked - Detection2DArray
    /faces/identified - Detection2DArray
    /faces/speaking - Detection2DArray
    /audio/identity - VoiceIdentity
    /fusion/state - std_msgs/String (JSON, published by identity_fusion_node)
    /system/response - std_msgs/String

------------------------------
Parameters:
    host (str) - interface to bind (default 0.0.0.0)
    port (int) - HTTP port (default 8080)
    max_fps (float) - maximum fps of the video stream
    threshold_respond / threshold_ask (float) - shown as markers on the bar

------------------------------
Author: Mario Casas Donjuan
Date: October 9, 2026
"""

import json
import threading
import time
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
from hri_vision_interfaces.msg import VoiceIdentity

from hri_common.people import load_people
from identity_fusion.ui_page import INDEX_HTML

TRACKED_TTL = 1.0
IDENTITY_TTL = 6.0
SPEAKING_TTL = 1.5
MAX_TURNS = 30


def ascii_text(text):
    """OpenCV fonts cannot draw accents: strips them for the video labels."""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()


def parse_boxes(msg, min_results=1):
    """Converts a Detection2DArray into dicts with id, center, size and the
    last result. With min_results=2 only detections that already carry an
    identity (face_id_node appends it after the tracker's result) get one."""
    boxes = []
    for d in msg.detections:
        box = {
            'id': d.id,
            'cx': d.bbox.center.position.x, 'cy': d.bbox.center.position.y,
            'w': d.bbox.size_x, 'h': d.bbox.size_y,
            'cls': None, 'score': 0.0, 'time': time.monotonic(),
        }
        if len(d.results) >= min_results:
            hyp = d.results[-1].hypothesis
            box['cls'], box['score'] = hyp.class_id, float(hyp.score)
        boxes.append(box)
    return boxes


def nearest(items, box, ttl, now):
    """Closest recent item to a box (by center), or None."""
    limit = max(box['w'], box['h']) * 0.8
    best, best_d = None, limit
    for it in items:
        if now - it['time'] > ttl:
            continue
        d = ((it['cx'] - box['cx']) ** 2 + (it['cy'] - box['cy']) ** 2) ** 0.5
        if d < best_d:
            best, best_d = it, d
    return best


def match(items, box, ttl, now):
    """Item for a box: same tracker id if available, else the closest center."""
    if box['id']:
        for it in items:
            if it['id'] == box['id'] and now - it['time'] <= ttl:
                return it
    return nearest(items, box, ttl, now)


class UiNode(Node):
    """Collects the system state and serves it over HTTP."""

    def __init__(self):
        super().__init__('ui_node')
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 8080)
        self.declare_parameter('max_fps', 15.0)
        self.declare_parameter('jpeg_quality', 75)
        self.declare_parameter('threshold_respond', 0.55)
        self.declare_parameter('threshold_ask', 0.35)
        self.declare_parameter('speaking_threshold', 0.02)

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.frame_cond = threading.Condition()
        self.running = True
        self.jpeg = None
        self.frame_id = 0
        self.last_frame = 0.0
        self.fps = 0.0

        self.tracked, self.identity, self.speaking = [], [], []
        self.voice = None
        self.turns = []
        self.turn_counter = 0
        self.people = {}
        self.people_time = 0.0

        self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, qos_profile_sensor_data)
        self.create_subscription(
            Detection2DArray, '/faces/tracked', self.tracked_callback, 10)
        self.create_subscription(
            Detection2DArray, '/faces/identified', self.identified_callback, 10)
        self.create_subscription(
            Detection2DArray, '/faces/speaking', self.speaking_callback, 10)
        self.create_subscription(
            VoiceIdentity, '/audio/identity', self.voice_callback, 10)
        self.create_subscription(
            String, '/fusion/state', self.fusion_callback, 10)
        self.create_subscription(
            String, '/system/response', self.response_callback, 10)

        self.start_server()

    # ---------- helpers ----------
    def display(self, person_id):
        """Display name of a person id."""
        now = time.monotonic()
        if now - self.people_time > 3.0:
            try:
                self.people = load_people()
            except Exception:
                pass
            self.people_time = now
        if not person_id or person_id == 'Unknown':
            return 'Desconocido'
        return self.people.get(str(person_id).lower(), str(person_id))

    def labels(self, now):
        """Tracked boxes with identity and speaking info (call with lock)."""
        out = []
        for box in self.tracked:
            if now - box['time'] > TRACKED_TTL:
                continue
            ident = match(self.identity, box, IDENTITY_TTL, now)
            speak = match(self.speaking, box, SPEAKING_TTL, now)
            person = ident['cls'] if ident else None
            known = bool(person) and person != 'Unknown'
            out.append({
                'box': box,
                'id': box['id'],
                'has_lips': speak is not None,
                'name': self.display(person) if known else 'Desconocido',
                'known': known,
                'score': ident['score'] if ident else 0.0,
                'speaking': bool(speak and speak['cls'] == 'speaking'),
                'activity': speak['score'] if speak else 0.0,
            })
        return out

    # ---------- subscriptions ----------
    def camera_callback(self, msg):
        """Draws boxes on the frame and publishes it as JPEG."""
        now = time.monotonic()
        if now - self.last_frame < 1.0 / self.get_parameter('max_fps').value:
            return
        dt = now - self.last_frame
        self.last_frame = now
        self.fps = 0.8 * self.fps + 0.2 * \
            (1.0 / dt) if self.fps else 1.0 / max(dt, 1e-3)

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8').copy()
        with self.lock:
            labels = self.labels(now)
        for item in labels:
            b = item['box']
            x1, y1 = int(b['cx'] - b['w'] / 2), int(b['cy'] - b['h'] / 2)
            x2, y2 = int(b['cx'] + b['w'] / 2), int(b['cy'] + b['h'] / 2)
            if item['speaking']:
                color = (90, 200, 60)
            elif item['known']:
                color = (237, 140, 60)
            else:
                color = (150, 150, 150)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = ascii_text(item['name'])
            if item['known']:
                text += f" {item['score']:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            top = max(y1 - th - 10, 0)
            cv2.rectangle(frame, (x1, top), (x1 + tw +
                          10, top + th + 10), color, -1)
            cv2.putText(frame, text, (x1 + 5, top + th + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if item['has_lips']:
                state = (f"HABLANDO (labios {item['activity']:.2f})" if item['speaking']
                         else f"callado (labios {item['activity']:.2f})")
                cv2.putText(frame, state, (x1, y2 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ok, buf = cv2.imencode(
            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.get_parameter('jpeg_quality').value])
        if ok:
            with self.frame_cond:
                self.jpeg = buf.tobytes()
                self.frame_id += 1
                self.frame_cond.notify_all()

    def tracked_callback(self, msg):
        """Stores the latest tracked boxes."""
        boxes = parse_boxes(msg)
        with self.lock:
            self.tracked = boxes

    def identified_callback(self, msg):
        """Stores the latest identities (keeps previous ones until TTL)."""
        boxes = parse_boxes(msg, min_results=2)
        with self.lock:
            self.identity = boxes

    def speaking_callback(self, msg):
        """Stores the latest lip-activity results."""
        boxes = parse_boxes(msg)
        with self.lock:
            self.speaking = boxes

    def voice_callback(self, msg):
        """Stores the latest voice identity."""
        with self.lock:
            self.voice = {'speaker': msg.speaker, 'score': float(msg.score),
                          'time': time.monotonic()}

    def fusion_callback(self, msg):
        """Starts a new turn from the fusion state."""
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        person = data.get('person')
        voice = data.get('voice') or {}
        turn = {
            'id': self.turn_counter,
            'time': time.strftime('%H:%M:%S'),
            'question': data.get('question', ''),
            'person': person,
            'name': self.display(person) if person else None,
            'confidence': data.get('confidence', 0.0),
            'state': data.get('state', 'negar'),
            'voice': {'speaker': voice.get('speaker'),
                      'name': self.display(voice.get('speaker')),
                      'score': voice.get('score', 0.0)} if voice else None,
            'faces': data.get('faces', []),
            'answer': None,
        }
        self.turn_counter += 1
        with self.lock:
            self.turns.append(turn)
            self.turns = self.turns[-MAX_TURNS:]

    def response_callback(self, msg):
        """Attaches the answer to the latest turn without one."""
        with self.lock:
            for turn in reversed(self.turns):
                if turn['answer'] is None:
                    turn['answer'] = msg.data
                    return
            self.turns.append({
                'id': self.turn_counter, 'time': time.strftime('%H:%M:%S'),
                'question': '', 'person': None, 'name': None, 'confidence': 0.0,
                'state': 'negar', 'voice': None, 'faces': [], 'answer': msg.data})
            self.turn_counter += 1

    # ---------- HTTP ----------
    def snapshot(self):
        """JSON-serializable state for the page."""
        now = time.monotonic()
        with self.lock:
            faces = [{'id': l['id'], 'name': l['name'], 'known': l['known'],
                      'score': l['score'], 'speaking': l['speaking'],
                      'has_lips': l['has_lips'], 'activity': l['activity']}
                     for l in self.labels(now)]
            turns = [dict(t) for t in self.turns[-15:]]
        live = now - self.last_frame < 2.0
        return {
            'fps': self.fps if live else 0.0,
            'thresholds': {'respond': self.get_parameter('threshold_respond').value,
                           'ask': self.get_parameter('threshold_ask').value},
            'speaking_threshold': self.get_parameter('speaking_threshold').value,
            'faces': faces,
            'turns': turns,
        }

    def start_server(self):
        """Starts the HTTP server in a daemon thread."""
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send_body(self, code, ctype, body):
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split('?')[0]
                if path in ('/', '/index.html'):
                    self.send_body(
                        200, 'text/html; charset=utf-8', INDEX_HTML.encode())
                elif path == '/state':
                    self.send_body(200, 'application/json',
                                   json.dumps(node.snapshot()).encode())
                elif path == '/video':
                    self.stream()
                else:
                    self.send_body(404, 'text/plain', b'not found')

            def stream(self):
                self.send_response(200)
                self.send_header('Content-Type',
                                 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                sent = -1
                try:
                    while node.running:
                        with node.frame_cond:
                            node.frame_cond.wait_for(
                                lambda: node.frame_id != sent or not node.running,
                                timeout=2.0)
                            jpeg, fid = node.jpeg, node.frame_id
                        if jpeg is None or fid == sent:
                            continue
                        sent = fid
                        self.wfile.write(
                            b'--frame\r\nContent-Type: image/jpeg\r\n'
                            b'Content-Length: %d\r\n\r\n' % len(jpeg))
                        self.wfile.write(jpeg)
                        self.wfile.write(b'\r\n')
                except (BrokenPipeError, ConnectionResetError):
                    pass

        host = self.get_parameter('host').value
        port = self.get_parameter('port').value
        try:
            self.server = ThreadingHTTPServer((host, port), Handler)
        except OSError as e:
            self.get_logger().error(f'Cannot start UI on {host}:{port}: {e}')
            self.server = None
            return
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.get_logger().info(f'UI ready: http://{host}:{port}')

    def stop(self):
        """Stops the HTTP server."""
        self.running = False
        with self.frame_cond:
            self.frame_cond.notify_all()
        if self.server:
            self.server.shutdown()


def main(args=None):
    """Node entry point."""
    rclpy.init(args=args)
    node = UiNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.stop()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
