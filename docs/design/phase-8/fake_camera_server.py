"""Throwaway probe harness (NOT part of the repo).

Runs the REAL backend/server.py (SharedState + create_app + start_server) with
a SYNTHETIC camera loop instead of risk_engine.py's real one, so what the Flet
probe talks to is the actual production MJPEG framing / route code, not an
imitation of it. Publishes a frame counter + moving bar so "is it live or is it
one frozen frame?" is answerable from a screenshot alone.
"""
import os, sys, time, threading
sys.path.insert(0, "/Users/shakedivgi/Projects/GuardianEye/backend")

import cv2
import numpy as np
from server import SharedState, CommandQueue, start_server

PORT = int(os.environ.get("PROBE_PORT", "8111"))
FPS = 15.0
W, H = int(os.environ.get("PROBE_W", "960")), int(os.environ.get("PROBE_H", "540"))

shared = SharedState()
commands = CommandQueue()

db_path = "/private/tmp/claude-501/-Users-shakedivgi-Projects-GuardianEye/076813af-c014-4ec3-bb4c-16a5d59c512f/scratchpad/probe.db"
start_server(shared, commands, host="127.0.0.1", port=PORT, db_path=db_path,
             clips_dir="/private/tmp/claude-501/-Users-shakedivgi-Projects-GuardianEye/076813af-c014-4ec3-bb4c-16a5d59c512f/scratchpad/clips/pending")
print(f"probe server on http://127.0.0.1:{PORT}", flush=True)

# --- SIMULATED additive endpoint: single-shot /frame.jpg on port 8112 -------
# This is NOT in backend/server.py. It exists only so option C (poll a
# single JPEG) can be measured honestly BEFORE asking to add the real thing.
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class FrameHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not self.path.startswith("/frame.jpg"):
            self.send_response(404); self.end_headers(); return
        jpeg, seq, status, age = shared.snapshot()
        if jpeg is None:
            self.send_response(503); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(jpeg)
    def log_message(self, *a):
        pass

threading.Thread(target=lambda: ThreadingHTTPServer(("127.0.0.1", 8112), FrameHandler).serve_forever(),
                 daemon=True).start()
print("simulated /frame.jpg on http://127.0.0.1:8112/frame.jpg", flush=True)

ZONES = ["none", "yellow", "orange", "red"]
n = 0
t0 = time.time()
try:
    while True:
        frame = np.full((H, W, 3), 24, dtype=np.uint8)
        if os.environ.get("PROBE_NOISE"):
            # real camera frames carry sensor noise and scene detail, which
            # dominates JPEG size - a flat synthetic frame compresses ~5x
            # smaller and would understate the bandwidth the UI must carry.
            frame = np.clip(frame.astype(np.int16) +
                            np.random.randint(-28, 28, frame.shape, dtype=np.int16),
                            0, 255).astype(np.uint8)
        # moving bar => a frozen stream is visually obvious
        x = int((n * 11) % (W - 120))
        cv2.rectangle(frame, (x, 40), (x + 120, 110), (0, 200, 255), -1)
        cv2.putText(frame, f"FRAME {n}", (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 4)
        cv2.putText(frame, time.strftime("%H:%M:%S"), (30, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (180, 255, 180), 3)
        ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        zone = ZONES[(n // 45) % len(ZONES)]
        status = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "camera": {"index": 0, "name": "ProbeCam", "width": W, "height": H},
            "risk": {"zone": zone, "value": 0.24, "person_id": 3, "hazard_id": 7,
                     "hazard_label": "object", "hazard_bbox": [120.0, 300.0, 260.0, 420.0]},
            "persons": [{"id": 3, "bbox": [400.0, 200.0, 520.0, 500.0],
                         "nearest_hazard_id": 7, "distance": 0.24, "zone": zone}],
            "hazards": [{"id": 7, "label": "object", "bbox": [120.0, 300.0, 260.0, 420.0],
                         "state": "pending", "origin": "scan", "is_first_scan": False,
                         "alerts_on_approach": True}],
            "hazard_counts": {"total": 9, "confirmed": 2, "pending_first_scan": 4,
                              "pending_new": 1, "dismissed": 2},
            "review_queue": {"length": 3, "current_id": 7, "queue": [7, 8, 9]},
            "alert": {"text": f"{zone.upper()} - person #3 near object", "active": zone == "red"},
            "diagnostics": {"fps": round(n / max(time.time() - t0, 1e-6), 2), "model": "probe",
                            "imgsz": 640, "conf": 0.35, "device": "probe",
                            "scan_enabled": True, "socket_detect_enabled": True,
                            "persistence_enabled": False},
        }
        shared.publish(enc.tobytes(), status)
        n += 1
        time.sleep(1.0 / FPS)
except KeyboardInterrupt:
    shared.mark_stopped()
