"""video.py - MJPEG stream reader for the Flet UI.

Read CLAUDE.md decision 1 before touching this file: Flet is a CLIENT of
FastAPI's /video_feed, over HTTP only. It does NOT push frames through
Flet's own state-update mechanism frame-by-frame (that is the bottleneck
CLAUDE.md rules out) and it does NOT hand ft.Image the MJPEG URL directly.

Handing ft.Image(src="http://.../video_feed") to Flet 0.86.5 was MEASURED
this session (docs/decision-log.md, 2026-08-30 "Phase 8 kickoff
measurements") to fail on both platforms, worse than a plain error:
  - desktop: permanently blank pane, no error_content event, ~25 MB/min
    unbounded Flutter-client memory growth (it buffers the endless
    multipart response forever).
  - web: blocked before decoding even starts, because Flutter web fetches
    images by XHR, not an <img> tag, so multipart/x-mixed-replace is
    cross-origin-blocked first.

What was measured to work, on BOTH platforms (13.9 fps against a 15 fps
source, flat memory over 1245 frames): parse the multipart stream in
Python on a background thread, and assign raw JPEG bytes straight to
Image.src (typed str | bytes in this Flet version - passing bytes is what
replaced the removed src_base64, so there's no base64 step). That is what
this module does; MjpegClient is the reusable wrapper around the exact
parser that was measured.
"""

from __future__ import annotations

import threading
import time
import urllib.request
from typing import Callable, Optional


def read_mjpeg(url: str, on_frame: Callable[[bytes], None], timeout: float = 10.0,
                should_stop: Optional[Callable[[], bool]] = None) -> None:
    """Parse multipart/x-mixed-replace using each part's Content-Length
    header, calling on_frame(jpeg_bytes) once per complete part.

    Do NOT scan for JPEG SOI/EOI markers instead - that over-counts frames
    (a truncated or re-started stream can contain marker bytes inside a
    still-incomplete part). Content-Length is authoritative here because
    backend/server.py's MJPEG writer always sends one, matching the
    multipart framing FastAPI's StreamingResponse produces.

    Blocks until the connection ends or should_stop() returns True. Meant
    to be run on its own background thread (see MjpegClient below) - never
    call this on Flet's own event loop thread.
    """
    resp = urllib.request.urlopen(url, timeout=timeout)
    try:
        buf = b""
        while True:
            if should_stop is not None and should_stop():
                return
            while b"\r\n\r\n" not in buf:
                if should_stop is not None and should_stop():
                    return
                chunk = resp.read(4096)
                if not chunk:
                    return
                buf += chunk
            head, buf = buf.split(b"\r\n\r\n", 1)
            length = next(
                (int(line.split(b":", 1)[1]) for line in head.split(b"\r\n")
                 if line.lower().startswith(b"content-length:")),
                None,
            )
            if length is None:
                # Malformed part - no way to know where it ends. Bail rather
                # than guess; the caller's reconnect loop will retry.
                return
            while len(buf) < length:
                if should_stop is not None and should_stop():
                    return
                chunk = resp.read(length - len(buf))
                if not chunk:
                    return
                buf += chunk
            on_frame(buf[:length])
            buf = buf[length:]
    finally:
        try:
            resp.close()
        except Exception:
            pass


class MjpegClient:
    """Background-thread MJPEG reader with auto-reconnect.

    Owns exactly one thread. start() is idempotent-ish (calling it again
    after stop() starts a fresh thread). on_frame is invoked from the
    background thread - the caller is responsible for hopping back onto
    Flet's event loop if it needs to touch async Flet APIs (see
    ui/app.py's asyncio.run_coroutine_threadsafe usage); plain control
    mutation + control.update() is safe to call directly from a worker
    thread in Flet 0.86.5, which is what this project relies on (measured
    in docs/design/phase-8/desktop_probe2.py).
    """

    def __init__(self, url: str, on_frame: Callable[[bytes], None],
                 reconnect_delay: float = 1.0, max_width: int = 0,
                 quality: int = 70, max_fps: float = 0.0):
        """`max_width`/`max_fps` exist for the WEB path and are measured, not
        precautionary. On desktop, Flet talks to a local bundled Flutter
        client and full-resolution frames are fine (verified). On web, every
        frame is msgpack'd over the page's WebSocket, and a real 1920x1080
        camera frame is ~0.2-0.5 MB: at ~15 fps that is several MB/s per
        viewer, which measurably KILLS the socket - the browser logs a
        repeating `WebSocket connection to ws://.../ws failed`, the session
        reconnects, and the video pane stays BLACK while the rest of the UI
        (which needs almost no bandwidth) still works. That asymmetry is
        exactly what a black video area beside a working action bar looks
        like, and it is what Shaked reported from a real tablet on
        2026-08-30. The same app against a 960x540 / ~19 KB-per-frame source
        rendered fine on the same browser, which is the A/B that identified
        the cause.

        Downscaling here rather than in cv/risk_engine.py is deliberate: the
        camera loop must not spend frame time re-encoding for a UI client,
        and CLAUDE.md decision 1 keeps the served frame a cv concern. This is
        presentation, done in the UI process, on the UI's own thread.
        """
        self.url = url
        self._on_frame = on_frame
        self._reconnect_delay = reconnect_delay
        self._max_width = max_width
        self._quality = quality
        self._min_interval = (1.0 / max_fps) if max_fps > 0 else 0.0
        self._last_push = 0.0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_error: Optional[str] = None

    def _shrink(self, jpeg: bytes) -> bytes:
        """Decode -> resize -> re-encode. Returns the original bytes on any
        failure: a video pane showing a heavy frame beats one showing none."""
        try:
            import cv2
            import numpy as np
            arr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if arr is None or arr.shape[1] <= self._max_width:
                return jpeg
            scale = self._max_width / float(arr.shape[1])
            small = cv2.resize(arr, (self._max_width, int(arr.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(".jpg", small,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
            return enc.tobytes() if ok else jpeg
        except Exception:
            return jpeg

    def _on_frame_wrapped(self, jpeg: bytes) -> None:
        if self._min_interval:
            now = time.monotonic()
            if now - self._last_push < self._min_interval:
                # Drop, do not queue. A backlog of stale frames is worse than
                # a lower frame rate - it arrives late AND keeps the socket
                # saturated.
                self.dropped_frames += 1
                return
            self._last_push = now
        if self._max_width:
            jpeg = self._shrink(jpeg)
        self.frame_count += 1
        self.last_error = None
        self._on_frame(jpeg)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                read_mjpeg(self.url, self._on_frame_wrapped,
                           should_stop=self._stop.is_set)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            if self._stop.is_set():
                return
            # Stream ended (server restarted, camera hiccup, etc.) - wait
            # and reconnect rather than leaving the video pane frozen.
            time.sleep(self._reconnect_delay)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
