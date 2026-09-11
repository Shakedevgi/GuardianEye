"""test_ui.py - headless self-test for the ui/ package's PURE logic.

Plain-assert script, NOT pytest (pytest is not installed in this project -
matches cv/test_risk_engine.py's and backend/test_server.py's convention).
Run directly:

    cd ui && ../.venv/bin/python3 test_ui.py

Covers exactly what CLAUDE.md's Phase 8 task asked for and no more: the
MJPEG multipart parser (ui/video.py), event-log grouping (ui/events.py),
and the API client's handling of 503/ok:false responses (ui/api_client.py)
- NOT Flet widget rendering, which needs a real page/client and is a
separate, manual verification step (see the app itself + the screenshot
check run against docs/design/phase-8/fake_camera_server.py).
"""

import io
import json
import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from events import GROUP_GAP_SECONDS, format_last_seen, group_events_by_hazard
from video import read_mjpeg


# ---------------------------------------------------------------------------
# video.py - MJPEG multipart parser
# ---------------------------------------------------------------------------

def test_read_mjpeg_parses_two_frames():
    frame1 = b"\xff\xd8AAA\xff\xd9"
    frame2 = b"\xff\xd8BBBBB\xff\xd9"
    stream = (
        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame1)).encode()
        + b"\r\n\r\n" + frame1 + b"\r\n"
        + b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame2)).encode()
        + b"\r\n\r\n" + frame2 + b"\r\n"
    )

    received = []
    _run_parser_against_bytes(stream, received)

    assert received == [frame1, frame2], received


def test_read_mjpeg_handles_chunked_delivery():
    """The parser must not assume one socket .read() returns one whole
    part - real HTTP delivers arbitrary chunk boundaries. Feed it 3 bytes
    at a time and confirm it still reassembles correctly."""
    frame = b"\xff\xd8" + (b"X" * 50) + b"\xff\xd9"
    stream = (
        b"--frame\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
    )

    received = []
    _run_parser_against_bytes(stream, received, chunk_size=3)
    assert received == [frame], received


def test_read_mjpeg_stops_cleanly_on_truncated_stream():
    """A stream that ends mid-header or mid-body (server died) must not
    raise - the caller (MjpegClient) is the one that decides to
    reconnect."""
    stream = b"--frame\r\nContent-Length: 100\r\n\r\n" + b"short"
    received = []
    _run_parser_against_bytes(stream, received)  # must not raise
    assert received == []


def test_read_mjpeg_bails_on_missing_content_length():
    stream = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + b"whatever\r\n"
    received = []
    _run_parser_against_bytes(stream, received)  # must not raise
    assert received == []


class _FakeResponse:
    def __init__(self, data: bytes, chunk_size: int):
        self._buf = data
        self._pos = 0
        self._chunk_size = chunk_size

    def read(self, n: int) -> bytes:
        size = min(n, self._chunk_size)
        chunk = self._buf[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk

    def close(self):
        pass


def _run_parser_against_bytes(data: bytes, out: list, chunk_size: int = 4096) -> None:
    import video as video_module

    fake_resp = _FakeResponse(data, chunk_size)
    real_urlopen = video_module.urllib.request.urlopen
    video_module.urllib.request.urlopen = lambda *a, **kw: fake_resp
    try:
        read_mjpeg("http://fake/video_feed", out.append)
    finally:
        video_module.urllib.request.urlopen = real_urlopen


# ---------------------------------------------------------------------------
# events.py - parent-view event grouping
# ---------------------------------------------------------------------------

def _row(id_, hazard_id, started_at, last_seen_at, peak_zone="orange", person_id=1,
         clip_triggered=False, hazard_label="object"):
    return {
        "id": id_, "hazard_id": hazard_id, "hazard_label": hazard_label,
        "person_id": person_id, "started_at": started_at, "last_seen_at": last_seen_at,
        "peak_zone": peak_zone, "clip_triggered": clip_triggered, "zone": peak_zone,
        "kind": "opened",
    }


def test_group_events_merges_person_id_churn_same_hazard_close_in_time():
    """The exact bug this exists for: one continuous approach split across
    person_ids by PersonTracker churn (backend/API.md's own documented
    caveat) must become ONE parent-facing entry."""
    events = [
        _row(3, hazard_id=7, started_at="2026-08-30T14:33:10", last_seen_at="2026-08-30T14:33:15",
             peak_zone="red", person_id=9),
        _row(2, hazard_id=7, started_at="2026-08-30T14:33:05", last_seen_at="2026-08-30T14:33:09",
             peak_zone="orange", person_id=5),
        _row(1, hazard_id=7, started_at="2026-08-30T14:33:00", last_seen_at="2026-08-30T14:33:04",
             peak_zone="yellow", person_id=3),
    ]
    groups = group_events_by_hazard(events)
    assert len(groups) == 1, groups
    g = groups[0]
    assert g["row_count"] == 3
    assert g["peak_zone"] == "red"  # worst zone across the group wins
    assert g["event_ids"] == [3, 2, 1]
    assert g["last_seen_at"] == "2026-08-30T14:33:15"


def test_group_events_does_not_merge_across_a_large_gap():
    from datetime import datetime, timedelta

    newer = datetime(2026, 8, 30, 15, 0, 0)
    older = newer - timedelta(seconds=GROUP_GAP_SECONDS + 30)
    events = [
        _row(2, hazard_id=7, started_at=newer.isoformat(), last_seen_at=newer.isoformat()),
        _row(1, hazard_id=7, started_at=older.isoformat(), last_seen_at=older.isoformat()),
    ]
    groups = group_events_by_hazard(events)
    assert len(groups) == 2, groups


def test_group_events_does_not_merge_different_hazards():
    events = [
        _row(2, hazard_id=8, started_at="2026-08-30T14:33:05", last_seen_at="2026-08-30T14:33:06"),
        _row(1, hazard_id=7, started_at="2026-08-30T14:33:00", last_seen_at="2026-08-30T14:33:04"),
    ]
    groups = group_events_by_hazard(events)
    assert len(groups) == 2, groups
    assert groups[0]["hazard_id"] == 8
    assert groups[1]["hazard_id"] == 7


def test_group_events_empty_input():
    assert group_events_by_hazard([]) == []


def test_format_last_seen_never_synthesizes_an_end_time():
    # There is deliberately no ended_at column (backend/db.py) - this
    # helper must only ever produce an approximate "last seen HH:MM"
    # label, never anything implying a precise close time.
    label = format_last_seen("2026-08-30T14:33:00+00:00")
    assert label.startswith("last seen "), label
    assert "ended" not in label.lower()

    # Malformed/unknown timestamps must degrade, not raise.
    assert format_last_seen("") == "last seen: unknown time"
    assert format_last_seen("not-a-timestamp") == "last seen: unknown time"


# ---------------------------------------------------------------------------
# api_client.py - 503 / ok:false handling
# ---------------------------------------------------------------------------

def _install_fake_urlopen(responder):
    """responder(request) -> (status_code, body_bytes). Raises
    urllib.error.HTTPError for non-2xx, matching real urllib behaviour, so
    ApiClient's except-HTTPError branches actually get exercised."""
    import api_client as api_client_module

    def fake_urlopen(req, timeout=None):
        status, body = responder(req)
        if 200 <= status < 300:
            return _FakeHttpOk(body)
        raise urllib.error.HTTPError(req.full_url, status, "err", {}, io.BytesIO(body))

    real = api_client_module.urllib.request.urlopen
    api_client_module.urllib.request.urlopen = fake_urlopen
    return real, api_client_module


class _FakeHttpOk:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_risk_status_503_raises_not_ready_not_a_plain_error():
    from api_client import ApiClient, NotReady

    real, mod = _install_fake_urlopen(lambda req: (503, json.dumps({"detail": "no frame published yet"}).encode()))
    try:
        client = ApiClient("http://fake")
        raised = False
        try:
            client.risk_status()
        except NotReady as exc:
            raised = True
            assert "no frame" in str(exc)
        assert raised, "expected NotReady, got no exception"
    finally:
        mod.urllib.request.urlopen = real


def test_review_503_raises_not_ready():
    from api_client import ApiClient, NotReady

    real, mod = _install_fake_urlopen(lambda req: (503, json.dumps({"detail": "no frame published yet"}).encode()))
    try:
        client = ApiClient("http://fake")
        try:
            client.review()
            assert False, "expected NotReady"
        except NotReady:
            pass
    finally:
        mod.urllib.request.urlopen = real


def test_review_command_503_raises_review_command_failed_not_silently_retried():
    from api_client import ApiClient, ReviewCommandFailed

    real, mod = _install_fake_urlopen(
        lambda req: (503, json.dumps({"detail": "frame loop did not respond in time"}).encode())
    )
    try:
        client = ApiClient("http://fake")
        try:
            client.review_confirm(7)
            assert False, "expected ReviewCommandFailed"
        except ReviewCommandFailed as exc:
            assert "did not respond" in str(exc)
    finally:
        mod.urllib.request.urlopen = real


def test_review_confirm_ok_false_raises_review_not_current():
    """The FIFO fail-closed contract (backend/API.md): confirm/dismiss on
    anything but current_id returns 200 {"ok": false, "reason": "..."}.
    This must NOT look like a success to the caller."""
    from api_client import ApiClient, ReviewNotCurrent

    body = json.dumps({"ok": False, "reason": "not the current review candidate"}).encode()
    real, mod = _install_fake_urlopen(lambda req: (200, body))
    try:
        client = ApiClient("http://fake")
        try:
            client.review_confirm(99999)
            assert False, "expected ReviewNotCurrent"
        except ReviewNotCurrent as exc:
            assert exc.reason == "not the current review candidate"
    finally:
        mod.urllib.request.urlopen = real


def test_review_confirm_ok_true_returns_normally():
    from api_client import ApiClient

    body = json.dumps({"ok": True, "id": 7, "new_state": "confirmed"}).encode()
    real, mod = _install_fake_urlopen(lambda req: (200, body))
    try:
        client = ApiClient("http://fake")
        result = client.review_confirm(7)
        assert result["ok"] is True
        assert result["new_state"] == "confirmed"
    finally:
        mod.urllib.request.urlopen = real


def test_clips_bad_status_400_raises_plain_api_error_not_not_ready():
    from api_client import ApiClient, ApiError, NotReady

    real, mod = _install_fake_urlopen(lambda req: (400, json.dumps({"detail": "bad status"}).encode()))
    try:
        client = ApiClient("http://fake")
        try:
            client.clips("bogus")
            assert False, "expected ApiError"
        except NotReady:
            assert False, "400 must not be treated as NotReady"
        except ApiError as exc:
            assert exc.status == 400
    finally:
        mod.urllib.request.urlopen = real


def run_all():
    tests = [obj for name, obj in list(globals().items())
             if name.startswith("test_") and callable(obj)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"ALL TESTS PASSED ({len(tests)} tests)")



# --- review crop: bbox is in CAMERA coordinates, the frame may be downscaled --

def _jpeg(width, height, color=(200, 30, 30)):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="JPEG")
    return buf.getvalue()


def test_crop_scales_bbox_when_the_displayed_frame_is_downscaled():
    """Shaked, live 2026-08-30: "most of the first detections are corrupted and
    we can't see them while classifying." Cause: hazard bboxes arrive in full
    camera coordinates (1920 wide) but the WEB path displays a frame
    downscaled to 960, and the crop used the raw bbox against the small
    frame. A bbox starting past 960 produced no crop at all; one straddling
    960 produced a half-cut image. Both render as a broken review card - the
    one screen where the parent must actually SEE the object to judge it."""
    from views.review import _try_crop
    from PIL import Image
    import io

    frame = _jpeg(960, 540)                 # what the web path displays
    bbox = [1200.0, 300.0, 1500.0, 500.0]   # camera-space, entirely past 960

    # Without the camera width this is the shipped bug: nothing to show.
    assert _try_crop(frame, bbox) is None

    # With it, the bbox maps into the displayed frame and a real crop appears.
    out = _try_crop(frame, bbox, camera_width=1920)
    assert out is not None, "a valid hazard produced no crop"
    img = Image.open(io.BytesIO(out))
    assert img.width > 0 and img.height > 0
    # 300px wide in camera space at 0.5 scale = 150px, plus context padding.
    assert 150 <= img.width <= 230, f"unexpected crop width {img.width}"


def test_crop_is_unchanged_on_the_desktop_path_where_scale_is_one():
    """Desktop displays the full-resolution frame, so passing the camera width
    must be a no-op there - the fix must not distort the path that worked."""
    from views.review import _try_crop
    from PIL import Image
    import io

    frame = _jpeg(1920, 1080)
    bbox = [400.0, 200.0, 700.0, 500.0]
    out = _try_crop(frame, bbox, camera_width=1920)
    assert out is not None
    img = Image.open(io.BytesIO(out))
    assert 300 <= img.width <= 380, f"unexpected crop width {img.width}"


def test_crop_clamps_a_bbox_that_runs_off_the_frame_edge():
    """A bbox partly outside the frame must still yield the visible part
    rather than failing - the parent still needs to see something."""
    from views.review import _try_crop
    frame = _jpeg(960, 540)
    out = _try_crop(frame, [1800.0, 900.0, 2400.0, 1300.0], camera_width=1920)
    assert out is not None



if __name__ == "__main__":
    run_all()
