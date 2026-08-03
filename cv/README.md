# cv/ - camera capture (Phase 1)

Owned by cv-agent. This is the raw camera capture piece only — no YOLO, no
risk logic yet (those come in later phases per `PHASE_PLAN.md`).

## One-time setup

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r cv/requirements.txt
```

## Find your USB camera's device index

OpenCV addresses cameras by a plain integer index and there's no reliable way
to know up front which index is your USB camera vs. the laptop's built-in
webcam. Run:

```bash
python cv/detect_cameras.py
```

It probes indices 0–5, opens each, and reports which ones actually deliver a
frame plus their resolution. Use the resolution/behavior (or unplug the USB
camera and re-run to see which index disappears) to figure out which index is
yours.

## Stream raw video

```bash
python cv/stream_camera.py --index <the index you found above>
```

A window opens showing the live raw feed. Quit with `q` (window focused) or
by closing the window.

If `--index` is omitted it defaults to `0`, which on a laptop is usually the
built-in webcam, not the USB camera — pass `--index` explicitly once you know
the right value.
