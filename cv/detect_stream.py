"""
detect_stream.py - Phase 2 main script.

Same live-camera capture loop as stream_camera.py (via camera.py), plus a
single YOLO26 pass per frame drawing boxes/labels/confidence for every stock
COCO class the model finds. This is Phase 2's entire goal: confirm the
detection pipeline actually sees what's in the room, in real time. No custom
classes (Phase 3), no hazard map / proximity scoring (Phase 4), no alerts or
clip saving (Phase 5) - see PHASE_PLAN.md.

Usage:
    python detect_stream.py                      # default index, mps, yolo26l
    python detect_stream.py --index 1
    python detect_stream.py --name Arducam       # select by device name (see below)
    python detect_stream.py --device cpu         # A/B against MPS (see below)
    python detect_stream.py --conf 0.4
    python detect_stream.py --model yolo26x.pt   # n/s/m/l/x
    python detect_stream.py --imgsz 1280         # more pixels for small objects

Selecting by --name is the more robust option for a fixed room camera:
AVFoundation camera indices are NOT stable across replugs or reboots (see
docs/phase-writeups/phase-2.md) - a numeric index can silently point at a
different physical camera later. --name re-resolves the current index every
time the camera is (re)opened, including on a mid-stream reconnect. If both
--index and --name are given, --name wins (see camera.py).

Why the --device escape hatch matters: there are known reports of Ultralytics
producing visibly wrong detections on MPS vs CPU. Defaulting to mps (this
machine's M5 Pro GPU) for speed, but --device cpu gives a one-word way to
rule MPS out as the cause of a bad detection before assuming the model or the
frame is at fault.

Keys (window must be focused):
    q   quit
    s   save the current frame to cv/captures/ - both the raw frame and the
        annotated one. Frames where detection fails are the raw material for
        Phase 3's fine-tuning dataset, so save them as you find them.
    t   self-timer: counts down (--timer seconds, default 10), then saves a
        single frame exactly like 's' does. For shots where the operator is
        also the subject (crouching/reaching poses 3m from the keyboard) and
        can't press 's' at the moment of capture. Press 't' again during the
        countdown to cancel it.

The overlay shows the running config (model, imgsz, conf, device) so you can
always tell which settings produced what you're looking at.

Bulk dataset collection (Phase 3 tooling):
    --interval N        Save a snapshot automatically every N seconds (float,
                         time.monotonic()-driven, not frame-counted) until the
                         script exits. Coexists with 's'/'t' - a manual save
                         never resets the interval clock. OFF by default; with
                         this flag absent, behaviour is unchanged.
    --session-tag TEXT   Optional string folded into saved filenames (e.g.
                         "kitchen-60cm-daylight") so a session's frames are
                         identifiable on disk without opening them. Sanitised
                         for filesystem safety.

After a bulk session, run cv/triage_captures.py over cv/captures/ to find
near-duplicate frames (an inevitable byproduct of interval capture) and
blurry frames before they go anywhere near a labelling tool.
"""

import argparse
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

# Must be set before torch is imported anywhere (including transitively via
# ultralytics) - lets unsupported MPS ops fall back to CPU instead of hard
# crashing. PyTorch's MPS operator coverage is still uneven, so this is
# insurance against a missing kernel taking the whole run down.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import torch
from ultralytics import YOLO

from camera import (
    DEFAULT_CAMERA_INDEX,
    CameraCapture,
    CameraSelectionError,
    startup_failure_message,
)

WINDOW_NAME = "GuardianEye - detection (Phase 2)"

# Weights live at a predictable path instead of wherever Ultralytics happens
# to drop them in the current working directory - keeps the repo layout
# independent of where you happened to run this script from.
MODELS_DIR = Path(__file__).resolve().parent / "models"

# Measured on this machine (M5 Pro, MPS, 1080p input): n 159fps, s 145,
# m 76, l 62, x 35. The camera itself caps at 25-30fps, so every size in the
# family already outruns the sensor and nano's speed advantage is headroom
# that cannot be spent. Defaulting to 'l' (55.0 COCO mAP vs nano's 40.9)
# buys accuracy for free while keeping ~2x margin over the camera for the
# Phase 4 risk engine and Phase 5 buffer to spend later.
DEFAULT_MODEL = "yolo26l.pt"

DEFAULT_CONF_THRESHOLD = 0.25

# YOLO downscales each frame to this square before inference. At the default
# 640 a 1920x1080 frame shrinks 3x, so a small distant object (a knife on a
# counter) can lose the detail it needs to be classified at all. Raising this
# costs frame rate but preserves small-object detail - see docs/decision-log.md.
DEFAULT_IMGSZ = 640

# Snapshots land here rather than the CWD so they're collectable as a dataset
# later. Gitignored - these are binaries, and there will be a lot of them.
CAPTURES_DIR = Path(__file__).resolve().parent / "captures"

OVERLAY_MARGIN = 10
OVERLAY_LINE_HEIGHT = 28
OVERLAY_COLOR = (0, 255, 0)
OVERLAY_OUTLINE = (0, 0, 0)

DEFAULT_TIMER_SECONDS = 10

# --session-tag is embedded in filenames, which have to stay filesystem-safe
# across macOS/Linux/Windows regardless of what the operator typed (spaces,
# slashes, punctuation). Anything outside this set is collapsed to '-'.
SESSION_TAG_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")
SESSION_TAG_MAX_LENGTH = 60

# --4k capture settings (see camera.py's module docstring and
# CAPABILITY_PROBE_FOURCCS comments for the underlying measurement): this
# project's Arducam only delivers its true 3840x2160 sensor resolution under
# MJPG - requesting a large size with no pixel format set yields 1920x1080
# instead, because uncompressed 4K exceeds USB bandwidth. MJPG is therefore
# not optional here, it's required to reach the requested size at all.
TARGET_4K_WIDTH = 3840
TARGET_4K_HEIGHT = 2160
TARGET_4K_FOURCC = "MJPG"

# The countdown/flash overlays (draw_countdown_overlay, draw_capture_flash)
# were sized by eye against a 1080p frame. Their font scale/thickness are
# fixed absolute pixel values, not proportional to frame size, so on a 4K
# frame (2x the linear resolution) they'd occupy half the on-screen
# proportion they were tuned for. Scaling by (actual height / this
# reference) keeps them a sensible relative size regardless of capture
# resolution; at exactly 1080p (today's default and this camera's normal
# delivered resolution) the scale factor is 1.0, so this is a no-op for the
# existing default path.
OVERLAY_REFERENCE_HEIGHT = 1080

# A 4K (or larger) frame won't fit most screens - this caps the window size
# for DISPLAY ONLY. Detection (model.predict) and save_snapshot() always
# keep operating on the full-resolution frame; only what's handed to
# cv2.imshow() is ever downscaled. See prepare_for_display().
DISPLAY_MAX_WIDTH = 1920

# How long the "captured" confirmation stays on screen. The operator who
# triggered a self-timer shot is across the room and can't see the console -
# this is their only feedback that the save actually happened.
CAPTURE_FLASH_SECONDS = 1.0

# FPS readout is smoothed with a simple exponential moving average so it
# doesn't jitter wildly frame to frame - alpha closer to 1 reacts faster,
# closer to 0 is smoother.
FPS_SMOOTHING_ALPHA = 0.1


def draw_overlay_line(image, text: str, line_number: int) -> None:
    """Draw one line of status text, with a dark outline behind it so it stays
    readable over both bright and dark parts of the frame.
    """
    origin = (OVERLAY_MARGIN, OVERLAY_MARGIN + line_number * OVERLAY_LINE_HEIGHT)
    for color, thickness in ((OVERLAY_OUTLINE, 4), (OVERLAY_COLOR, 2)):
        cv2.putText(
            image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness
        )


def _overlay_scale_factor(image) -> float:
    """How much to scale the big centered overlays (countdown/flash) relative
    to how they were tuned - see OVERLAY_REFERENCE_HEIGHT. At exactly the
    reference height (this camera's normal 1080p) this is 1.0, so it changes
    nothing for the existing default path.
    """
    height = image.shape[0]
    return height / OVERLAY_REFERENCE_HEIGHT


def draw_countdown_overlay(image, seconds_remaining: int) -> None:
    """Draw the remaining whole seconds large and centered in the frame.

    The corner overlay from draw_overlay_line() is sized to be read at the
    keyboard - unreadable from 3m across the room, which is exactly where the
    self-timer's subject needs to be. Same dark-outline-then-bright-fill
    approach so it stays legible against both a bright doorway and dark
    furniture, just scaled way up.

    Font scale/thickness are scaled by the frame's actual height (see
    _overlay_scale_factor) so this stays a sensible relative size whether the
    frame is today's default 1080p or a --4k capture, instead of shrinking to
    a small corner of a much bigger frame.
    """
    text = str(seconds_remaining)
    font = cv2.FONT_HERSHEY_SIMPLEX
    factor = _overlay_scale_factor(image)
    scale = 8.0 * factor
    thickness_outline, thickness_fill = max(1, round(20 * factor)), max(1, round(10 * factor))
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness_outline)
    h, w = image.shape[:2]
    origin = ((w - text_w) // 2, (h + text_h) // 2)
    for color, thickness in (
        (OVERLAY_OUTLINE, thickness_outline),
        (OVERLAY_COLOR, thickness_fill),
    ):
        cv2.putText(image, text, origin, font, scale, color, thickness)


def draw_capture_flash(image) -> None:
    """Confirm a self-timer capture actually happened - see
    CAPTURE_FLASH_SECONDS for why this exists at all.

    Scaled with frame height for the same reason as draw_countdown_overlay -
    see _overlay_scale_factor.
    """
    text = "CAPTURED"
    font = cv2.FONT_HERSHEY_SIMPLEX
    factor = _overlay_scale_factor(image)
    scale = 3.0 * factor
    thickness_outline, thickness_fill = max(1, round(14 * factor)), max(1, round(7 * factor))
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness_outline)
    h, w = image.shape[:2]
    origin = ((w - text_w) // 2, (h + text_h) // 2)
    for color, thickness in (
        (OVERLAY_OUTLINE, thickness_outline),
        (OVERLAY_COLOR, thickness_fill),
    ):
        cv2.putText(image, text, origin, font, scale, color, thickness)


def prepare_for_display(image):
    """Downscale ONLY for cv2.imshow() - a 4K (or larger) frame doesn't fit
    most screens. Detection (model.predict) and save_snapshot() always run on
    the full-resolution frame passed in elsewhere; this function's return
    value must never be used for anything but the imshow() call itself.

    A no-op (returns the same array, no copy) when the frame is already at or
    under DISPLAY_MAX_WIDTH - true for today's default 1920x1080 path, so
    this doesn't touch the existing default behaviour.
    """
    height, width = image.shape[:2]
    if width <= DISPLAY_MAX_WIDTH:
        return image
    scale = DISPLAY_MAX_WIDTH / width
    display_size = (DISPLAY_MAX_WIDTH, max(1, round(height * scale)))
    return cv2.resize(image, display_size, interpolation=cv2.INTER_AREA)


def sanitize_session_tag(raw: str) -> str:
    """Make a free-typed --session-tag value safe to embed in a filename on
    any of macOS/Linux/Windows: collapse anything that isn't alphanumeric,
    '_' or '-' to a single '-', strip leading/trailing separators, and cap
    the length so one long tag can't produce an unwieldy filename. Returns
    "" (falsy) if nothing safe survives, which callers treat as "no tag" -
    silently degrading is preferable to writing a garbage-named file.
    """
    collapsed = SESSION_TAG_SAFE_RE.sub("-", raw).strip("-_")
    return collapsed[:SESSION_TAG_MAX_LENGTH]


def save_snapshot(raw_frame, annotated_frame, args, session_tag: str = "") -> None:
    """Write the current frame to disk, both raw and annotated.

    The raw copy matters more than the annotated one: frames where detection
    fails are exactly the images Phase 3's fine-tuning dataset needs, and they
    have to be unannotated to be labelled. The filename records the config so
    a saved frame can always be traced back to what produced it.

    `session_tag`, if non-empty (already sanitized by sanitize_session_tag),
    is folded into the filename right after the timestamp so a whole
    collection session is identifiable on disk without opening any image -
    the existing timestamp+config naming is otherwise unchanged, so this is a
    no-op on the filename shape when no tag is given (--session-tag absent).
    """
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag_parts = [stamp]
    if session_tag:
        tag_parts.append(session_tag)
    tag_parts.append(f"{Path(args.model).stem}_imgsz{args.imgsz}")
    tag = "_".join(tag_parts)

    raw_path = CAPTURES_DIR / f"{tag}_raw.jpg"
    annotated_path = CAPTURES_DIR / f"{tag}_annotated.jpg"

    cv2.imwrite(str(raw_path), raw_frame)
    cv2.imwrite(str(annotated_path), annotated_frame)

    print(f"Saved {raw_path.name} + {annotated_path.name}")


def draw_interval_overlay(image, interval_seconds: float, saved_count: int) -> None:
    """Corner indicator that interval capture is armed, plus a running count
    of frames saved this session - sized/positioned like the existing
    FPS/config overlay lines (draw_overlay_line), just on the next line down,
    so it reads as part of the same status readout rather than a separate
    thing.
    """
    draw_overlay_line(
        image,
        f"INTERVAL CAPTURE ARMED ({interval_seconds:g}s)  saved this session: {saved_count}",
        3,
    )


def resolve_device(requested: str) -> str:
    """Confirm the requested device is actually usable, falling back to CPU
    with a clear warning rather than crashing. MPS op coverage in PyTorch is
    still uneven, so "requested" and "actually works" aren't the same thing.
    """
    if requested == "mps" and not torch.backends.mps.is_available():
        print(
            "Warning: --device mps requested but MPS is not available on "
            "this machine (torch.backends.mps.is_available() is False). "
            "Falling back to CPU."
        )
        return "cpu"
    return requested


def load_model(model_name: str, device: str) -> YOLO:
    """Load the model once, from a predictable on-disk path, before the
    capture loop starts - never re-load per frame.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = MODELS_DIR / model_name

    weights_already_present = weights_path.exists()
    if not weights_already_present:
        print(f"Weights not found at {weights_path} - downloading {model_name}...")

    model = YOLO(str(weights_path) if weights_already_present else model_name)

    if not weights_already_present:
        # A bare model name (not a path) makes Ultralytics download into the
        # current working directory - move it to our predictable location so
        # every future run finds it here instead of re-downloading.
        downloaded = Path(model_name)
        if downloaded.exists():
            downloaded.rename(weights_path)

    model.to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run stock YOLO26 detection on a live USB camera feed."
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help=f"OpenCV camera device index (default: {DEFAULT_CAMERA_INDEX} if "
        "neither --index nor --name is given). Run detect_cameras.py to find "
        "the right value for your USB camera. Ignored if --name is also given.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Select the camera by a substring of its device name (e.g. "
        "'Arducam') instead of a numeric index. Preferred for a fixed room "
        "camera: the index is re-resolved from the current device listing on "
        "every (re)open, so a replug that shuffles indices can't silently "
        "swap in the wrong camera. Wins over --index if both are given.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Inference device: 'mps' (default, Apple GPU) or 'cpu'. Use "
        "'cpu' as a one-word A/B if MPS detections look wrong.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help="Minimum detection confidence to draw a box "
        f"(default: {DEFAULT_CONF_THRESHOLD}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"YOLO26 weights to run (default: {DEFAULT_MODEL}). Sizes n/s/m/l/x "
        "trade accuracy for speed - all of them outrun this camera, so prefer "
        "a larger one unless you need the frame budget elsewhere.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"Inference resolution (default: {DEFAULT_IMGSZ}). Raise to 960/1280 "
        "to give small distant objects more pixels to be recognized by, at the "
        "cost of frame rate.",
    )
    parser.add_argument(
        "--4k",
        dest="fourk",
        action="store_true",
        help=f"Request {TARGET_4K_WIDTH}x{TARGET_4K_HEIGHT} capture with "
        f"{TARGET_4K_FOURCC} instead of today's default (no resolution/format "
        "requested at all, which this project's Arducam delivers as "
        "1920x1080). MJPG is required to reach 4K on this camera - "
        "uncompressed 4K exceeds USB bandwidth (see camera.py). OPT-IN and "
        "OFF by default: with this flag absent, the capture path is "
        "byte-for-byte identical to before this flag existed. The startup "
        "message always states the resolution actually measured from a "
        "real captured frame, never just the requested one - if the camera "
        "delivers less than requested, a clear warning names the actual "
        "resolution and the session continues at that resolution rather "
        "than failing. Only cv2.imshow()'s window is downscaled for "
        "display when this is on; detection and saved raw frames (`s`/`t`) "
        "always use the full captured resolution.",
    )
    parser.add_argument(
        "--timer",
        type=int,
        default=DEFAULT_TIMER_SECONDS,
        help="Self-timer countdown length in seconds for the 't' key "
        f"(default: {DEFAULT_TIMER_SECONDS}). Needs to be long enough to "
        "cross the room and get into a crouching/crawling/reaching pose "
        "before capture.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Bulk dataset collection: save a snapshot automatically every "
        "N seconds (float, e.g. 2.0) until the script exits, in addition to "
        "'s'/'t' - a manual save never resets this clock. Driven by "
        "time.monotonic(), not frame count, so it doesn't speed up/slow down "
        "with --imgsz. OFF by default (no flag = no behaviour change). "
        "Run cv/triage_captures.py afterward - interval capture will "
        "inevitably produce runs of near-identical frames.",
    )
    parser.add_argument(
        "--session-tag",
        type=str,
        default=None,
        help="Optional label folded into saved filenames (e.g. "
        "'kitchen-60cm-daylight') so a collection session's frames are "
        "identifiable on disk without opening them. Sanitised to "
        "alphanumeric/'-'/'_' for filesystem safety.",
    )
    args = parser.parse_args()

    session_tag = sanitize_session_tag(args.session_tag) if args.session_tag else ""
    if args.session_tag and not session_tag:
        print(
            f"Warning: --session-tag {args.session_tag!r} sanitized to "
            f"empty - nothing safe to embed in a filename. Proceeding "
            f"without a session tag."
        )
    elif args.session_tag and session_tag != args.session_tag:
        print(f"--session-tag sanitized to {session_tag!r} for filenames.")

    if args.interval is not None and args.interval <= 0:
        print(
            f"Error: --interval must be a positive number of seconds, got "
            f"{args.interval}."
        )
        return

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    print(f"Loading {args.model} (imgsz={args.imgsz}) ...")
    model = load_model(args.model, device)
    print("Model loaded.")

    # --4k is strictly opt-in: when absent, request_kwargs stays empty and
    # CameraCapture(...) is called exactly as it always was, so the default
    # path is byte-for-byte unchanged (see camera.py's request_width/
    # request_height/request_fourcc, which all default to None).
    request_kwargs = {}
    if args.fourk:
        print(
            f"--4k requested: asking for {TARGET_4K_WIDTH}x{TARGET_4K_HEIGHT} "
            f"({TARGET_4K_FOURCC}). This is a REQUEST - the resolution below "
            f"is what a real captured frame measured, which is the only "
            f"thing that actually matters."
        )
        request_kwargs = dict(
            request_width=TARGET_4K_WIDTH,
            request_height=TARGET_4K_HEIGHT,
            request_fourcc=TARGET_4K_FOURCC,
        )

    try:
        camera = CameraCapture(index=args.index, name=args.name, **request_kwargs)
    except CameraSelectionError as exc:
        print(str(exc))
        return

    ok, first_frame = camera.verify_startup()
    if not ok:
        print(startup_failure_message(camera.index))
        camera.release()
        return

    height, width = first_frame.shape[:2]
    device_label = camera.resolved_name or f"index {camera.index}"
    print(
        f"Streaming from camera {camera.index} ({device_label}) at "
        f"{width}x{height}. Press 'q' to quit."
    )

    smoothed_fps = None
    last_frame_time = time.monotonic()

    # Self-timer state. Driven by time.monotonic() deltas, not frame counts -
    # frame rate here ranges from ~35fps (yolo26x, imgsz 640) down to ~6fps
    # (imgsz 1600), so a frame-counted "10 seconds" would take wildly
    # different real time depending on --model/--imgsz. See requirement 6 in
    # the task and docs/phase-writeups/phase-2.md Round 2 for the waitKey
    # freeze bug this must not reintroduce - the loop still calls waitKey()
    # every iteration throughout the countdown.
    countdown_active = False
    countdown_remaining = 0.0
    countdown_last_tick = time.monotonic()
    countdown_last_announced = None
    countdown_signal_lost_notice = False
    capture_flash_until = None

    # Interval capture state (--interval). Also time.monotonic()-driven, for
    # the same reason as the self-timer above. `last_interval_save` anchors
    # to "now" (not incremented by a fixed step) each time a save fires, so a
    # camera outage or a long processing stall doesn't cause a burst of
    # catch-up saves the instant frames resume - see the loop below.
    # session_save_count counts every save this run makes regardless of
    # trigger ('s', 't', or interval) - the overlay's "saved this session"
    # readout and the console total both reflect the whole session, not just
    # interval-triggered saves.
    interval_active = args.interval is not None
    last_interval_save = time.monotonic()
    session_save_count = 0

    # Show the verified first frame before entering the loop below so a
    # window always exists. If the camera died the instant streaming
    # started, waitKey() in the loop would otherwise have no window to pump
    # events for, and neither 'q' nor the close button would do anything.
    cv2.imshow(WINDOW_NAME, prepare_for_display(first_frame))
    cv2.waitKey(1)

    try:
        for frame in camera.frames():
            # camera.frames() yields None while a read is failing/
            # reconnecting (see its docstring) - skip detection/display for
            # this iteration but still pump the GUI event loop and check for
            # quit below, or the window freezes and only Ctrl+C works.
            annotated = None

            if frame is None and countdown_active:
                # camera.frames() yields None while a read is failing/
                # reconnecting - it means "still trying," not "gone for
                # good" (see its docstring). A countdown that expired here
                # would have no frame to save, so freeze the clock instead:
                # reset the tick anchor so the outage isn't retroactively
                # counted as elapsed countdown time once frames resume.
                countdown_last_tick = time.monotonic()
                if not countdown_signal_lost_notice:
                    print(
                        "Camera signal lost - self-timer paused until "
                        "frames resume."
                    )
                    countdown_signal_lost_notice = True

            if frame is None and interval_active:
                # Same reasoning as the self-timer freeze just above, applied
                # to the interval clock: skip saving on a None frame (per
                # spec - there is nothing to save), and anchor the clock to
                # "now" so a long outage doesn't fire a burst of catch-up
                # saves the instant frames resume.
                last_interval_save = time.monotonic()

            if frame is not None:
                countdown_signal_lost_notice = False
                # Single YOLO pass per frame across all stock COCO classes at
                # once - per CLAUDE.md there is no separate person/hazard
                # detector, and Phase 2 shows every class found, not a
                # pre-filtered "hazard subset."
                results = model.predict(
                    frame, conf=args.conf, imgsz=args.imgsz, device=device, verbose=False
                )

                # Ultralytics' own plot() already draws each detection's box,
                # class label, and confidence - no need to hand-roll that, and
                # it stays correct if the model's class list changes later.
                annotated = results[0].plot()

                now = time.monotonic()
                instantaneous_fps = 1.0 / max(now - last_frame_time, 1e-6)
                last_frame_time = now
                smoothed_fps = (
                    instantaneous_fps
                    if smoothed_fps is None
                    else FPS_SMOOTHING_ALPHA * instantaneous_fps
                    + (1 - FPS_SMOOTHING_ALPHA) * smoothed_fps
                )

                # Both lines are on-screen deliberately: when you're A/B-ing
                # model sizes and resolutions, the window itself has to say
                # which config produced what you're looking at, or saved
                # frames become impossible to attribute after the fact.
                draw_overlay_line(annotated, f"FPS: {smoothed_fps:.1f}", 1)
                draw_overlay_line(
                    annotated,
                    f"{args.model}  imgsz={args.imgsz}  conf={args.conf}  {device}",
                    2,
                )

                if countdown_active:
                    dt = now - countdown_last_tick
                    countdown_last_tick = now
                    countdown_remaining -= dt
                    # ceil, not round: "1" should stay on screen for the
                    # entire last second, not disappear at 0.5s remaining.
                    remaining_display = max(0, math.ceil(countdown_remaining))
                    if remaining_display != countdown_last_announced:
                        print(f"Self-timer: {remaining_display}...")
                        countdown_last_announced = remaining_display

                    if countdown_remaining <= 0:
                        save_snapshot(frame, annotated, args, session_tag=session_tag)
                        session_save_count += 1
                        print(f"Running total saved this session: {session_save_count}")
                        countdown_active = False
                        capture_flash_until = now + CAPTURE_FLASH_SECONDS
                    else:
                        draw_countdown_overlay(annotated, remaining_display)

                if capture_flash_until is not None:
                    if now < capture_flash_until:
                        draw_capture_flash(annotated)
                    else:
                        capture_flash_until = None

                if interval_active:
                    # time.monotonic()-driven, not frame-counted - see the
                    # state comment above for why. A manual 's' below does
                    # NOT touch last_interval_save, so it can never reset or
                    # disturb this clock.
                    if now - last_interval_save >= args.interval:
                        save_snapshot(frame, annotated, args, session_tag=session_tag)
                        session_save_count += 1
                        last_interval_save = now
                        print(
                            f"[interval] auto-saved. Running total saved "
                            f"this session: {session_save_count}"
                        )
                    draw_interval_overlay(annotated, args.interval, session_save_count)

                # Downscale for the display window ONLY - `annotated` itself
                # stays full resolution and is what save_snapshot() writes to
                # disk below (raw AND annotated), and detection above already
                # ran on the full-resolution `frame`. See prepare_for_display.
                cv2.imshow(WINDOW_NAME, prepare_for_display(annotated))

            # waitKey also pumps the GUI event loop - required for imshow to
            # actually render and for the window's close button to register.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit key pressed - exiting.")
                break

            if key == ord("s") and frame is not None:
                # Manual save deliberately does not touch last_interval_save -
                # a manual 's' during interval capture must not reset or
                # disturb the interval clock (see --interval's help text).
                save_snapshot(frame, annotated, args, session_tag=session_tag)
                session_save_count += 1
                print(f"Running total saved this session: {session_save_count}")

            if key == ord("t"):
                if countdown_active:
                    # A fumbled/accidental start shouldn't force a wasted
                    # wait - pressing 't' again cancels it. 'q' still quits
                    # immediately during a countdown (checked above, not here).
                    countdown_active = False
                    print("Self-timer cancelled.")
                else:
                    countdown_active = True
                    countdown_remaining = float(args.timer)
                    countdown_last_tick = time.monotonic()
                    countdown_last_announced = None
                    countdown_signal_lost_notice = False
                    print(
                        f"Self-timer started: capturing in {args.timer}s. "
                        "Press 't' again to cancel."
                    )

            # Detect the user closing the window via the 'x' button. This
            # property disappears once the window is closed.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed - exiting.")
                break

    except KeyboardInterrupt:
        print("Interrupted (Ctrl+C) - exiting.")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
