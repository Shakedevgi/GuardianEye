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

The overlay shows the running config (model, imgsz, conf, device) so you can
always tell which settings produced what you're looking at.
"""

import argparse
import os
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


def save_snapshot(raw_frame, annotated_frame, args) -> None:
    """Write the current frame to disk, both raw and annotated.

    The raw copy matters more than the annotated one: frames where detection
    fails are exactly the images Phase 3's fine-tuning dataset needs, and they
    have to be unannotated to be labelled. The filename records the config so
    a saved frame can always be traced back to what produced it.
    """
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"{stamp}_{Path(args.model).stem}_imgsz{args.imgsz}"

    raw_path = CAPTURES_DIR / f"{tag}_raw.jpg"
    annotated_path = CAPTURES_DIR / f"{tag}_annotated.jpg"

    cv2.imwrite(str(raw_path), raw_frame)
    cv2.imwrite(str(annotated_path), annotated_frame)

    print(f"Saved {raw_path.name} + {annotated_path.name}")


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
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    print(f"Loading {args.model} (imgsz={args.imgsz}) ...")
    model = load_model(args.model, device)
    print("Model loaded.")

    try:
        camera = CameraCapture(index=args.index, name=args.name)
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

    # Show the verified first frame before entering the loop below so a
    # window always exists. If the camera died the instant streaming
    # started, waitKey() in the loop would otherwise have no window to pump
    # events for, and neither 'q' nor the close button would do anything.
    cv2.imshow(WINDOW_NAME, first_frame)
    cv2.waitKey(1)

    try:
        for frame in camera.frames():
            # camera.frames() yields None while a read is failing/
            # reconnecting (see its docstring) - skip detection/display for
            # this iteration but still pump the GUI event loop and check for
            # quit below, or the window freezes and only Ctrl+C works.
            annotated = None

            if frame is not None:
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

                cv2.imshow(WINDOW_NAME, annotated)

            # waitKey also pumps the GUI event loop - required for imshow to
            # actually render and for the window's close button to register.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit key pressed - exiting.")
                break

            if key == ord("s") and frame is not None:
                save_snapshot(frame, annotated, args)

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
