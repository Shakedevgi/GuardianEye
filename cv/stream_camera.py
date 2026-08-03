"""
stream_camera.py - Phase 1 main script.

Opens the USB camera at a given OpenCV device index and shows the raw,
unprocessed video feed in a window, continuously, until the user quits.

This is deliberately dumb: no detection, no risk scoring, nothing but
"can we reliably pull frames off this camera and show them." That's all
Phase 1 is asking for (see PHASE_PLAN.md). YOLO gets layered on top of this
same capture loop in Phase 2.

Usage:
    python stream_camera.py                # uses default index (see below)
    python stream_camera.py --index 1      # use camera index 1 explicitly

Run detect_cameras.py first if you don't know which index your USB camera is.

Quit with the 'q' key (window must be focused) or by closing the window.
"""

import argparse
import time

import cv2

# Fallback if --index isn't passed. This is very likely NOT the USB camera on
# a laptop (index 0 is usually the built-in webcam) - run detect_cameras.py
# and pass --index explicitly once you know the real value. It's a constant
# here (not hardcoded deep in the logic) specifically so it's easy to find
# and change.
DEFAULT_CAMERA_INDEX = 0

WINDOW_NAME = "GuardianEye - raw feed (Phase 1)"

# If frame reads start failing (e.g. USB camera briefly disconnects), how
# many consecutive failures we tolerate before trying to fully reopen the
# capture, and how long to wait between reopen attempts. Simple backoff -
# this is Phase 1, not a production reconnect strategy.
MAX_CONSECUTIVE_READ_FAILURES = 10
REOPEN_RETRY_DELAY_SECONDS = 1.0


def open_capture(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    return cap


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream raw frames from a USB camera.")
    parser.add_argument(
        "--index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help=f"OpenCV camera device index (default: {DEFAULT_CAMERA_INDEX}). "
        "Run detect_cameras.py to find the right value for your USB camera.",
    )
    args = parser.parse_args()

    camera_index = args.index

    cap = open_capture(camera_index)
    if not cap.isOpened():
        print(
            f"Could not open camera at index {camera_index}. "
            "Run detect_cameras.py to find a working index, or check that "
            "the camera is plugged in and macOS camera permissions are granted."
        )
        return

    print(f"Streaming from camera index {camera_index}. Press 'q' to quit.")

    consecutive_failures = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                consecutive_failures += 1
                print(
                    f"Warning: frame read failed ({consecutive_failures}/"
                    f"{MAX_CONSECUTIVE_READ_FAILURES})"
                )

                if consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    print("Too many failed reads in a row - reopening camera...")
                    cap.release()
                    time.sleep(REOPEN_RETRY_DELAY_SECONDS)
                    cap = open_capture(camera_index)
                    consecutive_failures = 0

                    if not cap.isOpened():
                        print("Reopen failed. Retrying in a moment...")
                        time.sleep(REOPEN_RETRY_DELAY_SECONDS)

                continue

            consecutive_failures = 0
            cv2.imshow(WINDOW_NAME, frame)

            # waitKey also pumps the GUI event loop - required for imshow to
            # actually render and for the window's close button to register.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit key pressed - exiting.")
                break

            # Detect the user closing the window via the 'x' button. This
            # property disappears once the window is closed.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed - exiting.")
                break

    except KeyboardInterrupt:
        print("Interrupted (Ctrl+C) - exiting.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
