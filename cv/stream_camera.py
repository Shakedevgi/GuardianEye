"""
stream_camera.py - Phase 1 main script.

Opens the USB camera at a given OpenCV device index and shows the raw,
unprocessed video feed in a window, continuously, until the user quits.

This is deliberately dumb: no detection, no risk scoring, nothing but
"can we reliably pull frames off this camera and show them." That's all
Phase 1 is asking for (see PHASE_PLAN.md). YOLO gets layered on top of this
same capture loop in Phase 2 (see detect_stream.py) - frame acquisition
itself now lives in camera.py so both scripts share one implementation.

Usage:
    python stream_camera.py                # uses default index (see below)
    python stream_camera.py --index 1      # use camera index 1 explicitly
    python stream_camera.py --name Arducam # select by device name (see below)

Run detect_cameras.py first if you don't know which index/name your USB
camera is.

Selecting by --name is the more robust option for a fixed room camera:
AVFoundation camera indices are NOT stable across replugs or reboots (see
docs/phase-writeups/phase-2.md) - a numeric index can silently point at a
different physical camera later. --name re-resolves the current index every
time the camera is (re)opened, including on a mid-stream reconnect. If both
--index and --name are given, --name wins (see camera.py).

Quit with the 'q' key (window must be focused) or by closing the window.
"""

import argparse

import cv2

from camera import DEFAULT_CAMERA_INDEX, CameraCapture, CameraSelectionError, startup_failure_message

WINDOW_NAME = "GuardianEye - raw feed (Phase 1)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream raw frames from a USB camera.")
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
    args = parser.parse_args()

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

    # Show the verified first frame before entering the loop below so a
    # window always exists. If the camera died the instant streaming
    # started, waitKey() in the loop would otherwise have no window to pump
    # events for, and neither 'q' nor the close button would do anything.
    cv2.imshow(WINDOW_NAME, first_frame)
    cv2.waitKey(1)

    try:
        for frame in camera.frames():
            # camera.frames() yields None while a read is failing/
            # reconnecting (see its docstring) - skip display but still
            # pump the GUI event loop and check for quit below, or the
            # window freezes and only Ctrl+C can end the session.
            if frame is not None:
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
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
