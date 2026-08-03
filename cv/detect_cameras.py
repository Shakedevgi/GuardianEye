"""
detect_cameras.py - Phase 1 helper.

We have (at least) two cameras on this machine: the laptop's built-in webcam
and a specific USB camera we actually want to use. OpenCV identifies cameras
by a plain integer "device index," and there's no reliable cross-platform way
to ask "which index is the USB one" up front - the OS just hands them out in
whatever order it discovered the devices.

This script iterates over a small range of indices, tries to open each one,
grabs a frame, and reports what it found (resolution + whether a frame read
succeeded) so a human can look at the reported resolutions/behavior and match
one of them to the USB camera by eye (e.g. unplug the USB camera and re-run,
or watch which index gives a preview window matching what the USB camera sees).

Run it, read the printed report, and note which index is the USB camera -
you'll pass that index into stream_camera.py.
"""

import cv2

# How many indices to probe. 6 is generous for a machine with a built-in
# webcam + one USB camera; raise this if you have more devices attached.
MAX_INDEX_TO_CHECK = 6

# How long (in frames) to try reading before giving up on an index. Some
# camera backends report "opened" successfully but need a frame or two to
# actually start delivering images.
READ_ATTEMPTS = 5


def probe_index(index: int) -> dict:
    """Try to open a camera at `index` and read one real frame from it.

    Returns a small report dict rather than raising - Phase 1 just needs a
    human-readable summary, not exceptions to handle.
    """
    report = {
        "index": index,
        "opened": False,
        "frame_read": False,
        "width": None,
        "height": None,
    }

    cap = cv2.VideoCapture(index)
    try:
        report["opened"] = cap.isOpened()
        if not report["opened"]:
            return report

        for _ in range(READ_ATTEMPTS):
            ok, frame = cap.read()
            if ok and frame is not None:
                report["frame_read"] = True
                report["height"], report["width"] = frame.shape[:2]
                break
    finally:
        cap.release()

    return report


def main() -> None:
    print(f"Probing camera indices 0..{MAX_INDEX_TO_CHECK - 1} ...\n")

    found_any_working = False

    for index in range(MAX_INDEX_TO_CHECK):
        report = probe_index(index)

        if not report["opened"]:
            print(f"[index {index}] could not open (no device here)")
            continue

        if not report["frame_read"]:
            print(f"[index {index}] opened but no frame could be read - skip")
            continue

        found_any_working = True
        print(
            f"[index {index}] WORKING - resolution "
            f"{report['width']}x{report['height']}"
        )

    print()
    if found_any_working:
        print(
            "Look at the resolutions/behavior above to figure out which index "
            "is your USB camera vs. the built-in webcam (tip: unplug the USB "
            "camera and re-run this script - whichever index disappears was "
            "the USB one). Then pass that index to stream_camera.py, e.g.:\n"
            "    python stream_camera.py --index 1"
        )
    else:
        print(
            "No working camera indices found. Check that the USB camera is "
            "plugged in and that macOS has granted camera permission to your "
            "terminal / Python (System Settings > Privacy & Security > Camera)."
        )


if __name__ == "__main__":
    main()
