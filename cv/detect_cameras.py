"""
detect_cameras.py - Phase 1 helper, rewritten in the 2026-08-05 "capability
matching" follow-up.

We have (at least) two cameras on this machine: the laptop's built-in webcam
and a specific USB camera we actually want to use. OpenCV identifies cameras
by a plain integer "device index," and there's no reliable cross-platform way
to ask "which index is the USB one" up front - the OS just hands them out in
whatever order it discovered the devices.

This script used to pair each OpenCV index with a device name by POSITION
(index 0 got whatever system_profiler listed first, etc.). That pairing was
field-tested and found to be actively wrong: system_profiler's listing order
and a second, independent enumeration (pyobjc/AVFoundation) agreed with EACH
OTHER, and both still disagreed with OpenCV's real index order on this same
machine (see docs/decision-log.md, 2026-08-05 "capability matching" entry,
for the measured table). A human reading the old output would have been told
"index 1 is the Arducam" while looking at the MacBook's webcam.

This version never pairs a name to an index by position. Instead it prints
two independent things and lets a human compare them by NUMBER:
  1. What each currently-connected, named device (via AVFoundation) is
     capable of - its maximum resolution, a property of the hardware.
  2. What each OpenCV index actually delivers when probed directly (opened,
     asked for an unreachable resolution, and read back what it really
     produced).
A device whose named capability and an index's probed capability match is
very likely the same physical camera - and, importantly, that comparison
holds regardless of what position either list happens to print things in.
See camera.named_device_capabilities() / camera.probe_index_capability() /
camera.resolve_name_to_index_by_capability() for the same logic used
automatically by --name in stream_camera.py / detect_stream.py.
"""

from camera import (
    CAPABILITY_PROBE_MAX_INDEX,
    named_device_capabilities,
    probe_index_capability,
)


def main() -> None:
    print(f"Probing OpenCV indices 0..{CAPABILITY_PROBE_MAX_INDEX - 1} ...\n")

    index_reports = []
    for index in range(CAPABILITY_PROBE_MAX_INDEX):
        report = probe_index_capability(index)
        index_reports.append(report)

        if not report["opened"]:
            print(f"[index {index}] could not open (no device here)")
            continue

        if not report["frame_read"]:
            print(
                f"[index {index}] opened but no frame could be read - skip "
                f"(dead connection, or macOS camera permission not yet "
                f"granted to this terminal app)"
            )
            continue

        print(
            f"[index {index}] WORKING - actual max delivered resolution "
            f"{report['width']}x{report['height']}"
        )

    print()

    # named_device_capabilities("") matches every currently-connected video
    # device's name (an empty substring is contained in every string) - a
    # convenient way to list all of them along with their AVFoundation-
    # reported capability, without claiming any of them corresponds to any
    # particular index above.
    named = named_device_capabilities("")

    if named is None:
        print(
            "(Device-name/capability lookup unavailable - either this isn't "
            "macOS, or pyobjc-framework-AVFoundation isn't installed (see "
            "cv/requirements.txt). You only have the index list above to go "
            "on; --name selection in stream_camera.py / detect_stream.py "
            "will also be unavailable for the same reason - use --index.)\n"
        )
    elif not named:
        print(
            "(No named camera devices found via AVFoundation - odd if any "
            "index above worked. --name selection will report the same and "
            "fall back to telling you to use --index.)\n"
        )
    else:
        print("Known device names (via AVFoundation), with their true capability:")
        for device in named:
            print(
                f"  '{device['name']}' - max resolution "
                f"{device['max_width']}x{device['max_height']}"
            )
        print()

    working_indices = [r for r in index_reports if r["frame_read"]]

    if working_indices and named:
        print(
            "Match a name above to a working index above by COMPARING THE "
            "NUMBERS (not the order they're printed in - position is not a "
            "reliable signal here, see this file's docstring). A name and "
            "an index whose resolutions are close are very likely the same "
            "physical device.\n"
        )

    if working_indices:
        print(
            "Preferred for a fixed room camera - select by name, since it's "
            "verified by capability (not position) every time it's opened, "
            "including on replug:\n"
            "    python stream_camera.py --name Arducam\n"
            "\n"
            "Selecting by index still works, but is not stable across "
            "replugs and carries no identity confirmation at all:\n"
            "    python stream_camera.py --index 1"
        )
    else:
        print(
            "No working camera indices found. Check that the USB camera is "
            "plugged in and that macOS has granted camera permission to your "
            "terminal application (not just Python) - System Settings > "
            "Privacy & Security > Camera - and that you've fully quit and "
            "reopened that terminal app since granting it."
        )


if __name__ == "__main__":
    main()
