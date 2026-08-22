"""
camera.py - shared frame-acquisition layer.

Extracted out of Phase 1's stream_camera.py so both the raw-feed script and
the Phase 2 detection script (and any later script that needs frames, e.g.
the Phase 5 rolling buffer) share one implementation of "open the device,
read frames, and recover from transient USB hiccups" instead of each having
its own copy of that logic to keep in sync.

This module owns exactly that concern and nothing else: no display, no
detection, no keypress handling. Callers loop over `CameraCapture.frames()`
and do whatever they want with each frame.

Phase 2 follow-up (2026-08-05): two bugs cost the team ~2 hours during Phase
2 bring-up (see docs/phase-writeups/phase-2.md, "bring-up hardships"), fixed
here:

1. `cap.isOpened()` returning True says nothing about whether frames are
   actually flowing. `CameraCapture.verify_startup()` reads real frames
   before a caller is allowed to claim success.
2. AVFoundation camera indices are NOT stable across replugs/reboots - the
   Arducam moved from index 1 to index 0 mid-session during bring-up. For a
   fixed room camera, a silent index shuffle can mean "monitoring the wrong
   camera" - a safety-relevant failure, not a cosmetic one. `CameraCapture`
   supports selection by device name, re-resolved on every reopen, not just
   at startup (see point 3 below for HOW that name is resolved - this has
   changed since first shipped).

Phase 2 follow-up round 3 (2026-08-05, "capability matching" - see
docs/decision-log.md for the full incident writeup): the *first* fix for (2)
above resolved a name to an index positionally - it enumerated devices via
`system_profiler SPCameraDataType` and assumed that listing's order matched
OpenCV's own index order, on the strength of having observed the two agree
during Phase 2 bring-up. That shipped, and then failed in the field: on a
later session, `system_profiler` order and a second independent enumeration
(pyobjc/AVFoundation's `devicesWithMediaType_`) agreed with EACH OTHER but
both disagreed with OpenCV's actual index order, and the running detection
script confidently printed "Camera selected: index 1 (Arducam-B0560-4K HDR)"
while displaying the MacBook's built-in webcam. Two independent enumeration
sources agreeing with each other was not evidence they matched OpenCV -
position is simply not a reliable proxy for OpenCV's index space, full
stop, no matter how it's obtained.

The fix implemented here abandons position entirely and identifies a device
by CAPABILITY instead: `named_device_capabilities()` asks AVFoundation what
resolution ceiling a *named* device has (a property of the hardware, not of
enumeration order), and `probe_index_capability()` asks each OpenCV index
directly, by opening it and reading back what it actually delivers when a
resolution far beyond any of these devices is requested. The two are
matched on that shared, position-independent fact. See
`resolve_name_to_index_by_capability()` below for the whole thing, and
`docs/decision-log.md`'s 2026-08-05 "capability matching" entry for the
measured numbers that motivated this.
"""

import platform
import time

import cv2

# pyobjc-framework-AVFoundation is macOS-only and used here for exactly one
# thing: asking a NAMED device what resolutions it supports (a hardware
# capability query), never for positional/index enumeration - that
# distinction matters, see the module docstring above. Guarded import so a
# non-macOS machine (or a macOS machine where this optional dependency
# hasn't been installed) degrades to "name-based selection unavailable",
# not a crash - see resolve_name_to_index_by_capability().
try:
    import AVFoundation
    import CoreMedia

    _AVFOUNDATION_AVAILABLE = True
except ImportError:
    _AVFOUNDATION_AVAILABLE = False

# Fallback if a caller doesn't pass an index or name. This is very likely NOT
# the USB camera on a laptop (index 0 is usually the built-in webcam) - run
# detect_cameras.py and pass the real index (or better, --name) explicitly
# once you know it. It's a constant here (not hardcoded deep in the logic)
# specifically so it's easy to find and change.
DEFAULT_CAMERA_INDEX = 0

# If frame reads start failing (e.g. USB camera briefly disconnects), how
# many consecutive failures we tolerate before trying to fully reopen the
# capture. Unrelated to STARTUP_VERIFY_* below: this is mid-stream hiccup
# recovery, not first-open validation.
#
# Read failures on a genuinely gone device are near-instant (cap.read()
# doesn't block waiting for hardware that isn't there), so this threshold
# alone does NOT bound how often we attempt a reopen - that's what the
# REOPEN_* backoff below is for.
MAX_CONSECUTIVE_READ_FAILURES = 10

# 2026-08-05 bring-up fix (see docs/decision-log.md): a fixed 1s delay
# between reopen attempts, combined with instant read failures, meant a long
# unplug hammered reopen attempts back-to-back - each one shelling out to
# system_profiler *twice* (once for the name lookup, once for the reverse
# name-for-index lookup) at ~1s each. Escalating, capped backoff keeps a long
# outage from spinning the CPU and spamming system_profiler while still
# reconnecting within a couple seconds of the device coming back for a short
# hiccup.
REOPEN_INITIAL_DELAY_SECONDS = 1.0
REOPEN_MAX_DELAY_SECONDS = 8.0
REOPEN_BACKOFF_MULTIPLIER = 2.0

# Settle time after release() and before creating the new cv2.VideoCapture.
# On macOS/AVFoundation, a VideoCapture recreated on the same index
# immediately after release() can report isOpened() == True while never
# actually delivering a frame - the OS hasn't finished tearing down the old
# capture session yet. This mirrors exactly the "isOpened() lies" lesson
# STARTUP_VERIFY_* below already encodes for first-open; the mid-stream
# reopen path previously trusted isOpened() alone and inherited that same
# failure mode (see docs/decision-log.md, 2026-08-05 follow-up).
REOPEN_SETTLE_SECONDS = 1.0

# How many real frames to try reading right after a reopen before trusting
# it actually worked - same idea as STARTUP_VERIFY_ATTEMPTS but shorter,
# since a failed attempt here just falls back into the backoff loop rather
# than being a fatal startup error.
REOPEN_VERIFY_ATTEMPTS = 5
REOPEN_VERIFY_RETRY_DELAY_SECONDS = 0.2

# Reopen attempts are logged loudly the first few times (so a real replug is
# immediately visible), then throttled to every Nth attempt so a long unplug
# doesn't produce thousands of near-identical lines. This is a logging
# concern only - every attempt still happens, only the printing is skipped.
REOPEN_LOG_ALWAYS_FIRST_N = 3
REOPEN_LOG_EVERY_N_ATTEMPTS = 5

# How many times to try reading one real frame before concluding a freshly
# opened camera is dead, and how long to wait between attempts. This exists
# because cap.isOpened() can return True while .read() never delivers a
# frame - macOS camera permission not granted to the terminal app, wrong
# device index, or the camera already held by another process all look
# identical from isOpened()'s point of view. Phase 2's bring-up lost ~2
# hours to exactly this (see docs/phase-writeups/phase-2.md). Some UVC
# cameras also genuinely need a few reads to "warm up" before frames start
# flowing - one failed read is not conclusive proof of a dead camera either,
# hence multiple attempts rather than a single check. 15 attempts x 0.2s is
# generous (3s worst case) without hanging the script for a truly dead
# camera.
STARTUP_VERIFY_ATTEMPTS = 15
STARTUP_VERIFY_RETRY_DELAY_SECONDS = 0.2


def open_capture(
    index: int,
    request_width: int | None = None,
    request_height: int | None = None,
    request_fourcc: str | None = None,
) -> cv2.VideoCapture:
    """Open OpenCV index `index`, optionally requesting a capture resolution
    and/or pixel format before returning.

    All three request_* args default to None, matching the original
    zero-arg behaviour byte-for-byte for every existing caller that doesn't
    pass them (the capability probe below constructs its own VideoCapture
    inline rather than through here, so it's unaffected either way).

    FOURCC is set before width/height, mirroring probe_index_capability()
    below and for the same reason: on this project's Arducam, requesting a
    4K size with no pixel format negotiated first silently falls back to
    1920x1080 (uncompressed 4K exceeds USB bandwidth) - MJPG has to be
    requested for the large size to actually be reachable at all. Setting a
    cv2.VideoCapture property is a REQUEST, not a guarantee - callers must
    verify what was actually delivered by reading a real frame, never trust
    cap.get() (see this module's docstring and CameraCapture below).
    """
    cap = cv2.VideoCapture(index)
    if request_fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*request_fourcc))
    if request_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, request_width)
    if request_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, request_height)
    return cap


# How many OpenCV indices to probe when resolving a name by capability.
# Matches detect_cameras.py's own MAX_INDEX_TO_CHECK - both scripts are
# probing the same small local index space (built-in webcam + a couple of
# USB/Continuity cameras), so they're kept in sync deliberately rather than
# picking two different numbers that could silently drift apart.
CAPABILITY_PROBE_MAX_INDEX = 6

# Requested resolution used purely to force AVFoundation to reveal a
# device's true ceiling: bigger than any consumer webcam/UVC camera this
# project expects to encounter (well beyond the Arducam's own 4K max), so
# whatever comes back is the device's real maximum, not an artifact of
# under-asking. 8K, chosen with headroom above the Arducam's measured
# 3840x2160 ceiling.
CAPABILITY_PROBE_REQUEST_WIDTH = 7680
CAPABILITY_PROBE_REQUEST_HEIGHT = 4320

# Pixel formats to try during a capability probe, best-first; the probe keeps
# the largest frame any of them delivers. MJPG is not optional here - a 4K UVC
# camera can't send uncompressed frames at full resolution within USB
# bandwidth, so its top mode is only reachable in a compressed format.
# Measured on this project's Arducam during Phase 2 bring-up: no FOURCC gave
# 1920x1080, MJPG gave the true 3840x2160. None = leave the format alone, for
# devices that are happiest untouched.
CAPABILITY_PROBE_FOURCCS = ("MJPG", None)

# How many real frames to try reading during a capability probe, and how
# long to wait between attempts - same "isOpened() can lie, only a real
# frame counts" lesson as STARTUP_VERIFY_*, applied here to a device we
# don't yet know the identity of rather than one already selected.
CAPABILITY_PROBE_READ_ATTEMPTS = 10
CAPABILITY_PROBE_READ_DELAY_SECONDS = 0.2

# Relative tolerance (fraction of pixel area) allowed between an
# AVFoundation-reported max resolution and what OpenCV actually delivers for
# the same physical device when asked for CAPABILITY_PROBE_REQUEST_*. In
# principle both paths go through the same underlying AVFoundation session,
# so an exact match would be expected - this tolerance exists as a margin
# for format-negotiation quirks (e.g. a slightly different "best" format
# picked under an unreachable request) rather than because exact matching is
# known to fail. NOT verified against real hardware from this environment -
# no camera permission is available here (see docs/decision-log.md,
# 2026-08-05 "capability matching" entry, "what was and wasn't verified").
CAPABILITY_MATCH_AREA_TOLERANCE = 0.10


def named_device_capabilities(name_substring: str) -> list[dict] | None:
    """Ask AVFoundation what resolution ceiling every currently-connected
    camera whose name contains `name_substring` (case-insensitive) actually
    has - a property of the physical hardware, queried live each call (never
    cached) so a replug is picked up.

    This deliberately does NOT return, or rely on, any notion of "index" or
    "position" - see the module docstring's "capability matching" section
    for why: positional enumeration order (whether from system_profiler or
    from this exact same AVFoundation API) was field-tested and found to NOT
    correspond to OpenCV's own index space. This function is used for
    exactly one thing: "what should device X be capable of," never "what
    position is device X at."

    Returns a list of dicts (possibly empty - that name just isn't
    connected right now): {"name", "uid", "max_width", "max_height"}, where
    max_width/max_height is the largest-area format the device's own
    `AVCaptureDevice.formats()` reports supporting.

    Returns None, distinct from [], if capability lookup isn't possible at
    all: non-macOS, pyobjc-framework-AVFoundation not installed, or an
    unexpected AVFoundation-side error. Callers MUST distinguish this from
    [] - None means "we cannot use this method here, fall back to
    --index," not "nothing matched."
    """
    if platform.system() != "Darwin" or not _AVFOUNDATION_AVAILABLE:
        return None

    needle = name_substring.lower()
    try:
        devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(
            AVFoundation.AVMediaTypeVideo
        )
        matches = []
        for device in devices:
            name = str(device.localizedName())
            if needle not in name.lower():
                continue

            max_width, max_height = 0, 0
            for fmt in device.formats():
                dims = CoreMedia.CMVideoFormatDescriptionGetDimensions(
                    fmt.formatDescription()
                )
                if dims.width * dims.height > max_width * max_height:
                    max_width, max_height = dims.width, dims.height

            matches.append(
                {
                    "name": name,
                    "uid": str(device.uniqueID()),
                    "max_width": max_width,
                    "max_height": max_height,
                }
            )
        return matches
    except Exception:
        # Any unexpected AVFoundation-side failure degrades to "lookup
        # unavailable," same as an import failure - callers already have to
        # handle None, so there's no separate error path to add here.
        return None


def probe_index_capability(
    index: int,
    attempts: int = CAPABILITY_PROBE_READ_ATTEMPTS,
    delay: float = CAPABILITY_PROBE_READ_DELAY_SECONDS,
    request_width: int = CAPABILITY_PROBE_REQUEST_WIDTH,
    request_height: int = CAPABILITY_PROBE_REQUEST_HEIGHT,
) -> dict:
    """Open OpenCV index `index` fresh, request an unreachably large
    resolution, read back a REAL frame, and report what the device actually
    delivered - a capability fingerprint read through OpenCV itself, not
    inferred from any enumeration source.

    Returns {"index", "opened", "frame_read", "width", "height"}.
    width/height come from the actual frame's `.shape` (never `cap.get()`,
    which - like `isOpened()` - can report values that don't reflect what
    `.read()` is really delivering) and are None if no real frame was read.
    """
    report = {
        "index": index,
        "opened": False,
        "frame_read": False,
        "width": None,
        "height": None,
    }

    # Try each pixel format in turn and keep the LARGEST frame any of them
    # delivered. MJPG is tried first and matters more than it looks: a 4K UVC
    # camera physically cannot push uncompressed frames at full resolution
    # over USB bandwidth, so it only offers its top mode in a compressed
    # format. Measured on this project's Arducam during Phase 2 bring-up:
    # requesting a large size with no FOURCC delivered 1920x1080, while MJPG
    # at 3840x2160 delivered true 4K. Probing without MJPG would therefore
    # fingerprint the camera at a third of its real capability and fail to
    # match what AVFoundation reports it can do - which would make every
    # --name lookup for that camera fail as "capability unconfirmed."
    for fourcc in CAPABILITY_PROBE_FOURCCS:
        cap = open_capture(index)
        try:
            if not cap.isOpened():
                continue
            report["opened"] = True

            if fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, request_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, request_height)

            for _ in range(attempts):
                ok, frame = cap.read()
                if ok and frame is not None:
                    height, width = frame.shape[:2]
                    previous_area = (report["width"] or 0) * (report["height"] or 0)
                    if width * height > previous_area:
                        report["frame_read"] = True
                        report["width"], report["height"] = width, height
                    break
                time.sleep(delay)
        finally:
            cap.release()

    return report


def _capability_areas_match(
    expected_width: int,
    expected_height: int,
    actual_width: int | None,
    actual_height: int | None,
    tolerance: float = CAPABILITY_MATCH_AREA_TOLERANCE,
) -> bool:
    """Compare an AVFoundation-reported max resolution against an actual
    OpenCV-delivered one by pixel area, within `tolerance` - see
    CAPABILITY_MATCH_AREA_TOLERANCE for why a tolerance exists at all.
    """
    if not actual_width or not actual_height:
        return False
    expected_area = expected_width * expected_height
    if expected_area == 0:
        return False
    actual_area = actual_width * actual_height
    return abs(actual_area - expected_area) / expected_area <= tolerance


def resolve_name_to_index_by_capability(
    name_substring: str,
    probe_indices=range(CAPABILITY_PROBE_MAX_INDEX),
) -> tuple[int, str, str]:
    """Resolve a device name to a CONFIRMED OpenCV index by matching
    capability, not position - see the module docstring's "capability
    matching" section for the full incident this replaces.

    Every non-success outcome raises CameraSelectionError with a message
    naming exactly what's wrong, rather than ever guessing:
    - capability lookup unavailable at all (non-macOS / pyobjc missing) -
      told explicitly, told to use --index instead.
    - name matches zero currently-connected devices.
    - name matches MORE THAN ONE currently-connected device (ambiguous by
      name alone).
    - the named device's expected capability matches zero currently
      reachable OpenCV indices (could not confirm identity at all).
    - the named device's expected capability matches MORE THAN ONE OpenCV
      index (ambiguous by capability - e.g. two devices with a similar
      resolution ceiling).

    On success, returns (index, confirmed_device_name, detail) - `detail`
    is a human-readable string of exactly what was compared, suitable for
    printing so a match is never asserted without showing its evidence.
    """
    named_matches = named_device_capabilities(name_substring)

    if named_matches is None:
        raise CameraSelectionError(
            f"Cannot select a camera by name ('{name_substring}'): "
            f"capability lookup via AVFoundation is unavailable on this "
            f"machine (not macOS, pyobjc-framework-AVFoundation isn't "
            f"installed, or AVFoundation itself errored - see "
            f"cv/requirements.txt). Use --index instead and confirm the "
            f"physical device with detect_cameras.py; do NOT fall back to "
            f"positional name matching - that was tried, shipped, and "
            f"observed to disagree with OpenCV's real index order in the "
            f"field (see docs/decision-log.md, 2026-08-05)."
        )

    if len(named_matches) == 0:
        raise CameraSelectionError(
            f"No currently-connected camera's name contains "
            f"'{name_substring}'. Run detect_cameras.py and confirm the "
            f"camera is plugged in and its name."
        )

    if len(named_matches) > 1:
        listed = ", ".join(f"'{m['name']}'" for m in named_matches)
        raise CameraSelectionError(
            f"'{name_substring}' matches more than one currently-connected "
            f"camera by name ({listed}) - ambiguous, refusing to guess. "
            f"Use a more specific --name substring."
        )

    target = named_matches[0]

    candidates = []
    for index in probe_indices:
        probe = probe_index_capability(index)
        if not probe["frame_read"]:
            continue
        if _capability_areas_match(
            target["max_width"], target["max_height"], probe["width"], probe["height"]
        ):
            candidates.append((index, probe))

    if len(candidates) == 0:
        raise CameraSelectionError(
            f"Found '{target['name']}' via AVFoundation (expected max "
            f"resolution {target['max_width']}x{target['max_height']}), "
            f"but no OpenCV index in "
            f"{list(probe_indices)[0]}..{list(probe_indices)[-1]} "
            f"delivered a matching resolution when probed directly - "
            f"could not confirm this device is reachable through OpenCV "
            f"right now. It may be held by another process, or macOS "
            f"camera permission may not be granted to this terminal app "
            f"(see startup_failure_message)."
        )

    if len(candidates) > 1:
        indices = [c[0] for c in candidates]
        raise CameraSelectionError(
            f"'{target['name']}' (expected max resolution "
            f"{target['max_width']}x{target['max_height']}) matches more "
            f"than one OpenCV index by capability: {indices}. Ambiguous - "
            f"another connected camera has a similar resolution ceiling. "
            f"Refusing to guess; run detect_cameras.py to compare their "
            f"actual delivered resolutions and pick one with --index."
        )

    index, probe = candidates[0]
    detail = (
        f"confirmed via OpenCV: probe requested "
        f"{CAPABILITY_PROBE_REQUEST_WIDTH}x{CAPABILITY_PROBE_REQUEST_HEIGHT} "
        f"and index {index} delivered {probe['width']}x{probe['height']}, "
        f"matching '{target['name']}''s AVFoundation-reported max "
        f"({target['max_width']}x{target['max_height']})"
    )
    return index, target["name"], detail


class CameraSelectionError(RuntimeError):
    """Raised when a camera was requested by name and no device currently
    matches. Distinct from "device opened but delivered no frames" (that's
    handled by verify_startup) - this is "we don't even know which index to
    try."
    """


class CameraCapture:
    """Owns a cv2.VideoCapture and hands out frames with reconnect-on-failure
    built in, so callers get that behavior for free instead of re-implementing
    the read-loop/backoff logic themselves.

    Selection is either by numeric index (`index=`) or by device name
    substring (`name=`) - not both; see __init__. Selecting by name is the
    safety-relevant option for a fixed room camera: AVFoundation indices are
    not stable across replugs/reboots (observed first-hand during Phase 2
    bring-up), so a name is re-resolved to a (possibly different) index every
    time this class reopens the device, including after a mid-stream
    reconnect - see `frames()`.

    `request_width`/`request_height`/`request_fourcc` optionally request a
    specific capture resolution/pixel format, re-applied on EVERY open
    (initial open and every `_reopen()` - see `open_capture()`) so a
    mid-session reconnect can't silently drop back to a lower resolution
    unnoticed. All default to None, which is the original "negotiate
    nothing" behaviour, unchanged. What was actually delivered is measured
    from a REAL frame (never trusted from `cap.get()`) and recorded in
    `delivered_width`/`delivered_height`; see `_note_delivered_resolution()`
    for the "request vs. measured" distinction and its warning.

    Usage:
        camera = CameraCapture(name="Arducam")
        ok, frame = camera.verify_startup()
        if not ok:
            ... report error, bail ...
        for frame in camera.frames():
            ... do something with frame ...
            if done:
                break
        camera.release()
    """

    def __init__(
        self,
        index: int | None = None,
        name: str | None = None,
        request_width: int | None = None,
        request_height: int | None = None,
        request_fourcc: str | None = None,
    ):
        if name is not None and index is not None:
            # Both passed: name wins. Name identifies a physical device
            # regardless of which index it currently sits at, which is
            # exactly the safety property --index can't offer - so if a
            # human passes both, we assume the name is the more deliberate,
            # trustworthy choice rather than silently picking one at random.
            print(
                f"Both --index {index} and --name '{name}' were given - "
                f"selecting by name (more robust to index shuffling). "
                f"Pass only --index if you specifically want the raw index."
            )
            index = None

        if name is None and index is None:
            index = DEFAULT_CAMERA_INDEX

        self._selector_name = name  # None => selecting by fixed index
        self._selector_index = index  # only meaningful when _selector_name is None
        self._capability_detail = None  # human-readable evidence for a name match
        self.resolved_name = None

        # Optional capture-resolution/format request, applied on EVERY open
        # (initial open below and every _reopen() - see open_capture() and
        # _reopen()). All default to None, which reproduces the original
        # "just open the index, negotiate nothing" behaviour exactly - a
        # caller that never passes these (e.g. the default no-flag path in
        # detect_stream.py) gets byte-for-byte the same capture as before
        # this feature existed. When set, a request is just that - a
        # request; delivered_width/delivered_height below record what a
        # REAL frame actually measured, which can be smaller (see
        # _note_delivered_resolution).
        self._request_width = request_width
        self._request_height = request_height
        self._request_fourcc = request_fourcc
        self.delivered_width = None
        self.delivered_height = None
        # Whether resolved_name reflects a capability match confirmed THIS
        # round (vs. a stale value kept from an earlier successful
        # resolution while a later re-resolution attempt failed - see
        # _reopen). _print_resolved_device uses this to make sure it never
        # asserts a name that isn't currently backed by evidence.
        self._name_confirmed_this_round = False

        self.index = self._resolve_index()
        self.cap = open_capture(
            self.index,
            self._request_width,
            self._request_height,
            self._request_fourcc,
        )

        self._print_resolved_device()

    def _resolve_index(self) -> int:
        """Work out which OpenCV index to (re)open right now, at
        construction time. Raises CameraSelectionError (uncaught here - the
        caller decides how to handle it) if name-based resolution fails, so
        construction never silently proceeds with an unconfirmed guess.

        Re-runs capability-based resolution from scratch (see
        resolve_name_to_index_by_capability) rather than caching - a replug
        can change the answer between calls, and re-confirming by capability
        every time is the entire point of this fix, not something to
        shortcut for speed.
        """
        if self._selector_name is None:
            self.resolved_name = None
            self._capability_detail = None
            self._name_confirmed_this_round = False
            return self._selector_index

        index, confirmed_name, detail = resolve_name_to_index_by_capability(
            self._selector_name
        )
        self.resolved_name = confirmed_name
        self._capability_detail = detail
        self._name_confirmed_this_round = True
        return index

    def _print_resolved_device(self) -> None:
        """Print what was actually selected, loudly, at startup and on every
        reopen - a wrong match needs to be obvious immediately, not
        discovered later by reviewing useless footage. This is deliberately
        printed before frame verification: it's "what we resolved," not "what
        we confirmed works" (see verify_startup for that).

        CRITICAL property, and the actual point of the 2026-08-05
        "capability matching" fix: this must NEVER print a device name that
        hasn't been confirmed by actually reading a frame at a matching
        capability through OpenCV itself (see
        resolve_name_to_index_by_capability). A confidently-printed name
        backed by nothing but positional enumeration is exactly the bug
        that let this project silently monitor the wrong physical camera
        while claiming otherwise - this function is the one place that bug
        showed up to a human, so it is the one place that must never repeat
        it, in either direction (name-based selection prints its own
        evidence; index-based selection says plainly that nothing was
        verified).
        """
        if self._selector_name is not None and self._name_confirmed_this_round:
            print(
                f"Camera selected: index {self.index} ({self.resolved_name}) - "
                f"{self._capability_detail}"
            )
        elif self._selector_name is not None:
            # Selected by name, but this round's re-resolution didn't run or
            # didn't succeed (see _reopen) - resolved_name, if set, is a
            # stale value from an earlier successful resolution and must not
            # be presented as currently confirmed.
            print(
                f"Camera selected: index {self.index} - could not confirm "
                f"this is '{self._selector_name}' by capability this round "
                f"(last confirmed identity, if any: {self.resolved_name!r}, "
                f"now stale)."
            )
        else:
            print(
                f"Camera selected: index {self.index} - name NOT verified "
                f"(selected by --index; capability-based identity "
                f"confirmation only runs for --name selection). If this is "
                f"meant to be a specific fixed room camera, prefer --name so "
                f"a replug can't silently swap in the wrong physical camera "
                f"without anyone noticing."
            )

    def _note_delivered_resolution(self, frame, context: str) -> None:
        """Measure what a REAL frame actually is and record it, warning
        loudly if a resolution was requested (request_width/request_height)
        but not fully honoured.

        This is the "verify, don't assume" half of the resolution-request
        feature: cap.set(CAP_PROP_FRAME_WIDTH, ...) is a negotiation, not a
        guarantee, and cap.get() can echo back the requested value even when
        .read() delivers something smaller (the exact lesson this module's
        docstring and probe_index_capability() already encode for capability
        probing - applied here to the live capture path instead). A frame
        smaller than requested is NOT treated as fatal: a slightly-smaller
        capture is still a usable session, so this prints a clear warning
        and lets the caller keep streaming rather than raising.
        """
        if self._request_width is None and self._request_height is None:
            return

        height, width = frame.shape[:2]
        self.delivered_width, self.delivered_height = width, height

        wants_more_width = (
            self._request_width is not None and width < self._request_width
        )
        wants_more_height = (
            self._request_height is not None and height < self._request_height
        )
        if wants_more_width or wants_more_height:
            print(
                f"WARNING ({context}): requested "
                f"{self._request_width}x{self._request_height} capture but "
                f"a real captured frame measured {width}x{height} instead. "
                f"This is a MEASUREMENT (frame.shape), not a report from "
                f"cap.get() - the camera is genuinely not delivering the "
                f"requested resolution (commonly: the requested size needs "
                f"MJPG and didn't get it, or exceeds this device's/USB "
                f"bandwidth's real ceiling). Continuing to stream at "
                f"{width}x{height} rather than failing the session - do "
                f"NOT assume the requested resolution was achieved."
            )

    @staticmethod
    def _should_log_reopen_attempt(attempt: int) -> bool:
        """Rate-limit reopen-attempt logging so a long unplug doesn't spam
        thousands of lines, while keeping the first few attempts fully
        visible so a real replug's recovery is never ambiguous.
        """
        return (
            attempt <= REOPEN_LOG_ALWAYS_FIRST_N
            or attempt % REOPEN_LOG_EVERY_N_ATTEMPTS == 0
        )

    def _reopen(self, attempt: int):
        """Fully tear down and recreate the capture once, re-resolving a
        name-based selection fresh by capability (the whole point: a replug
        can move the device to a different index, or hand it a different
        physical device entirely, between attempts).

        This re-probes EVERY candidate OpenCV index via
        resolve_name_to_index_by_capability on every single reopen attempt,
        not just a cheap check of the last-known index - deliberately, since
        a shortcut here (e.g. "just re-check self.index") would silently
        reintroduce a position-shaped trust assumption for the one
        candidate not re-verified. This costs more per reopen than the old
        system_profiler-based lookup did; that cost is accepted on purpose
        (see docs/decision-log.md, 2026-08-05 "capability matching" entry) -
        correctness of WHICH physical camera this is beats reopen latency
        for a child-safety monitor. If re-resolution fails (e.g. the device
        is genuinely still unplugged), this falls back to retrying the
        last-known index, but marks the identity as UNCONFIRMED for this
        round rather than reasserting a possibly-stale name - see
        _print_resolved_device.

        Returns (True, frame) if the reopened capture actually delivered a
        real frame within REOPEN_VERIFY_ATTEMPTS tries, (False, None)
        otherwise. Checking only `cap.isOpened()` here would repeat the
        exact false-positive this module's STARTUP_VERIFY_* logic exists to
        avoid for the initial open - AVFoundation can report a freshly
        recreated capture as "opened" before its underlying session has
        actually settled.
        """
        verbose = self._should_log_reopen_attempt(attempt)

        if self._selector_name is not None:
            try:
                index, confirmed_name, detail = resolve_name_to_index_by_capability(
                    self._selector_name
                )
                self.index = index
                self.resolved_name = confirmed_name
                self._capability_detail = detail
                self._name_confirmed_this_round = True
            except CameraSelectionError as exc:
                self._name_confirmed_this_round = False
                if verbose:
                    print(
                        f"Warning (reopen attempt {attempt}): could not "
                        f"re-confirm '{self._selector_name}' by capability "
                        f"this round ({exc}). Retrying with last-known "
                        f"index {self.index} in case it's still valid - its "
                        f"identity is UNCONFIRMED until the next successful "
                        f"resolution."
                    )

        if verbose:
            print(f"Reopen attempt {attempt}: tearing down old capture...")

        self.cap.release()
        # Explicitly drop the reference (not just reassign) before the
        # settle sleep - belt-and-suspenders against relying on a capture
        # object whose underlying AVFoundation session hasn't actually
        # finished closing yet.
        self.cap = None
        time.sleep(REOPEN_SETTLE_SECONDS)

        if verbose:
            self._print_resolved_device()

        self.cap = open_capture(
            self.index,
            self._request_width,
            self._request_height,
            self._request_fourcc,
        )

        ok, frame = False, None
        if self.cap.isOpened():
            for _ in range(REOPEN_VERIFY_ATTEMPTS):
                read_ok, read_frame = self.cap.read()
                if read_ok and read_frame is not None:
                    ok, frame = True, read_frame
                    # A resolution request is re-applied on every reopen
                    # (see open_capture() call above) precisely so a
                    # mid-session reconnect can't silently drop back to a
                    # lower resolution unnoticed - measure it here every
                    # time, not just at first startup.
                    self._note_delivered_resolution(frame, context=f"reopen attempt {attempt}")
                    break
                time.sleep(REOPEN_VERIFY_RETRY_DELAY_SECONDS)

        if verbose:
            if ok:
                print(f"Reopen attempt {attempt}: succeeded, frames flowing again.")
            else:
                print(
                    f"Reopen attempt {attempt}: capture reopened but delivered "
                    f"no frame within {REOPEN_VERIFY_ATTEMPTS} tries - device "
                    f"likely still absent. Backing off before retrying."
                )

        return ok, frame

    @property
    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def verify_startup(
        self,
        attempts: int = STARTUP_VERIFY_ATTEMPTS,
        delay: float = STARTUP_VERIFY_RETRY_DELAY_SECONDS,
    ):
        """Confirm the camera actually delivers frames, not just that
        cap.isOpened() returned True. Callers must call this (and check the
        result) before printing any "streaming started" success message -
        isOpened() alone caused ~2 hours of confusion during Phase 2 bring-up
        because it can be True while every subsequent .read() fails.

        Returns (True, frame) on the first successful read within `attempts`
        tries (some UVC cameras need a few reads to warm up), or
        (False, None) if every attempt failed.
        """
        if not self.is_opened:
            return False, None

        for _ in range(attempts):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                self._note_delivered_resolution(frame, context="startup")
                return True, frame
            time.sleep(delay)

        return False, None

    def frames(self):
        """Generator yielding frames indefinitely.

        Contract (every caller, including future phases, must respect this):
        each iteration yields either a real frame, or `None` when the camera
        is currently failing/reconnecting. Callers MUST still execute their
        full loop body on a `None` yield - in particular they must still
        call `cv2.waitKey()` (or equivalent) and check for a quit request -
        just skipping the display/processing step. `cv2.waitKey` is not only
        the keypress handler, it's also the only thing that pumps the OpenCV
        GUI event loop; a caller that only acts on real frames leaves the
        window frozen and unkillable (except via Ctrl+C) for the entire
        duration of a failure, however long that takes. Callers do not need
        to implement any reconnect logic themselves - that all happens
        inside this generator.

        On repeated read failures this releases and reopens the capture
        device with escalating, capped backoff (see REOPEN_* constants) -
        it never gives up permanently, it just keeps retrying, less
        aggressively the longer the outage runs. Callers control when to
        stop by breaking out of their `for` loop (or letting an exception
        propagate); either way the generator's caller is responsible for
        eventually calling `release()`.

        If this camera was selected by name, the index is re-resolved from
        scratch on every reopen (not just at construction) - a mid-stream
        replug is exactly the scenario that can silently move the physical
        camera to a different index, and reopening the *old* index without
        re-resolving would mean the reconnect logic itself starts streaming
        the wrong camera (e.g. the laptop's built-in webcam) without anyone
        noticing.
        """
        consecutive_failures = 0
        reopen_attempt = 0
        reopen_delay = REOPEN_INITIAL_DELAY_SECONDS

        while True:
            ok, frame = self.cap.read()

            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures <= MAX_CONSECUTIVE_READ_FAILURES:
                    print(
                        f"Warning: frame read failed ({consecutive_failures}/"
                        f"{MAX_CONSECUTIVE_READ_FAILURES})"
                    )

                if consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    reopen_attempt += 1
                    reopen_ok, reopened_frame = self._reopen(reopen_attempt)
                    consecutive_failures = 0

                    if reopen_ok:
                        reopen_attempt = 0
                        reopen_delay = REOPEN_INITIAL_DELAY_SECONDS
                        # This frame was already consumed verifying the
                        # reopen worked - yield it instead of discarding it,
                        # then resume the normal read loop next iteration.
                        yield reopened_frame
                        continue

                    time.sleep(reopen_delay)
                    reopen_delay = min(
                        reopen_delay * REOPEN_BACKOFF_MULTIPLIER,
                        REOPEN_MAX_DELAY_SECONDS,
                    )

                # Yield control back to the caller even mid-failure/reopen -
                # see the docstring contract above. Without this, the caller
                # never gets to pump the GUI event loop or notice a quit
                # request for however long the outage lasts.
                yield None
                continue

            consecutive_failures = 0
            yield frame

    def release(self) -> None:
        self.cap.release()


def startup_failure_message(index: int) -> str:
    """Shared failure text for both scripts' "camera never delivered a
    frame" path. Causes are listed in priority order based on what actually
    cost the team time during Phase 2 bring-up (see
    docs/phase-writeups/phase-2.md): the macOS permission issue was by far
    the most time-consuming and least obvious, so it's named first and most
    specifically rather than buried in a generic "check permissions" line.
    """
    return (
        f"Camera at index {index} opened but never delivered a real frame.\n"
        f"Most likely causes, in priority order:\n"
        f"  1. macOS camera permission is granted to your TERMINAL APPLICATION\n"
        f"     (Terminal/iTerm/VS Code/etc.), not to Python itself, and a new\n"
        f"     grant only takes effect after you fully QUIT AND REOPEN that\n"
        f"     terminal app - not just re-running the script. Check System\n"
        f"     Settings > Privacy & Security > Camera.\n"
        f"  2. Index {index} points at the wrong device (AVFoundation indices\n"
        f"     shift when cameras are plugged/unplugged - they are NOT stable\n"
        f"     across reboots or replugs). Run detect_cameras.py to see what's\n"
        f"     currently at each index, or select by --name instead of --index.\n"
        f"  3. Another application (or another instance of this script) is\n"
        f"     already holding the camera open."
    )
