# cv/ - camera capture + detection (Phase 1 + Phase 2)

Owned by cv-agent. Camera capture (Phase 1) plus stock YOLO/COCO detection on
the live feed (Phase 2). No custom hazard classes, no risk/proximity logic,
no alerts yet - those come in later phases per `PHASE_PLAN.md`.

## One-time setup

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r cv/requirements.txt
```

`torch`/`torchvision`/`ultralytics` are pinned exactly (see
`cv/requirements.txt` for why). The first run of `detect_stream.py` downloads
the requested weights (a few MB to ~130MB depending on size) into
`cv/models/` - that directory and `*.pt` files are gitignored, so each machine
fetches its own copy rather than the weights living in git.

## Find your USB camera (2026-08-05: identified by capability, not position)

OpenCV addresses cameras by a plain integer index, and — this matters for a
fixed room camera — **that index is not stable**. AVFoundation reassigns
indices on replug/reboot; during Phase 2 bring-up the same physical Arducam
moved from index 1 to index 0 mid-session (see
`docs/phase-writeups/phase-2.md`). For a monitor that's supposed to watch one
fixed room unattended, a silent index shuffle can mean silently monitoring
the laptop's webcam instead. Prefer selecting by **name**, not index.

**Important history, because it changes what "select by name" actually
means here:** the first version of name-based selection matched a device
name to an OpenCV index *positionally* (via `system_profiler`'s listing
order). That shipped, then failed in the field: on a later session,
`system_profiler`'s order and a second, independent enumeration
(pyobjc/AVFoundation) agreed with EACH OTHER, and both still disagreed with
OpenCV's actual index order — the script confidently printed "index 1
(Arducam-B0560-4K HDR)" while streaming the MacBook's built-in webcam
instead. Two enumeration sources agreeing was never evidence they matched
OpenCV. See `docs/decision-log.md`'s 2026-08-05 "capability matching" entry
for the full measured writeup.

Name-based selection now works differently: it identifies a device by what
it can actually **do** (its maximum resolution, a hardware property, queried
via `pyobjc-framework-AVFoundation`), then opens every OpenCV index directly,
asks each for an unreachable resolution, reads back a real frame, and
matches the two by capability - never by list position. If it can't confirm
a unique match, it says so and refuses to guess, rather than asserting a
name it hasn't verified.

```bash
python cv/detect_cameras.py
```

It probes indices 0–5 directly through OpenCV (opens each, requests an
unreachable resolution, reads back a real frame) and reports the actual
delivered resolution per index - the device's true capability, not a
default/negotiated mode. Separately, it lists every currently-connected
camera's name (via AVFoundation) alongside ITS true capability. **It does
NOT pair a name with an index by print order** - match them yourself by
comparing the resolution numbers (e.g. "index 0 delivered 3840x2160" next
to "'Arducam-B0560-4K HDR' - max resolution 3840x2160" is a real match;
their position in either list is not evidence of anything). `--name` in the
scripts below performs this same match automatically and refuses to guess
if it's ambiguous.

## Stream raw video (Phase 1, no detection)

```bash
python cv/stream_camera.py --name Arducam       # preferred: survives replugs
python cv/stream_camera.py --index <index>      # still works, not replug-safe
```

A window opens showing the live raw feed. Quit with `q` (window focused) or
by closing the window.

If neither `--index` nor `--name` is given, it defaults to index `0`, which
on a laptop is usually the built-in webcam, not the USB camera. If both
`--index` and `--name` are given, `--name` wins (a name identifies a physical
device regardless of which index it currently sits at — that's the whole
point). The script now reads and verifies a real frame before printing any
"streaming" success message, and prints the resolved device index + name +
confirmed resolution once it has (e.g. `Streaming from camera 0
(Arducam-B0560-4K HDR) at 1920x1080`) — a camera that opens but never
delivers frames (dead USB connection, macOS permission not yet granted to
this terminal app, wrong index) now fails loudly with the likely causes
listed, instead of a false "success" message.

## Stream with detection (Phase 2)

```bash
python cv/detect_stream.py --name Arducam       # preferred: survives replugs
python cv/detect_stream.py --index <index>      # still works, not replug-safe
```

Same capture loop as `stream_camera.py`, plus a YOLO26 pass on every frame:
boxes, class labels, and confidence scores are drawn for every stock COCO
class the model finds (not filtered to a "hazard subset" - the point of
Phase 2 is confirming the model sees whatever is actually in the room). The
overlay shows a live FPS readout plus the running config, so you can always
tell which settings produced what you're looking at.

Keys (window must be focused): `q` quits, `s` saves the current frame to
`cv/captures/` - both raw and annotated. Frames where detection fails are the
raw material for Phase 3's fine-tuning dataset, so save them as you find them.

Flags:

- `--index` / `--name` - same as `stream_camera.py` (default `--index 0` if
  neither is given; `--name` wins if both are given).
- `--device` - `mps` (default, Apple GPU on this machine) or `cpu`. There are
  known reports of Ultralytics producing visibly wrong detections on MPS vs
  CPU, so `--device cpu` is a one-word A/B to rule that out.
- `--conf` - minimum confidence to draw a box (default `0.25`).
- `--model` - which YOLO26 weights to run (default `yolo26l.pt`; sizes
  `n`/`s`/`m`/`l`/`x`). Measured here at imgsz 640: n 159fps, s 145, m 76,
  l 62, x 35. The camera caps at 25-30fps, so every size already outruns the
  sensor - prefer a larger one unless a later phase needs the frame budget.
- `--imgsz` - inference resolution (default `640`). This matters more than
  model size for small distant objects: YOLO downscales a 1920x1080 frame to
  a 640px square, and scissors ~150px tall were undetectable by *every* model
  size at 640, but reached 0.734 confidence on `yolo26x` at `--imgsz 1600`.
  Raising it costs frame rate (x drops to ~7fps at 1600).

For the highest-quality detection this stock model can do:

```bash
python cv/detect_stream.py --name Arducam --model yolo26x.pt --imgsz 1600
```

If MPS isn't actually available on the machine you're running on, the script
prints a warning and falls back to CPU automatically rather than crashing.

`camera.py` holds the shared frame-acquisition logic that both
`stream_camera.py` and `detect_stream.py` build on: open (by index or by
re-resolved name), read loop, consecutive-failure counting,
reopen-with-backoff (re-resolving a name-based selection by capability on
every reopen, not just at startup - see below), and startup frame
verification (a real frame must be read before either script is allowed to
print a success message — see `docs/phase-writeups/phase-2.md` for why that
distinction mattered in practice). `detect_cameras.py` uses
`camera.named_device_capabilities()` and `camera.probe_index_capability()`
to report device names/capabilities and per-index capabilities
independently (macOS + `pyobjc-framework-AVFoundation` for the former).

### Camera identity is confirmed by capability, not position (2026-08-05)

`--name` does NOT trust any enumeration source's listing order to mean
anything about OpenCV's index space — that was tried, shipped, and observed
to fail in the field (see `docs/decision-log.md`, 2026-08-05 "capability
matching" entry, for the measured incident). Instead:

1. `camera.named_device_capabilities(name)` asks AVFoundation (via
   `pyobjc-framework-AVFoundation`, macOS-only) what resolution ceiling the
   NAMED device supports - a property of the hardware, not of enumeration
   order.
2. `camera.probe_index_capability(index)` opens each OpenCV index directly,
   requests an unreachable resolution, reads back a REAL frame, and reports
   what that index actually delivers.
3. The two are matched by capability. If the match is unambiguous, the
   index is used and the exact evidence (requested vs. delivered
   resolution) is printed. If it's NOT unambiguous - the name isn't
   currently connected, the name matches more than one device, no OpenCV
   index confirms the expected capability, or more than one does - the
   script refuses to guess and prints exactly why (`CameraSelectionError`).

This re-runs in full on every reopen, including mid-stream reconnects after
a replug, not just at startup - re-confirming by capability every time is
the entire point, so this is not shortcut for speed. Plain `--index`
selection now explicitly prints "name NOT verified" rather than a
best-effort name hint - identity is either confirmed with evidence, or
stated plainly as unconfirmed. Never both silently blurred, which is what
caused the original incident.

If `pyobjc-framework-AVFoundation` isn't installed, or this isn't macOS,
`--name` fails with a clear message telling you to use `--index` instead -
it does NOT fall back to any positional guess.

### Unplug/replug and quitting mid-failure (2026-08-05 fix)

Live testing surfaced two bugs in the above, both now fixed:

- **`q` and the window close button now work even while the camera is
  failing or reconnecting.** Previously the read-failure path never handed
  control back to the script's loop, so `cv2.waitKey()` — which both reads
  keys and pumps the OpenCV window's event loop — never ran during a
  failure, freezing the window until Ctrl+C. `CameraCapture.frames()` now
  yields `None` on a failed read so both scripts can keep pumping the GUI
  and checking for quit throughout an outage, however long it runs.
- **Unplug/replug recovery is more robust and more observable.** Reopen
  attempts now verify a real frame is actually flowing before declaring
  success (not just `cap.isOpened()`, which can be `True` before an
  AVFoundation capture has actually settled), enumerate the device list
  once per attempt instead of twice, and back off on an escalating,
  capped schedule (1s up to 8s) instead of a flat 1s — so a long unplug
  doesn't spin the CPU or spam `system_profiler`. Reopen attempts print
  loudly for the first few tries, then every 5th, so it's always possible
  to tell from the console whether a reopen is being attempted and whether
  it's succeeding.
- This was verified with a monkeypatched fake camera, not real hardware —
  see `docs/decision-log.md`'s 2026-08-05 "round 2" entry for exactly what
  was and wasn't proven.
