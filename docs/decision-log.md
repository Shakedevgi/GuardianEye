# GuardianEye — Decision Log

One line per real architectural decision, dated, with a one-sentence reason.
Locked decisions from the original project scaffold (CLAUDE.md) are not
re-logged here individually — this log tracks decisions made *during* the
build, phase by phase.

---

### 2026-08-03 — Phase 1

- **venv + `requirements.txt` over conda/poetry.** Simplest, zero-dependency
  option for a two-person student project on a shared codebase; nothing here
  needs conda's binary/env-management muscle.
- **Flat `cv/` directory (no `src/` nesting, no package `__init__.py` yet).**
  Phase 1 has exactly two scripts; premature package structure would add
  ceremony with no payoff until there's enough code to justify it.
- **Camera identified by a plain integer OpenCV device index, resolved by a
  human via `detect_cameras.py`, not hardcoded or auto-detected.** There is no
  reliable cross-platform way to ask "which index is the USB camera" — the OS
  hands out indices in enumeration order, which varies by machine and can
  change if devices are re-plugged. Keeping `DEFAULT_CAMERA_INDEX` a named
  constant (not buried in logic) and exposing `--index` as a CLI flag keeps
  the machine-specific bit machine-specific instead of baked into the code.
- **Reconnect-with-backoff (not crash, not silent hang) on repeated frame-read
  failures.** `stream_camera.py` tolerates up to
  `MAX_CONSECUTIVE_READ_FAILURES` (10) failed reads before releasing and
  reopening the capture device, with a fixed retry delay. This satisfies
  Phase 1's "no crashes for several minutes" bar for transient USB hiccups
  without building a production-grade reconnect strategy that later phases
  don't need yet.
- **`.gitignore` added at repo root** (`.venv/`, `__pycache__/`, `.DS_Store`,
  editor directories) so environment/OS cruft never gets committed as the
  team grows past one contributor's machine.

### 2026-08-04 — Phase 2

- **`yolo26n.pt` (Ultralytics YOLO26, nano size) as the Phase 2 baseline
  model.** Nano is the smallest sufficient model for a detection sanity
  check ("does the pipeline see a person/knife/oven at all"), not peak
  accuracy - model size gets revisited in Phase 3 once real hazard-class
  fine-tuning is in scope and accuracy actually starts to matter.
  **Superseded the next day — see 2026-08-05 below.** This reasoning assumed
  model-size-vs-speed would be the thing worth revisiting "in Phase 3." In
  practice the bring-up session below found the tradeoff that actually
  mattered (input resolution, not model size) within the same phase, and
  nano was replaced as the default before Phase 3 even started. Left here
  rather than deleted so the log shows the actual reasoning path, including
  the part that didn't hold up for even 24 hours.
- **`--device mps` by default, with `--device cpu` as an explicit escape
  hatch**, on this M5 Pro machine. MPS is the fast path, but there are known
  reports of Ultralytics producing visibly wrong detections on MPS vs CPU -
  the escape hatch exists so a bad-looking detection can be triaged in one
  extra word instead of guessing whether the model or the backend is at
  fault.
- **`torch`/`torchvision`/`ultralytics` pinned exactly, departing from
  `opencv-python`'s existing `>=` style.** This stack (YOLO26, torch 2.13,
  Python 3.14/cp314 wheels) is new enough that an unpinned install could
  silently land on an untested combination for the next person who runs
  `pip install`; exact pins make "it works on my machine" reproducible.
- **Frame acquisition extracted into `cv/camera.py`, shared by
  `stream_camera.py` and `detect_stream.py`** instead of each script keeping
  its own copy of the open/read/reconnect-with-backoff loop. Two copies of
  that logic would inevitably drift out of sync as later phases (e.g. the
  Phase 5 rolling buffer) also need frames.

### 2026-08-05 — Phase 2 (bring-up findings)

- **Default model moved from `yolo26n` to `yolo26l`, plus a `--model` flag.**
  Measured on this machine (MPS, 1080p in, imgsz 640): n 159fps, s 145,
  m 76, l 62, x 35. The camera itself caps at 25–30fps, so every size in the
  family already outruns the sensor — nano's speed advantage is headroom that
  cannot be spent, and it was costing 14 mAP for nothing.
- **`--imgsz` flag added (default 640).** Input resolution, not model size,
  turned out to be the dominant factor for small distant objects: scissors
  sitting ~150px tall in a 1080p frame were undetectable by *every* model
  size at imgsz 640 even at conf=0.01, but reached 0.734 confidence on
  `yolo26x` at imgsz 1600. YOLO downscales 1920x1080 to 640 by default,
  destroying exactly the detail that small hazards depend on.
- **AVFoundation camera indices are NOT stable across replugs.** Observed
  first-hand: the Arducam moved from index 1 to index 0 mid-session when
  replugged, silently swapping which physical camera a given `--index`
  refers to. Phase 1's decision log assumed a human-chosen index was stable;
  it is not. For a fixed room camera that must survive reboots, selection
  should eventually be by device name/unique ID, with index as fallback.
  Not yet implemented — logged so it isn't rediscovered painfully later.
- **OpenCV 5 exonerated as a capture-failure cause.** Frames read fine on
  both 5.0.0 and 4.14.0 once the correct device index was used; the
  `opencv-python>=4.9` floor stays as-is.

### 2026-08-05 — Phase 2 follow-up round 2 (live testing found two more bugs)

- **`CameraCapture.frames()` now yields `None` on a failed read instead of
  `continue`-ing silently, and both callers were updated to still call
  `cv2.waitKey()`/check the window every iteration even when the yielded
  value is `None`.** Live testing found that `q` and the window close
  button went completely dead during any read-failure/reconnect period,
  because the old `frames()` never returned control to the caller on
  failure - the caller's loop body (where `cv2.waitKey()` lives) simply
  never ran, and `waitKey` is the only thing that pumps the OpenCV GUI
  event loop as well as reading keys. This is now a documented contract on
  `frames()` itself, since every later phase's caller (Phase 5's rolling
  buffer included) depends on it: skip display on a `None` yield, but never
  skip `waitKey`/quit-checking. Both scripts also now `cv2.imshow()` the
  frame from `verify_startup()` once before entering the loop, so a window
  exists even if the camera dies on its very first post-verification read.
- **Mid-stream reopen now enumerates devices once per attempt (not twice),
  uses escalating capped backoff (not a flat 1s), and verifies the
  reopened capture actually reads a real frame before declaring success
  (not `cap.isOpened()` alone).** Live testing found that unplug/replug
  never recovered in practice even though the reopen code looked correct
  on paper. Reading the code closely against the module's own documented
  Phase 2 lesson ("`isOpened()` can be `True` while every `.read()` fails")
  surfaced that the *mid-stream* reopen path had never actually applied
  that lesson to itself - it trusted `isOpened()` alone right after
  recreating the capture, which is exactly the false-positive this module
  exists to guard against, and is very plausibly why the loop looked
  "stuck" rather than "slow": a reopened-but-not-yet-settled capture would
  report success, reset the failure counter, and then immediately start
  failing reads again from the top, indistinguishable in the logs from
  the original failure. Fixed by reusing `verify_startup`'s "read a few
  real frames, don't trust the flag" approach for the reopen path too, via
  a new `_reopen()` helper. Separately, failed reads on a truly-gone
  device are near-instant (no hardware to wait on), so ten "failures"
  could elapse before the flat 1s delay ever mattered, and each reopen
  attempt shelled out to `system_profiler` twice (once to resolve the
  name, once to reverse-resolve it for the print) - under a fast failure
  loop this could genuinely thrash. Both fixed: one `list_camera_devices()`
  call per attempt, reused for both lookups, and backoff that escalates
  from 1s up to a capped 8s the longer an outage runs, resetting back to
  1s the moment a reopen actually succeeds. Reopen-attempt logging is
  loud for the first few attempts, then throttled to every 5th, so a long
  outage doesn't spam thousands of lines while still leaving `q`/window-
  close observable throughout (see above). **The safety property is
  preserved**: name-based selection still re-resolves the name to a
  (possibly new) index on every single reopen attempt, not just the first.
  **Caveat, stated plainly**: this was verified against a monkeypatched
  fake capture object (see the cv-agent PR/session notes), not real
  hardware - no camera permission was available to physically unplug/
  replug during this fix. The AVFoundation "needs a settle period before a
  freshly recreated `VideoCapture` on that index works" assumption is
  therefore still an assumption, mitigated by `_reopen()`'s post-recreate
  real-frame verification rather than fully proven.

### 2026-08-05 — Phase 2 follow-up (camera-layer fixes)

- **Both scripts now verify a real frame before printing any success
  message.** `CameraCapture.verify_startup()` reads up to
  `STARTUP_VERIFY_ATTEMPTS` (15, up from `detect_cameras.py`'s old
  5-attempt `READ_ATTEMPTS`, which bring-up showed was itself too tight for
  a camera that needed a few extra reads to warm up) real frames before
  either script claims "streaming." `cap.isOpened()` returning `True` says
  nothing about whether frames are actually flowing — that gap cost the
  team ~2 hours during Phase 2 bring-up (see
  `docs/phase-writeups/phase-2.md`). On failure, the new
  `startup_failure_message()` names causes in the priority order bring-up
  actually encountered them: macOS terminal-app camera permission first
  (least obvious, most time-consuming), wrong/shifted index second, another
  process holding the camera third. This is startup validation only — it
  does not touch `CameraCapture.frames()`'s existing mid-stream
  reconnect-with-backoff behavior, which is a separate concern (recovering
  from a hiccup after streaming has already been confirmed working).
- **Camera selection by device name added (`--name`), index (`--index`)
  kept working unchanged; name wins if both are given.** AVFoundation
  indices are not stable across replugs/reboots (observed first-hand: the
  Arducam moved from index 1 to index 0 mid-session) — for a monitor
  specified as a fixed room camera, a silent index shuffle after a reboot
  or replug can mean silently monitoring the wrong physical camera (e.g.
  the laptop's built-in webcam), which is a real safety failure for a child
  monitor, not a cosmetic one. Name wins on conflict because it identifies
  the physical device regardless of what index it currently sits at, which
  is the exact property `--index` can't offer.
- **Name→index resolution happens fresh on every reopen, including
  mid-stream reconnects, not just at startup.** The existing
  reconnect-with-backoff path used to reopen the same integer index it
  started with; if a replug is what triggered the reconnect in the first
  place, that index may now point at a different physical camera, meaning
  the reconnect logic itself could silently swap the room camera for the
  webcam without anyone noticing. `CameraCapture` now stores *how* it was
  selected (name or index) and, when selected by name, re-runs name
  resolution before every reopen — so a replug is handled the same way
  whether it happens before startup or mid-session.
- **Device-name enumeration via `system_profiler SPCameraDataType`, not
  `pyobjc-framework-AVFoundation`.** Considered enumerating through
  AVFoundation directly, since that's the same underlying framework
  OpenCV's own macOS backend uses (a stronger guarantee than two unrelated
  tools happening to agree) — but that would add a new macOS-only pinned
  dependency for a two-person student project, and `system_profiler`
  (already on every Mac, no install required) was empirically verified
  during bring-up to agree with OpenCV's index ordering for this team's
  three real devices (Arducam, MacBook Pro Camera, iPhone Continuity
  Camera). This is a positional correspondence, not a documented contract —
  `camera.list_camera_devices()`'s docstring states the assumption
  explicitly, and both scripts still verify a name-resolved index by
  reading an actual frame rather than trusting the name match alone.
  Switching to pyobjc is the documented escape hatch if this assumption is
  ever observed to break. Enumeration degrades to `[]` (index-only
  selection) on non-macOS platforms or if `system_profiler`'s output can't
  be parsed.
  **This entire approach — and its documented escape hatch — failed in the
  field the very next day. Superseded below (2026-08-05, "capability
  matching"); left here, not deleted, so the record shows the reasoning
  that didn't hold up, not just the fix.**

### 2026-08-05 — Phase 2, "capability matching" (positional name→index
resolution shipped, then failed in the field)

- **What happened.** Running `detect_stream.py --name Arducam ...`, the
  script printed `Camera selected: index 1 (Arducam-B0560-4K HDR)` and then
  streamed the MacBook's built-in webcam, not the Arducam. For a monitor
  whose entire premise is watching one fixed room, confidently asserting the
  wrong physical camera's name is worse than not having a name feature at
  all — nobody would think to distrust text that says "success."

- **Measured, on the user's machine, at the time of the failure:**

  | enumeration source | order |
  |---|---|
  | `system_profiler SPCameraDataType` | `0 MacBook Pro Camera`, `1 Arducam-B0560-4K HDR`, `2 iPhone Camera` |
  | pyobjc `AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)` | `0 MacBook Pro Camera`, `1 Arducam-B0560-4K HDR`, `2 iPhone Camera` — identical ordering, each with a distinct `uniqueID` (values not recorded here: they're persistent per-device hardware identifiers, including for a team member's personal phone, and the ordering is the finding — the IDs add nothing to it) |
  | **OpenCV's actual index space** (confirmed by the user visually, repeatedly) | **index 0 = Arducam, index 1 = MacBook webcam** |

  Both independent enumeration sources agreed with EACH OTHER and both
  disagreed with OpenCV. Earlier the same day, all three had agreed
  (Arducam first, everywhere) — the AVFoundation-side ordering shifted
  between sessions while OpenCV's did not. **This directly falsifies the
  previous entry's documented escape hatch**: switching `list_camera_devices()`
  from `system_profiler` to pyobjc would NOT have fixed this, since pyobjc
  produced the identical wrong order. Two tools agreeing with each other was
  never evidence they matched OpenCV — position is not a reliable proxy for
  OpenCV's index space, full stop, regardless of how it's obtained. This
  generalizes past this one bug: no future "let's enumerate a different way"
  idea should be trusted for this purpose without a fresh empirical check
  against OpenCV itself, on the actual machine, at the actual time of use.

- **The fix: identify a device by CAPABILITY, not position.** The physical
  devices are distinguishable by what they can actually deliver — a
  property of the hardware, not of enumeration order. Measured maximum
  resolutions (via pyobjc, reading `AVCaptureDevice.formats()`) on this
  machine: Arducam 3840x2160, MacBook Pro Camera 1552x1552 (its widest
  active format is a square crop, not classic 16:9 — notably its max pixel
  *area* is actually larger than 1080p's, so "1080p vs 4K" was too coarse a
  simplification; full max-resolution dims are compared instead), iPhone
  Continuity Camera 1920x1440. `camera.named_device_capabilities()` looks up
  a NAMED device's expected max resolution via pyobjc (capability query,
  never positional enumeration); `camera.probe_index_capability()` opens
  each OpenCV index directly, requests an unreachable 7680x4320, reads back
  a real frame, and reports the actual delivered resolution — the device's
  true ceiling, observed through OpenCV itself. `resolve_name_to_index_by_capability()`
  matches the two (within a documented tolerance, `CAPABILITY_MATCH_AREA_TOLERANCE`)
  and refuses to guess — raising a specific, honest `CameraSelectionError`
  for every non-unambiguous outcome (name not connected, name ambiguous
  among multiple devices, no index confirms the capability, or more than
  one index does).
- **`CameraCapture` never prints an unverified name again.** Selecting by
  `--index` alone now prints "name NOT verified" explicitly rather than a
  best-effort positional hint — the false confidence, not the absence of a
  name, was the actual bug. Selecting by `--name` prints the exact evidence
  behind a match (requested resolution vs. what the matched index
  delivered), and re-runs the full capability resolution on every reopen
  (mid-stream reconnects included), not just at startup, accepting the
  extra latency this costs over the old positional lookup as a deliberate
  tradeoff — correctness of which physical camera this is matters more than
  reopen speed for a child-safety monitor.
- **`list_camera_devices()` and `resolve_name_to_index()` (system_profiler-
  based, positional) were removed from `camera.py`**, not just deprecated —
  leaving a proven-unsafe positional shortcut callable in the codebase was
  judged a bigger risk (a future caller reintroducing it) than losing a
  system_profiler-only fallback with no proven purpose left to serve.
  `detect_cameras.py` was rewritten the same way: it no longer pairs a name
  with an index by print order: it lists each OpenCV index's actual probed
  capability and each AVFoundation-known name's expected capability
  side-by-side, and tells the human to match them by comparing the numbers,
  not by position.
- **New dependency: `pyobjc-framework-AVFoundation==12.2.1`, macOS-only,**
  pinned exactly in `cv/requirements.txt` with a `sys_platform == "darwin"`
  marker (confirmed to install cleanly under this project's pinned Python
  3.14/cp314 combination; 12.2.1 was the latest version resolvable at
  pinning time). Used strictly for capability queries on a named device,
  never for positional enumeration — see the module docstring in
  `camera.py` for why that distinction is load-bearing this time, not
  cosmetic.
- **What was and wasn't verified for this fix.** Verified directly: the
  AVFoundation-side name→capability lookup returns the real, distinct
  per-device max resolutions above (re-run and confirmed, not just quoted
  from the original incident). Verified: the not-found, ambiguous-by-name,
  and capability-lookup-unavailable error paths all raise the intended,
  specific message. **Not verified: actually reading a real frame through
  OpenCV on this development machine**, whether during the original
  incident's follow-up or in later testing — this environment has no
  camera permission granted to it, so `cv2.VideoCapture(...).isOpened()`
  returns `False` for every index here, and the "no OpenCV index confirms
  this capability" error path is what was actually exercised, not the
  successful-match path. The `CAPABILITY_MATCH_AREA_TOLERANCE` tolerance
  value is therefore still unvalidated against a real device delivering a
  real frame at the requested resolution — someone with camera permission
  on real hardware needs to confirm `--name Arducam` resolves to the
  correct index and prints the expected match evidence before this is
  fully trusted, not just logically sound.

### 2026-08-05 — Phase 2, capability matching: live hardware confirmation
(post-sign-off docs-agent audit update)

- **The gap flagged immediately above is now closed for the core claim.**
  Per the orchestrator: the user has run both `detect_cameras.py` and
  `detect_stream.py --name Arducam` on real hardware and confirmed both
  work correctly, including that the resolved camera identity is actually
  right, not just that a match was printed. This is genuine progress from
  "logically sound" to "demonstrated" for the specific failure mode this
  redesign exists to prevent. Docs-agent has no camera/shell access and did
  not watch this run directly, so this line records what was reported, not
  an independent confirmation — logged here rather than left buried in a
  chat transcript because it changes the honest status of a previously
  flagged open risk.
- **Still NOT confirmed, and not the same claim as the one above: a real
  physical unplug/replug recovering correctly under the current
  capability-matching reopen path.** The Round 2 entry above ("follow-up
  round 2") states its reopen fix was verified against a monkeypatched fake
  capture object, not real hardware. Nothing since has claimed a real
  unplug/replug test against the current capability-matching `_reopen()`.
  Cold-start name resolution and mid-stream reconnect exercise the same
  underlying logic under different conditions (one confirmed candidate
  index at rest, vs. a full re-probe of every candidate index while frames
  were already flowing and something has just gone wrong) — proving one
  does not prove the other.
- **New risk, found during this docs-agent audit, not previously flagged by
  anyone.** A full `--name` reopen now re-runs the entire capability probe
  (up to `CAPABILITY_PROBE_MAX_INDEX = 6` OpenCV indices, up to 2 FOURCC
  settings each, up to `CAPABILITY_PROBE_READ_ATTEMPTS = 10` reads each at
  0.2s apart) inside a single blocking call to `_reopen()`. Worst case this
  is on the order of tens of seconds, during which the caller's loop —
  where `cv2.waitKey()` lives — never runs. This means the Round 2 fix
  ("`q`/window-close keep working during a failure") does not apply
  *during an active reopen attempt* for `--name` selection specifically;
  it applies before the first reopen attempt and in the gaps between
  attempts, not during one. This is a narrower recurrence of the exact bug
  Round 2 fixed, reintroduced as a side effect of Round 3's heavier
  identity check, and as far as docs-agent can tell nobody has weighed in
  yet on whether to accept it, bound the probe sweep, or otherwise address
  it. See `docs/phase-writeups/phase-2.md`, "After sign-off" section, for
  the full writeup. This is a flagged risk, not a decision either way.
