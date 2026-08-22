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
- **NOW CONFIRMED (2026-08-05, superseding the caveat below): a real
  physical unplug/replug recovers correctly under the current
  capability-matching reopen path.** The user performed the physical test
  and reported it working. This closes the last unverified path in the
  camera layer — every safety-relevant claim about camera identity is now
  backed by a test on real hardware rather than by a mocked simulation.
  The original caveat is kept below so the log shows what was and wasn't
  known at the time, rather than reading as if it had always been verified.
- ~~**Still NOT confirmed, and not the same claim as the one above: a real
  physical unplug/replug recovering correctly under the current
  capability-matching reopen path.**~~ The Round 2 entry above ("follow-up
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

### 2026-08-05 — Phase 3 scoping (decided before the phase starts)

- **Labelling and training stay entirely local; no cloud upload of training
  images.** `PHASE_PLAN.md` previously said "Roboflow or similar." Training
  images for this project are photographs of a real home and, unavoidably,
  of a child. CLAUDE.md decision 8's privacy requirement is written about
  event logs and video clips rather than training data, so this isn't
  strictly a violation — but uploading family photos to a third-party
  service plainly contradicts its spirit, and the gap existed only because
  nobody had thought about training data when decision 8 was written.
  Locally-run labelling (Label Studio / self-hosted CVAT) plus Ultralytics
  fine-tuning on the M5 Pro via MPS means no stage of Phase 3 needs cloud.
- **Phase 3 starts by measuring, not collecting.** Phase 2's late discovery
  that input resolution dominates small-object detection invalidated every
  earlier "COCO can't detect this" observation, since they were all taken at
  imgsz 640. The list of classes actually needing fine-tuning is therefore
  unknown until re-tested at `yolo26x --imgsz 1600`. Collecting and
  labelling before producing that list risks spending the phase's effort on
  classes the stock model already handles.
- **Dataset must span varied camera placement.** There is no single
  permanent mounting position — the camera moves between testing areas at
  different heights and angles. A dataset shot from one fixed setup would
  train for a scene that rarely recurs.

### 2026-08-09 — Phase 3 Step 0 measurement (findings, not yet decisions)

- **Step 0 measures offline on raw frames, not by re-shooting live.**
  `--imgsz` only affects inference; the frame `detect_stream.py` writes on
  `s` is always full-resolution. So one live capture can be re-measured at
  any model/resolution/confidence afterwards. Built
  `cv/measure_detection.py` to do this: sweeps model x imgsz x conf over
  `cv/captures/*_raw.jpg`, one CSV row per detection. It also serves Phase
  3's later regression check, where the fine-tuned model has to be compared
  against this stock baseline on identical frames.
- **Measurement runs at conf 0.01, not 0.25.** Every capture before this was
  taken at 0.25, which cannot tell "the model emitted nothing" from "the
  model emitted a 0.04 box we discarded." Those need different remedies, so
  the distinction has to be measurable.
- **`person` is degraded by high resolution, not improved by it.** Across 10
  frames, `person` scores 0.90-0.98 at imgsz 640 and collapses to 0.03-0.37
  on half of them at imgsz 1600. This reverses
  `docs/phase-3-scoping-notes.md`'s "person detection is weaker than
  expected" — that observation was real but was measured only at 1600, the
  one setting that breaks it. Consequence: the CLAUDE.md decision 2
  single-pass tension flagged in Phase 2's write-up is now confirmed by
  measurement rather than hypothesised — `person` wants 640-960, small
  hazards want 1280-1600, and a single pass at either sacrifices a layer.
  **Flagged for Phase 4, deliberately not resolved here.**
- **`knife` detection is anti-correlated with danger.** Held in a hand it
  scores 0.48-0.885; resting unattended on a surface or shelf it scores
  0.011-0.049. The hazard this project exists to catch is the unattended
  knife within reach, which is precisely the failing case.
- **`scissors` is unstable rather than absent.** Five frames of the same
  scissors in the same spot score 0.029-0.795 at the same resolution. One
  genuine hard failure too: white scissors on a white wall top out at 0.045
  at any resolution. This corrects the scoping notes' stronger claim that
  scissors were a flat "complete miss."
- **Electrical (sockets, power strips, adapters, trailing cables) is a true
  absence.** No COCO class, and checked for a substitute mislabel: one
  spurious `remote` at 0.012 across the whole sweep, i.e. noise. So it is
  "no class," not "consistently misnamed as something else."
- **Recommendation logged, decision deferred to the orchestrator: CLAUDE.md
  decision 7 needs a narrow amendment.** It sorts classes into "COCO already
  fits it" vs "COCO lacks it," and names `knife` and `scissors` as examples
  of the former. Measurement puts both in a third bin decision 7 has no room
  for: COCO has the class and detects it too unreliably to alert on. The
  proposed change is to the sorting criterion only — from "does COCO have
  this class?" to "does COCO detect it reliably in this deployment?" The
  hybrid approach and "no training from scratch" are unaffected.
- **Not measured, and explicitly not to be assumed:** `oven` and `sink`
  (decision 7 names `oven` as already-fitting; no capture contains one and
  neither exists in the accessible testing areas, so that claim is untested,
  not confirmed), stairs, small choking hazards, chemical/medicine bottles,
  and any toddler-sized subject at room distance. All `person` numbers above
  are an adult at close range.

### 2026-08-09 — CLAUDE.md amended (Shaked, this session)

- **Decision 7 amended: the fine-tuning test is reliability, not availability.**
  It previously sorted classes by whether COCO knew them, and named `knife`
  and `scissors` as examples of classes that already fit. Step 0 measured both
  and neither does. New wording splits classes into three measured bins;
  `knife` and `scissors` move into the fine-tuning set. This makes Phase 3
  larger than originally scoped — accepted deliberately, because a knife
  resting on a counter scoring ~0.03 means the alert never fires in the case
  the product exists for, and no confidence threshold fixes that.
  `oven` and `sink` recorded as unverified rather than assumed working.
- **CLAUDE.md's decisions are explicitly changeable, with a required
  procedure.** The header said "locked — do not silently change," which read
  as "do not change." Shaked's correction: the decisions are engineering
  calls, not scripture — the actual requirement is to ask first and log the
  change, never to alter one silently. Header and preamble updated to say
  that. Questioning a decision when evidence contradicts it is now stated as
  expected behaviour rather than something to be avoided.

### 2026-08-09 — Phase 3 Step 0, session 2 (21 live captures)

- **Small choking hazards are confidently MISLABELLED, not missed.** A car key
  in an open palm reads as `cell phone` at 0.92; a lighter as `cell phone`
  0.47-0.60; small objects on the floor as `sports ball` up to 0.74. This is
  worse than a miss: a gap in the hazard map is visibly empty, but a
  high-confidence benign label fills it with a "phone" the risk engine has no
  reason to flag. **Fine-tuning these classes must therefore also suppress the
  competing COCO labels**, not merely add new ones — which changes how the
  training set has to be labelled.
- **This does NOT generalise to the electrical gap.** Sockets/power strips
  remain a true absence with no substitute mislabel (one stray `remote` at
  0.012). The two "COCO has no class" cases behave differently and should not
  be treated as one bucket.
- **`knife` failure confirmed across two independent sessions.** Max knife
  confidence anywhere in session 2 was 0.244, on a frame where a person is
  reaching for it; staged resting-knife shots scored 0.012-0.186. Nothing
  clears the 0.25 default. No longer provisional.
- **`person` confirmed at imgsz 640** — 0.94-0.977 across crouching, reaching
  and partially-occluded poses, reproducing session 1's inversion finding.
  Frame 102954 is the phase in miniature: a person reaching toward a knife on
  a ledge scores `person` 0.977 and `knife` 0.244.
- **`--4k` did not take** — all 21 frames are 1920x1080. Cause not yet
  diagnosed; the flag was verified only against a fake camera.
- **Block D (camera height/angle variation) not shot** — USB cable too short to
  reposition the camera. Deferred until an extender is available. This is a
  real gap against PHASE_PLAN's requirement that the dataset span placements,
  not an optional extra.

### 2026-08-09 — Phase 3 Step 0, session 3 (4K, two camera heights)

- **`--4k` confirmed working on real hardware** (12 frames at 3840x2160). The
  previous session's 1080p output was a forgotten flag, not a bug.
- **Capture at 4K, infer at 640-1600. Never above 1600.** Raising imgsz to
  2560 to "match" 4K capture degrades nearly everything: microwave 0.950 ->
  0.126, oven 0.872 -> 0.261, person 0.959 -> 0.075. YOLO26 is trained at 640
  and large departures from that scale hurt. The extra capture pixels are
  worth keeping for the dataset - they are detail to label and train on, and
  cannot be recovered later - but they are not an inference setting.
- **`knife` failure is a CONTEXT failure, not a size/resolution one.** On a
  kitchen counter beside appliances it reaches 0.827. On a floor it scores
  0.06-0.29. Frame 104604 controls every other variable and settles it: 4K,
  camera at 60cm, large unoccluded red knife beside a reaching hand - `knife`
  0.214, while `scissors` in the *same frame* hits 0.975. COCO knows knives in
  kitchens; it does not know a knife on a floor. This is precisely the
  emergency case, so the anti-correlation with danger is now explained rather
  than merely observed.
- **`oven` measured for the first time and decision 7's claim holds for it** -
  0.872 as `oven`, 0.965 as `microwave`, consistently across five frames. It
  fires as *both* classes simultaneously, so Phase 4 should treat oven/
  microwave as one hazard group rather than distinct classes. Caveat: this is
  a countertop toaster oven; a floor-level built-in oven is still unmeasured.
  `sink` remains unverified (max 0.085, none present in any frame).
- **Camera height 60cm is viable** - `person` 0.913-0.956 at toddler eye level,
  so a low mount costs Layer B nothing.

### 2026-08-09 — Phase 3 class list merged (Shaked)

- **`knife` + `scissors` merge into one `sharp_object` class.** Step 0b
  measured that both stock COCO and open-vocabulary models confuse the two on
  floors: on frame 104604 the model labelled the knife `scissors` at 0.78 with
  `knife` explicitly offered as a competing prompt. Merging removes a
  discrimination we have measured we cannot win, and doubles the training
  examples per class. The parent-facing alert ("sharp object near your child")
  is unchanged in usefulness. Cost: the system can no longer report which kind
  of sharp object - accepted deliberately, since the specific label is the one
  we cannot reliably deliver.
- **All small choking hazards merge into one `small_swallowable` class**
  (keys, coins, batteries, bottle caps, lighters, small toy parts). Same
  reasoning: one hazard, one alert, and five sparse classes become one
  well-populated one.
- **Electrical is removed from the fine-tuning scope entirely.** Step 0b
  measured YOLO-World's `wall socket` prompt at 0.48-0.90 across every
  electrical frame, against stock COCO's zero - no training data needed. It
  runs in ~30ms so it is live-viable. It must run as an ADDITIONAL detector
  alongside stock YOLO26, never as a replacement: open-vocab `person` clears
  0.5 in under half of frames and must never touch Layer B.
- **Net effect: Phase 3's labelling scope drops from six-plus classes to two.**

### 2026-08-10 — Round 1 fine-tune fails to generalise; VLM safety net rejected

- **Measured generalisation failure.** The round-1 model (92 office frames)
  scores precision 0.246 / recall 0.147 on 32 labelled frames from a second
  building, against 0.902 / 0.607 on its own validation split. It memorised one
  room. No confidence threshold recovers it.
- **A leak-free train/val split is necessary but not sufficient.** The
  duplicate-group split was correct and still could not detect this - 13
  validation images from the same building cannot distinguish learning from
  memorisation. Only a second location can. Any future claim about model
  quality must be measured on a held-out *location*, not a held-out split.
- **Earlier decisive-frame numbers were measured on training-building frames**
  (floor knife 0.861, car key 0.930) and were presented as evidence the
  approach generalises. They were not. Corrected in
  `docs/phase-3-step0-findings.md`.
- **The 32 home frames are now the held-out test set** - fully labelled, from an
  independent building. Every future change is measurable against 0.246/0.147.
- **Local VLM (Florence-2) rejected as Layer A's safety net.** Its prompted
  grounding has no absence case: asked for "banana" on a wall-socket frame it
  returns a confident, tight box on the socket labelled banana; asked for
  "knife" on a car-key frame it boxes the key. Four absent-object probes, four
  confident boxes. A presence check that cannot say no is unusable for safety.
  Unprompted discovery is honest and produced one real win (correctly naming a
  floor-level power outlet, which COCO cannot), but is sparse and mislabelled
  the floor knife as scissors. Phase 4 builds on fine-tuned YOLO alone.
- **Plan revised:** home frames move into training; a third location becomes the
  new test set; public data (Open Images `Knife` + `Scissors` -> class 0
  `sharp_object`) is pulled for the visual diversity two buildings cannot
  supply, trained two-stage so our own frames get the last word.

### 2026-08-10 — Reframe: detect objects-on-surfaces, not named classes (Shaked)

- **The detection problem was the wrong shape.** Three days went into
  classifying named objects (`sharp_object`, `small_swallowable`), reaching
  precision 0.179 / recall 0.216 on held-out frames. Shaked's correction: the
  requirement was never "is that a knife" - it is "**is there something on a
  reachable surface that shouldn't be there.**" A knife, a coin, a bottle cap, a
  lighter, or an object nobody has thought of are all the same signal.
  Classification fails on anything outside its training set, which also makes
  "add hazards as we think of them" impossible; anomaly detection does not have
  that failure mode. **The orchestrator carried CLAUDE.md decision 7's class-list
  framing forward without asking whether classification was the right shape for
  the actual requirement — that question should have preceded the class list.**
- **Reachable surfaces, not just floor** - floor, low tables, low shelves.
  Objects on a coffee table are in scope (visible in the home frames).
- **Fixed hazards are in better shape than the class-list work suggested:**
  `oven` already measures 0.872 on stock COCO; `wall socket` measures 0.48-0.90
  via YOLO-World prompting; "fan/heater running" is a *motion* signal rather
  than an object-detection one; an open window is static once located. Three of
  the four named fixed hazards are close to solved and were filed away only
  because the phase was chasing a class list.
- **Two-input risk model (Shaked) - supersedes pure proximity for Phase 4.**
  Severity has two independent axes, not one:
  - **Base risk from accessibility** - an object on the *floor* is available to
    a crawling child at any moment (high); the same object on a *low table*
    needs reaching or climbing (medium).
  - **Escalation from the child** - proximity and reaching raise either
    baseline to red.
  CLAUDE.md decision 5 currently describes only the proximity axis. This adds
  the surface-height axis, which sets a hazard's risk level *before* the child
  is anywhere near it. **Not yet an amendment to decision 5 - recorded here as
  the intended Phase 4 design, to be raised with Shaked and Yahli before
  changing CLAUDE.md.**
- **Useful consequence:** the surface segmentation needed to find
  objects-on-surfaces also yields *which* surface each object sits on - so the
  detector that finds a hazard also supplies its base risk level, at no extra
  cost.
- **Not wasted:** the 124 labelled frames, the 32-frame held-out test set, the
  verified IoU scorer, and Step 0's findings all carry over and are how the new
  approach gets evaluated. What changes is the target, not the measurement
  infrastructure.

### 2026-08-10 — CLAUDE.md decision 7 rewritten: find objects, don't name them (Shaked)

- **Approved by Shaked.** The old wording ("fine-tune a small number of custom
  classes") is replaced. The goal is now stated as: **find anything on a surface
  a child can reach, whether or not we can name it.** Naming is a bonus, not the
  requirement.
- Detection order of preference is now written into the decision:
  class-agnostic first (change detection works and is ~1.4ms/frame; the
  already-there-at-boot case is still unsolved), named classes where they
  already work without training (`person`, `oven`, `chair`, `refrigerator`,
  wall sockets via prompting), fine-tuning only where those fall short and only
  after measuring.
- **Added as a standing rule:** the test for any detector is measured
  performance in a building the model has not seen. A leak-free train/val split
  is necessary and not sufficient - only a held-out *location* detects a model
  that memorised one room. This is the lesson that cost a day.
- Also recorded: some hazards are not object detection at all ("fan running" is
  motion; an open window is a fixed region once located) and should not be
  forced into a class list.
- Shaked's direction: **keep training AND change the approach** - the mixed
  classifier run continues, and class-agnostic detection is developed alongside
  it rather than replacing it outright.

### 2026-08-11 — Round 3 (mixed + oversampling) is the worst run; oversampling by duplication ruled out

- Mixed training with our 79 office frames repeated 14x scored **pooled recall
  0.017 / precision 0.133** on the held-out home frames, against round 2's
  0.216 / 0.179. `small_swallowable` produced zero true positives at every
  threshold.
- **Not a training failure** - the run was healthy on its own validation split
  (mAP50 0.73 at best epoch). It learned the office frames and transferred
  nothing.
- **Cause: duplication-driven memorisation.** At 14x, the model saw each of 79
  frames 364 times before its best checkpoint. Round 2's sequential schedule
  accidentally protected against this by spending 95 epochs on 1,078 *distinct*
  public images first.
- **Rule adopted: do not oversample by duplicating file entries.** Use loss
  weighting if class balance needs addressing.
- **Round 2 (sequential) is the best model to date** - `sharp_object` recall
  0.339 on an unseen building.
- **The finding that matters:** three training schedules over the same 92
  frames produced recall 0.147, 0.216, and 0.017. The variable that never
  changed is the one that counts - the number of distinct rooms seen. This
  supports the class-agnostic reframe rather than competing with it.

### 2026-08-11 — Setup becomes a guided, parent-confirmed walkthrough (Shaked)

- **The startup-detection problem was reframed, not solved by a sixth
  detector.** Five methods were tried for "look at one unfamiliar frame and
  autonomously judge which objects are hazards": named classification, colour
  clustering, texture objectness, VLM presence-checking, and class-agnostic
  segmentation with a compactness filter. All five failed at that specific
  job - full detail and numbers in `docs/phase-3-step0-findings.md`. Two failed
  by missing most hazards; three failed by flagging furniture edges, door
  handles, or patterned cushions as confidently as real hazards. This is not a
  gap in this project's engineering - it is an open problem in computer vision
  (noticing an unfamiliar object with no prior information), and no more
  single-frame variants should be attempted without a genuinely new idea.
- **Shaked's correction: the model was never supposed to judge alone.**
  CLAUDE.md decision 4 already specified that setup-phase hazards are logged
  for the parent to review - that hand-off already existed. What decision 7's
  work had drifted into was requiring the model to be *correct* about what an
  object is and whether it matters before the parent ever sees it. Removing
  that requirement removes the hard problem: the model only needs to flag
  "this reachable surface is not empty," which needs far less certainty than
  "this specific object is a hazard, named and judged correctly."
  Over-flagging is now acceptable, since a human filters it in seconds.
- **Setup review changes from passive log to an interactive guided
  walkthrough.** The parent confirms hazard/fine per flagged spot in a
  two-to-three-minute session when the room is first set up, not a log
  reviewed whenever. Confirmed hazards seed Layer A's hazard map for the rest
  of the session. This is decision 4's existing hand-off made concrete and
  immediate, not a new mechanism.
- **Change detection was believed to be the solved half at the time of this
  entry — later found not to be.** The claim here ("~1.4ms/frame, correctly
  boxed a car key and lighter") turned out unbacked and partly wrong on
  re-measurement; see the 2026-08-11 entries below for the correction and the
  real numbers.
- CLAUDE.md decisions 3 and 4 amended accordingly, same turn.

### 2026-08-11 — Phase 3 closed as a scoping/architecture phase, not a
working-detection phase (docs-agent audit)

- **`PHASE_PLAN.md`'s Phase 3 "done when" bar rewritten and status set to
  `[~]`, not `[x]`.** The original bar ("the fine-tuned model reliably
  detects the new classes... without badly regressing COCO classes")
  described a deliverable three measured fine-tuning rounds did not reach
  (best pooled recall 0.216 on an unseen building) and, per this phase's own
  evidence, was not going to reach with more of the same approach. New bar:
  done when the fine-tuning question is closed by held-out-location
  measurement AND, if it doesn't generalise, a replacement architecture is
  decided and logged with evidence — which is what actually happened. Full
  reasoning in `docs/phase-writeups/phase-3.md`.
- **Phase 4's scope note added in `PHASE_PLAN.md`**: Layer A now depends on
  two mechanisms (change detection, guided walkthrough) that Phase 3
  validated by measurement/decision but did not build as code. This is new
  work for Phase 4 that wasn't in its original scope.
- **Audit finding, not previously flagged: CLAUDE.md decision 7's and this
  log's own "full detail in `docs/phase-3-step0-findings.md`" pointers are
  not fully accurate.** That findings document contains the three
  fine-tuning rounds and the VLM rejection in traceable, spot-checkable
  detail, but does **not** contain: the five-method comparison for the
  single-frame-judgment problem (colour clustering, texture objectness,
  class-agnostic segmentation numbers), the walkthrough reframe's reasoning,
  or any paragraph describing how/when change detection was tested. Two of
  those five methods (colour clustering, texture objectness) have **no
  committed code anywhere in the repository** — their numbers (recall 0.09;
  recall 0.79 / ~33% eyeballed precision) exist only as bare assertions in
  CLAUDE.md and this log. The class-agnostic segmentation method **does**
  have real, committed code (`cv/measure_segmentation.py`, read in full and
  confirmed to implement what's described), but its output numbers aren't
  written up in any findings prose a reader can check against a CSV, the way
  Step 0's numbers are. Confirmed directly: `cv/*.py` (12 files, listed by
  glob) contains no frame-differencing/change-detection module and nothing
  named walkthrough/setup-flow; a repo-wide grep for "absdiff", "change
  detection", "frame diff" and "walkthrough" matches nothing under `cv/`.
  The specific change-detection evidence quoted in CLAUDE.md decision 7
  ("~1.4ms/frame", "correctly boxed a car key and a lighter... at 0.92")
  appears in exactly two places in the whole repo — CLAUDE.md itself and
  this log — and nowhere else, including no CSV or script. This is not a
  claim that the underlying work wasn't done; per the task that produced
  this phase, it was done deliberately in a scratchpad outside the repo.
  It is a claim that **it isn't independently verifiable by anyone reading
  only this repository**, which is a materially different evidentiary
  standing than the rest of this phase's committed, spot-checkable work, and
  should be treated as such rather than cited with the same confidence.
- **Not changed:** the underlying architecture decisions (CLAUDE.md 3, 4, 7)
  themselves are not being questioned here — the measured, traceable parts of
  this phase (three fine-tuning rounds, the VLM rejection, the segmentation
  method's existence) genuinely support the reframe's direction. What's
  flagged is narrower: some of the specific numbers cited *as if* fully
  measured and documented are not currently backed by anything checkable in
  the repo, and Phase 4 should re-measure change detection properly, as
  committed code, before leaning on the "~1.4ms/frame, correctly boxed a car
  key" claim as settled fact.
- A smaller housekeeping finding, also from this audit: the top-level
  `runs/` directory (containing round 1 and round 2-stage-1's weights, an
  artifact of a relative `project=` path bug already diagnosed and fixed in
  `cv/train_mixed.sh`'s header comment) shows up as untracked in `git
  status`, not ignored — `.gitignore` covers `cv/runs/` but not a bare
  top-level `runs/`. Nothing unsafe is at risk (the separate top-level
  `*.pt` rule still covers the weight files), but it's visible clutter worth
  a one-line `.gitignore` fix.

### 2026-08-11 — Change detection re-measured, twice, with real code (docs-agent
audit follow-through)

- **First re-measurement** (`cv/measure_change_detection.py`, first version,
  real person-overlap suppression implemented for the first time — it had
  been named in the original claim but never actually built). Result: the
  original decision 7 claim ("~1.4ms/frame, and it correctly boxed a car key
  and a lighter that stock COCO called `cell phone` at 0.92") does not hold
  up. Pooled recall against three labelled bursts: **0.030** — structurally
  low, because those bursts are static-scene labelling shots ~3s apart of
  objects already present in both frames, which frame-differencing cannot
  find (nothing to threshold for something unmoving in both frames). Precision
  by eye on a stratified sample: **~28–33%** real objects, the rest floor-tile
  speckle and edges. Checked the car-key/lighter claim directly, by name, on
  the actual named frames: every candidate blob is suppressed as
  person-overlap (the object moves inside the hand/arm silhouette when
  placed) or simply doesn't track the object's outline even before
  suppression. **The claim is refuted specifically as a change-detection
  result.** Most likely explanation: it's a real, correctly-measured number
  from a different method (stock YOLO reading the car key as `cell phone`
  0.92 — Session 2, Block B) that got attributed to change detection when
  decision 7 was written, conflating two results measured in the same
  session. Timing: pure frame-diff on 1080p confirmed at **~1.4ms/frame** —
  that part of the old claim was real — but real person-suppression (2 YOLO
  calls) costs ~37ms more, so **~43ms/pair combined**, not ~1.4ms.
- **Second fix** (same script, rewritten): single-pair person-overlap
  suppression was found to delete real hazards (an object moving in an open
  palm is inside the exact region being suppressed) and the labelled test
  bursts were found to structurally be unable to exercise "object placed and
  left" at all. Replaced with persistence tracking: a candidate blob must
  still be present, in roughly the same place, 2–3 frames later, AND pass a
  visual-stability re-check (not just location — location-only was tried
  first and kept 57 of 69 pure-noise blobs on a true-negative control pair,
  all floor grout/glare, because those are the highest-contrast edges in the
  frame and relight slightly on every exposure). Tested on a new,
  purpose-shot continuous clip (`changetest`, 22 frames ~1s apart, scissors
  placed then removed): **the detector genuinely catches the real placed
  object**, confirmed by eye against the raw pixels at a correctly-shaped
  bounding box. But precision is still ~33% (same ballpark, different false
  positives — mostly person-adjacent settling artifacts now, not lighting
  flicker), and recall on the older sparse-capture bursts got **worse**
  (0.030 → 0.011 pooled), because persistence requires closely-spaced frames
  those bursts don't have.
- **CLAUDE.md decision 7 corrected in the same session** to state the real,
  current numbers (real detection confirmed on a continuous clip; ~1 in 3
  detections is a real object; ~39–43ms/pair combined cost; the car-key/
  lighter claim retracted and reattributed to stock classification) in place
  of the original unbacked claim. Verified directly, by reading the current
  file, that the correction is actually present and matches
  `docs/phase-3-step0-findings.md`'s numbers.

### 2026-08-11 — Phase 3 closed, for real this time: known-weak change
detector accepted, Phase 4 begins with live-system testing (Shaked)

- **The pattern that triggered this decision.** Across three fine-tuning
  rounds and two change-detection fix rounds — five separate attempts,
  spanning two different mechanisms — each fix improved one metric at the
  direct cost of another: round 2's public-data-first schedule fixed
  `sharp_object` recall and broke `small_swallowable`; round 3's fix for
  that broke both. The first change-detection fix (person-suppression) was
  measured to delete real hazards along with the person; the second fix
  (persistence tracking) fixed that specific failure but made recall worse
  on every previously-labelled test burst. No offline, single-method,
  isolated-test iteration ever reached a bar clearly usable for an alert,
  across any of the five rounds.
- **Shaked's call: stop iterating on isolated offline/photo tests, close
  Phase 3 with what is honestly true today, and use Phase 4's live-camera
  testing to find out what actually needs fixing next.** Reasoning, stated
  plainly: static single-method tests have been tried five times without the
  system ever being run as a whole, and each fix's tradeoff only became
  visible after it shipped — a symptom that isolated testing had stopped
  producing new information and started just moving the same problem
  around. Real-world behavior on the running system is expected to surface
  what specifically breaks (which this many rounds of static testing have
  not settled), not because live testing is assumed superior in general, but
  because *this specific* debugging loop had stopped converging.
- **This is not a claim that change detection is fixed, or that Phase 4 is
  guaranteed to fix it.** Per CLAUDE.md's own preamble, nothing here is
  permanent — this is a decision about *how* the team learns what to fix
  next (by integration testing, not more isolated photo tests), not a
  verdict that the current ~33% precision is acceptable to ship, and not a
  guarantee Phase 4's approach will resolve it either.
- `PHASE_PLAN.md`'s Phase 3 status set to `[x]`. The scoping/architecture
  question was already closed by the first attempt; what changed since is
  that the previously-unbacked change-detection claim now has real,
  committed, spot-checked evidence (two full re-measurement rounds), and the
  team has an explicit, logged reason to stop refining it further in
  isolation. Full reasoning in `docs/phase-writeups/phase-3.md`.

### 2026-08-12 — Phase 4 kickoff: pure-proximity Slice 1, two-axis risk model
deferred (Shaked)

- **Slice 1 scope agreed:** a live risk overlay running only against the real
  camera feed — one YOLO26 pass per frame for `person` plus the named hazards
  that already work with no training (oven/microwave treated as one group,
  refrigerator, chair); an in-memory Layer A hazard map that starts empty
  every run (decision 3) and is populated by auto-adding named-hazard
  detections plus a manual `--seed-hazard x,y,w,h,label` flag for
  deterministic testing; and Layer B proximity scoring exactly as decision 5
  currently specifies (bbox-center Euclidean distance, normalized by frame
  diagonal, smoothed over a 5-10 frame rolling window). Red/orange/yellow
  zone thresholds ship as a first-guess, explicitly not yet tuned against
  anything.
- **The two-axis risk model (accessibility base-risk from surface height,
  escalated by proximity), proposed 2026-08-10, stays unadopted for now.**
  Raised with Shaked per that entry's own instruction; his call: ship pure
  proximity first since there's nothing running yet to judge whether the
  accessibility axis is even needed in practice once real zone thresholds
  are watched live - revisit once Slice 1 exists. CLAUDE.md decision 5 is
  unchanged.
- **Explicitly deferred past Slice 1:** the interactive parent-confirm
  walkthrough (needs a candidate generator - most likely the existing
  class-agnostic segmentation tooling - plus a UI decision, since Flet
  doesn't exist yet), change-detection integration into a live loop (still
  ~1/3 precision, only offline-tested so far), and anything from Phase 5
  (alerts, rolling video buffer).
- `PHASE_PLAN.md`'s Phase 4 status set to `[~]`.

### 2026-08-12 — Phase 4 Slice 3: Grounding DINO committed, broadened
sharp-object detection, change-detection confirmation (cv-agent)

- **Grounding DINO promoted from a scratch-venv research result
  (docs/phase-4-detection-research.md, Part 3) to committed, reproducible
  code.** `cv/measure_grounddino.py` (new) loads `grounding-dino-tiny`/
  `-base` via `transformers` and scores against the same 32-frame held-out
  home set, same IoU>=0.5 methodology as `evaluate_home_frames.py`
  (imported unchanged). `transformers==5.15.0` pinned in
  `cv/requirements.txt` against the existing `torch==2.13.0` pin - verified
  in THIS repo's own `.venv` (not copied from the research session) that it
  loads and runs real MPS inference before being pinned.
- **Reproduction confirmed close for `sharp_object` (the doc's actual
  recommendation), on both checkpoints** - e.g. tiny @ conf>=0.20:
  TP49/FP395/FN13 here vs. the doc's TP49/FP398/FN13. **`small_swallowable`
  on `grounding-dino-base` did NOT reproduce in the same range** (~10x more
  TPs here than the doc's table, deterministic on rerun - traced to
  `grounding-dino-base` emitting several near-duplicate, self-repeating
  compound labels per real object on this prompt list, which the
  now-nonexistent original scratch script may have deduplicated
  differently). Flagged in the doc rather than silently resolved either
  way; does not change any recommendation, since `small_swallowable`
  broadening was already rejected in Part 2 regardless of model.
- **Broadened live sharp-object detection shipped as a SEPARATE
  open-vocabulary pass, not folded into the existing wall-socket pass** -
  own model instance, own cadence (`--sharp-scan-interval`), imgsz 640/conf
  0.25 (the measured best-precision config from the research doc), prompted
  with ONLY `knife`/`scissors` (abstract prompts measured zero true
  positives). Reason for a separate pass rather than one combined
  prompt list: the two hazards' measured-best imgsz differ (socket 1280,
  sharp_object 640) - sharing one config would sacrifice one or the other,
  the same single-pass tension CLAUDE.md already names for
  person-vs-small-hazard resolution. Folded into `HAZARD_SOURCE_AUTO`, same
  reliability tier as wall_socket.
- **Change-detection confirmation via a Grounding DINO crop check added as
  a new mechanism** (`HAZARD_SOURCE_CHANGE_CONFIRMED`): when
  `ChangeDetector` confirms a persistent blob, the region is cropped
  (padded, frame-clamped) and checked against a hazard-relevant prompt list
  via Grounding DINO, event-triggered only (never per-frame, never on the
  whole frame). **Policy decision, left open by the task and resolved
  here:** if nothing matches, the hazard-map entry is SUPPRESSED entirely,
  not kept as a third "unconfirmed" tier - reasoning: a low-confidence
  plain CHANGE_DETECTED tier already exists and already communicates
  "unconfirmed, ~1/3 precision"; a second, even-weaker tier would add
  clutter without new information. `--disable-change-confirm` reverts to
  the plain Slice 2 behavior.
- **Offline sanity check against the real `changetest` capture burst
  (22 frames, scissors placed then removed) - honest result, not a clean
  win.** The real placed-object event IS correctly confirmed with a clean,
  specific label ("scissors"). But in this kitchen scene, the
  research-doc-sourced prompt list's "cabinet handle"/"drawer handle"/
  "cabinet knob"/"handle" prompts fire liberally against real background
  cabinetry that's visible in many crops (one heavy-motion frame produced
  24 raw candidates, 20 of which "confirmed" - almost certainly not 20
  genuinely new objects). Net observed suppression across the burst: 206
  raw ChangeDetector confirmations -> 113 gdino-confirmed (45% suppressed),
  but that aggregate number is dominated by two noisy frames and should not
  be read as a general precision figure - this mechanism has NOT been
  measured for precision/recall the way Parts 1/2 were, only sanity-checked
  for "does it run and does it catch the real case." Logged here rather
  than left only as a code comment, per the task's explicit instruction to
  be honest about whether this looks like it's reducing false positives.
- `cv/test_risk_engine.py` extended with headless tests for the three new
  pure-logic pieces (`compute_crop_region`, `label_contains_any_keyword`,
  `resolve_change_confirmation`) - 31 tests total, all passing. `python
  cv/risk_engine.py --help` and full model-loading (all four models) verified
  directly, without a camera.

### 2026-08-12 — Phase 4 Slice 3: two more bugs found on a live-camera
recording, fixed in `cv/risk_engine.py` (cv-agent)

- **Bug 1 - two distinct sharp objects placed near each other were
  collapsing into one hazard-map entry.** A knife on the table, tracked
  correctly as a `sharp_object` hazard, stopped showing as a separate,
  correctly-positioned box once scissors were placed right next to it.
  Root cause confirmed by reading the real matching code: CLAUDE.md
  decision 7 deliberately merges `knife`/`scissors` into one hazard-map
  label (`sharp_object`) so a parent doesn't need a reliably-named class to
  be warned, but `find_match`'s per-label matching let two DISTINCT
  physical objects sharing that label and placed close together fold into
  the SAME entry - decision 7 never intended that. Fixed by threading the
  sharp-object open-vocab pass's own per-detection underlying class name
  (`results[0].names[cls_id]` - same pattern `measure_openvocab.py`
  already uses) through as a new, opt-in `source_class` field on
  `HazardEntry`/`update_from_detection`/`find_match`: two detections only
  merge if they share BOTH the hazard-map `label` AND this underlying
  class. Every other call site (oven/microwave, wall_socket, `--seed-
  hazard`, change detection) never passes it and is unaffected. The
  displayed/alerting label stays the single merged `sharp_object` concept
  either way, unchanged. Not a CLAUDE.md decision change - decision 7's
  merge is about not needing to *name* which sharp object it is, not about
  conflating two simultaneously-present distinct objects into one tracked
  entry.
- **Bug 2 - stale hazard-map entries never disappeared, even after being
  visibly moved.** Scissors detected on the floor, then physically moved
  to the table, left an empty box at the original floor position for the
  rest of the session - `HazardMap`'s original design (decision 3)
  deliberately never expires an entry. Shaked flagged this as real
  clutter. Resolved with a SCOPED fix, not a blanket reversal of decision
  3: `HAZARD_SOURCE_AUTO` entries (oven/microwave/refrigerator/
  wall_socket/sharp_object) come from detectors that genuinely re-check
  the same spot on a schedule (every frame, or every `--socket-scan-
  interval`/`--sharp-scan-interval`), so several missed re-checks in a row
  is real, repeated evidence the object is gone - these are now removed
  via `HazardMap.remove_stale_auto()` (new method, called every frame from
  `main()`), gated by a new `--auto-hazard-stale-seconds` flag (default
  10.0s - "a few missed scan cycles' worth, not just one," reasoned
  relative to the socket/sharp passes' own 2.0s default cadence and their
  measured non-trivial miss rates; see `DEFAULT_AUTO_HAZARD_STALE_SECONDS`'
  comment). `HAZARD_SOURCE_SEED`/`CHANGE_DETECTED`/`CHANGE_CONFIRMED`
  entries are deliberately LEFT sticky/non-expiring: nothing periodically
  re-checks those specific spots, so removing them on a timer would be
  guessing, not responding to evidence, and would undercut decision 3's
  actual intent (something the model saw once should stay flagged even
  while temporarily out of view). This remains a known, explicitly
  documented simplification for those three sources: a seeded/change-
  detected hazard moved away for good still incorrectly stays on the map
  for the rest of the session. Flagged in this same task rather than
  silently picked, per the task's explicit instruction, in case this
  scoping is judged wrong later.
- `cv/test_risk_engine.py` extended with 9 new headless tests (46 total,
  all passing): 3 for `source_class` disambiguation (distinct classes stay
  separate, same class still merges, default/unfiltered call sites
  unaffected) and 5 for `remove_stale_auto` (AUTO entry expires past the
  window, repeatedly-rematched AUTO entry survives, `CHANGE_DETECTED`/
  `CHANGE_CONFIRMED`/`SEED` entries are never removed by this mechanism
  regardless of elapsed time). `python cv/risk_engine.py --help` verified
  directly, without a camera - `--auto-hazard-stale-seconds` appears with
  its default and reasoning.

### 2026-08-13 — Scope reset: five models down to two, one rule for hazards
(Shaked, full authority delegated on mechanism)

- **What triggered it.** Shaked stopped the session mid-debugging: *"we've been
  doing that for a few days... it shouldn't be that of a problem to set a camera
  and make it detect stuff and alarm them... I think we got lost and stuck on
  specific things and we need to rethink it."* He restated the requirement in
  its original form — **a camera detects objects; a human classifies which are
  dangerous; the system detects new objects appearing after that; it alerts** —
  and delegated the technical mechanism entirely, with three standing
  conditions: explain in plain language, justify against that requirement
  specifically rather than in the abstract, and document the *current*
  architecture rather than layering more history on top of it.
- **The diagnosis.** The live loop had accumulated five models running at once
  (per-frame YOLO, two YOLO-World instances for sockets and sharp objects,
  Grounding DINO for crop confirmation, FastSAM for the walkthrough). Every one
  passed its own isolated test. Nearly every live bug across the phase —
  the multi-second freeze, boxes covering blank walls, duplicate hazard-map
  entries, and (suspected, to be confirmed by measurement) the collapse to
  ~2fps — traced back to **one** component: the automatic pixel-differencing
  change detector, known weak since it was built (~1 in 3 detections real).
  Days went into patching that one component rather than asking whether the
  system needed it to carry that weight.
- **The mechanism that replaced it.** Layer A is now maintained by a **periodic
  class-agnostic re-scan compared against what is already known**: a spot in the
  new list but not the old means something arrived; a known spot absent from the
  new list means something was removed. The insight that makes this better
  rather than merely different: pixel differencing compares *brightness*, so a
  shadow, an auto-exposure adjustment, or a passing person are indistinguishable
  from an object appearing — which is literally why blank walls filled with
  boxes. A re-scan compares *objects*, so a wall is never "new," and the scan's
  own false positives (sofa texture, floor grout) self-cancel because they
  appear in both lists at the same place. It also yields **removal** for free,
  which the previous design could only approximate with a 10-second timer.
- **The rule that removed the most complexity: nothing auto-adds a hazard.**
  Every detector proposes; the parent disposes. This collapsed five hazard
  sources — each with its own colour, matching logic, staleness rule, and alert
  behaviour — into one kind of hazard with one rule. It is also just a restatement
  of the requirement Shaked gave, which decision 4 had specified from the start.
- **Cut, with reasoning** (all of it working in isolation; none of it surviving
  five-models-in-one-loop): live pixel change detection (replaced, above); the
  Grounding DINO crop-confirm step (existed only to filter the above before a
  human saw it — with the human as the filter by design it had no job, and at
  ~500ms/call was the prime suspect for the 2fps collapse); the `sharp_object`
  knife/scissors detector (the scan proposes those objects anyway and the parent
  confirms, one fewer model for the same outcome — and it measured 0.382/0.210
  on an unseen building, worse than simply showing the parent a box).
- **Kept:** person tracking and proximity/zone scoring (both live-verified
  repeatedly across four recordings, never a source of a bug); the scan +
  keyboard confirm loop (works, needs a duplicate fix); `oven`/`microwave`/
  `refrigerator` **read out of the person pass at zero extra cost**, which
  strengthens decision 2 rather than bending it; wall sockets on a slow cadence
  as the one named detector earning its keep, because a socket is flush with a
  wall and therefore structurally invisible to an "objects on surfaces" scan.
- **Two behaviours Shaked added on review, both correcting real holes in the
  proposal:**
  1. **An unreviewed new object is treated as dangerous until judged.** The
     first draft alerted only on parent-confirmed hazards, which meant an object
     appearing while nobody watched the screen would sit silent as the child
     approached — the exact accident the product exists to prevent. The cost
     asymmetry settles it: a false alert costs one tap; a missed one is the
     failure that matters. Exception, or the system is unusable: items from the
     *first* scan do not alarm while pending, since that is the room's normal
     state with the parent present reviewing it.
  2. **A new object alerts on appearance, not only on approach** — the parent
     should not have to be watching to learn something arrived.
- **CLAUDE.md decision 4's "acknowledgment never silences a proximity alert"
  line removed, with Shaked's explicit agreement.** It predated the existence of
  an explicit dismiss action; read literally against the current design it made
  the "not a hazard" button meaningless (a dismissed sofa cushion would alarm
  every time the child climbed on the sofa).
- **Known limitation, flagged rather than hidden, and Shaked's ruling on it.**
  Dismissals are recorded per-spot, so a real hazard placed exactly where
  something harmless was dismissed could inherit that dismissal. Shaked: *"keep
  it as double and even triple mark — better safe than sorry."* Resolution: a
  dismissal records the spot's appearance, and a later scan finding that spot
  materially changed returns it to the review queue. Reuses the existing
  crop-comparison logic rather than inventing a new mechanism.
- **CLAUDE.md decisions 2, 3, 4 and 7 rewritten in the same turn**, per this
  file's own procedure. Decision 7 in particular is now a description of what
  runs, not an accumulated record of what was tried — the phase write-ups and
  `docs/phase-4-detection-research.md` keep that history.
- **Still open, being investigated rather than assumed:** the named hazard boxes
  (oven/fridge/socket/sharp) did not appear once across a 110-second live
  recording despite scissors being in plain view the whole time. Leading
  hypothesis is not a bug — the scissors in that recording were white on a white
  bedsheet, and Phase 3 measured exactly that case at ~0.045 confidence (i.e.
  effectively blind). To be confirmed by measurement before it is written down
  as fact; if true it is further evidence for not depending on named detection.

### 2026-08-13 — Scope reset implemented in `cv/risk_engine.py` (cv-agent, full
authority delegated on mechanism per the task above)

- **Step 0 finding: NOT a bug, confirmed directly against the pre-refactor
  code, not just re-asserted.** Pushed a synthetic AUTO/named-class detection
  through the exact same `update_from_detection`/`remove_stale_auto` sequence
  `main()` ran every frame: it persists correctly across 50 continuous
  re-detections (never silently dropped) and expires correctly once genuinely
  unmatched past the staleness window - no gating bug found anywhere in that
  path. Separately, `DEFAULT_SHARP_CONF` (0.25) sat nowhere near the
  ~0.045 confidence Phase 3 measured for white scissors on a white surface -
  that gap alone fully explains a 0-detection session for that specific
  object without needing a bug. The "0 auto all session" observation is most
  parsimoniously explained by no oven/fridge/socket being in that particular
  (bedroom) frame at all, combined with the scissors' measured blind spot -
  consistent with, not contradicting, the existing leading hypothesis. This
  is now moot for `sharp_object`/change-detection specifically (both deleted
  below), but matters because it clears the `oven`/`microwave`/`refrigerator`
  path, which survives the refactor, of any lurking bug.
- **Five detectors down to two, three hazard sources down to three STATES,
  as directed.** `ChangeDetector`, the Grounding DINO crop-confirmation step,
  and the `sharp_object` open-vocab pass are removed from the live pipeline
  entirely (offline research scripts `measure_change_detection.py`,
  `measure_grounddino.py`, `measure_segmentation.py` untouched on disk, per
  the task). `HazardEntry` now carries `state` (PENDING/CONFIRMED/DISMISSED)
  and `origin` (NAMED/SCAN/SEED) instead of five `HAZARD_SOURCE_*` values -
  `match_by_label`/`source_class` (built to fix duplicate-merging bugs
  specific to the two deleted mechanisms) are deleted as genuinely
  unnecessary now: named-class labels are already distinct per physical
  class, and scan-origin entries share one generic label matched only
  against other scan-origin entries, so the collision those two parameters
  guarded against can no longer occur.
- **The periodic room scan is now the sole "detect new"/"detect removed"
  mechanism** (`HazardMap.apply_scan_candidates`), replacing both the
  one-shot `--walkthrough` flag and `remove_stale_auto`'s timer. Two
  mechanism choices made under the task's delegated authority, stated
  explicitly per its own instruction:
  - **Two-consecutive-scans jitter guard** via a provisional (seen-once,
    not-yet-a-HazardEntry) list, replaced wholesale each scan cycle - a
    candidate must match the previous cycle's provisional list to be
    promoted, and a known entry's absence counter resets to 0 the instant it
    reappears. Untested against real scan-to-scan camera jitter (no camera in
    this session) - flagged as unverified, not assumed correct.
  - **Duplicate-candidate dedup** (`dedupe_candidates`, IoU>=0.4, unmeasured
    first guess) runs on each scan's raw candidate list before any matching,
    fixing the reported "asked about the same object twice" bug at the
    source rather than in the review queue.
  - **"First scan" is tracked per-entry, not by wall-clock alone.**
    Scan-origin entries carry `is_first_scan` from the provisional record
    that promoted them (so a spot present at scan cycle 0 is correctly
    tagged even though promotion itself happens on cycle 1, one cycle later).
    Named-class/socket entries use a simpler global `first_scan_done` flag
    (flipped once scan cycle 0 completes) - a coarser proxy, since those
    detectors run every frame rather than on the scan's own cadence; flagged
    in the module docstring as an accepted approximation, not measured.
- **Dismissal re-raise reuses `region_change_frac` exactly as instructed**,
  via a new `DISMISS_REAPPEAR_CHANGE_FRAC = 0.15` threshold - deliberately
  LOWER (more sensitive) than that function's own tuning elsewhere
  (`STABILITY_MAX_CHANGE_FRAC = 0.3`, which asks the opposite question -
  "did this stay the same" - when confirming persistence), so re-raising
  trips more readily, in the direction Shaked's "better safe than sorry"
  ruling calls for. Missing/degenerate input errs toward re-raising, not
  toward silence, for the same reason.
- **Named-class and wall-socket detections now also produce PENDING entries
  requiring human review, not automatic hazards** - a reading of CLAUDE.md
  decision 7's "nothing auto-adds a hazard; every detector proposes, the
  parent disposes" as applying to every detector, not only the room scan
  (the task's own Step 2 says so explicitly: "every detector - the per-frame
  named-class pass, the periodic scan, the socket pass - produces PENDING
  entries only"). This is a real behavior change from every earlier slice,
  where oven/fridge/socket detections went straight onto the map as
  alert-eligible. Functionally the difference is narrow (an unreviewed
  arrival already alerts per decision 4's table regardless of whether a
  human ever gets to it), but it does mean a fridge present at startup sits
  as PENDING/no-alert, identically to a scanned floor object, until either
  reviewed or the room settles - flagged here in case that reading is judged
  wrong.
- **`--seed-hazard` now creates a CONFIRMED entry directly**, not a PENDING
  one - unchanged in spirit from every earlier slice (it exists purely for
  deterministic Layer B testing, never meant to exercise the review queue).
- **`cv/test_risk_engine.py` rewritten**: every test naming a deleted
  mechanism (`ChangeDetector`, Grounding DINO confirm/suppress,
  `source_class`/`match_by_label`, `WalkthroughState`'s fixed-list model) is
  gone; kept/adapted geometry, `find_match`, `PersonTracker`, rolling
  window/`score_frame`, `--seed-hazard` parsing; added tests for
  arrived/removed diffing (including the two-consecutive-scan guard and
  candidate dedup), the dismissal re-raise (both directions), and every
  branch of decision 4's proximity-alert-eligibility table. 42 tests, all
  passing. `python risk_engine.py --help` verified directly.
- **File size**: `risk_engine.py` 2901 -> 1295 lines, `test_risk_engine.py`
  1485 -> 603 lines - smaller as the task required, not merely rearranged.
- **Not verified, stated plainly**: none of this has run against a real
  camera this session (no hardware access here) - the live loop, the
  overlay's actual legibility, real scan-to-scan jitter behavior, the
  2-consecutive-scan guard's effect on a genuinely moving/settling room, and
  actual frame rate (including whether removing the three deleted models
  actually fixes the ~2fps collapse, which is the leading hypothesis but
  unconfirmed without a camera) all still need Shaked's live testing, same
  as every previous slice.
