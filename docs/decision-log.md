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

### 2026-08-22 — Phase 4 live-verified on the two-model build; Phase 4 closed
(docs-agent audit, following Shaked's recorded live test)

- **The gap left open on 2026-08-13 is now closed for the headline claim.**
  Nine days after the rewrite above (no intermediate session in between - the
  next message in the session transcript is this test), Shaked ran
  `risk_engine.py --name Arducam` against real hardware and recorded a 54.6s
  clip. Docs-agent independently re-checked the claim rather than taking the
  chat summary on faith: pulled every frame via `ffmpeg`, ran an automatic
  freeze check across the full clip (none found), and read the on-screen FPS
  readout directly off multiple frames.
- **FPS: 15.0-15.2 for the full clip, confirmed by direct frame inspection,
  not just reported.** This was 2.0-2.5 before the 2026-08-13 reset. Confirms
  the reset's leading hypothesis (Grounding DINO plus the two YOLO-World
  passes were the cost, not the scan or person tracking) was correct.
- **Live-verified, by watching real pixels, not just trusting overlay text:**
  a new object placed in frame was picked up by the periodic scan on its own,
  boxed orange as `NEW-UNREVIEWED`, with `ALERT: New object detected (room
  scan)` firing - no button touched. Approaching a CONFIRMED hazard produced
  `RISK: RED` with the connector line drawn to it. Removing a confirmed
  object made its box disappear on its own (hazard count dropped 17 -> 11)
  without any timer - the scan-comparison mechanism, not a stale-entry
  timeout, per the 2026-08-13 redesign. Duplicate boxes on the same physical
  object are "mostly gone," per Shaked directly - not claimed as fully
  solved; `SCAN_DUPLICATE_IOU_THRESHOLD=0.4` is still an unmeasured first
  guess in the code, and this is the first live evidence it's roughly the
  right ballpark, not a tuned result.
- **Also caught live, not specifically asked for: the dismissal re-raise
  rule fired for real.** The overlay read `ALERT: New object detected (spot
  changed since dismissal)` at one point in the clip - CLAUDE.md decision 4's
  "better safe than sorry" re-raise (Shaked, 2026-08-13) behaving correctly
  against a real dismissed spot that later looked different, not just
  passing its unit test (`test_apply_scan_candidates_dismissed_entry_reraised_on_material_change`).
- **One specific, safety-critical path is still NOT live-verified, and this
  is being logged rather than smoothed over:** approaching an *unreviewed*
  (PENDING, arrived-after-first-scan) hazard is supposed to alert RED the
  same as a confirmed one (`hazard_alerts_on_approach`, CLAUDE.md decision
  4's "unreviewed means dangerous" row). Every RED event in this clip was
  against an already-confirmed hazard, because everything on screen had
  already been reviewed by the time the approach happened. The code
  implements this path (`hazard_alerts_on_approach` returns `True` for
  `PENDING` when `is_first_scan` is `False`, and `main()` filters hazards
  through exactly that function before handing them to `score_frame`,
  which does not itself look at `.state` at all) and it has a direct unit
  test (`test_hazard_alerts_on_approach_pending_arrived_later_does_alert`).
  What does not exist is a test, live or unit, that composes the two - an
  unreviewed arrived entry actually scored into a "red" zone by
  `score_frame`. This is a real, narrow, named gap, not a hidden one - flagged
  in the phase write-up as the next thing to check on camera.
- **Three items handed off explicitly, not silently dropped** (full detail
  in `docs/phase-writeups/phase-4.md`):
  1. **Debug telemetry (FPS, model/imgsz/conf, raw distances) is drawn
     directly onto the same `annotated` frame pixels** (`risk_engine.py`
     `draw_overlay_line` calls in `main()`) that Phase 7's MJPEG stream is
     meant to serve. Per CLAUDE.md decision 1 (Flet is a pure client of
     FastAPI, never touching frame pixels), this needs to become
     `/risk_status` JSON fields the UI decides whether to render - a
     concrete Phase 7 architecture note, not cosmetic. **Proposed, not
     applied: no CLAUDE.md edit made here** - flagging this as something
     Phase 7 should account for when designing `/risk_status`, for
     Shaked/Yahli to confirm before Phase 7 starts.
  2. **The h/n/s keypress review loop is a developer stand-in, not the
     parent-facing feature PHASE_PLAN.md's Phase 4 done-when (a)
     describes.** The scan/confirm *logic* is real, tested, and live-verified
     above; the *interaction* a parent would actually use (tapping through
     objects in the Flet app) does not exist yet and is explicitly Phase 8's
     job. Phase 4's done-when (a) is being read as satisfied at the logic
     layer, not as "a parent could use this today" - same reading Phase 3
     used for its own "demoably works" bar.
  3. **Overlapping bounding-box labels when boxes cluster (e.g. the
     top-left corner) is genuinely cosmetic** - Phase 8's problem, noted so
     it isn't forgotten, no action taken now.
- **`PHASE_PLAN.md`'s Phase 4 status set to `[x]`.** Reasoning in full in
  the phase write-up; short version: (b) and (c) of the done-when bar are
  now live-verified on real hardware with independently-checked evidence
  (not just a chat summary), (a) is met at the logic layer with the
  parent-interaction gap explicitly named as Phase 8's job rather than
  silently assumed done, and the one specific safety-critical path not yet
  live-verified (approach-to-unreviewed-object) is real code with a real
  unit test, flagged as the first thing to check in Phase 5 or a future
  recording, not hidden inside a claim of full verification.

## 2026-08-22 — Decision 1 clarified: diagnostics are JSON, boxes are pixels

**Decision:** CLAUDE.md decision 1 now states explicitly what may be drawn into
the frame FastAPI serves. Hazard/person boxes may be (they are product — a
parent should see what is flagged). Diagnostics — FPS, model/`imgsz`/`conf`,
device, raw normalized distances — may not; they become `/risk_status` JSON
fields and the UI decides whether to render them.

**Reason:** Pixels are one-way. Anything painted into the frame is permanent by
the time Flet receives it, so a choice made in the cv layer silently becomes a
choice the UI layer cannot undo — the exact coupling decision 1 exists to
prevent. Raised by docs-agent at Phase 4 close (see the previous entry, item 1,
where it was logged as a proposal awaiting sign-off).

**Scope — this changes no code today.** `risk_engine.py`'s `draw_overlay_line`
telemetry currently feeds only the local `cv2.imshow` debug window, which is a
developer tool and stays verbose. The clarification is a constraint on Phase 7:
build `/video_feed` from a clean annotated frame plus `/risk_status`, rather
than reusing the debug window's image. Cost is ~nil now and grows once Phase 8
is built on top of it, which is why it was settled before Phase 7 rather than
during it.

**Approval:** Approved by Shaked 2026-08-22; Shaked is briefing Yahli. Recorded
here rather than treated as a fait accompli, per CLAUDE.md's rule that decisions
may be questioned freely but not changed without asking.

## 2026-08-22 — Unreviewed-approach path live-verified; frame_risk carries hazard identity

**What happened:** Shaked recorded the exact follow-up test Phase 4's write-up
asked for (`cv/captures/Screen Recording 2026-08-22 at 13.34.16.mov`): placed
a new object, walked toward it without confirming or dismissing anything. At
0:20 the overlay reads `RISK: RED person #1 vs object (normalized dist 0.05)`
against a hazard map with **0 confirmed entries** — the red connector line
points to an orange `NEW-UNREVIEWED` box. Verified by pulling frames directly
via ffmpeg, not from a chat summary. This closes the one gap
`docs/phase-writeups/phase-4.md` left open: the "unreviewed means dangerous"
rule (CLAUDE.md decision 4) was previously code-correct and unit-tested but
had never been watched fire on real pixels. See the write-up's Addendum
section for the full account.

**Two small code changes came out of reviewing that same clip:**

1. **Connector-line behavior confirmed as correct, not changed.** The debug
   overlay draws a line only to each person's nearest hazard, even when
   several hazards are in a scored zone at once. Checked and confirmed this
   loses no safety information: `classify_zone` is purely distance-based and
   monotonic, so the nearest hazard is always the highest-risk one by
   construction, and `score_frame` scores every pair regardless of what gets
   drawn. No code change; recorded so the reasoning doesn't need re-deriving
   later.
2. **`score_frame`'s `frame_risk` now includes `hazard_id` and `hazard_bbox`**
   (`cv/risk_engine.py:887`), not just `hazard_label`. Reason: `/risk_status`
   (Phase 7) needs to name *which* hazard is driving the current risk zone so
   a Flet client (Phase 8) can highlight it — the debug window's drawn
   connector line is not something the UI can inherit, since decision 1 keeps
   Flet a pure JSON/MJPEG client. No scoring-logic change; the two existing
   `score_frame` tests were extended with the new fields rather than
   duplicated. All 42 tests in `cv/test_risk_engine.py` pass.

**Why this is a decision-log entry and not just a commit message:** it closes
the specific open item the Phase 4 write-up named as a precondition for
treating decision 4's table as fully proven, and it extends the API surface
Phase 7 will build on — worth being able to find later without re-reading the
whole write-up.

## 2026-08-22 — Phase 5 kickoff: three measurements, then the alert lifecycle built on top

**Per CLAUDE.md's measure-before-build culture, three things were measured
before any code was written, all with real data (real captures from
cv/captures/, the real 2026-08-22 recordings, real subprocess/afplay calls),
not synthetic estimates:**

1. **Rolling buffer memory.** Raw 1920x1080 frames would cost ~467MB for a
   5s/75-frame buffer (6.22MB/frame). JPEG-encoding into the deque at q75,
   at FULL capture resolution (no downscaling), measured across 10 real
   captures from cv/captures/, averages ~200KB/frame (111-245KB range) - a
   5s buffer is ~15MB, a 32x reduction with no resolution loss, at ~2.7ms
   encode cost/frame (~4% of one 15-FPS frame's 66.7ms budget). Downscaling
   to 960x540 was considered and rejected: it saves another ~10MB against an
   already-negligible number, in exchange for a visibly lower-resolution
   saved clip - a bad trade. **Decision (Shaked, 2026-08-22): JPEG q75, full
   capture resolution, no downscaling.**

2. **Audio playback must not block the frame loop - verified, not assumed.**
   `subprocess.run(["afplay", ...])` (blocking) measured at ~1.9s wall time
   for a 1.0s clip - at 15 FPS that's ~29 dropped frames, and would have
   reproduced Phase 4's FPS-collapse crisis for every single alert.
   `subprocess.Popen(["afplay", ...])` (fire-and-forget) measured at 2-5ms
   call cost (~4-7% of one frame budget), with zero frames over budget in a
   90-frame simulated loop that fired sounds mid-loop three times.
   `AppKit.NSSound.play()` was also measured and rejected: its own `.play()`
   call cost up to 112ms, over one full frame budget by itself.
   **Decision: `subprocess.Popen(["afplay", path])`, no new dependency.**
   Caveat logged rather than hidden: afplay carries ~0.4-0.9s of fixed
   process/CoreAudio startup latency before sound is actually audible (a
   0.05s clip still took 0.72s wall via a blocking call) - this cost is paid
   by the OS process, not the Python caller, so it does NOT block the frame
   loop, but a parent may perceive a brief lag between the visual alert and
   the voice. Not yet judged against real perception on hardware.

3. **Alert cooldown/debounce policy - measured against a real recording, not
   guessed.** Pulled the risk-zone readout at 4Hz from the actual
   2026-08-22 unreviewed-approach clip (cv/captures/Screen Recording
   2026-08-22 at 13.34.16.mov) and found one continuous walk toward one
   object produced ~8.5s of RED that flickered RED->ORANGE->RED twice, with
   dips up to 0.75s - DESPITE the existing 8-frame rolling window already
   smoothing the raw distance. A pure edge-triggered design (alert only on
   the none->red transition) was considered and rejected on this same
   evidence: it would have re-armed after each dip and produced 3 separate
   alerts/clips for what a parent would experience as one approach.
   **Decision: exit hysteresis, not entry debounce - `ALERT_HOLD_SECONDS =
   2.0` (~2.7x margin over the worst dip observed), keyed per
   (person_id, hazard_id) so a second, different hazard is never silenced by
   the first's cooldown, with escalation (yellow->orange->red) re-signaling
   but de-escalation staying silent within the hold window.** A separate,
   global `CLIP_MIN_INTERVAL_SECONDS = 30.0` backstop caps how often a NEW
   clip file can start across different events, independent of the per-event
   hold logic - a disk-safety floor, not a replacement for it.

**What got built on top of these three decisions** (`cv/risk_engine.py`,
`cv/test_risk_engine.py`, both cv-agent-owned per PHASE_PLAN.md):

- **`AlertEvent`/`AlertSignal`/`AlertManager`** replace the pre-Phase-5
  single overwritable `alert_text`/`alert_until` slot with real event
  identity, the hysteresis state machine above, and a decide-and-commit
  `should_trigger_clip()` gate (CLAUDE.md decision 6: red-only, once per
  event). Pure state machine, no I/O - unit-tested the same way HazardMap
  is, with a direct regression test modeling the measured 0.75s dip.
- **`RollingBuffer`/`ClipRecorder`/`write_clip()`** implement decision 6's
  ~5s-buffer + ~2s-tail clip, JPEG-buffered per measurement 1, written on a
  background thread because encode+write of a 7s clip measured at
  ~130-200ms (2-3 frame budgets) - a real hitch if done inline, the same
  class of problem Phase 4's crisis was about. H.264 (`avc1`, falling back
  to `mp4v`) chosen for file size (~0.75MB vs ~2.28MB for a 7s clip,
  measured on real footage) and because Phase 7/8 will want these playable
  in a browser.
- **`AudioPlayer`** wraps the non-blocking Popen call from measurement 2;
  missing audio files log a warning once and are skipped rather than
  crashing, since recording them was a parallel human dependency.
- **A drawing-order change, not a new mechanism**: buffer/clip frames are
  captured immediately after hazard/person boxes are drawn but BEFORE the
  connector line and any diagnostic overlay (FPS, model config, risk
  readout) - the same "boxes are product, diagnostics are pixels" split
  from the 2026-08-22 decision-1 clarification, now enforced by construction
  rather than left as a Phase 7 TODO. This also means the buffered frame is
  already the "clean annotated frame" Phase 7's `/video_feed` needs to be
  built from, one phase early, at zero extra cost. **Decision (Shaked,
  2026-08-22): the connector line is excluded from saved clips** - Phase 8
  is not expected to draw one, so a clip should look like what the product
  actually shows; boxes stay in.
- **Clip path convention (Shaked, 2026-08-22): `cv/clips/pending/
  <timestamp>_event<id>_<hazard_label>.mp4`**, agreed now so Phase 6
  (backend-agent, SQLite + clip lifecycle) inherits the convention rather
  than renaming it later.
- **Voice clips**: `cv/audio/hazard_detected.wav` / `baby_getting_close.wav`
  / `immediate_danger.wav`, per CLAUDE.md decision 4's three-clip set.
  Recorded by Shaked same-day; delivered as mono PCM16 at 24kHz (not the
  44.1kHz originally suggested - afplay handles this format transparently,
  confirmed by direct playback through the real `AudioPlayer` class) and
  1.39-2.34s each (two slightly over the 2.0s target flagged when the spec
  was written - not re-litigated, since the target was a latency judgment
  call, not a hard requirement, and no live evidence yet says 2.3s is too
  long in practice).

**Not yet live-verified on camera**: the buffer/clip/alert pipeline has been
exercised by the unit suite (72 tests, all pass - 30 new for Phase 5) and by
a standalone synthetic wiring smoke test (no camera - simulated per_person
dicts driving the exact call sequence main() makes), but not yet by an actual live
`risk_engine.py --name Arducam` run with a real critical event. That is the
next step before Phase 5 can be marked closed, per PHASE_PLAN.md's own
done-when bar ("a simulated critical event produces a correct saved clip
file and the right alert fires - visually and audibly").

## 2026-08-26 — Phase 5 live test found a real alert-storm + a visibility bug; both fixed

Shaked ran `risk_engine.py --name Arducam` live (recording:
`cv/captures/Screen Recording 2026-08-26 at 19.26.36.mov`, 130s). Watched
the clip directly rather than taking the report on faith - pulled the
terminal log visible in the last ~10s of the recording and read the overlay
drawing code against it. Two real, code-confirmed problems, one visual
fix, both closed same day:

**1. Alert storm - one physical spot re-alerted repeatedly, landing on top
of a real proximity escalation.** The terminal log showed hazard entry
**#15 dismissed and re-raised four times in a row** in a few seconds:

```
ALERT: New object detected (spot changed since dismissal): object
Dismissed entry #15 (not a hazard).
ALERT: New object detected (spot changed since dismissal): object
Dismissed entry #15 (not a hazard).
[...] x4, then:
ALERT: RISK ORANGE: object approaching (person #1)
ALERT: RISK RED: object approaching (person #1)
```

Root cause, confirmed by reading `HazardMap.apply_scan_candidates`: the SAME
`HazardEntry` (same id) flips DISMISSED->PENDING every time
`fingerprint_changed` trips against `DISMISS_REAPPEAR_CHANGE_FRAC = 0.15` -
deliberately sensitive per Shaked's 2026-08-13 "better safe than sorry"
ruling, so one ambiguous/hard-to-segment object can cycle repeatedly. **That
sensitivity was NOT changed** - the actual bug was one layer up:
`AlertManager.open_new_object()` had zero memory across re-raises, so every
single one fired its own independent, unthrottled alert. This burst then
coincided with a real ORANGE->RED escalation, and felt like six alarms at
once.

**Decision (Shaked, 2026-08-26): a global alert-pacing arbiter, separate
from AlertManager's per-pair hysteresis.**
- `GLOBAL_ALERT_MIN_INTERVAL_SECONDS = 2.5` - at most one alert is actually
  voiced/bannered per this many seconds, GLOBALLY (not per-hazard,
  not per-pair).
- Priority order when several signals compete inside one window: **RED
  (immediate danger) > yellow/orange (getting close) > new object** -
  exactly Shaked's stated order. The highest-priority candidate seen during
  a blocked window wins once it reopens; lower-priority alternatives are
  dropped, not queued for later.
- **RED always bypasses the pacing entirely** (Shaked, explicit call,
  recommended by docs/cv-agent and confirmed rather than assumed):
  immediate-danger alerts must never wait their turn. A RED interrupt also
  clears anything currently held, on the reasoning that a stale
  lower-priority alert isn't worth surprise-firing right after a RED.

Implemented as a new `AlertArbiter` class (`cv/risk_engine.py`), sitting
between `AlertManager` (unchanged - still decides WHETHER a pair's state
genuinely changed) and the actual `speak()` call. Clip-saving was
deliberately left OUT of the arbiter - `should_trigger_clip`'s own
once-per-event/30s-cooldown gate is independent, since a critical moment is
worth recording even on a frame where the audio/banner was suppressed
because something else just spoke. 9 new tests, including a direct
regression test reproducing the four-re-raise-plus-RED scenario end to end
and asserting it collapses to the correct 2 spoken alerts (not 6-7).
81/81 tests pass total.

**2. Review-candidate white outline was genuinely invisible against light
backgrounds.** Shaked reported "couldn't see some of the white lines of the
hazard number and marking." Confirmed directly in `draw_hazard_box`: the
white outline marking which hazard is currently awaiting an h/n/s decision
was a bare 1px `cv2.rectangle` call with no dark halo behind it - unlike
every text label in this file (`draw_label`), which already draws a black
outline pass first specifically for this reason. Against the water heater,
the light tile floor, or ordinary video compression, a 1px pure-white line
with nothing behind it disappears. **Fix: the same two-pass halo technique
`draw_label` already uses** - a thicker `OVERLAY_OUTLINE` (black) rectangle
drawn first, the white rectangle on top. No behavior change, no new
constant, verified by rendering the fixed box against a synthetic light
background before/after.

**Not changed, flagged rather than silently left alone:**
`DISMISS_REAPPEAR_CHANGE_FRAC` (the actual reason hazard #15 kept
flip-flopping) - the arbiter fix solves Shaked's stated complaint (the
alerting behavior) without touching Layer A's detection sensitivity, which
was a deliberate safety-first call. Worth a dedicated measurement later if
this keeps happening on other ambiguous objects, but out of scope for this
fix.

## 2026-08-26 — Follow-up: the white marking still disappeared - edge clipping, not just missing contrast

Same-day follow-up. Shaked confirmed the alert-storm and contrast fixes
above worked, but reported one thing left: "still can't see some of the
white marks of hazard number and which one i need to confirm and deny...
make sure it doesn't disappear above or below the screen." Watched the new
recording (`cv/captures/Screen Recording 2026-08-26 at 19.58.26.mov`)
before touching anything - a single frame at t=8s showed **multiple hazard
boxes near the top of frame with visibly missing top borders**, confirmed
directly against `draw_hazard_box`'s code rather than guessed at.

**Two distinct, compounding bugs, both purely geometric (no scoring/logic
change):**

1. **The review-outline rectangle expands OUTWARD by 2px** (`x1-2, y1-2` to
   `x2+2, y2+2`) before the previous fix's halo is drawn. For a box near any
   frame edge, that outward offset pushes part of the rectangle off-canvas -
   the affected side (e.g. the top border, for a box near y=0) simply isn't
   drawn at all, since it has no on-canvas pixels to render. **Fix**: clamp
   all four outline coordinates to `[0, frame_width-1] x [0, frame_height-1]`
   before drawing, instead of using the raw ±2 offsets directly.
2. **`cv2.putText`'s origin is the text BASELINE, not a bounding-box
   corner** - glyphs are drawn extending UPWARD from it. Every label in this
   file that sits "above" a box (`max(0, y1 - 8)` for the state tag,
   `max(0, cy1 - 24)` for the REVIEW candidate text) was clamping to y=0 when
   the box was near the top - but y=0 is not a valid position for text meant
   to appear ABOVE that point; nearly the entire glyph ends up off-canvas
   regardless of the clamp. This is why labels "disappeared," not a contrast
   problem this time. **Fix**: `label_anchor_y()` - a shared helper that
   flips the label to sit BELOW its anchor point instead of clamping to an
   invalid position, when there isn't enough clearance above
   (`LABEL_TOP_CLEARANCE = 14`). Applied to `draw_hazard_box`'s state label,
   `draw_person_box`'s "person #N" label (same bug, same fix, not
   separately reported but confirmed identical by reading the code), and
   the REVIEW-candidate label in `main()`. The two labels use different
   above/below offsets (8/16 for the state tag, 24/40 for REVIEW) so they
   don't land on top of each other when both flip below near the top edge.

Verified by rendering synthetic boxes at the exact edge position seen in
the recording, before/after, and by re-deriving the actual glyph height via
`cv2.getTextSize` rather than guessing the offset constants. 4 new tests
(`test_label_anchor_y_*`, `test_draw_hazard_box_review_outline_does_not_
crash_near_frame_edges`), all pass; 85/85 total. Left/right-edge label
clipping (same origin-is-left-edge mechanism, just horizontal) was noticed
while reading this code but not reported live and not fixed here - flagged
for later if it turns out to matter in practice.

## 2026-08-26 — Follow-up #2: dismissed spots re-raising repeatedly on single-scan noise

Shaked, after confirming the arbiter and visibility fixes worked: "we still
have like the same thing i've marked as hazard or no hazard showing up as a
hazard detected 5 times every few seconds and each time i say no... what CAN
we do to avoid quadruple detection to the same thing." This is the same
underlying cause named (but deliberately not touched) in the first
2026-08-26 entry: `HazardMap.apply_scan_candidates`'s dismissal re-raise
runs `fingerprint_changed()` fresh on every scan and re-raises on a SINGLE
positive - for one visually ambiguous object, scan-to-scan noise (lighting,
a shifted segmentation box, compression) was enough to trip
`DISMISS_REAPPEAR_CHANGE_FRAC = 0.15` repeatedly, forcing a fresh 'n' every
~5s scan cycle. The alert-arbiter fix quieted the VOICED alert but did
nothing to the underlying re-raise/re-enqueue rate, which is the actual
review burden Shaked is describing here.

**Three options presented, with tradeoffs, before writing any code:**
- **A - require 2 consecutive scans of "changed" before re-raising**,
  reusing the exact jitter-guard shape already proven for brand-new
  arrivals (`SCAN_CONSECUTIVE_SCANS_REQUIRED`). Doesn't change what counts
  as "changed" at all - only requires it to persist. Cost: a REAL hazard
  swap now takes one extra scan cycle (~5s) to be caught.
- **B - grace period right after a dismissal** (skip the check entirely for
  a short window post-dismiss).
- **C - raise `DISMISS_REAPPEAR_CHANGE_FRAC` itself** (0.15 -> ~0.25) - most
  direct, but also makes a smaller real hazard swap easier to miss
  entirely, not just slower to catch.

**Decision (Shaked, 2026-08-26): Option A.**

**Implemented**: `HazardEntry` gained `changed_scans: int = 0` (meaningful
only while `state == DISMISSED`, same shape as `absent_scans`), and a new
constant `DISMISS_REAPPEAR_CONSECUTIVE_SCANS_REQUIRED = 2`.
`apply_scan_candidates`'s dismissed-match branch now increments
`changed_scans` on a positive `fingerprint_changed()` read, resets it to 0
on a negative read (the change didn't persist - not evidence of a real
swap), and only re-raises once the counter reaches the threshold, clearing
it back to 0 on re-raise. `HazardMap.dismiss()` resets the counter on every
fresh dismissal so a stale count can never carry over. `DISMISS_REAPPEAR_
CHANGE_FRAC` itself (the sensitivity of what counts as "changed") was left
exactly as it was - the fix targets how easily one noisy scan can act
alone, not what counts as evidence.

6 new/updated tests, including one modeling the exact live failure mode
(changed/unchanged/changed/unchanged - never two in a row - must never
re-raise regardless of the total count) and one confirming a fresh dismiss
resets any leftover counter. 87/87 tests pass total. Not yet live-verified
against the specific object that was flagging repeatedly - next live test
should specifically watch that spot rather than only trusting the unit
suite.

## 2026-08-26 — Follow-up #3: the 2-consecutive-scan fix didn't work; found and fixed the real cause

Shaked ran a fresh live test (no recording - the exact terminal output was
pasted directly) and hazard entry #13 was dismissed **5 times** in a short
span, with "ALERT: New object detected (spot changed since dismissal)"
firing on almost every scan in between. This is the exact object from the
first 2026-08-26 entry, now tested again WITH that entry's fix
(`DISMISS_REAPPEAR_CONSECUTIVE_SCANS_REQUIRED = 2`) already in place - and
it still happened.

**Honest diagnosis: the previous fix targeted the wrong failure mode.**
Requiring "2 consecutive scans of changed" only helps if the noise is
occasional. This log shows it firing on very close to *every* scan - which
means the underlying `fingerprint_changed()` comparison itself was
unreliable, not just noisy, and "2 in a row" is trivially satisfied every
time by a comparison that is failing consistently.

**Root cause, found by re-reading the code with Shaked's own framing in
mind ("if nothing came or moved in the frame, nothing should be alarted -
it should be as simple as that"):** `fingerprint_changed(match.fingerprint,
frame, candidate)` was cropping the comparison region using `candidate` -
**that scan's own fresh segmentation box**, not a fixed reference. FastSAM's
segmentation boundary is not pixel-identical run to run even for a
completely static scene, especially for a visually irregular/thin shape (a
cable, a ribbed surface). So the check was comparing *different pixels*
scan to scan - measuring "we sampled a slightly different patch of the
image this time" as "the scene changed," which is not a real signal at all.

**Reproduced and verified directly, not assumed:** built a synthetic
fine-striped test frame (a flat color and a smooth gradient both turned out
too shift-tolerant, after `region_change_frac`'s Gaussian blur, to
reproduce this - a striped/textured pattern was needed to match a real
ribbed/textured object). Confirmed the OLD candidate-bbox comparison flags
"changed" on every single 2-4px horizontal shift tried; confirmed the fix
below reads "unchanged" on every one of the same shifts.

**Fix:** `HazardEntry` gained `fingerprint_bbox` - the STABLE bbox recorded
at dismiss time, reused for cropping every later scan's comparison frame
too, instead of that scan's own wobbling candidate bbox. `fingerprint_changed()`
now always compares the SAME patch of the frame across scans; what counts
as "changed" (`DISMISS_REAPPEAR_CHANGE_FRAC`) and the 2-consecutive-scan
requirement from the previous entry are both unchanged and now layered on
top of a comparison that's actually measuring the right thing. Also added
an opt-in diagnostic (`fingerprint_changed(..., label=...)`) that prints
the real computed change fraction during a live run, so if this still
misfires the next debugging pass has real numbers instead of another guess.

2 new regression tests (one confirming box jitter alone never re-raises
across 6 different shift amounts on a real adversarial pattern, one
confirming a genuine change still re-raises correctly even with jittering
boxes) plus updated `fingerprint_bbox` assertions on the existing tests.
89/89 tests pass total. Not yet live-verified against entry #13's actual
object - that object is the next thing to specifically re-test.

### 2026-08-26 — Phase 5 reviewed and closed (docs-agent audit)

- **`PHASE_PLAN.md`'s Phase 5 status set to `[x]`.** Full reasoning in
  `docs/phase-writeups/phase-5.md`. Short version: the code in
  `cv/risk_engine.py` (`AlertEvent`/`AlertManager`/`AlertArbiter`,
  `RollingBuffer`/`ClipRecorder`/`write_clip`, `AudioPlayer`,
  `label_anchor_y`, `HazardEntry.fingerprint_bbox`) matches every claim in
  this session's five decision-log entries above, checked directly against
  the file rather than the entries alone - including a character-for-
  character cross-check of `banner_text_for_signal`'s output against the
  exact terminal-log text quoted in the 2026-08-26 entries, which confirms
  those logs are genuine output from this code, not paraphrase. Real
  artifacts on disk were confirmed to exist (both 2026-08-26 recordings,
  the three audio files, the saved clip at
  `cv/clips/pending/20260826-204245_event8_object.mp4`) via the `Read`
  tool's binary-file existence check.
- **Two things this audit could NOT do, stated plainly rather than
  papered over:** no shell/Bash access was available this session (`Grep`/
  `Glob` both failed with `rg not found` on every call), so docs-agent
  could not execute `cv/test_risk_engine.py` (89 tests counted directly by
  reading the file, matching this session's own final count, with several
  of the newest tests - `label_anchor_y`, the `AlertArbiter` priority
  logic - hand-traced against the real function arithmetic rather than
  just read) or re-run `ffprobe`/frame-extraction against the saved clip
  the way Phase 4's audit did. This is a materially weaker form of
  verification than Phase 4 got for the equivalent claims, named
  explicitly in the write-up rather than presented as equal rigor.
- **Audibility is the one piece of this phase's own done-when bar
  ("visually and audibly") that is not confirmed on the record anywhere.**
  `AudioPlayer` is code-verified to launch `afplay` non-blockingly and to
  fail closed on a missing file; nobody has logged actually hearing a
  voice clip play. Named as the first thing to check next, the same
  pattern Phase 4 closed under with its own one named safety-relevant gap.
- **A documentation-discipline gap, distinct from the engineering:** the
  most recent of this session's five entries (Follow-up #3, the
  `fingerprint_bbox` fix) ends "not yet live-verified against entry #13's
  actual object" - but no sixth entry exists recording that re-test
  happening, even though the task briefing that produced this write-up
  describes a "Live test 5" with a successful outcome and an independently
  ffprobe'd clip. Docs-agent confirmed the referenced clip file exists but,
  per the tooling gap above, could not independently confirm its claimed
  frame-count/duration/codec details this session, and could not confirm
  from the repo alone whether the missing entry is an oversight or
  evidence the retest hasn't actually been logged yet. Recommend closing
  this specific gap in the same turn this write-up is reviewed, per
  CLAUDE.md's own "log a decision in the same turn it's made" rule.
- **Read as a genuine instance of this project's "measure, don't assume"
  culture, not a process failure:** the dismissal-re-raise bug was fixed
  twice in this session, and the first fix's failure is logged plainly
  (Follow-up #2 vs. Follow-up #3) rather than folded into a single clean
  narrative. The second fix's diagnosis came from re-reading the code
  against the literal live symptom rather than guessing at a second
  plausible cause, and was reproduced with a purpose-built synthetic test
  pattern after two simpler patterns failed to trigger the bug at all -
  the same disciplined-debugging shape as Phase 3's fine-tuning rounds and
  Phase 4's five-models reset, applied one layer down (one function's
  implicit assumption, not a whole subsystem).

## 2026-08-26 — Missing entry, added late: the fingerprint_bbox fix re-tested live and confirmed

Filling the exact gap docs-agent flagged in its Phase 5 close-out audit
(see the entry immediately above and `docs/phase-writeups/phase-5.md`):
Follow-up #3 shipped the `fingerprint_bbox` fix and ended "not yet live-
verified against entry #13's actual object." That re-test happened the
same day, in the same working session, but the confirming entry was never
written at the time - a real process miss, not a fabricated result. Adding
it now, later than CLAUDE.md's own "log it the same turn" rule asks for,
rather than leaving the record inconsistent with what actually happened.

**What was checked, by whom:** Shaked ran `python risk_engine.py --name
Arducam` live and pasted the full terminal output directly into the
session (not a recording - no video artifact exists for this specific run).
That output showed hazard entries #4, #6, #9, #10, #11, and #13 - #13 being
the entry that had re-raised repeatedly in both prior failed-fix rounds -
with `fingerprint_changed`'s new opt-in diagnostic printing real
`change_frac` values throughout: entry #13 ranged 0.000-0.049 across many
scans, entry #10 similarly low, every other dismissed entry read 0.000
almost the whole session. **Zero** "ALERT: New object detected (spot
changed since dismissal)" lines appear anywhere in the pasted log - the
fix held for the full session, not just briefly. The same log also shows a
real RISK ORANGE -> RISK RED escalation firing correctly and triggering a
saved clip.

**Independently checked in the orchestrator session, not taken on the
log's word alone:** the referenced clip file,
`cv/clips/pending/20260826-204245_event8_object.mp4`, was inspected
directly with `ffprobe` (codec h264, 1920x1080, exactly 105 frames,
duration 7.000000s - matching decision 6's 5s-buffer + 2s-tail spec
exactly) and by extracting frames from it with `ffmpeg` (showed real,
clean hazard boxes on the floor with no diagnostic overlay or connector
line baked in - confirming the decision-1 drawing-order claim against the
actual saved artifact, not just the source code).

**This closes the specific gap docs-agent's audit correctly identified as
unconfirmed** - the clip and the change_frac numbers were real and already
checked before this session's docs-agent review ran, but never logged
here, which is exactly the discrepancy the audit flagged rather than
guessed at. Recorded now for the same reason the rest of this log exists:
so the next person doesn't have to take an orchestrator's chat summary on
faith.

### 2026-08-28 — Phase 6 kickoff: schema shape and four decisions (Shaked)

Before writing any migration/schema code, the orchestrator re-verified two
things left open at Phase 5 close and proposed a schema + four decisions
for Shaked to make, per this project's measure/decide-before-build culture.

**Re-verified, closing two Phase 5 gaps:**
- `cd cv && python test_risk_engine.py` actually run this session (Phase
  5's docs-agent audit couldn't - no shell access that session): **89/89
  tests pass**, for real, not just counted by reading the file.
- `afplay cv/audio/hazard_detected.wav` run directly - no error, so the
  file is valid, playable audio and `afplay` launches correctly outside
  the risk_engine.py process too. **Still not independently confirmed
  audible by a human** - Shaked was asked directly in this same turn;
  answer not yet on the record. Carried forward as the same open item
  Phase 5 flagged, one step closer.

**The core design problem and its resolution.** `AlertEvent.id` and
`HazardEntry.id` are session-local by design (decision 3 - Layer A/B carry
no state across process restarts), so they cannot be a database primary
key on their own - two different camera runs both produce an "event #8."
Resolution: the `events` table gets its own `INTEGER PRIMARY KEY
AUTOINCREMENT`; the session-local id is kept as an informational column
alongside `run_started_at` (this process's real start time). No separate
`sessions` table - an in-process dict (`session_local_id -> db_id`),
scoped to the lifetime of the running `risk_engine.py` process, is
sufficient to route "escalated" signals to the right existing row, and
that dict's lifetime already matches exactly how long the session-local id
stays meaningful. A related, easy-to-miss issue: `AlertManager` times
everything in `time.monotonic()`, which has no fixed relationship to a
wall-clock date across process restarts - so persisted timestamps are a
fresh `time.time()` read taken at the moment persistence actually writes
the row, not a conversion of the monotonic `now` passed around internally.

**Four decisions (Shaked, this session):**
1. **Undecided-pending-clip auto-delete timeout: 24 hours.** (Proposed
   24h/48h/72h with reasoning; 24h chosen.)
2. **New `backend/` directory**, not inside `cv/`. Matches CLAUDE.md's team
   split - backend-agent already owns "SQLite schema and queries" as its
   own domain, separate from cv-agent's pipeline code - and gives Phase 7's
   FastAPI app a home waiting for it instead of a later move. SQLite file:
   `backend/guardianeye.db`, gitignored (same privacy reasoning as
   `cv/captures/`/`cv/clips/` - this is runtime data from a real home, not
   source).
3. **The events table persists only alert signals that were actually
   voiced/bannered to the parent - not every AlertManager state
   transition.** The orchestrator's recommendation (log everything plus an
   `ever_voiced` flag) was NOT taken; Shaked chose the narrower "what the
   parent was told" scope instead. **Consequence, worth being explicit
   about:** `AlertManager`'s "closed" signal (a proximity pair's hysteresis
   window elapsing) is never voiced by design (`audio_for_signal` returns
   None for it), so under this policy it is never persisted either - the
   `events` table has no `ended_at`/closing record at all. `last_seen_at`
   on the most recent voiced row is the closest available signal for "how
   recent was this," not a precise close time. This is a direct, accepted
   consequence of the decision, not an oversight - the table is a record of
   what the parent was actually told, not a mirror of the full in-memory
   state machine.
4. **Two small additive changes to Phase 5's `ClipRecorder`/`write_clip`,
   approved** (trigger/tail timing unchanged): `ClipRecorder.poll(now)`
   returns `(path, event)` pairs instead of bare paths, so a clip row can
   be linked to the `AlertEvent` that triggered it; `write_clip()` gains an
   optional `on_complete` callback so a clip's DB row can learn
   success/frame-count once the background write thread actually finishes,
   instead of the row staying permanently "writing."

**Schema (events, clips) and the `cv/clips/kept/` convention** (parent
"keep" moves the file out of `pending/`, mirroring the existing
`pending/` naming) are specified in full in the same-turn implementation
task handed to backend-agent. `clips.status` gets a fourth value,
`expired` (an automatic timeout), kept distinct from `discarded` (an
explicit parent "no") - the DB row survives either way as an audit trail
even after the underlying file is deleted.

**Scope boundary held deliberately**: no FastAPI, no HTTP, no UI - Phase
6 is the data layer only, per `PHASE_PLAN.md`'s own division of labor.
`PHASE_PLAN.md` status set to `[~]`.

### 2026-08-28 — Phase 6 implemented: `backend/` persistence layer, both
Phase 5 hook-ins wired (backend-agent)

Implements the schema and four decisions from the same-day "Phase 6
kickoff" entry above. New files: `backend/db.py` (schema + connection
handling), `backend/persistence.py` (EventWriter, clip lifecycle, queries),
`backend/test_persistence.py` (16 tests, plain-assert style matching
`cv/test_risk_engine.py`), `backend/API.md` (function-surface reference for
Phase 7). Both `cd cv && ../.venv/bin/python3 test_risk_engine.py` (93/93 -
89 original + 4 new covering the two Phase 5 changes below) and `cd backend
&& ../.venv/bin/python3 test_persistence.py` (16/16) actually run this
session, not just read.

**Judgment calls made along the way, not already pinned down by the
kickoff entry:**

- **Module layout is flat** (`backend/db.py`, `backend/persistence.py`,
  no `__init__.py`, no `backend` treated as an importable package) -
  matches this repo's own Phase 1 precedent ("flat cv/ directory, no src/
  nesting, no package ceremony until there's enough code to justify it").
  `cv/risk_engine.py` adds `backend/` to `sys.path` explicitly (not
  relying on CWD, which differs depending on whether the script is
  launched from `cv/` or the repo root) and does `from persistence import
  ...` / `from db import ...` - flat module imports, exactly like its
  existing `from camera import ...` / `from detect_stream import ...`.
- **Every function opens its own short-lived `sqlite3.Connection`** (via
  `db.get_connection`), never a connection shared across calls or threads -
  the task's own threading note made this the only safe default, and at
  this write volume (alerts/clips, not per-frame) the overhead is a
  non-issue, matching the task's own framing.
- **`EventWriter` is a small class, not a bare function + external dict.**
  It owns the `session_local_id -> events.id` map itself (one instance per
  `risk_engine.py` process, constructed once in `main()`) rather than
  making the caller thread a dict through every call - the map's lifetime
  and the writer's lifetime are identical by construction, which was the
  kickoff entry's own reasoning for why an in-process dict suffices.
- **A verified, not assumed, reason the clip/event id-linking race is safe
  in practice**: `ClipRecorder.poll()` returns `(path, event)` pairs, and
  the clip DB row is inserted in `main()` right after, resolving
  `event.id` to `events.id` via `EventWriter.db_id_for()`. This only works
  if the triggering event's row already exists by then.
  `AlertManager.should_trigger_clip()` only fires for PROXIMITY events at
  `peak_zone == "red"`, and `alert_priority()` gives RED the top tier,
  which `AlertArbiter.offer()` (read in full for this reason) returns
  *immediately, never held* - RED is explicitly exempt from the pacing
  window. So the same call to `handle_alert_signal()` that triggers the
  clip always also gets `voiced is not None` on the same frame, and the
  event row is persisted before `poll()` ever returns that clip's path.
  Guarded anyway (skip the clip insert + print a warning if `db_id_for`
  somehow returns `None`) rather than trusting this silently, since
  `clips.event_id` is `NOT NULL` and a crash here would be worse than a
  skipped clip row.
- **`write_clip()`'s `on_complete` fires on every exit path, including
  every failure path** (empty frame list, undecodable first frame,
  `VideoWriter` failing to open), with `frame_count=0` for all of them -
  not just on success. `clips.write_status` has an explicit `'failed'`
  value in the schema; a callback that only ever fired on success would
  leave a failed write's row stuck at `'writing'` forever.
- **`ClipRecorder._start_write()` builds a per-clip closure** (capturing
  `path`/`event`) around the single `on_write_complete` hook given at
  construction, rather than requiring a second "on-started" hook - matches
  the task's literal code sketch (`_start_write` "passes your callback
  into the `threading.Thread(...)` call") rather than inventing a third
  hook point not in that sketch.
- **`update_clip_write_result` matches by `path` (UNIQUE), not by a clip id
  passed back across the thread boundary** - simpler than trying to hand a
  freshly-inserted id into a closure built before the insert has
  necessarily run, and `path` is known at closure-construction time either
  way. A small bounded retry (`_CLIP_ROW_RETRY_ATTEMPTS = 5`, 0.05s apart)
  guards the theoretical (not expected, per the reasoning above) race where
  the write finishes before the row exists.
- **`keep_clip` also updates `clips.path`** to the new `cv/clips/kept/`
  location (not specified explicitly in the task, but implied by "keep
  moves the file" - a `path` column that stops pointing at a real file
  after "keep" would make `get_clip`/future Phase 7 downloads silently
  broken). `_kept_dir_for()` derives the sibling `kept/` directory from the
  clip's own stored path rather than a hardcoded constant, so it's correct
  regardless of `--clips-dir`/CWD.
- **New CLI flags on `risk_engine.py`**: `--db-path` (default
  `backend/guardianeye.db`, resolved via `backend.db.DEFAULT_DB_PATH`),
  `--disable-persistence` (isolates the persistence layer for debugging,
  mirrors the existing `--disable-scan`/`--disable-audio` pattern),
  `--sweep-interval` (default 60s, `DEFAULT_SWEEP_INTERVAL_SECONDS` -
  polling cadence for checking the 24h timeout, not the timeout itself).
- **The auto-delete sweep runs inline in the frame loop**, not on its own
  thread - it's a handful of DB rows at most per call at this write volume,
  and it's Layer-A-adjacent housekeeping, not on Layer B's latency-critical
  path, so the task's own "not latency-critical" framing for periodic scans
  applies here too.
- **Verified end-to-end, not just per-function**: a manual smoke test
  (event insert -> clip trigger -> background write -> on_complete DB
  update -> status query -> auto-delete sweep) run against real
  `risk_engine.py` classes (`AlertEvent`, `AlertSignal`, `ClipRecorder`,
  real `cv2`-encoded frames) and a temp SQLite DB, confirming the full
  round trip actually works end to end, not just that each piece's own
  unit tests pass in isolation.

**Not done, flagged rather than silently skipped**: this session did not
run `risk_engine.py` against a real camera with `--db-path` pointed at a
real DB - the smoke test above exercises the same classes/functions but
constructs its own synthetic `AlertEvent`s rather than driving them through
a live `main()` loop. Someone with camera access should run a real session
and confirm `backend/guardianeye.db` actually accumulates rows and a real
RED escalation produces both a clip file and a linked clip row, before this
is trusted the way Phase 5's live-camera confirmations were.

## 2026-08-28 — Phase 6 live test: persistence works, and found a real bug
that only live data could find (half of all voiced alerts were never saved)

Shaked ran `python risk_engine.py --name Arducam` with persistence enabled
(`run_started_at=2026-08-28T09:24:52`), pasted the full terminal output, and
the orchestrator queried the resulting `backend/guardianeye.db` directly
rather than taking "it ran without errors" as success. **Doing that
comparison is what found the bug below** - the run looked completely healthy
from the terminal alone.

**First, a correction to this log's own record.** The two Phase 6 entries
above were originally dated **2026-08-26**; the actual date was
**2026-08-28**. The orchestrator anchored on the Phase 5 dates it had just
been reading instead of the current date, and backend-agent inherited the
wrong date from the task briefing it was given. Both headings and every
cross-reference to them (in `backend/db.py`, `backend/persistence.py`,
`backend/API.md`, `cv/risk_engine.py`) were corrected. Recorded here rather
than silently fixed, because a dated log whose dates are wrong is worse than
one that admits it got them wrong - and because "the orchestrator misdated
it and the subagent copied that" is a real, repeatable multi-agent failure
mode worth having on the record for the workflow write-up.

**Audio audibility CONFIRMED - closes an item open since Phase 5.** The
orchestrator played `cv/audio/hazard_detected.wav` via `afplay` and asked
Shaked directly; Shaked confirmed: "yes i did heard it works."
`PHASE_PLAN.md`'s Phase 5 done-when bar ("the right alert fires - visually
**and audibly**") is now met on both halves. This had been carried as the
single named gap through Phase 5's close-out write-up and both Phase 6
entries above. It cost one command and one question, having sat open for
two phases.

**Persistence confirmed working on real hardware.** `backend/guardianeye.db`
accumulated 5 event rows from the live session - both `new_object` (reason
`room scan`) and `proximity` kinds, correct zones/peak_zones, correct
`run_started_at`, `hazard_bbox` round-tripping through JSON. This closes the
"not yet live-verified" gap backend-agent flagged in the entry immediately
above. No clips: no RED escalation occurred this run (the session peaked at
ORANGE), so `clips` is legitimately empty - not a failure, and the clip path
was already live-proven in Phase 5.

**THE BUG: only ~5 of ~10 voiced alerts were persisted.** Cross-checking the
terminal log's `ALERT:` lines against the DB rows showed the persisted
`session_local_id` values were **1, 3, 6, 7, 9** - gaps at 2, 4, 5, 8. Those
gaps are `AlertEvent`s that `AlertManager` really created and that the
parent really was told about (they appear as `ALERT:` lines in the log),
but which never reached the database. Specifically missing: the
`named detection: refrigerator` new-object alert, person #1's
YELLOW->ORANGE escalation, and two of the ORANGE alerts.

**Root cause**: `speak()` is reached from **two** call sites, and Phase 6's
wiring only persisted at one of them. `AlertArbiter.offer()` returning a
winner goes through `handle_alert_signal()`, which recorded correctly. But
`AlertArbiter.poll()` - which releases a signal that was *held back* by the
2.5s pacing window, and is called separately in the main frame loop - called
`speak(held)` with no `record()` beside it. So every alert that lost its
pacing race and was released a moment later was voiced and bannered to the
parent while silently never being saved. Given
`GLOBAL_ALERT_MIN_INTERVAL_SECONDS = 2.5`, that is a large fraction of any
busy session, which matches the roughly-half loss observed.

**Fix: move the persistence call INSIDE `speak()`**, rather than adding a
second `record()` beside the second call site. Decision 3's requirement is
"persist exactly what the parent was actually told," and `speak()` *is* the
function that tells them - so putting the write there makes the invariant
structural instead of something every present and future call site has to
remember. The narrower patch (record at both call sites) would have fixed
this instance and left the same trap armed for the next one.

**Why the 93-test suite passed the whole time, stated plainly rather than
glossed:** this is a *wiring* bug living in `main()`'s local closures, not a
logic bug inside any class. `AlertArbiter.poll()` is correct and unit-tested;
`EventWriter.record()` is correct and unit-tested; the defect was only in how
`main()` connected them, and `main()` has no test coverage (it needs a
camera). This is the same shape as Entry 6 in `docs/agent-workflow-notes.md`
(individually-correct components composing badly) and the same shape as
Phase 5's `fingerprint_bbox` bug - both found by live data, neither findable
by the unit suite. **No new unit test was added for this**, deliberately and
with the tradeoff named: a test that mirrors `main()`'s call sequence would
duplicate the wiring rather than test it, and would pass even if `main()`
later drifted. Making this genuinely testable means extracting `main()`'s
alert wiring into an injectable object - a real refactor, logged here as a
candidate for Phase 7 (which will need to drive these same paths from a
FastAPI process anyway), not something to improvise during a phase close.

**The fix is NOT itself live-verified yet.** It is a three-line structural
change whose correctness is readable, and the full suite still passes
(93/93 cv, 16/16 backend), but nobody has re-run the camera and confirmed
the previously-missing alerts now appear. **That re-run is the next thing to
do**, and it should specifically compare `ALERT:` line count against row
count the way this entry did - the check that found the bug is the check
that confirms the fix.

**A second finding, not fixed, flagged with new evidence: `PersonTracker` ID
churn is now visibly polluting the persisted event log.** Rows 2, 3 and 4
are three *separate* event rows against the **same hazard #3**, with
`person_id` 1, 4 and 5 - one continuous approach by (almost certainly) one
person, recorded as three unrelated events, because `AlertManager` keys
proximity events on `(person_id, hazard_id)` and a new person id starts a
brand-new event. This was flagged as a forward pointer in Phase 5's write-up
(`PERSON_STALE_SECONDS = 1.0` suspected but unconfirmed) on the basis of one
terminal log; it has now recurred and, more importantly, has a *consequence*
it didn't visibly have before - it corrupts the event history a parent will
eventually read in Phase 8, and inflates event counts. Still not fixed here
(it is cv-agent's Layer B territory, not persistence), but it has graduated
from "cosmetic log noise" to "wrong data in the database," which is a
stronger reason to schedule it.

**One question asked and deliberately answered with no change** (Shaked):
whether events/clips should be batched on a 3-5s cadence rather than
"logging every millimetre of movement," since two clips of a child at 2.0m
and 1.9m from the same object would be wasteful. Answered: that case cannot
occur, because neither writes are per-movement. An event is written once per
*zone crossing*, not per frame - a child moving 2.0m->1.9m inside the same
zone produces zero database activity, and the whole yellow->orange->red
ladder is at most 1 insert plus 2 updates for an entire episode. Clips are
already hard-capped at one per event by `should_trigger_clip`'s
decide-and-commit gate plus a 30s global cooldown (CLAUDE.md decision 6,
Phase 5). A fixed 3-5s batch would in fact be *worse* on both ends: it would
still write during a long unchanging approach that currently writes nothing,
and it would blur the exact moment of the red crossing - the one timestamp
that matters - into an arbitrary time bucket. No change made; recorded
because the question is a reasonable one that will recur, and the reasoning
should not have to be reconstructed next time.

## 2026-08-28 — Phase 6's top open item closed: the persistence fix
re-verified live, and a correction to how it was verified

Shaked ran a second live session (`run_started_at=2026-08-28T09:46:47`)
specifically to re-check the `speak()` persistence fix from the entry
above. The orchestrator's first attempt at checking it - "diff the
`ALERT:` line count against the DB row count" - turned out to be the wrong
test: the log had 10 `ALERT:` lines, the DB had 9 rows, and a raw
mismatch would read as the bug recurring. It doesn't - it's exactly what
`EventWriter.record()`'s insert-on-open/update-on-escalate design is
supposed to produce once an event is voiced more than once before closing.

**Correct verification, and the actual evidence:** queried
`started_at`/`last_seen_at` per row rather than just counting rows. One
row (person #4, hazard "object") has `started_at=09:47:56` and
`last_seen_at=09:47:59` - three seconds apart, proving that row was
inserted once (at YELLOW) and updated once (to ORANGE), i.e. the YELLOW
and ORANGE `ALERT:` lines for person #4 correctly collapsed into a single
row reflecting current state. Every other row's `started_at`/`last_seen_at`
are identical (single insert, never escalated further). Full accounting:
5 new-object inserts + person #1 YELLOW (insert) + person #1 ORANGE
(insert, as a **separate** event - see below) + person #3 ORANGE (insert)
+ person #4's insert-then-update = 9 rows, 10 record() calls, matching all
10 `ALERT:` lines with nothing missing. **The fix holds** - this closes the
single highest-priority open item from `docs/phase-writeups/phase-6.md`.

**A correction to the verification method itself, for the record**: "count
ALERT lines vs. count DB rows" is not the right test once an event can be
voiced across more than one zone crossing before closing - the right test
is "does every voiced signal's underlying event appear in the DB, either
as its own row or folded into an update of an already-open one." Anyone
re-running this check in the future should use the started_at/last_seen_at
comparison above, not a bare count.

**Not a new bug, flagged as a loose thread**: person #1's YELLOW (09:47:46)
and ORANGE (09:47:48) landed as two *separate* event rows (session-local
ids 6 and 9), 2 seconds apart - right at the `ALERT_HOLD_SECONDS = 2.0`
boundary, meaning the (person, hazard) pair most likely dropped out of a
scored zone for a moment and the event genuinely closed and reopened. This
is plausibly an ordinary real-world blip and plausibly connected to the
already-flagged `PersonTracker` ID churn - not distinguishable from this
log alone, not claimed as a new finding, left for whoever picks up the
churn issue.

### 2026-08-28 — Phase 6 reviewed and closed (docs-agent audit)

`PHASE_PLAN.md`'s Phase 6 status set to `[x]`. Full reasoning in
`docs/phase-writeups/phase-6.md`; summary here, not a substitute for it.

- **Schema, `EventWriter`'s insert/update dedup, timestamp handling
  (fresh wall-clock reads, never a converted `time.monotonic()` value), and
  the sweep's real file I/O were all read directly against the three Phase 6
  decision-log entries above and matched**, including hand-tracing
  `EventWriter.record()` and `sweep_expired_clips()` against their own tests'
  actual assertions, not just confirming the functions exist.
- **The `speak()` fix for the persistence-drop bug was independently traced,
  not taken on the log's word.** Confirmed: `speak()` is now the single
  choke point for both voicing and persisting an alert; its two call sites
  (`handle_alert_signal()`, and the main loop's `AlertArbiter.poll()`
  release) are mutually exclusive per signal, so no double-record path
  exists; and even a hypothetical duplicate call would be masked as a
  harmless idempotent `UPDATE` by `EventWriter.record()`'s own
  insert-vs-update branching rather than surfaced - a real, if narrow, blind
  spot worth naming on its own. **The fix itself remains code-verified, not
  live-re-verified** - nobody has re-run the camera and diffed `ALERT:` line
  count against DB row count a second time. This is named as the single
  most important open item from this phase, ahead of everything below,
  because it's a path that was tried, found broken, and patched - not a path
  that simply hasn't been tried yet (the shape of Phase 4's and Phase 5's
  named gaps).
- **A self-correction claim was checked, not trusted, and found
  incomplete.** The entry immediately above states the 2026-08-26→2026-08-28
  date-typo fix "landed everywhere" (`backend/db.py`, `backend/persistence.py`,
  `backend/API.md`, `cv/risk_engine.py`). A repo-wide grep found
  `backend/API.md` line 6 still read 2026-08-26 at review time. Fixed
  directly during this audit (mechanical, no judgment call) rather than only
  flagged - but the miss itself is the finding: a log entry asserting "I
  checked everywhere and fixed it" is a claim like any other, and this is a
  concrete instance of that claim being wrong by one file. Filed in
  `docs/agent-workflow-notes.md` as its own multi-agent-process entry.
- **`PersonTracker` ID churn graduated from a suspected cause (Phase 5,
  based on one unverified log) to confirmed against real persisted rows**:
  three DB rows this session, same hazard, three different `person_id`s,
  almost certainly one continuous approach. Not Phase 6's mechanism to fix
  (it's cv-agent's Layer B), but flagged as a precondition for Phase 8's
  event-history view specifically, since it now corrupts exactly the data
  that view will display.
- **The missing `ended_at` column is honestly documented in the three files
  most likely to be read first** (`db.py`, `persistence.py`, `API.md`) but
  its specific Phase 8 consequence - a parent can be told approximately when
  an event stopped (`last_seen_at` + `ALERT_HOLD_SECONDS`), never precisely
  - doesn't appear to have been weighed against what Phase 8's UI actually
  needs to show. Not a blocker for Phase 6; worth a short explicit
  conversation before Phase 8's event-log view is designed around this
  schema.
- **The "no unit test for the wiring bug" reasoning was assessed as correct
  but incomplete**, not simply accepted or simply rejected: duplicating
  `main()`'s exact call sequence in a test genuinely wouldn't catch a future
  drift, but an invariant-level test (every signal reaching `speak()` also
  reaches `record()`, exercised with the real `AlertArbiter`/`EventWriter`
  classes rather than `main()`'s closures) is possible and doesn't exist.
  The named Phase 7 plan (extract `main()`'s alert wiring into an injectable
  object) is judged a credible plan rather than deferral-in-name-only,
  because Phase 7 needs that same extraction independently of this bug (to
  drive these paths from FastAPI instead of a frame loop) - but it remains a
  plan, not yet a fact, and worth re-checking specifically at Phase 7 close.
- **Phase 5's audibility gap is now closed on the record**: Shaked confirmed
  hearing `hazard_detected.wav` during this session ("yes i did heard it
  works"). `PHASE_PLAN.md`'s Phase 5 section amended in place (not
  rewritten) to note this; Phase 5's own `[x]` and history are untouched.

### 2026-08-28 — Phase 7 kickoff: the concurrency model is the phase, and it
is the reverse of the obvious one (Shaked approved all seven)

Phase 7 (FastAPI serving layer) opened by reading Phases 4-6 back rather than
adding routes. The finding that shaped everything: `main()` is a
single-threaded blocking loop that owns the camera, both YOLO models,
`hazard_map`, `review_queue`, the annotated frame, and a `cv2.imshow` window
driven by a blocking `cv2.waitKey()`. ASGI needs to serve HTTP concurrently
with that. There is no "just add routes" version of this phase — the
concurrency model IS the design problem, the same way `AlertEvent.id`
collision was the design problem at the start of Phase 6.

**1. The camera loop keeps the MAIN thread; uvicorn runs on a background
daemon thread. This was measured, not assumed, and it inverts the obvious
arrangement.** The natural proposal — camera loop to a background thread,
uvicorn on main — was tested before being built on:

```
cv2.error: Unknown C++ exception from OpenCV code
```

`cv2.imshow` from a non-main thread fails on this machine. OpenCV's macOS
highgui is Cocoa-backed and Cocoa windows must be created on the main thread.
Moving the loop off main would therefore have killed the local debug window —
which CLAUDE.md decision 1 explicitly protects ("this does not constrain the
local OpenCV debug window"). Ten seconds of measurement replaced an assumption
that would have been discovered as a crash halfway through the phase, and it
also dissolved the second trap raised at kickoff: because the loop stays on
main, the `h`/`n`/`s` keys and the debug window keep working exactly as they
do today, unchanged, alongside the server.

State crosses the thread boundary two ways, and HTTP handlers never touch
`HazardMap`/`ReviewQueue`/`PersonTracker`/`AlertManager` directly:
- **Reads**: a `SharedState` guarded by one `threading.Condition`. The loop
  publishes `(jpeg, status_dict, frame_seq)` once per frame — rebind two
  references, bump a counter, `notify_all`, release. The dict is built fresh
  each frame and never mutated after publish, so readers grab a reference
  under the lock and serialize outside it. No torn reads, no lock held across
  I/O, and the loop never blocks on HTTP.
- **Writes**: a lock-protected command queue the loop drains once per frame,
  right beside the existing `waitKey` handlers, calling the same functions
  those keys call.

**Why writes go through a queue rather than the HTTP thread calling
`HazardMap` directly** — this is load-bearing, not stylistic:
`hazard_map.dismiss(entry_id, frame)` *requires the current camera frame* to
compute its dismissal fingerprint (`cv/risk_engine.py:846`), and only the loop
thread has one. A dismiss from the HTTP thread would silently skip the
fingerprint and break the "spot changed since dismissal" re-raise rule —
exactly the rule Phase 5 spent two failed debugging rounds getting right
(2026-08-26 follow-ups #2 and #3). The queue also keeps exactly one writer to
Layer A/B state, which is the simplest model that is actually correct.

**2. `/video_feed` serves the JPEG the rolling buffer already encoded.**
`rolling_buffer.append(annotated, now)` (`risk_engine.py:2137`) already returns
the encoded clean annotated frame, and MJPEG is by definition a stream of
JPEGs. Publishing that same object costs zero additional encode (~2.7ms/frame
already spent, measured at `ROLLING_BUFFER_JPEG_QUALITY`) and — more
importantly — makes **CLAUDE.md decision 1 structurally enforced rather than a
matter of discipline**. The served bytes are literally the same object the
rolling buffer stores, captured above the diagnostics line, so a future
diagnostic overlay physically cannot leak into the stream without also
corrupting saved clips, where it would be caught immediately. This is the
payoff of the capture point cv-agent placed one phase early, on purpose.

**3. `/risk_status`'s shape — smaller new work than kickoff assumed, and worth
correcting.** The kickoff brief stated none of this is assembled into a dict
anywhere. Half of it already is: `score_frame()` returns `frame_risk` with
`zone`/`value`/`person_id`/`hazard_label`/`hazard_id`/`hazard_bbox`, and its
docstring (`risk_engine.py:1109`) already names `/risk_status` as its
consumer. What genuinely does not exist is everything around it — FPS, model/
`imgsz`/`conf`/`device`, hazard-map counts (computed inline inside
`draw_risk_readout` and thrown straight at pixels), review-queue state, and
the person list. Assembled by a new `build_risk_status()` in
`risk_engine.py` — it lives cv-side because it reads `HazardEntry`/
`PersonEntry`, keeping `backend/server.py` a pure serving layer that knows
nothing about them. `hazard_counts` is factored out and shared with
`draw_risk_readout` rather than copied, so the pixel readout and the JSON
cannot drift apart.

**4. Hazard-review endpoints ARE in Phase 7, expanding `PHASE_PLAN.md`'s
literal goal line.** Phase 4's close named the keyboard `h`/`n`/`s` loop "a
developer stand-in... explicitly Phase 8's job" to become a real interaction —
and Phase 8 cannot build that against nothing. Shipping Phase 7 without it
means Phase 8's first act is reopening Phase 7. Marginal cost is near zero:
the command queue has to exist for decision 1 regardless, so this is three
routes on infrastructure already being built. `PHASE_PLAN.md`'s Phase 7
section amended in place to record the expanded scope.

**5. `/clips` serves video bytes, on a separate route.** `/clips` itself stays
metadata-only JSON; `GET /clips/{id}/video` returns the file. This is not
scope creep — clips are *already* written as `avc1`/H.264 specifically for
this, and `risk_engine.py:497` says so in as many words ("avc1 is chosen ...
because Phase 7/8 will want these playable in a browser"). The alternative is
Phase 8 reading the local filesystem directly, which breaks decision 1's
"Flet is a client of FastAPI." The path comes only from the DB row, never from
the client, and is `realpath`-verified to sit under the clips directory before
being served — a client-supplied path here would be a plain directory-traversal
read of the whole disk.

**6. Dependencies pinned in a new `backend/requirements.txt`**, installed into
the same shared `.venv` (one virtualenv for the project, not one per
component). `fastapi==0.141.1`, `uvicorn==0.52.4`, `starlette==1.6.0`,
`pydantic==2.13.4`, `httpx==0.28.1`. Pinned exactly, matching
`cv/requirements.txt`'s stated policy for fast-moving stacks — `/video_feed`'s
correctness rests on one specific documented Starlette behaviour (a *sync*
generator passed to `StreamingResponse` is iterated in a threadpool, so
blocking inside it does not stall the event loop; an `async def` generator that
blocked would freeze every other endpoint in the process), which is exactly the
kind of load-bearing assumption that should not silently ride an auto-upgrade.
Deliberately plain `uvicorn`, **not** `uvicorn[standard]` — the extra pulls
uvloop/httptools/watchfiles/websockets/PyYAML and this server needs none of
them; fewer moving parts matters more than marginal throughput on a machine
already carrying a delicate torch/MPS stack. `httpx` is a test-only dependency
(Starlette's `TestClient` transport) pinned explicitly rather than left
transitive, so a clean-venv rebuild cannot produce a backend whose test suite
silently cannot run. Verified additive: both suites still pass 93/16 after the
install, torch and ultralytics untouched.

**7. The server binds `127.0.0.1` by default, and this is a privacy decision
rather than a default (Shaked).** Offered three options — localhost-only, LAN
with no auth, or LAN plus a shared token. Shaked chose localhost-only. The feed
is live video of a room with a child in it plus every saved clip, with no
authentication; binding `0.0.0.0` would expose that to every device on the WiFi
including guests and anything compromised, which sits badly against CLAUDE.md
decision 8's stance that privacy here is "a requirement, not just a convenience
choice." A `--host` flag exists to opt into the LAN. **Phase 8 must decide
authentication before it is used**, since Phase 8's own done-when ("reachable
from a tablet/phone browser on the same WiFi") requires LAN binding — recorded
here so that decision is made deliberately rather than by someone reaching for
the flag mid-demo.

**One thing deliberately NOT done, and one open question it leaves.** Phase 6's
write-up and decision log both recorded "extract `main()`'s alert wiring into
an injectable object" as Phase 7's job, the remedy for the bug that ate half of
Phase 6's alerts. Having read `main()`: **Phase 7 does not actually need it.**
What Phase 7 needs is small and additive (publish a snapshot, drain a queue);
the alert-wiring extraction is a separate refactor of the exact three closures
that just received a live-verified bug fix. Shaked's call: do it, but as its own
step *after* `/video_feed` is live-verified, so a failure tells you which change
caused it rather than nothing. Recorded explicitly because two documents carry
it as a Phase 7 commitment and letting that quietly lapse is precisely the drift
docs-agent flagged at Phase 6's close.

**Proposed CLAUDE.md amendment, NOT yet made, awaiting Shaked/Yahli.**
Decision 1 names the JSON API as "(`/events`, `/risk_status`, `/clips`)".
Phase 7 also serves `/review` (+ confirm/dismiss/skip) and `/health`. Following
the same precedent as the 2026-08-22 diagnostics entry, this is logged as a
proposal rather than edited into CLAUDE.md unilaterally. Nothing in the
implementation contradicts decision 1 — the addition is to its illustrative
list, not to its rule.

**Two carried-forward data-quality issues that `/events` now exposes to a
client for the first time. Neither is fixed here, and neither is papered over.**
There is no `ended_at` column (deliberate, Phase 6 decision 3) — `/events` will
NOT synthesize one from `last_seen_at + ALERT_HOLD_SECONDS`, because that would
be the serving layer inventing data and would quietly convert a deliberate
decision into a fake column. And `PersonTracker` ID churn means one continuous
approach can surface as several rows with different `person_id`s. Both are
documented at the route so whoever meets them in Phase 8 recognises them
instead of filing a fresh Phase 7 bug.

### 2026-08-28 — CLAUDE.md decision 1 amended: `/review` and `/health` added
to the API list (Shaked approved)

The 2026-08-28 "Phase 7 kickoff" entry above logged this as a proposal per
CLAUDE.md's own rule (ask before changing it, log in the same turn). Shaked
confirmed. Decision 1's illustrative route list now reads `/events`,
`/risk_status`, `/clips`, `/review`, `/health` instead of the original three.
Nothing about the rule itself changed — the video/control decoupling and the
"diagnostics are JSON, boxes are pixels" split are untouched; this only
updates the example list to match what actually shipped.

### 2026-08-28 — Phase 7 follow-up: `main()`'s alert wiring extracted into
`AlertDispatcher`, closing Phase 6's named debt

Done as its own step AFTER Phase 7's concurrency work was committed and
live-verified (`b768510`), per Shaked's call at kickoff — deliberately not
folded into the same pass, so that if something broke it would be
attributable to one change rather than two structural edits to the same
code at once.

**What this closes.** `docs/phase-writeups/phase-6.md`'s gap #4 and the
2026-08-28 Phase 6 live-test entry both recorded the same debt: the
persistence-drop bug lived in `main()`'s closures, no unit test could reach
it, and the stated remedy was "extract `main()`'s alert wiring into an
injectable object." Phase 6's own reasoning for not adding a test then —
"a test that mirrors `main()`'s call sequence would duplicate the wiring
rather than test it" — was judged by docs-agent as *correct but incomplete*,
because an invariant-level test using the real classes was possible and
simply didn't exist. It exists now.

**The extraction.** `AlertDispatcher` (cv/risk_engine.py, next to
`AlertArbiter`) owns `speak()`, `offer()`, `poll()` and the banner state that
used to be `alert_text`/`alert_until` locals. It takes the arbiter, an
audio player, and an `event_recorder` (anything with `.record(signal)` —
`EventWriter` in production, `None` under `--disable-persistence`), plus an
injectable `clock` so banner expiry is testable without sleeping.

Deliberately scoped to the VOICING path only. **Clip triggering stays in
`main()`**, because it is intentionally not gated by the arbiter (Phase 5: a
critical moment is worth recording even on a frame where we chose not to
re-announce it audibly, and `should_trigger_clip()` has its own
once-per-event/30s gate). Folding it in would have coupled two things Phase 5
deliberately separated.

**Behaviour is unchanged, and this was verified line by line rather than
asserted.** Every removed line — the two banner locals, `raise_alert()`,
`speak()`, and the four call sites — has an exact counterpart inside the new
class, including the detail that the banner's expiry clock is a *fresh*
`time.monotonic()` read rather than the `now` passed into `offer()`/`poll()`,
matching what `raise_alert()` did. One incidental simplification: `speak()`
used to test `args.persistence_enabled` before calling
`event_writer.record()`, while `event_writer` was *already*
`EventWriter`-or-`None` from the same flag — the dispatcher checks the
recorder itself, so there is one source of truth instead of two conditions
that could drift apart.

**Five new tests (98 → 103).** The headline one,
`test_alert_dispatcher_every_voiced_signal_is_also_recorded_including_held_release`,
drives a real `AlertArbiter` through the exact failure shape: voice one
signal, offer a second inside the 2.5s pacing window so it is *held*, confirm
it is neither voiced nor persisted, then release it via `poll()` after the
window reopens and assert `recorder.recorded == spoken`. That held-then-
released path is precisely what was silently unpersisted before the fix. It
observes the real code path by wrapping `speak` on the instance rather than
subclassing, so it tests the shipping implementation, not a reimplementation
of it — and it never imports or re-types `main()`'s call sequence, which was
the whole objection to the naive version of this test.

Also covered: RED bypassing the pacing window is still recorded (a *third*
route into `speak()`, and the structural point of the extraction is that a
new route gets persistence for free); `--disable-persistence` still voices;
banner expiry on an injected clock; and a pinned check that `"closed"`
signals never produce audio, since a `"closed"` signal reaching `speak()`
would create exactly the row `db.py`'s missing `ended_at` column says should
not exist.

**Not yet live-re-verified, and that matters here specifically.** Phase 7's
browser/stream/keyboard live test (Shaked, 2026-08-28) ran against
`b768510`, i.e. BEFORE this refactor. 103/26/16 tests pass and the removed-
line audit is exact, but this is a change to the alert path, and this
project's own record is that a clean suite is not proof for this particular
code — the Phase 6 bug passed 93/93 the entire time it was losing half the
session's alerts. A short camera run confirming alerts still voice, banner,
and land in the DB should happen before docs-agent closes the phase.

### 2026-08-28 — AlertDispatcher refactor live-re-verified: alerts still
voice, banner, and persist correctly

Shaked ran a live camera session against `c9c3046` (the `AlertDispatcher`
extraction commit) specifically to close the one open item that commit
left on the record — the Phase 7 stream/keyboard live test had run against
the prior commit, before this refactor touched the alert path.

**Method: the same one that caught the original Phase 6 bug** — diff
`ALERT:` terminal lines against `events` rows for the run's own
`run_started_at`, not a bare pass/fail read of the session. The terminal
log shows 6 `ALERT:` lines (3 `New object detected (room scan): object`,
3 `RISK ORANGE: object approaching` for persons #1/#2/#3). The DB shows
6 rows for that exact `run_started_at`, in the same order, same kind,
same hazard/person ids, same zones — a clean one-to-one match:

```
id=27 new_object  hazard=15               reason="room scan"
id=28 new_object  hazard=16               reason="room scan"
id=29 new_object  hazard=17               reason="room scan"
id=30 proximity   hazard=4 person=1 zone=orange
id=31 proximity   hazard=4 person=2 zone=orange
id=32 proximity   hazard=4 person=3 zone=orange
```

`session_local_id` has two gaps (5, 6 — between the last new-object row's 3
and the first proximity row's 4/7) - two `AlertEvent`s were opened
internally but never won voicing (held by `AlertArbiter`'s pacing window,
then superseded or the window never reopened before another candidate took
priority). This is normal arbiter behaviour, not a discrepancy: per the fix,
a row is written only for what actually reached `speak()`, and both this
session and the entry above confirm nothing that *did* reach `speak()` went
unrecorded.

This closes the one item the AlertDispatcher commit left open. Phase 7 is
now fully live-verified: the stream/keyboard test (against `b768510`) and
the alert-persistence invariant (against `c9c3046`, this entry) both
checked on real hardware, not just a green test suite - matching this
project's standing "a clean run is not proof" position, applied here
specifically because the Phase 6 bug this refactor touches passed 93/93 the
entire time it was live-dropping alerts.
