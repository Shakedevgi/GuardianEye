# Phase 2 — Baseline detection

**Status: signed off (2026-08-05). Three further rounds of camera-layer work
happened after that sign-off, audited here in a dedicated section below —
"After sign-off." Read that section even if you already read this write-up
once: it contains the single most important story of the whole phase, and it
happened after the point where this document originally stopped.**

This write-up is deliberately not sanitized. Phase 2's bring-up was the
messiest thing this project has hit so far, and the orchestrator was explicit
that the hardships and false leads are as much the deliverable as the working
code. If you're reading this later wondering why a "run YOLO on a webcam"
phase generated this much narrative: because most of the phase's real time
went into hardware and OS problems that had nothing to do with YOLO, and
that's worth learning from as much as the detection code itself. That is even
more true of the post-sign-off rounds than of the original bring-up: a
carefully-designed, carefully-reasoned fix shipped, was asserted with printed
"evidence," and was still wrong — watching the correct-looking chain of
reasoning fail anyway is more instructive than any of it succeeding on the
first try would have been.

## What Phase 2 was supposed to prove

Per `PHASE_PLAN.md`: stock YOLO (COCO classes) running on the live feed from
Phase 1, drawing boxes for whatever COCO objects are actually in the room
(person, oven, knife, etc.), in real time. No custom classes, no risk logic —
those are Phase 3 and Phase 4. The done-bar is "live feed shows correct
bounding boxes and labels for whatever COCO objects are actually in the room."

## What was actually built

Three files under `cv/`, all read line-by-line for this audit:

- **`cv/camera.py`** — the Phase 1 capture loop (open device, read frames,
  reconnect-with-backoff on repeated failures) extracted into a shared
  module so it isn't duplicated between `stream_camera.py` and the new
  detection script. It owns exactly that concern: no display, no detection,
  no keypress handling.

- **`cv/stream_camera.py`** — refactored to build on `camera.py`'s
  `CameraCapture` class instead of owning its own `cv2.VideoCapture` and
  retry logic directly. Externally, nothing changed: same window, same `q`
  to quit, same window-close detection, same reconnect behavior.

- **`cv/detect_stream.py`** — the actual Phase 2 deliverable. Same capture
  loop, plus one YOLO26 pass per frame (`model.predict(...)`, called once per
  iteration, no second pass for a different class subset). Ultralytics'
  own `results[0].plot()` draws every detected class it finds, unfiltered —
  Phase 2's whole point is confirming the model sees what's actually in the
  room, not previewing a hazard-specific filter that belongs to a later
  phase. Flags: `--index`, `--device` (`mps`/`cpu`), `--conf`, `--model`,
  `--imgsz`. An on-screen overlay shows live FPS and the exact running
  config (model file, imgsz, conf, device) so a screenshot or saved frame
  can always be traced back to what produced it. Pressing `s` saves both the
  raw and annotated current frame to `cv/captures/` (gitignored) — raw
  frames where detection fails are exactly the material Phase 3's
  fine-tuning dataset needs.

## Model and environment

Ultralytics **YOLO26** (released 2026-01-14, NMS-free end-to-end), on
`ultralytics==8.4.115`, `torch==2.13.0`, `torchvision==0.28.0`, all pinned
exactly rather than floor-pinned like `opencv-python`, because this
combination (YOLO26, torch 2.13, Python 3.14/cp314 wheels) is new enough that
an unpinned install risks silently landing on an untested combination for the
next person who runs `pip install`. Running on Python 3.14.4 / arm64, with
device `mps` (Apple Metal on this machine's M5 Pro GPU) by default and a
`--device cpu` escape hatch — there are known reports of Ultralytics
producing visibly wrong detections on MPS versus CPU, so this exists as a
one-word way to rule the backend in or out as the cause of a bad detection,
not as an untested option nobody would reach for.

The default model started as `yolo26n` (nano) and was changed to `yolo26l`
(large) during the same phase, once benchmarking showed why nano's speed
advantage was headroom nobody could spend (see below).

## The bring-up hardships — read this section, it's the point

**Roughly two hours were lost to a camera that reported success and delivered
nothing.** `cv2.VideoCapture(index).isOpened()` returned `True`, but
`.read()` never produced a usable frame. Two things were tangled together as
the cause:

1. macOS's camera permission (TCC) is granted to the *terminal application*,
   not to the Python process running inside it — and a permission grant only
   takes effect after a full restart of that terminal app, not just the
   Python process. This is exactly the kind of failure mode that looks like
   a code bug and isn't.
2. AVFoundation camera indices are **not stable**. The Arducam moved from
   index 1 to index 0 mid-session when it was unplugged and replugged,
   silently changing which physical camera a given `--index` pointed at.
   Phase 1's decision log assumed a human-chosen index was stable once
   found; it isn't. For a fixed room camera that has to survive reboots
   without a human re-diagnosing it each time, a silent switch to the
   laptop's built-in webcam is a real safety failure for a child monitor —
   not a cosmetic inconvenience. The durable fix (select by device
   name/unique ID, index only as fallback) is logged in the decision log
   but **not yet implemented**.

**The single most actionable code lesson of the phase:** both
`stream_camera.py` and `detect_stream.py` print `"Streaming from camera
index N. Press 'q' to quit."` immediately after `camera.is_opened` (i.e.
`cap.isOpened()`) comes back true — before a single frame has ever actually
been read. I confirmed this by reading both files directly (`cv/camera.py`
lines 59–61 for `is_opened`, and the print statements immediately following
`if not camera.is_opened: ... return` in both scripts). That one message
asserting success on a check that says nothing about frames actually flowing
is what caused most of the two-hour confusion — a human staring at "success"
text while the camera delivered zero frames has no reason to suspect the
camera. **This is still unfixed** — see open items below.

**False leads that cost real time before the actual causes were found, each
disproven by direct evidence rather than argued away:**
- *OpenCV 5.0.0.93 suspected of an AVFoundation regression* — disproven;
  frames read fine on both 5.0.0 and 4.14.0 once the correct index was used.
- *A pixel-format/FOURCC negotiation failure* — disproven; every format
  combination worked once the right index was in play.
- *ffmpeg failing to capture* — looked significant, turned out to be a red
  herring caused by ffmpeg's own default capture-mode selection, unrelated
  to the actual problem.

**Process lesson worth keeping:** the orchestrator twice inferred which
OpenCV index corresponded to the physical USB camera by reasoning from
`system_profiler` / ffmpeg device-enumeration order, and was wrong both
times. The user's direct empirical check (plug/unplug and watch which index
disappears, or read an actual frame) was correct both times. Enumeration
order is not a reliable proxy for what OpenCV will actually hand you at a
given index — a lesson that generalizes past this specific bug.

**Benchmark estimation was also unreliable without direct measurement.**
Published benchmark ratios, scaled from Ultralytics' own CPU/ONNX figures,
predicted `yolo26x` at roughly 12 FPS on this machine. Measured reality was
35 FPS — off by about 3x. Only measuring directly on the actual target
hardware was trustworthy; scaling someone else's numbers wasn't.

## What was measured

All benchmarks below are on this machine (M5 Pro, MPS, 1080p input):

| model | imgsz 640 | imgsz 1280 | imgsz 1600 |
|---|---|---|---|
| n | 159 fps | — | — |
| s | 145 fps | — | — |
| m | 76 fps | 26.6 fps | 17.0 fps |
| l | 62 fps | 21.7 fps | 13.9 fps |
| x | 35 fps | 11.2 fps | 6.9 fps |

The camera itself caps at 25 fps at 4K and 30 fps at 1080p, so at the default
`imgsz 640` every model size in the family already outruns the sensor —
nano's speed advantage over `l`/`x` at that resolution is headroom nobody can
spend. That's the actual argument logged for defaulting to `yolo26l`: it
trades some of that unusable headroom for 55.0 COCO mAP versus nano's 40.9,
while still leaving roughly 2x margin over the camera for Phase 4's risk
engine and Phase 5's buffer to spend later.

**The central technical finding of the phase:** scissors sitting about 150px
tall in a 1080p frame were undetectable by *every* model size at the default
`imgsz 640`, even lowering the confidence threshold to 0.01. Raising input
resolution surfaced them — `yolo26x` reached a 0.734 confidence score on
scissors at `imgsz 1600`. The reason is mechanical, not mysterious: YOLO
downscales a 1920x1080 frame to a 640px square by default, and that
downscale destroys exactly the pixel detail small objects depend on to be
classified at all. Model size and resolution compound rather than
substitute for each other — `m` and `l` only reached confidence scores
around 0.17–0.20 on the scissors even at high `imgsz`, while `x` was the
only size where this became a genuinely usable detection, not a lucky one.

## Where the accuracy question actually landed

This evolved over the course of the session and should not be presented as
settled in either direction:

1. From one close-up test frame, the orchestrator initially concluded stock
   COCO detection of small hazards was inadequate and would require Phase
   3 fine-tuning to fix.
2. That was retracted once the `imgsz` mechanism above was found — the
   apparent failure was a resolution artifact, not necessarily a domain gap.
3. The user then tested across multiple scenes at default settings and
   reported detection was still inaccurate, which pointed back toward a
   real stock-COCO gap on small hazards, not purely a resolution artifact.
4. After switching to the high-resolution / larger-model configuration and
   capturing more scenes, the user reported detection was "way better."

**Net position, stated honestly: this is unresolved and trending positive,
not settled.** The correct statement for the record is that stock COCO's
adequacy for small hazards at realistic room distance depends on
configuration (model size and inference resolution both matter, and
compound), and whether it's *good enough as-is* versus *needs Phase 3
fine-tuning* is an open empirical question that Phase 3's systematic dataset
work needs to answer — not something this phase decided.

## An open architectural tension, flagged, not resolved

CLAUDE.md decision 7 assumes stock COCO classes (person, oven, knife,
scissors, ...) already fit well enough that fine-tuning is only needed for
what COCO doesn't have at all (small choking-hazard objects, stairs). This
phase put real pressure on that assumption — the scissors finding above is
exactly the kind of "COCO has the class but doesn't reliably detect it at
this size/distance/resolution" case decision 7 didn't anticipate — without
resolving it either way. I am flagging this as an open question for the
orchestrator to weigh going into Phase 3. **I am explicitly not amending
CLAUDE.md** — that decision belongs to Shaked and Yahli, not to me.

There's a second, related tension worth raising for Phase 4: CLAUDE.md
decision 2 requires a single YOLO pass per frame covering both the child and
all hazards together. But this phase's own numbers show person detection
holding up fine at cheap settings (`imgsz 640`), while small hazards need
`imgsz 1600` and a much heavier model — the cheap, fast configuration and the
thorough, small-object-capable configuration are not the same pass. It's
possible decision 3 already resolves this — Layer A's hazard map doesn't
need to update every frame the way Layer B's child-proximity tracking does,
so a cheap-and-frequent pass for the child plus a heavier-and-less-frequent
pass for hazards might satisfy both decisions without violating either. I am
raising this as a question for the orchestrator, not deciding it — this is a
Phase 4 design question, not a Phase 2 finding to be adjudicated here.

## Audit findings

I read `cv/camera.py`, `cv/stream_camera.py`, `cv/detect_stream.py`,
`cv/requirements.txt`, `cv/README.md`, `.gitignore`, and
`docs/decision-log.md` directly, plus `cv/detect_cameras.py` for continuity
with Phase 1.

**What checks out:**
- `cv/camera.py` preserves Phase 1's constants and behavior exactly:
  `DEFAULT_CAMERA_INDEX = 0`, `MAX_CONSECUTIVE_READ_FAILURES = 10`,
  `REOPEN_RETRY_DELAY_SECONDS = 1.0`, and the same reopen-with-backoff loop.
  This is a genuine extraction, not a rewrite — `stream_camera.py`'s
  user-visible behavior (prompts, key handling, `finally`-block cleanup) is
  unchanged; it now delegates frame acquisition to `CameraCapture` instead
  of owning a `cv2.VideoCapture` directly.
- `cv/detect_stream.py` does exactly what Phase 2 claims and nothing more.
  One `model.predict(...)` call per frame (single pass, confirmed by
  reading the loop body once), `results[0].plot()` draws every class found
  with no filtering to a hazard subset, and there is no risk/proximity
  code, no hazard map, no alert logic, and no clip buffer anywhere in the
  file. No stubs, no TODOs standing in for later-phase work.
- The model is loaded once (`load_model()`, before `camera = CameraCapture(...)`
  and before the `for frame in camera.frames()` loop starts), not per frame.
- `cv/requirements.txt`'s pins (`torch==2.13.0`, `torchvision==0.28.0`,
  `ultralytics==8.4.115`) match what was reported as actually installed and
  tested. I could not independently re-verify the installed environment
  against these pins — docs-agent has no shell access in this project, only
  Read/Grep/Glob/Write/Edit — so this is a check of the file's content, not
  independent confirmation of the running venv. Worth someone running
  `pip freeze` against this file at some point rather than assuming.
- `docs/decision-log.md`'s Phase 2 entries were internally contradictory
  before this review: the 2026-08-04 section states `yolo26n.pt` as the
  Phase 2 default without qualification, while the 2026-08-05 section
  changes the default to `yolo26l.pt` — a reader skimming just the first
  section would come away believing something the code no longer does.
  I've added a note to the 2026-08-04 entry pointing forward to the
  supersession rather than leaving the contradiction standing.

**Gaps found that nobody had flagged before this review:**
- **`cv/README.md` is stale relative to the current code.** It still says
  "the first run of `detect_stream.py` downloads `yolo26n.pt`" and "plus a
  YOLO26n pass on every frame," both of which describe the pre-bring-up
  default. The actual default is `yolo26l.pt`. The README's "Flags" section
  also only documents `--index`, `--device`, and `--conf` — it's missing
  `--model` and `--imgsz` entirely, which are exactly the two flags the
  bring-up session showed actually matter most for hazard-detection
  quality. Someone reading only the README today would be pointed at the
  wrong default model and wouldn't know the two most consequential flags
  exist. This should be routed back for a quick fix; I haven't changed
  `cv/README.md` myself since it isn't one of my named deliverables for
  this phase, but it should not wait for Phase 3 to be corrected.
- **Minor comment overclaim, not a functional bug:** `detect_stream.py`'s
  comment on `PYTORCH_ENABLE_MPS_FALLBACK` says the env var is set "per
  CLAUDE.md / Phase 2 spec." I re-read `CLAUDE.md` in full for this review
  and it says nothing about MPS, PyTorch fallback behavior, or Apple GPU
  handling at all — that's a reasonable engineering decision on its own
  merits, but attributing it to CLAUDE.md specifically overstates the
  source. Small, but the kind of thing worth catching before "per CLAUDE.md"
  becomes a habit for justifying choices CLAUDE.md never actually made.

**Does this match CLAUDE.md's architecture?** Yes, on the things Phase 2 is
actually responsible for: decision 2 (single YOLO pass per frame) is
implemented correctly and verified by reading the loop body, not inferred
from a docstring. Decisions 3–8 aren't yet implemented and shouldn't be —
Phase 2 correctly stayed out of hazard-map, risk-scoring, alerting, and
storage territory. Nothing here overreaches into backend-agent or ui-agent
scope.

## Open items — not done, deliberately deferred (status as of original sign-off)

- `stream_camera.py` and `detect_stream.py` should read and verify one real
  frame before printing any success message, and should fail with a message
  naming the actual likely causes (macOS camera permission tied to the
  terminal app, wrong/shifted device index, device already held by another
  process) instead of a bare "could not open camera" message. This is the
  single highest-value fix coming out of this phase and is not yet done.
  **Done — see "After sign-off, Round 1" below.**
- Camera selection by device name/unique ID, with numeric index as a
  fallback only — needed because AVFoundation indices are not stable across
  replugs, which is a safety-relevant gap for a fixed room camera, not
  cosmetic. **Done, then found to be built on a flawed assumption, then
  redone — see "After sign-off, Rounds 1 and 3" below. This is the part of
  the phase most worth reading if you only read one section.**
- Whether Phase 4's risk engine can honor CLAUDE.md decision 2 (one YOLO
  pass covering person + hazards together) given that reliable person
  detection and reliable small-hazard detection don't currently share a
  cheap configuration — raised above as a question for the orchestrator,
  not resolved here. **Still open** — nothing in the post-sign-off rounds
  touched detection code at all; they were entirely about camera selection
  and reconnect reliability.
- `cv/README.md`'s stale model default and missing `--model`/`--imgsz` flag
  documentation (see audit findings above). **Fixed** — re-read as part of
  this update: `cv/README.md` now correctly documents `yolo26l.pt` as
  default and both `--model` and `--imgsz` in its Flags section.

## After sign-off: three more rounds, and the most important story in the phase

Phase 2 was signed off above on 2026-08-05 with two explicit follow-ups
outstanding: the misleading `isOpened()`-only success message, and
device-name selection. Both got built. One of them then failed in the field
in a way that could have mattered for a child-safety monitor, and had to be
rebuilt on a different foundation. This section covers all three rounds,
audited against the current state of `cv/camera.py`, `cv/stream_camera.py`,
`cv/detect_stream.py`, `cv/detect_cameras.py`, `cv/requirements.txt`, and
`cv/README.md`, all re-read in full for this update — not summarized from
memory of the earlier audit.

### Round 1 — closing the two follow-ups

**Startup verification.** `CameraCapture.verify_startup()` (`cv/camera.py`,
confirmed present) now reads up to `STARTUP_VERIFY_ATTEMPTS = 15` real frames,
0.2s apart, before either script is allowed to print anything claiming
success. Both `stream_camera.py` and `detect_stream.py` call it and only
print "Streaming from camera..." after it returns `True` — confirmed by
reading both scripts' `main()` directly. On failure, `startup_failure_message()`
lists causes in the order bring-up actually hit them: terminal-app camera
permission first (it cost the most time and is least obvious), then
wrong/shifted index, then another process holding the device. The old
`isOpened()`-only success print is gone from both scripts; I searched for it
and found no remaining trace.

**Name-based selection (first version).** `--name` was added, resolving a
substring like `"Arducam"` to an OpenCV index. The first implementation did
this *positionally*: it shelled out to `system_profiler SPCameraDataType`,
took the Nth device in that listing, and assumed N was also the OpenCV index
of the same physical device — on the strength of having watched the two agree
by hand during the original bring-up. This is the version Round 3 below
replaced; see there for what happened to it.

### Round 2 — two bugs live testing found that no design review would have

**Bug: `q` did nothing during a read-failure streak.** Root cause, confirmed
by reading the old code: `frames()`'s failure branch ended in `continue`
without ever `yield`ing, so control never returned to the caller's loop body
— and the caller's loop body is where `cv2.waitKey()` lives, which is not
just the key handler but the *only* thing that pumps the OpenCV GUI event
loop. No `waitKey`, no rendering, no responding to the window's close button,
no responding to `q` — the window would sit frozen for however long the
failure lasted, recoverable only by `Ctrl+C`. This is a defect the Phase 2
refactor introduced without anyone noticing: Phase 1's original inline loop
called `waitKey` unconditionally every iteration; pulling the loop out into a
generator silently changed that contract, and nothing caught it until a human
watched a frozen window.

The fix, confirmed present in the current `frames()`: every failed read now
falls through to `yield None; continue` instead of a bare `continue`, and
both callers check `if frame is not None:` before displaying, but call
`cv2.waitKey()` and the window-close check unconditionally, every iteration,
regardless of what was yielded. The docstring on `frames()` now states this
contract explicitly as binding on every future caller, including the Phase 5
rolling buffer. I verified this contract is actually honored by both current
callers, not just documented — read both `main()` functions directly.

**Bug: replug never actually recovered.** Root cause, confirmed by reading
the old reopen code: the mid-stream reopen path — recreating the
`cv2.VideoCapture` after too many consecutive failures — checked only
`cap.isOpened()` to decide the reopen had worked. This is the *exact*
false-positive the module's own docstring already warned about for the
*first* open (`isOpened()` can be `True` while every subsequent `.read()`
fails) — but the lesson had not been applied to the reopen path itself. A
reopen would report success, reset the failure counter to zero, and then
immediately start failing reads again from the top — indistinguishable in
the logs from the original failure ever having stopped. From the outside this
looked like "stuck," when it was actually "resetting and failing, over and
over, silently."

Fixed, confirmed present: the reopen path (`_reopen()`) now reuses the same
"read real frames, don't trust the flag" approach as `verify_startup()` —
`REOPEN_VERIFY_ATTEMPTS = 5` reads before declaring a reopen successful.
Separately fixed in the same round: escalating backoff
(`REOPEN_INITIAL_DELAY_SECONDS = 1.0` up to `REOPEN_MAX_DELAY_SECONDS = 8.0`,
capped, resetting to 1s on a successful reopen) instead of a flat 1s delay,
and reopen-attempt logging throttled after the first few attempts
(`REOPEN_LOG_ALWAYS_FIRST_N = 3`, then every `REOPEN_LOG_EVERY_N_ATTEMPTS =
5`th) so a long outage doesn't spam the console. **Stated caveat, worth
repeating rather than letting it quietly drop:** the decision log records
this fix was verified against a monkeypatched fake capture object, not real
unplug/replug hardware, because no camera permission was available in that
session. I have not seen evidence since that the *reopen-recovers-from-a-
real-physical-replug* path specifically has been exercised on real hardware —
see "What I'd still want proven" near the end of this section.

### Round 3 — the feature shipped, failed in the field, and was redesigned

This is the round that actually matters most for the paper.

**What happened.** Running `detect_stream.py --name Arducam`, the script
printed `Camera selected: index 1 (Arducam-B0560-4K HDR)` — evidence,
printed, looking exactly like the "never assert an unverified name" design
was working — while the video window showed the MacBook's built-in webcam.
Root cause: Round 1's name resolution matched a name to an index by
*position* in `system_profiler`'s device listing, on the strength of having
verified that position agreed with OpenCV's actual index order earlier that
same day. It stopped agreeing. Measured at the moment of failure (this table
is also in `docs/decision-log.md` and is the load-bearing evidence for the
whole redesign, so it's worth reproducing here too):

| enumeration source | order |
|---|---|
| `system_profiler SPCameraDataType` | MacBook 0, Arducam 1, iPhone 2 |
| pyobjc `AVCaptureDevice.devicesWithMediaType_` | MacBook 0, Arducam 1, iPhone 2 (identical) |
| OpenCV's actual index space | **Arducam 0, MacBook 1** |

Both independent enumeration sources agreed with each other and both
disagreed with OpenCV. This is the detail that makes the incident more than
a one-off bug: **the documented escape hatch in the original code was "if
this positional assumption breaks, switch to pyobjc"** — and pyobjc produced
the identical wrong order. The fallback plan rested on the same unproven
premise as the thing it was a fallback *for*: that some enumeration order,
gotten some other way, could be trusted to correspond to OpenCV's index
space. It couldn't, regardless of which tool produced it. I read
`cv/camera.py`'s current module docstring and `docs/decision-log.md`'s entry
for this incident and confirmed both now state this explicitly rather than
leaving it as an implied lesson.

**The redesign: identify a device by capability, not position.** I read
`cv/camera.py` end to end to verify this is actually what the code does, not
just what the docstring claims:

- `named_device_capabilities(name_substring)` asks AVFoundation what
  resolution ceiling a *named*, currently-connected device supports — a
  property of the hardware itself, queried live, never cached, never
  positional.
- `probe_index_capability(index)` opens an OpenCV index directly, requests
  an unreachable 7680x4320, reads back a **real frame**, and reports the
  actual delivered resolution (from `frame.shape`, explicitly never from
  `cap.get()` — see the theme section below for why that's not incidental).
- `resolve_name_to_index_by_capability()` matches the two by pixel area
  within a documented tolerance (`CAPABILITY_MATCH_AREA_TOLERANCE = 0.10`)
  and raises a specific `CameraSelectionError` for every non-unique outcome
  — name not connected, name ambiguous, no index confirms the capability, or
  more than one index does. I confirmed by reading the function that it
  never falls through to a best-effort guess; every branch either returns a
  single confirmed match or raises.
- I confirmed the two positional functions this replaced —
  `list_camera_devices()` and `resolve_name_to_index()` — are not merely
  deprecated but **actually gone**: a repo-wide search for both names turns
  up nothing outside historical prose (docstrings, README, decision log)
  describing what used to exist. No `subprocess` import remains in
  `camera.py` either. This was a real removal, not a "kept around just in
  case" shortcut left callable for someone to stumble back into.

**Measured signatures, confirmed distinct** (per the decision log, reproduced
in `cv/camera.py`'s comments): Arducam 3840x2160, MacBook Pro Camera 1552x1552
(a square crop — its widest active format is not classic 16:9, so the
orchestrator's original "4K vs 1080p" framing was too coarse; cv-agent's
correction to compare full dimensions rather than a resolution-tier label is
the right call and is what the code actually does), iPhone Continuity Camera
1920x1440. Worth flagging as a minor, not disqualifying, observation: the
MacBook (2,408,704 px) and iPhone (2,764,800 px) areas differ by about 15%,
against a 10% match tolerance — comfortably distinct today, but not by a wide
margin. If a fourth camera with a similar pixel count ever entered the mix,
capability matching could become genuinely ambiguous rather than just
theoretically so. Not a bug — the code already handles "ambiguous" by
refusing to guess and raising — but worth knowing the margin isn't huge.

**A bug caught before it ever reached a user.** The capability probe
originally set requested resolution but not pixel format (FOURCC). Cross-
referencing an unrelated diagnostic run from earlier in the *same* session
showed the Arducam delivers 1920x1080 with no format specified, and
3840x2160 only under MJPG — true 4K exceeds USB bandwidth uncompressed. Left
as-is, the probe would have fingerprinted the Arducam at a third of its real
capability, permanently mismatching AVFoundation's reported 4K ceiling and
making every `--name Arducam` lookup fail with "capability unconfirmed,"
never with the wrong-camera failure it was built to prevent, but wrong
nonetheless. I confirmed the fix is in place: `CAPABILITY_PROBE_FOURCCS =
("MJPG", None)`, tried in that order, keeping whichever format returns the
larger frame. **Notably, this bug was caught by the orchestrator, not by
cv-agent** — it required remembering evidence from an entirely separate part
of the same session that the subagent doing the capability-matching redesign
had no way to see. See `docs/agent-workflow-notes.md` for why I think this is
one of the more important process observations of the whole phase.

**What's confirmed working now, and what isn't.** Per the orchestrator, the
user has since run `detect_cameras.py` and `--name Arducam` on real hardware
and confirmed both work, including the camera identity being correct — I
have not personally watched this happen (I have no camera or shell access),
but the claim is consistent with everything I read in the code, and I've
recorded it in the decision log as a live-verification update. What I have
**not** seen claimed or evidenced anywhere: that the reopen/reconnect path
specifically has been exercised with a real physical unplug/replug under the
current capability-matching code. The decision log's Round 2 entry states
plainly that the reopen fix itself was only verified against a mocked
capture object. Given that a full name re-resolution on reopen now means
probing up to `CAPABILITY_PROBE_MAX_INDEX = 6` indices, each under up to two
FOURCC settings, each with up to `CAPABILITY_PROBE_READ_ATTEMPTS = 10` read
attempts at 0.2s apart — a worst-case single reopen attempt for `--name`
selection is now on the order of tens of seconds, not the ~2 seconds an
`--index` reopen takes. That is a real, currently undocumented-as-a-tradeoff
consequence: **during that entire window, the caller is blocked inside a
single call to `_reopen()`, so `cv2.waitKey()` is not being called, so the
Round 2 "`q` and window-close now work during failures" fix does not apply
during an active reopen attempt** — it applies between reopen attempts and
during the ordinary failure-counting that precedes the first reopen, not
during the reopen call itself. This is a narrower version of the exact
problem Round 2 fixed, reopened (no pun intended) by Round 3's heavier
identity check, and as far as I can tell nobody has flagged it yet. I'm
flagging it now: worth a deliberate decision (accept it, given how rare a
mid-session replug during active recording should be; or bound the probe
sweep so a reopen can't block for that long) rather than leaving it as an
unnoticed side effect of an otherwise sound redesign.

### The theme: "isOpened() lies" shows up three times, not once

It's worth naming as a single recurring lesson rather than three unrelated
incidents, because that's actually more useful for the next person than
treating each as its own bug:

1. **At first open** (the original bring-up, before sign-off):
   `cap.isOpened()` returned `True` while `.read()` never produced a frame,
   for two hours, because of a macOS permission scoping issue. Fixed by
   `verify_startup()` reading real frames before declaring success.
2. **At reopen** (Round 2): the exact same false-positive, in the exact same
   codebase, in a code path added *after* the lesson from (1) had already
   been learned and written into the module's own docstring — and still
   wasn't applied there until live testing forced the issue. Fixed by giving
   `_reopen()` its own real-frame verification.
3. **In the capability probe** (Round 3): `probe_index_capability()`
   deliberately reads `frame.shape` for width/height and explicitly never
   calls `cap.get(cv2.CAP_PROP_FRAME_WIDTH/HEIGHT)` — because `cap.get()` is
   subject to the identical class of lie as `isOpened()`: it can report a
   requested or negotiated value that doesn't reflect what `.read()` is
   actually handing back. The comment in the code states this explicitly.
   This is the same lesson applied *proactively* for once, rather than
   learned the hard way a third time — worth noting as the one place in this
   whole saga where a past failure mode was correctly anticipated rather
   than rediscovered.

The generalizable version: **a status flag or a negotiated setting is a
claim, not a measurement.** Anywhere this codebase needs to know something is
actually true about a camera — that it's open, that it delivered a frame,
that it's a specific physical device, that it's running at a given
resolution — the only thing trusted is a real frame read back and inspected.
That is a genuinely disciplined pattern by the end of Round 3, and it took
three rounds of getting burned to get there.

### The other theme: liveness is not identity

The three rounds also draw out a distinction worth stating plainly, because
it's easy to blur: **confirming a camera delivers frames says nothing about
which camera it is.** `verify_startup()` and the Round 2 reopen fix both
solve *liveness* — "is something actually streaming." Round 3 exists entirely
because a name resolution can be perfectly live (it opened, it read frames,
nothing crashed) while confidently being the wrong physical device. The
positional name-resolution bug never failed a liveness check — the MacBook
webcam is a perfectly good, perfectly live camera. It failed an identity
check that Round 1's version never actually performed, despite printing text
that implied it had. For a monitor whose entire premise is watching one
specific, fixed room, identity is the property that actually matters, and
it's the harder, later-solved one of the two.

### What I'd still want proven before trusting this fully

- ~~Physical unplug/replug recovery, on real hardware, under the *current*
  capability-matching reopen path~~ — **done (2026-08-05). The user ran the
  physical unplug/replug test against the capability-matching reopen path
  and it recovered correctly.** With this, every safety-relevant claim about
  camera identity in this phase rests on a real-hardware test rather than a
  mock. The remaining open item below (reopen responsiveness) is about the
  window briefly freezing, not about monitoring the wrong camera.
- Whether the tens-of-seconds worst-case reopen latency for `--name`
  selection (flagged above) is something the team wants to accept, bound, or
  address — right now it's an unflagged side effect, not a decision anyone
  has weighed in on.
- The `CAPABILITY_MATCH_AREA_TOLERANCE = 0.10` value is logically sound but,
  by the decision log's own admission, was "not verified against real
  hardware" at the time it was set — the recent live confirmation of
  `--name Arducam` is good evidence it works for *this* team's three
  devices today, but it hasn't been stress-tested against a closer pair of
  resolutions than the ~15% margin the MacBook/iPhone pair happens to have.

## Verdict

**Phase 2 is functionally complete and matches what `PHASE_PLAN.md` and
CLAUDE.md ask of it — code-wise, sign-off stands.** The detection pipeline
does one clean YOLO pass per frame across all stock COCO classes, with no
scope creep into later phases and no stubs standing in for unfinished work.
The camera-capture refactor is a genuine, behavior-preserving extraction, not
a rewrite in disguise. Both follow-ups outstanding at original sign-off are
now closed: the misleading `isOpened()`-only success print is gone, and
camera selection is now identity-verified by capability rather than asserted
positionally.

**What I would not want waved past silently, updated for the current state:**
the reopen-during-capability-reprobe responsiveness gap described above (a
narrower recurrence of the exact freeze Round 2 fixed), and the fact that
real-hardware verification of *reconnect* specifically (as opposed to
cold-start identity resolution) still hasn't been demonstrated as far as I
can tell. Neither blocks the camera layer from supporting Phase 3's dataset
collection — Phase 3 is mostly steady-state capture, not aggressive
unplug/replug testing — but both should be watched, not forgotten, before
this code is asked to run truly unattended (which matters more starting
Phase 4/5, when the system is meant to run without anyone standing over it).

**Is the camera layer sound enough to carry Phase 3?** Yes, with the two
caveats above named rather than swept past. Identity resolution is now
verified on real hardware, not just logically argued for; the quit-key
freeze that would have made a bad session actively annoying to recover from
is fixed for the common case; and the one remaining gap (reopen
responsiveness during an active capability re-probe) is a latency/UX issue,
not a "wrong camera" or "silent corruption" issue — it degrades to "the
window is briefly unresponsive during a replug," not to a safety-relevant
misidentification. That is a meaningfully different risk class than the one
this whole saga was about, which is why I don't think it should block
starting Phase 3, only that it shouldn't be forgotten before Phase 4/5.

## Questions to check your own understanding

1. `detect_stream.py` calls `model.predict()` exactly once per frame and
   draws every class it returns, with no filtering. Why does CLAUDE.md
   insist on this — what would go wrong architecturally (not just
   performance-wise) if you ran a separate, second pass just for the
   person class?
2. The phase found that raising `imgsz` mattered more than raising model
   size for detecting small objects like scissors. Explain *why*, in terms
   of what YOLO actually does to a 1920x1080 frame at `imgsz=640` — and
   then explain why that fact might complicate Phase 4's single-pass
   requirement rather than just being a Phase 2 footnote.
3. The camera index instability bug (Arducam silently moving from index 1
   to index 0) wasn't caught by any test — it was caught because a human
   physically replugged the camera mid-session. What kind of check, if any,
   could `camera.py` run at every reconnect to at least detect (not
   necessarily fix) that the physical device behind an index has changed?
4. Round 3's positional name-resolution bug shipped with a documented escape
   hatch ("switch to pyobjc if this breaks") that turned out to rest on the
   exact same flawed assumption as the code it was meant to rescue. What
   made that assumption wrong in a way that no amount of *additional*
   positional enumeration — from any tool — could have fixed? Put another
   way: what would you have needed to check, before writing that escape
   hatch into the code, to notice it wasn't actually a fallback?
5. `verify_startup()` and `probe_index_capability()` both refuse to trust a
   status flag (`isOpened()`, `cap.get()`) and insist on reading a real
   frame instead. Explain the difference between what each of those two
   functions is actually trying to prove — one proves *liveness*, the other
   proves *identity* — and why a camera that passes the first can still
   completely fail the second.
