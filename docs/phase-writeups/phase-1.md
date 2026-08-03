# Phase 1 — Foundation

**Status: complete (audited 2026-08-03).**

## What Phase 1 was supposed to prove

Per `PHASE_PLAN.md`, Phase 1 has nothing to do with AI or risk scoring yet.
It exists to prove one boring but load-bearing thing: that you can reliably
pull frames off the actual USB camera hardware, on this actual machine, and
show them in a window, continuously, without the program crashing. The
"done-bar" is explicitly: run one script, see a live raw video window, no
crashes, for several minutes.

Why bother with a phase this small? Because everything from Phase 2 onward
(YOLO, risk scoring, MJPEG streaming) is built *on top of* whatever capture
loop gets written here. If frame capture itself is flaky — wrong device
index, camera drops out under load, window handling is fragile — that bug
would otherwise get discovered three phases later, buried under YOLO and risk
logic where it's much harder to isolate. Phase 1 isolates it now, with nothing
else in the way.

## What was actually built

Two scripts and supporting scaffolding, all under `cv/`:

- **`cv/detect_cameras.py`** — a throwaway diagnostic tool. It probes OpenCV
  device indices 0 through 5, tries to open each and read a real frame, and
  prints a report (opened? frame readable? resolution?) for each index. This
  exists because OpenCV identifies cameras with a plain integer index, and
  there is no reliable cross-platform way to know in advance which index
  belongs to the built-in laptop webcam versus a specific external USB
  camera — the OS just hands them out in whatever order it discovered the
  devices. A human has to look at the report and figure out which index is
  which (the script even suggests unplugging the USB camera and re-running to
  see which index disappears).

- **`cv/stream_camera.py`** — the actual Phase 1 deliverable. Takes
  `--index <n>` (defaulting to `0`, explicitly commented as "probably wrong
  on a laptop — figure out your real index with `detect_cameras.py`"), opens
  that camera, and loops: read a frame, show it in an OpenCV window, check for
  the `q` key or the window's close button, repeat. If a frame read fails
  repeatedly (say, a brief USB disconnect), it doesn't crash — after 10
  consecutive failures it releases the capture device, waits a second, and
  reopens it, then keeps going. Ctrl+C and window-close are both handled
  cleanly in a `finally` block that releases the camera and destroys the
  window.

- **`cv/requirements.txt`** — just `opencv-python`. Nothing else. A comment in
  the file explicitly notes that YOLO/torch/ultralytics are deliberately held
  back for Phase 2, per the phase plan — a small but real discipline check:
  it would have been easy to install everything up front "to save time later,"
  and the team didn't.

- **`cv/README.md`** — setup and run instructions matching what the code
  actually does (venv creation, `pip install`, run `detect_cameras.py` first,
  then `stream_camera.py --index <n>`).

- **Root `.gitignore`** — keeps `.venv/`, `__pycache__/`, `.DS_Store`, and
  editor directories out of version control.

## How it was verified

This is the part that can't be faked or automated: Shaked physically ran
`python cv/detect_cameras.py`, found the real USB camera at index 0 on his
machine, then ran `python cv/stream_camera.py --index 0` and watched a live
window feed off the actual hardware for several minutes without a crash. No
subagent can validate "does a window pop up showing my actual room" — that
required a human at the keyboard, and that's exactly what happened here rather
than an agent asserting the code "should work."

## Audit findings

I read every line of `cv/detect_cameras.py`, `cv/stream_camera.py`,
`cv/requirements.txt`, `cv/README.md`, and the root `.gitignore` directly (not
a summary of them). Findings:

- **No stubs, no fake/placeholder logic.** Both scripts do exactly what their
  docstrings claim. There's no "TODO: implement real reconnect logic" or
  silently-swallowed exception hiding a real problem.
- **Matches CLAUDE.md's implicit expectations for this layer.** CLAUDE.md
  doesn't dictate camera-capture internals directly (that's a Phase 2+
  concern once YOLO enters), but Phase 1's job — a clean, isolated capture
  loop that later phases build on — is exactly what got delivered, with
  nothing from later phases (YOLO, risk scoring, FastAPI, Flet) leaking in
  early.
- **Reconnect logic is honestly scoped.** The code and its comments are
  explicit that this is a "simple backoff, not a production reconnect
  strategy" — an accurate self-description rather than an overclaim. Good
  practice: it tells the next phase's implementer (also cv-agent) exactly
  what corners were cut on purpose.
- **One minor thing worth having on the record, not a blocker:** the reconnect
  backoff (10 failed reads, 1 second retry delay) is untested against a *real*
  disconnect-reconnect cycle — the "no crashes for several minutes" success
  criterion was verified under normal operation, not under a deliberately
  induced camera dropout. That's fine for Phase 1's bar as literally written,
  but it's a latent gap the team should know they haven't exercised yet if a
  flaky USB cable becomes a real symptom in the demo room.
- **Scope stayed inside cv-agent's lane.** Nothing here touches FastAPI,
  Flet, or SQLite, and no AI/model dependency was pulled in ahead of Phase 2.

No blockers. Phase 1 is marked complete in `PHASE_PLAN.md`.

## Questions to check your own understanding

1. Why does `detect_cameras.py` exist as a separate script instead of just
   hardcoding a camera index into `stream_camera.py`? What would break, and
   for whom, if you hardcoded it?
2. The reconnect logic waits for 10 consecutive failed frame reads before
   reopening the capture device, rather than reopening on the very first
   failure. Why might reopening on the first failure actually be *worse*
   behavior for a camera that's just having one bad frame?
3. `requirements.txt` deliberately does not include YOLO/torch yet, even
   though we know Phase 2 needs them imminently. What's the actual argument
   for waiting, beyond "the phase plan says so" — what could go wrong if
   Phase 1 and Phase 2 dependencies got installed and tested together instead
   of in isolation?
