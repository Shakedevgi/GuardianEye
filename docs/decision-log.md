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
