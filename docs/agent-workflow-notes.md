# GuardianEye — Multi-Agent Workflow Notes

Running narrative of how the "simulated engineering team of Claude Code
subagents" approach is actually going, as its own deliverable (per CLAUDE.md's
team-structure section, this is not a footnote — it's material for the final
paper).

---

## Entry 1 — Scaffold + Phase 1 (2026-08-03)

### The initial scaffold

The project started with a single commit (`a181780`) that laid down
`CLAUDE.md` (architecture decisions, locked unless deliberately revised),
`PHASE_PLAN.md` (9 phases with explicit dependencies and a hard rule: no
phase N+1 before phase N is "closed" — demoed, docs-agent reviewed, write-up
produced), and four subagent role files (`cv-agent`, `backend-agent`,
`ui-agent`, `docs-agent`, later moved into `.claude/agents/` in commit
`89b4246`). The intent, stated explicitly in CLAUDE.md, is dual: build
GuardianEye correctly, *and* generate a real case study in orchestrating
multiple AI agents on non-trivial, order-dependent work. Treating that
meta-goal as a first-class deliverable from commit one — rather than
retrofitting it later — is itself worth noting: it means role boundaries and
phase gates were defined before any code existed, not reverse-engineered from
however the work happened to get done.

### The rhythm (as it played out in Phase 1)

The division of labor is strict by design: cv-agent owns camera/YOLO/risk
logic, backend-agent owns FastAPI/SQLite, ui-agent owns Flet, and
docs-agent (me) is explicitly barred from being "just a scribe" — CLAUDE.md
and PHASE_PLAN.md both state a phase isn't complete until docs-agent has
independently verified the code matches the claim and produced a write-up.
That's a meaningful check on a known failure mode of agent-driven
development: an agent (or a human skimming its output) declaring victory
because the code *looks* plausible, without anyone re-reading it adversarially
against the original spec.

### How Phase 1's delegation actually went

This is the part I can verify directly, having read the code myself rather
than taking the "what was built" summary on faith.

**Scope discipline was clean.** cv-agent stayed inside its lane: two scripts
(`detect_cameras.py`, `stream_camera.py`), a minimal `requirements.txt`
(`opencv-python` only — a comment in the file explicitly notes YOLO/torch are
deliberately deferred to Phase 2, not smuggled in early), a scoped README, and
a root-level `.gitignore`. Nothing touches FastAPI, Flet, or SQLite territory,
and nothing pulls in AI/model dependencies ahead of where PHASE_PLAN.md says
they belong. That's a good sign for the "agents don't quietly encroach on each
other's territory" question this project is implicitly testing.

**Hardware-dependent verification correctly stayed human-in-the-loop.** No
subagent can plug in a USB camera or judge whether a live OpenCV window is
actually rendering a picture. The workflow handled this correctly: cv-agent
wrote the two scripts, and Shaked was the one who ran
`python cv/detect_cameras.py` to find the real device index, then ran
`python cv/stream_camera.py --index 0` and watched a real live feed for
several minutes. This is exactly the kind of step no agent should be trusted
to self-certify, and it wasn't — the loop closed with a human hands-on-camera,
not an agent claiming it "should work."

**The code itself is honest about its own limits**, which matters more than it
might sound: `stream_camera.py`'s reconnect logic is commented as "simple
backoff — this is Phase 1, not a production reconnect strategy," and the
default camera index is flagged in-code as "very likely NOT the USB camera on
a laptop." That kind of self-aware, undersell-don't-oversell commenting makes
my auditor job easier — there was no gap between what the code claimed to do
and what it does, which is exactly the thing I'm supposed to be checking for
and exactly the thing that's easy for an agent to get sloppy about under
pressure to "finish a phase."

### What I'd flag for the team to think about going forward, not urgent yet

- Nothing in Phase 1 required cross-agent coordination (only cv-agent
  touched code), so this entry can't yet speak to how well the agents
  *hand off* work to each other — that's untested until Phase 5+ (cv-agent
  → backend-agent clip storage hookup) or Phase 9 (all agents). Worth
  watching closely when that first happens.
- The phase-gate rule ("docs-agent signs off before N+1 starts") is only as
  strong as someone actually enforcing it before typing "start Phase 2." Phase
  1 did wait for this review, which is the right precedent to keep.

### For the final paper

If I had to summarize this entry for someone who wasn't in the room: the
single most interesting design choice so far isn't the code, it's that the
team defined "done" (demo + independent audit + write-up) *before* writing any
code, and set up an agent whose only job is to be the skeptical reader. Phase
1 is too small a sample to know if that holds up under real pressure (a
messier phase, a deadline, a disagreement about scope) — but it held for
Phase 1.
