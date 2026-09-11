"""alert_audio.py - decides WHEN to play an alert sound. No Flet import here
on purpose: this is pure logic (a dict in, a filename or None out), fully
testable without a running app, a device, or audio hardware.

/risk_status's `alert` object carries `text`/`active`/`clip` (see
backend/API.md). Polled at RISK_STATUS_POLL_HZ (4/sec, see ui/app.py), so
naively "play `clip` whenever `active` is true" would replay the same sound
roughly 16 times over one ALERT_BANNER_SECONDS window (4s at 4Hz) - once is
correct, not sixteen.

TWO layers of "don't spam," deliberately separate:

1. Identity dedup - play once per DISTINCT active alert, identified by
   (text, clip) together, not by the active->true transition alone. Two
   reasons this is NOT the same as "play on the rising edge of active":
     - A second, different alert can arrive and refresh the banner (a new
       speak() call resets banner_until) WHILE the previous one's banner
       is still showing - `active` never dips false in between, so an
       edge-only check would miss the second, genuinely new alert
       entirely.
     - `alert.clip` can be null while `alert.active` is true (a "closed"
       signal still shows banner text but plays no sound, per
       cv/risk_engine.py's AlertDispatcher.current_clip() docstring) -
       this must never be mistaken for "nothing changed, stay quiet" and
       must never itself be treated as something to play.

2. Cooldown - a minimum wall-clock gap (ALERT_SOUND_COOLDOWN_SECONDS,
   default matching cv/risk_engine.py's own GLOBAL_ALERT_MIN_INTERVAL_
   SECONDS = 2.5s, not picked freshly - if that pacing is right for the
   Mac's own speaker, it's the right starting point here too) between any
   two sounds, even for two genuinely DIFFERENT alerts arriving in quick
   succession. Identity dedup alone would let a fast flurry of distinct
   hazards each make a sound; a real device (unlike the backend's own
   AlertArbiter, which already paces SIGNAL GENERATION) has no equivalent
   guard on what actually comes out of its speaker, so this adds one. The
   cooldown clock is NOT reset by a brief active->False->True blip - it is
   purely "how long since a sound last actually played," independent of
   banner state bouncing.

An alert suppressed by the cooldown is NOT remembered as "played" - once
the cooldown window passes, whatever alert is active at that moment (which
may be a later, different one) gets its own chance to play.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

# Mirrors cv/risk_engine.py's GLOBAL_ALERT_MIN_INTERVAL_SECONDS deliberately
# - see the module docstring's "Cooldown" section for why that number,
# not a freshly-invented one.
ALERT_SOUND_COOLDOWN_SECONDS = 2.5


class AlertSoundPolicy:
    """Call `next_clip(status.get("alert"))` once per /risk_status poll.
    Returns the clip filename to play now, or None to do nothing.
    """

    def __init__(self, cooldown_seconds: float = ALERT_SOUND_COOLDOWN_SECONDS,
                 clock: Callable[[], float] = time.monotonic):
        self._last_played_key: Optional[tuple] = None
        self._last_play_time: Optional[float] = None
        self._cooldown = cooldown_seconds
        self._clock = clock

    def next_clip(self, alert: Optional[dict], now: Optional[float] = None) -> Optional[str]:
        if now is None:
            now = self._clock()
        alert = alert or {}
        active = bool(alert.get("active"))
        clip = alert.get("clip")

        if not active:
            # Reset the IDENTITY memory on every inactive poll, not just
            # once: the same alert (same text/clip) can legitimately recur
            # much later as a genuinely new occurrence, and must play again
            # then - it can only do that if "already played this one" is
            # cleared while nothing is active in between. The COOLDOWN
            # clock is deliberately NOT reset here - see module docstring.
            self._last_played_key = None
            return None

        if not clip:
            # Active but nothing to play (the "closed"-signal case) - not a
            # new key to remember, so a REAL sound-bearing alert arriving
            # right after this one is not mistaken for a repeat of it.
            return None

        key = (alert.get("text"), clip)
        if key == self._last_played_key:
            return None

        if self._last_play_time is not None and (now - self._last_play_time) < self._cooldown:
            # Suppressed for spam control - deliberately NOT remembered as
            # "played" (self._last_played_key untouched), so once the
            # cooldown clears, whatever is active THEN (this alert, if it's
            # still going, or a newer one) gets a real chance to play.
            return None

        self._last_played_key = key
        self._last_play_time = now
        return clip
