"""token_store.py - remember a passed-PIN token client-side so a tablet
doesn't have to re-enter the PIN every visit (Phase 8 decision, 2026-08-30:
"Remember a token client-side... PIN configurable via CLI flag/env, not
hardcoded").

Two paths, honestly different in how well-verified they are:

1. Desktop (page.web is False): a plain local file under the user's home
   directory. This machine also runs the camera/server, so a local file is
   uncontroversially "this device's own storage" - verified working in
   this session (plain file I/O, nothing Flet-specific to break).

2. Web (a tablet/phone hitting the Flet web server over LAN): tried
   ft.SharedPreferences() first, since it is Flet 0.86.5's documented
   replacement for the removed client_storage and is meant to persist via
   the browser's own storage. **Measured this session, against the real
   flet-desktop client connecting to a local ft.run(): SharedPreferences
   times out with no listener response** - the desktop client apparently
   doesn't bundle whatever plugin backs it, at least not in this
   installation. It was NOT possible to verify the web-browser path the
   same way in this sandbox (no real browser available to drive here, and
   curling the page doesn't establish the Flutter session SharedPreferences
   needs). So this module tries it, on a short timeout, and falls back to
   an in-memory, THIS-TAB-ONLY remember (a fresh page load / new tab will
   re-ask for the PIN) if it doesn't respond in time. This is a real,
   named gap, not a silently-assumed success - see ui/app.py's docstring
   and the final report for the same caveat.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import flet as ft

_DESKTOP_TOKEN_FILE = Path.home() / ".guardianeye" / "ui_token"
_SHARED_PREFS_KEY = "guardianeye_pin_token"
_SHARED_PREFS_TIMEOUT = 3.0

# In-memory fallback for the web path when SharedPreferences doesn't
# respond - keyed by nothing durable, so it only survives as long as this
# Python process's session object does (i.e. this browser tab, this run).
_memory_fallback: dict[int, str] = {}


async def load_token(page: ft.Page) -> Optional[str]:
    if not page.web:
        return _read_desktop_file()
    # Web: in-memory only, deliberately. See the comment on
    # _try_shared_prefs_get for the measured reason SharedPreferences is
    # never mounted here.
    return _memory_fallback.get(id(page))


async def save_token(page: ft.Page, token: str) -> None:
    if not page.web:
        _write_desktop_file(token)
        return
    _memory_fallback[id(page)] = token


def clear_token_sync(page: ft.Page) -> None:
    """Best-effort synchronous clear for the 'forget this device' settings
    action - doesn't await the SharedPreferences round trip, matching the
    fire-and-forget nature of a logout button."""
    if not page.web:
        try:
            _DESKTOP_TOKEN_FILE.unlink(missing_ok=True)
        except Exception:
            pass
    _memory_fallback.pop(id(page), None)


def _read_desktop_file() -> Optional[str]:
    try:
        return _DESKTOP_TOKEN_FILE.read_text().strip() or None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _write_desktop_file(token: str) -> None:
    try:
        _DESKTOP_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        _DESKTOP_TOKEN_FILE.write_text(token)
    except Exception:
        pass


# MEASURED, and the reason the web path no longer touches SharedPreferences
# at all: mounting ft.SharedPreferences() into page.overlay renders
# "Unknown control: SharedPreferences" as a full red Flutter error box
# COVERING THE PIN SCREEN in a real browser (verified this session against
# the flet-web client, which is the tablet path Phase 8's done-when is about).
# It is not merely "storage might not persist" - it makes the first screen a
# tablet ever shows look broken, and the session then drops its WebSocket.
# Re-asking for the PIN on a fresh page load is both the safe behaviour for a
# device that watches a child and the honest one. These two helpers are kept
# only so the reasoning stays attached to the code; nothing calls them.
async def _try_shared_prefs_get(page: ft.Page) -> Optional[str]:
    try:
        sp = ft.SharedPreferences()
        page.overlay.append(sp)
        page.update()
        value = await asyncio.wait_for(sp.get(_SHARED_PREFS_KEY), timeout=_SHARED_PREFS_TIMEOUT)
        return value if isinstance(value, str) else None
    except Exception:
        return None


async def _try_shared_prefs_set(page: ft.Page, token: str) -> bool:
    try:
        sp = ft.SharedPreferences()
        page.overlay.append(sp)
        page.update()
        await asyncio.wait_for(sp.set(_SHARED_PREFS_KEY, token), timeout=_SHARED_PREFS_TIMEOUT)
        return True
    except Exception:
        return False
