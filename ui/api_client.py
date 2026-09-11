"""api_client.py - the ONE place this app talks HTTP to backend/server.py.

CLAUDE.md decision 1: Flet is a client of FastAPI. Every route this file
calls is documented in backend/API.md; nothing here re-derives, guesses at,
or reimplements backend/cv logic. If a response shape here and the real
running server ever disagree, that is a contract-break bug to report, not
a cue to silently adapt (see backend/API.md's own header note).

Plain urllib, not requests - avoids adding an undeclared dependency to
ui/requirements.txt for a handful of small JSON calls (requests happens to
be present in the shared .venv only because fastapi/starlette pull it in
transitively for other purposes; nothing here should rely on that).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional


class ApiError(Exception):
    """A real error talking to the backend - not a 503 'still booting'."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class NotReady(Exception):
    """503 'no frame published yet' - the camera loop hasn't started.

    Per backend/API.md: /risk_status and /review both return this before
    the first frame is published. Treat as "still booting," not an error
    to alarm on (Phase 8 decisions, this session).
    """


class ReviewCommandFailed(Exception):
    """A review confirm/dismiss/skip call whose backend replied with a
    real 503 'frame loop did not respond' - a genuine wedged-loop signal,
    per backend/API.md. NOT retried silently by this client; the caller
    decides how to surface it."""


class ReviewNotCurrent(Exception):
    """confirm/dismiss returned {"ok": false, "reason": "..."} because the
    id acted on was not the current FIFO candidate. Should be rare if the
    UI only ever acts on /review's current_id, but the backend fails
    closed on this rather than silently succeeding, so the client
    surfaces it distinctly instead of treating it as a plain ApiError."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class ApiClient:
    """Thin synchronous HTTP wrapper. Every method call is a single
    request/response - no request fan-out, matching the Phase 8 polling
    decision (one /risk_status call renders the whole live view)."""

    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- low-level -----------------------------------------------------

    def _get(self, path: str) -> Any:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read()
            detail = _extract_detail(body)
            if exc.code == 503:
                raise NotReady(detail or "not ready") from exc
            raise ApiError(detail or f"HTTP {exc.code}", status=exc.code) from exc
        except urllib.error.URLError as exc:
            raise ApiError(f"connection failed: {exc.reason}") from exc

    def _post(self, path: str) -> Any:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method="POST", data=b"")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read()
            detail = _extract_detail(body)
            if exc.code == 503:
                raise ReviewCommandFailed(detail or "frame loop did not respond") from exc
            if exc.code == 404:
                raise ApiError(detail or "not found", status=404) from exc
            raise ApiError(detail or f"HTTP {exc.code}", status=exc.code) from exc
        except urllib.error.URLError as exc:
            raise ApiError(f"connection failed: {exc.reason}") from exc

    # -- routes ----------------------------------------------------------

    def risk_status(self) -> dict:
        """GET /risk_status. Raises NotReady on 503 (still booting)."""
        return self._get("/risk_status")

    def health(self) -> dict:
        """GET /health. Never raises NotReady - /health always answers,
        that's the point of it (distinguishes dark-room from dead-loop)."""
        return self._get("/health")

    def events(self, limit: int = 50) -> list[dict]:
        data = self._get(f"/events?limit={limit}")
        return data["events"]

    def clips(self, status: str) -> list[dict]:
        """status must be one of pending/kept/discarded/expired - a bad
        value 400s server-side; this client does not pre-validate it, the
        server is the source of truth for the allowed set."""
        data = self._get(f"/clips?status={status}")
        return data["clips"]

    def clip(self, clip_id: int) -> dict:
        return self._get(f"/clips/{clip_id}")

    def keep_clip(self, clip_id: int) -> dict:
        return self._post(f"/clips/{clip_id}/keep")

    def discard_clip(self, clip_id: int) -> dict:
        return self._post(f"/clips/{clip_id}/discard")

    def clip_video_url(self, clip_id: int) -> str:
        """Not fetched by this client - handed to the system browser
        (Phase 8 decision 4: no flet-video dependency)."""
        return f"{self.base_url}/clips/{clip_id}/video"

    def review(self) -> dict:
        """GET /review. Raises NotReady on 503.

        Returns the FULL response verbatim: {length, current_id, current,
        queue}. Callers MUST use `queue` as-is for FIFO position display
        and MUST NOT re-derive it from /risk_status's hazards list (see
        backend/API.md's explicit warning - this was a real Phase 7 bug)."""
        return self._get("/review")

    def review_confirm(self, entry_id: int) -> dict:
        return self._review_command(f"/review/{entry_id}/confirm")

    def review_dismiss(self, entry_id: int) -> dict:
        return self._review_command(f"/review/{entry_id}/dismiss")

    def review_skip(self) -> dict:
        return self._review_command("/review/skip")

    def _review_command(self, path: str) -> dict:
        result = self._post(path)
        if isinstance(result, dict) and result.get("ok") is False:
            raise ReviewNotCurrent(result.get("reason", "rejected"))
        return result


def _extract_detail(body: bytes) -> Optional[str]:
    try:
        data = json.loads(body)
        if isinstance(data, dict):
            return data.get("detail")
    except Exception:
        pass
    return None
