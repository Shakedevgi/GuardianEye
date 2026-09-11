"""events.py - pure event-log grouping logic (no Flet, no HTTP).

CLAUDE.md decision 1 / backend/API.md: /events rows are per-(person_id,
hazard_id) tracking segments, not one row per real-world "approach". Phase
7's API doc states this outright: PersonTracker ID churn means one
continuous approach by the same child can produce several rows with
different person_ids. Phase 8 decisions (2026-08-30, item 5) say the
PARENT view must coalesce consecutive rows sharing a hazard_id within a
short window into one entry, labelled by the hazard, using peak_zone for
severity and approximate language for recency ("last seen 14:33") - never
a synthesized end time, since events.ended_at genuinely does not exist
(backend/db.py's module docstring, by design).

The DEVELOPER view shows the raw, ungrouped rows with their person_ids -
this module's grouping is a presentation choice for the parent view only;
nothing about the underlying data changes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

# How close together (in seconds) two rows sharing a hazard_id have to be
# for the parent view to treat them as the same continuous approach rather
# than two separate visits. Not a backend contract - purely a UI judgment
# call, kept here (not buried inline) so it's easy to find and reconsider.
GROUP_GAP_SECONDS = 45.0

_ZONE_SEVERITY = {"red": 3, "orange": 2, "yellow": 1, "none": 0}


def _parse_ts(value: str) -> Optional[datetime]:
    """Best-effort ISO-8601 parse. Returns None (never raises) on anything
    unparseable - a parent view should degrade to 'unknown time', not
    crash, on a timestamp format quirk."""
    if not value:
        return None
    text = value
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _severity(zone: str) -> int:
    return _ZONE_SEVERITY.get(zone, 0)


def group_events_by_hazard(events: list[dict]) -> list[dict]:
    """events: rows as returned by GET /events, most-recent-first (the
    order the backend already returns - this function does not re-sort).

    Returns a list of grouped entries, most-recent-first, each shaped:
        {
            "hazard_id": int,
            "hazard_label": str,
            "peak_zone": str,           # worst zone across the group
            "started_at": str,          # earliest row's started_at in the group
            "last_seen_at": str,        # latest row's last_seen_at in the group
            "clip_triggered": bool,     # true if ANY row in the group triggered a clip
            "row_count": int,           # how many raw rows were coalesced
            "event_ids": [int, ...],    # raw events.id values, for traceability
        }

    Consecutive here means adjacent in the given (already time-ordered)
    list - this function does not itself re-sort by time, matching
    /events' own contract that it's already ordered by started_at DESC.
    """
    groups: list[dict] = []
    current: Optional[dict] = None
    prev_started: Optional[datetime] = None

    for row in events:
        hazard_id = row.get("hazard_id")
        started = _parse_ts(row.get("started_at", ""))
        last_seen = _parse_ts(row.get("last_seen_at", ""))

        same_hazard = current is not None and current["hazard_id"] == hazard_id
        within_window = False
        if same_hazard and started is not None and prev_started is not None:
            # events are most-recent-first, so "gap" is prev row's started
            # minus this (older) row's started.
            within_window = abs((prev_started - started).total_seconds()) <= GROUP_GAP_SECONDS
        elif same_hazard and (started is None or prev_started is None):
            # Can't measure the gap - fail toward NOT merging rather than
            # silently coalescing rows that might be unrelated visits.
            within_window = False

        if same_hazard and within_window:
            current["started_at"] = row.get("started_at", current["started_at"])
            if last_seen and (current.get("_last_seen_dt") is None or last_seen > current["_last_seen_dt"]):
                current["last_seen_at"] = row.get("last_seen_at", current["last_seen_at"])
                current["_last_seen_dt"] = last_seen
            if _severity(row.get("peak_zone", "none")) > _severity(current["peak_zone"]):
                current["peak_zone"] = row.get("peak_zone", "none")
            current["clip_triggered"] = current["clip_triggered"] or bool(row.get("clip_triggered"))
            current["row_count"] += 1
            current["event_ids"].append(row.get("id"))
        else:
            current = {
                "hazard_id": hazard_id,
                "hazard_label": row.get("hazard_label", "object"),
                "peak_zone": row.get("peak_zone", "none"),
                "started_at": row.get("started_at", ""),
                "last_seen_at": row.get("last_seen_at", ""),
                "_last_seen_dt": last_seen,
                "clip_triggered": bool(row.get("clip_triggered")),
                "row_count": 1,
                "event_ids": [row.get("id")],
            }
            groups.append(current)

        prev_started = started

    for g in groups:
        g.pop("_last_seen_dt", None)
    return groups


def format_last_seen(last_seen_at: str) -> str:
    """'last seen 14:33' style approximate label. Never claims an end
    time - there is deliberately no ended_at column (backend/db.py)."""
    dt = _parse_ts(last_seen_at)
    if dt is None:
        return "last seen: unknown time"
    local = dt.astimezone() if dt.tzinfo else dt
    return f"last seen {local.strftime('%H:%M')}"
