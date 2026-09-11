"""live.py - the Console direction's main column: risk meter, video panel,
recent-events strip.

Translates docs/design/phase-8/DirectionB_Idle.dc.html (calm) and
DirectionB_Alert.dc.html (red) into Flet controls. The persistent left
rail is a SEPARATE widget (views/rail.py) - this class owns only the main
column, matching the mock's own division into a fixed-width rail plus a
flex-grow main area.

Differences from the mock, noted rather than silently dropped:

  - The approved DirectionB_Alert mock's rail still shows a "Mute 2 min"
    row. That contradicts Shaked's 2026-08-30 decision (recorded in
    docs/decision-log.md, "The alert screens have no acknowledge/mute
    button") which says the mock had already been republished with that
    control removed from all three artboards. Since no backend route can
    act on it (see the same entry), this build follows the WRITTEN
    decision rather than the stale row still present in the .dc.html file
    on disk, and omits it. Flagged for Shaked/Yahli in the final report
    rather than silently resolved either way.
  - The mock's alert sentence hardcodes "He is at the low table". This
    build renders it from risk.hazard_label (already real data, already
    used the same way in review copy) rather than a fixed string.
  - The meter's fill percentage in the mock is a fixed illustration value
    per state (8% none / 94% red). /risk_status's risk.value is a
    resolved-instant proximity number, not defined to map linearly onto a
    "how alarming is the room right now" gauge, so this build uses the
    same per-zone fixed proportions the mock itself hardcodes (8/35/65/94)
    rather than inventing a formula from risk.value. risk.value itself is
    still shown as text for anyone who wants the real number (developer
    view only, since it is a raw distance CLAUDE.md decision 1 keeps out
    of the parent view).
"""

from __future__ import annotations

from typing import Callable, Optional

import flet as ft

from colors import (
    ALERT_RED,
    ALERT_RED_BORDER,
    ALERT_RED_DARK,
    ALERT_SUBTEXT_ON_RED,
    ALERT_TEXT_ON_RED,
    BG_PANEL,
    BG_VIDEO_FRAME,
    BORDER_SUBTLE,
    TEXT_DIM,
    TEXT_LABEL,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    ZONE_COLOR,
)
from events import format_last_seen, group_events_by_hazard
from placeholder import PLACEHOLDER_JPEG

CALM_BORDER_WIDTH = 1
ALERT_BORDER_WIDTH = 3

_ZONE_METER_PCT = {"none": 8, "yellow": 35, "orange": 65, "red": 94}
_ZONE_LABEL = {"none": "NONE", "yellow": "YELLOW", "orange": "ORANGE", "red": "RED"}


class LiveView:
    """Owns the video Image control and the main-column chrome around it.

    The Image control is created once, here, and is never removed from
    the control tree afterwards - frames are pushed to it from a
    background thread (see ui/video.py) for the lifetime of the app.
    """

    def __init__(self, page: ft.Page, on_open_clip: Callable[[], None]):
        self.page = page
        self.dev_mode = False
        self._on_open_clip = on_open_clip
        self._is_red_alert = False

        self.video_image = ft.Image(
            src=PLACEHOLDER_JPEG,
            fit=ft.BoxFit.CONTAIN,
            gapless_playback=True,
            expand=True,
        )

        # -- risk meter bar (header of the main column) --------------------
        self.meter_label = ft.Text("RISK", size=11, color=TEXT_LABEL, style=ft.TextStyle(letter_spacing=1.1))
        self.meter_fill = ft.Container(width=8, height=8, bgcolor=ZONE_COLOR["none"], border_radius=4)
        self.meter_track = ft.Container(
            width=240, height=8, border_radius=4, bgcolor=BORDER_SUBTLE, clip_behavior=ft.ClipBehavior.HARD_EDGE,
            content=ft.Row(controls=[self.meter_fill]),
        )
        self.meter_zone_txt = ft.Text("NONE", size=14, color=TEXT_MUTED, weight=ft.FontWeight.W_500,
                                       font_family="IBM Plex Mono")
        self.meter_status_txt = ft.Text("no one in frame", size=13, color=TEXT_DIM)
        self.meter_bar = ft.Container(
            height=46,
            padding=ft.Padding.symmetric(horizontal=20),
            border=ft.Border.only(bottom=ft.BorderSide(1, BORDER_SUBTLE)),
            content=ft.Row(spacing=14, vertical_alignment=ft.CrossAxisAlignment.CENTER, controls=[
                self.meter_label, self.meter_track, self.meter_zone_txt,
                ft.Container(expand=True), self.meter_status_txt,
            ]),
        )

        # -- alert sentence band (RED only) ---------------------------------
        self.alert_sentence = ft.Text(size=32, color="#FFFFFF", spans=[])
        self.alert_band = ft.Container(
            height=90, visible=False, bgcolor=ALERT_RED_DARK,
            border=ft.Border.only(bottom=ft.BorderSide(1, ALERT_RED_BORDER)),
            padding=ft.Padding.symmetric(horizontal=20),
            alignment=ft.Alignment.CENTER_LEFT,
            content=self.alert_sentence,
        )

        # -- video panel -----------------------------------------------
        # A black video pane is ambiguous - "the room is dark", "the camera
        # died", and "video is not reaching THIS device" look identical, and
        # that ambiguity has now cost two debugging rounds (0.0.0.0, then the
        # iPad). /health was added at Phase 7 for exactly this distinction on
        # the camera side; this is the client side of the same idea: say
        # whether any frame has ever arrived HERE.
        self.no_video_text = ft.Text("Connecting to the camera...", size=16, color=TEXT_SECONDARY)
        self.no_video_hint = ft.Text("", size=13, color=TEXT_MUTED,
                                      font_family="IBM Plex Mono")
        self.no_video_overlay = ft.Container(
            expand=True, alignment=ft.Alignment.CENTER, bgcolor=BG_VIDEO_FRAME,
            content=ft.Column(
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8,
                controls=[self.no_video_text, self.no_video_hint],
            ),
        )
        self.video_frame = ft.Container(
            expand=True,
            margin=ft.Margin.symmetric(horizontal=20, vertical=14),
            bgcolor=BG_VIDEO_FRAME,
            border=ft.Border.all(CALM_BORDER_WIDTH, BORDER_SUBTLE),
            content=ft.Stack(expand=True, controls=[self.video_image, self.no_video_overlay]),
        )

        self.diag_text = ft.Text("", size=11, color=TEXT_MUTED, font_family="IBM Plex Mono")
        self.diag_panel = ft.Container(
            visible=False, padding=8, bgcolor="#00000090",
            content=self.diag_text,
        )
        self.video_stack = ft.Stack(expand=True, controls=[self.video_frame, self.diag_panel])

        # -- bottom area: recent events (calm) / show-the-clip (alert) -----
        self.recent_label = ft.Text("RECENT", size=11, color=TEXT_LABEL, style=ft.TextStyle(letter_spacing=1.1))
        self.recent_list = ft.Column(spacing=9, controls=[])
        self.recent_block = ft.Container(
            height=176, padding=ft.Padding.only(left=20, right=20, bottom=16),
            content=ft.Column(spacing=9, controls=[self.recent_label, self.recent_list]),
        )

        self.show_clip_btn = ft.Container(
            height=52, padding=ft.Padding.symmetric(horizontal=24),
            border=ft.Border.all(1, BORDER_SUBTLE), alignment=ft.Alignment.CENTER,
            on_click=lambda e: self._on_open_clip(),
            content=ft.Text("Show the clip", size=17, color=TEXT_PRIMARY),
        )
        self.alert_actions_block = ft.Container(
            height=78, visible=False, padding=ft.Padding.only(left=20, right=20, bottom=16),
            content=ft.Row(controls=[self.show_clip_btn]),
        )

        self.control = ft.Column(
            expand=True, spacing=0, horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            controls=[
                self.meter_bar,
                self.alert_band,
                self.video_stack,
                self.recent_block,
                self.alert_actions_block,
            ],
        )

    # -- external updates --------------------------------------------------

    def set_dev_mode(self, enabled: bool) -> None:
        self.dev_mode = enabled
        self.diag_panel.visible = enabled

    def push_frame(self, jpeg: bytes) -> None:
        """Called from the MJPEG background thread. Plain control mutation
        + control-level update() (not page.update()) - measured safe to
        call directly from a worker thread in Flet 0.86.5."""
        self.video_image.src = jpeg
        self.video_image.update()
        if self.no_video_overlay.visible:
            self.no_video_overlay.visible = False
            self.no_video_overlay.update()

    def report_video_state(self, pushed: int, err) -> None:
        """Called once a second by the shell. Only ever changes what the
        OVERLAY says - once a frame has actually rendered the overlay is
        gone for good, so this can never cover live video."""
        if not self.no_video_overlay.visible:
            return
        if err:
            self.no_video_text.value = "No video from the camera"
            self.no_video_hint.value = str(err)[:90]
        elif pushed > 0:
            # The server has handed frames to this session and none has
            # painted - so the break is on this device, not upstream.
            self.no_video_text.value = "Video is not displaying on this device"
            self.no_video_hint.value = f"{pushed} frames sent, none rendered"
        else:
            self.no_video_text.value = "Connecting to the camera..."
            self.no_video_hint.value = ""
        self.no_video_overlay.update()

    def apply_status(self, status: dict) -> None:
        risk = status.get("risk", {}) or {}
        zone = risk.get("zone", "none")
        alert = status.get("alert", {}) or {}
        camera = status.get("camera", {}) or {}

        self._is_red_alert = zone == "red" and bool(alert.get("active"))

        zone_color = ZONE_COLOR.get(zone, ZONE_COLOR["none"])
        pct = _ZONE_METER_PCT.get(zone, 8)

        if self._is_red_alert:
            self.meter_bar.bgcolor = ALERT_RED
            self.meter_track.bgcolor = "#00000048"
            self.meter_fill.bgcolor = "#FFFFFF"
            self.meter_label.color = "#FFFFFFBF"
            self.meter_zone_txt.color = "#FFFFFF"
            self.meter_status_txt.color = ALERT_SUBTEXT_ON_RED
            self.meter_status_txt.value = "saving a clip"
        else:
            self.meter_bar.bgcolor = None
            self.meter_track.bgcolor = BORDER_SUBTLE
            self.meter_fill.bgcolor = zone_color
            self.meter_label.color = TEXT_LABEL
            self.meter_zone_txt.color = zone_color if zone != "none" else TEXT_MUTED
            self.meter_status_txt.color = TEXT_DIM
            self.meter_status_txt.value = {
                "none": "no one in frame",
                "yellow": "in the area",
                "orange": "getting close",
            }.get(zone, "no one in frame")

        self.meter_track.content.controls = [
            ft.Container(width=self._track_px(pct), height=8, bgcolor=self.meter_fill.bgcolor,
                         border_radius=4),
        ]
        self.meter_zone_txt.value = _ZONE_LABEL.get(zone, "NONE")

        self.alert_band.visible = self._is_red_alert
        self.recent_block.visible = not self._is_red_alert
        self.alert_actions_block.visible = self._is_red_alert
        if self._is_red_alert:
            # Same class-agnostic caveat as views/review.py's headline: the
            # room scan does not name what it finds and its generic label is
            # the literal string "object", so "He is at the object" reads as
            # a failed identification rather than an alert. Only the
            # per-frame pass yields real class names (oven/microwave/
            # refrigerator) worth saying out loud.
            _raw = (risk.get("hazard_label") or "").strip()
            hazard_label = (_raw.replace("_", " ") if _raw and _raw != "object"
                            else "something you flagged")
            self.alert_sentence.spans = [
                ft.TextSpan("He is at ", ft.TextStyle(color="#FFFFFF", size=32)),
                ft.TextSpan(hazard_label, ft.TextStyle(color="#FF6B6B", size=32)),
            ]

        border_width = ALERT_BORDER_WIDTH if self._is_red_alert else CALM_BORDER_WIDTH
        self.video_frame.border = ft.Border.all(border_width,
                                                  ALERT_RED if self._is_red_alert else BORDER_SUBTLE)

        if self.dev_mode:
            diag = status.get("diagnostics", {}) or {}
            self.diag_text.value = (
                f"fps={diag.get('fps')}  model={diag.get('model')}  imgsz={diag.get('imgsz')}  "
                f"conf={diag.get('conf')}  device={diag.get('device')}\n"
                f"frame_seq={status.get('frame_seq')}  zone={zone}  dist={risk.get('value')}\n"
                f"camera={camera}\n"
                f"hazards: {status.get('hazard_counts')}"
            )

        self._update_all()

    @staticmethod
    def _track_px(pct: int) -> int:
        return max(2, int(240 * pct / 100))

    def set_booting(self) -> None:
        self.meter_status_txt.value = "starting up - waiting for the first frame"
        self._update_all()

    def apply_events(self, events: list[dict]) -> None:
        """/events, on-demand refresh (not part of the 4 Hz /risk_status
        poll - see ui/app.py's slower recent-events timer)."""
        groups = group_events_by_hazard(events)[:3]
        self.recent_list.controls.clear()
        if not groups:
            self.recent_list.controls.append(ft.Text("Nothing yet.", size=13, color=TEXT_DIM))
        for g in groups:
            zone_color = ZONE_COLOR.get(g["peak_zone"], ZONE_COLOR["none"])
            if g["peak_zone"] == "none":
                headline = f"Went near the {g['hazard_label']}"
            else:
                headline = f"Went near the {g['hazard_label']} · up to {g['peak_zone']}"
            note = "1 approach" if g["row_count"] <= 1 else f"{g['row_count']} approaches"
            self.recent_list.controls.append(
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=12, vertical=9), bgcolor=BG_PANEL,
                    border=ft.Border.only(left=ft.BorderSide(2, zone_color)),
                    content=ft.Row(spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER, controls=[
                        ft.Text(format_last_seen(g["last_seen_at"]).replace("last seen ", ""),
                                size=12, color=TEXT_SECONDARY, font_family="IBM Plex Mono", width=52),
                        ft.Container(expand=True, content=ft.Text(headline, size=14, color=TEXT_PRIMARY)),
                        ft.Text(note.upper(), size=11, color=TEXT_DIM, font_family="IBM Plex Mono"),
                    ]),
                )
            )
        self._update_all()

    def _update_all(self) -> None:
        try:
            self.control.update()
        except Exception:
            pass
