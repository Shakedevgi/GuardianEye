"""rail.py - the Console direction's persistent left rail.

Translates docs/design/phase-8/DirectionB_Idle.dc.html's left column: room
state, the hazard-map legend as a compact colour legend WITH COUNTS, and
the review-queue count with a way in. This exists specifically to fix
defect 4 from the rebuild brief ("persistent state belongs in the left
rail; do not just move the same sparse strip around") and defect 1
("settings was a tiny grey icon in a far corner - give it a real, labelled,
findable place").

This rail is shown for every screen EXCEPT review, which the approved mock
(DirectionB_Review.dc.html) replaces with its own rail (see
views/review.py) - the review queue is strictly FIFO and reviewing one
entry is a different task than watching the room, so it earns a different
rail rather than trying to cram both into one.

Deliberate deviation from the mock, noted rather than silently made: the
mock hardcodes "Living room" as the room name. Nothing in backend/API.md
offers a configurable room name - the only identifying string available is
`camera.name` from /risk_status. Rather than fabricate a room name the
backend never sent, this rail labels the row "CAMERA" and shows the real
`camera.name` value.
"""

from __future__ import annotations

from typing import Callable, Optional

import flet as ft

from colors import (
    BG_CHIP,
    BG_RAIL,
    BORDER_SUBTLE,
    TEXT_DIM,
    TEXT_LABEL,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    ZONE_COLOR,
)

RAIL_WIDTH = 260

_LEGEND_ROWS = [
    # (hazard_counts key, legend label, box colour)
    ("confirmed", "Confirmed", "#FF0000"),
    ("pending_new", "New, unreviewed", "#FF8C00"),
    ("pending_first_scan", "Was already here", "#00FFFF"),
    ("dismissed", "Dismissed", "#8C8C8C"),
]


def _section_label(text: str) -> ft.Text:
    return ft.Text(text.upper(), size=11, color=TEXT_LABEL, weight=ft.FontWeight.W_500,
                    style=ft.TextStyle(letter_spacing=1.1))


def _divider() -> ft.Container:
    return ft.Container(height=1, bgcolor=BORDER_SUBTLE)


def _nav_row(icon: str, label: str, on_click) -> ft.Container:
    return ft.Container(
        height=44,
        border_radius=6,
        padding=ft.Padding.symmetric(horizontal=10),
        on_click=on_click,
        content=ft.Row(spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER, controls=[
            ft.Icon(icon, size=18, color=TEXT_SECONDARY),
            ft.Text(label, size=15, color=TEXT_PRIMARY),
        ]),
    )


class PersistentRail:
    """The always-visible left rail for the live / log / clips / settings
    screens. One instance, mutated in place - never rebuilt from scratch,
    so control identity (and Flet's diffing) stays stable across the 4 Hz
    /risk_status poll.
    """

    def __init__(self, page: ft.Page, on_nav: Callable[[str], None]):
        self.page = page
        self._on_nav = on_nav
        self._alert_mode = False

        # -- room / camera ---------------------------------------------
        self.camera_name_txt = ft.Text("Camera", size=17, color=TEXT_PRIMARY)
        self.camera_dot = ft.Container(width=8, height=8, border_radius=4, bgcolor="#8C8C8C")
        self.camera_status_txt = ft.Text("connecting...", size=12, color=TEXT_SECONDARY,
                                          font_family="IBM Plex Mono")
        room_block = ft.Column(spacing=5, controls=[
            _section_label("camera"),
            self.camera_name_txt,
            ft.Row(spacing=7, vertical_alignment=ft.CrossAxisAlignment.CENTER, controls=[
                self.camera_dot, self.camera_status_txt,
            ]),
        ])

        # -- hazard map legend --------------------------------------------
        self.hazard_map_label = _section_label("hazard map · 0")
        self.legend_rows: dict[str, ft.Text] = {}
        legend_controls = [self.hazard_map_label]
        for key, label, color in _LEGEND_ROWS:
            count_txt = ft.Text("0", size=14, color=TEXT_PRIMARY, font_family="IBM Plex Mono")
            self.legend_rows[key] = count_txt
            legend_controls.append(
                ft.Row(spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER, controls=[
                    ft.Container(width=11, height=11, border=ft.Border.all(2, color)),
                    ft.Container(expand=True, content=ft.Text(label, size=14, color=TEXT_PRIMARY)),
                    count_txt,
                ])
            )
        self.hazard_map_block = ft.Column(spacing=11, controls=legend_controls)

        # -- "in play" block, shown instead of the hazard map legend during
        # a RED alert (docs/design/phase-8/DirectionB_Alert.dc.html) -------
        self.in_play_hazard_label = ft.Text("", size=15, color="#FFFFFF")
        self.in_play_hazard_sub = ft.Text("confirmed hazard", size=12, color="#FF8080",
                                           font_family="IBM Plex Mono")
        self.in_play_hazard_box = ft.Container(
            padding=12, border=ft.Border.all(1, "#FF0000"), bgcolor="#FF000018",
            content=ft.Column(spacing=3, controls=[self.in_play_hazard_label, self.in_play_hazard_sub]),
        )
        self.in_play_person_label = ft.Text("Child", size=15, color=TEXT_PRIMARY)
        self.in_play_person_sub = ft.Text("", size=12, color=TEXT_SECONDARY, font_family="IBM Plex Mono")
        self.in_play_person_box = ft.Container(
            padding=12, border=ft.Border.all(1, BORDER_SUBTLE),
            content=ft.Column(spacing=3, controls=[self.in_play_person_label, self.in_play_person_sub]),
        )
        self.in_play_block = ft.Column(spacing=8, visible=False, controls=[
            _section_label("in play"),
            self.in_play_hazard_box,
            self.in_play_person_box,
        ])

        # -- review queue ----------------------------------------------
        self.review_count_txt = ft.Text("0", size=32, color=TEXT_PRIMARY, weight=ft.FontWeight.W_500,
                                         font_family="IBM Plex Mono")
        self.review_open_btn = ft.Container(
            height=40, border_radius=4, bgcolor=BG_CHIP, alignment=ft.Alignment.CENTER,
            on_click=lambda e: self._on_nav("review"),
            content=ft.Text("Open review", size=14, color=TEXT_PRIMARY),
        )
        self.review_block = ft.Column(spacing=9, controls=[
            _section_label("review queue"),
            ft.Row(spacing=8, vertical_alignment=ft.CrossAxisAlignment.END, controls=[
                self.review_count_txt, ft.Text("waiting", size=13, color=TEXT_SECONDARY),
            ]),
            self.review_open_btn,
        ])

        # -- navigation: findable, labelled, no tiny corner icons -------
        nav_block = ft.Column(spacing=2, controls=[
            _section_label("more"),
            ft.Container(height=6),
            _nav_row(ft.Icons.HISTORY, "Event log", lambda e: self._on_nav("log")),
            _nav_row(ft.Icons.MOVIE_OUTLINED, "Saved clips", lambda e: self._on_nav("clips")),
            _nav_row(ft.Icons.SETTINGS_OUTLINED, "Settings", lambda e: self._on_nav("settings")),
        ])

        self.footer_txt = ft.Text("parent view", size=11, color=TEXT_MUTED, font_family="IBM Plex Mono")

        self.column = ft.Column(
            spacing=22,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            controls=[
                room_block,
                _divider(),
                self.hazard_map_block,
                self.in_play_block,
                _divider(),
                self.review_block,
                _divider(),
                nav_block,
                ft.Container(expand=True),
                self.footer_txt,
            ],
        )

        self.control = ft.Container(
            width=RAIL_WIDTH,
            bgcolor=BG_RAIL,
            border=ft.Border.only(right=ft.BorderSide(1, BORDER_SUBTLE)),
            padding=ft.Padding.symmetric(horizontal=18, vertical=20),
            content=self.column,
        )

    # -- external updates --------------------------------------------------

    def set_dev_mode(self, enabled: bool) -> None:
        self.footer_txt.value = "developer view" if enabled else "parent view"

    def apply_status(self, status: dict) -> None:
        camera = status.get("camera", {}) or {}
        self.camera_name_txt.value = camera.get("name") or "Camera"

        counts = status.get("hazard_counts", {}) or {}
        total = counts.get("total", 0)
        self.hazard_map_label.value = f"HAZARD MAP · {total}"
        for key, txt in self.legend_rows.items():
            txt.value = str(counts.get(key, 0))

        review_queue = status.get("review_queue", {}) or {}
        length = review_queue.get("length", 0)
        self.review_count_txt.value = str(length)

        risk = status.get("risk", {}) or {}
        alert = status.get("alert", {}) or {}
        is_red_alert = risk.get("zone") == "red" and bool(alert.get("active"))
        self._set_alert_mode(is_red_alert, risk, status)

        self.control.update()

    def _set_alert_mode(self, is_red_alert: bool, risk: dict, status: dict) -> None:
        if is_red_alert == self._alert_mode and not is_red_alert:
            return
        self._alert_mode = is_red_alert
        self.hazard_map_block.visible = not is_red_alert
        self.review_block.visible = not is_red_alert
        self.in_play_block.visible = is_red_alert
        if is_red_alert:
            hazards = {h.get("id"): h for h in status.get("hazards", []) or []}
            hazard = hazards.get(risk.get("hazard_id"))
            label = (hazard or {}).get("label") or risk.get("hazard_label") or "flagged spot"
            self.in_play_hazard_label.value = label.replace("_", " ").title()
            state = (hazard or {}).get("state", "confirmed")
            self.in_play_hazard_sub.value = {
                "confirmed": "confirmed hazard",
                "pending": "unreviewed - treated as dangerous",
            }.get(state, state)
            person_id = risk.get("person_id")
            self.in_play_person_label.value = f"Child #{person_id}" if person_id is not None else "Child"

    def set_in_play_duration(self, text: str) -> None:
        """Client-observed 'closing since HH:MM:SS' - see LiveView, which
        owns the wall-clock timestamp this rail displays (this rail has no
        polling loop of its own)."""
        self.in_play_person_sub.value = text

    def apply_health(self, health: Optional[dict], reachable: bool) -> None:
        if not reachable:
            self.camera_dot.bgcolor = "#8C8C8C"
            self.camera_status_txt.value = "server unreachable"
        elif health is None:
            self.camera_dot.bgcolor = "#8C8C8C"
            self.camera_status_txt.value = "connecting..."
        elif health.get("camera_running"):
            self.camera_dot.bgcolor = "#00FF00"
            self.camera_status_txt.value = "camera live"
        else:
            self.camera_dot.bgcolor = "#FF0000"
            self.camera_status_txt.value = "camera not running"
        self.control.update()
