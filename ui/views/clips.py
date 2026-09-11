"""clips.py - saved-clip review: keep/discard in-app, playback handed off
to the system browser.

Console direction: replaces the main column's content while the
persistent left rail stays visible (see views/log.py's docstring for the
same reasoning).

Phase 8 decision 4 (2026-08-30): Video is not in Flet 0.86.5's core and
flet-video is deliberately not being added (ui/requirements.txt). Playback
opens /clips/{id}/video in the system/default browser via
page.launch_url() - keep/discard still happen in-app through
POST /clips/{id}/keep and /discard.
"""

from __future__ import annotations

from typing import Callable

import flet as ft

from common import back_button

from colors import BG_PANEL, BG_CHIP, BORDER_SUBTLE, TEXT_DIM, TEXT_PRIMARY, TEXT_SECONDARY

STATUSES = ["pending", "kept", "discarded", "expired"]


class ClipsPanel:
    def __init__(self, page: ft.Page, on_close: Callable[[], None],
                 on_keep: Callable[[int], None], on_discard: Callable[[int], None],
                 on_refresh: Callable[[str], None], video_url_fn: Callable[[int], str]):
        self.page = page
        self._on_close = on_close
        self._on_keep = on_keep
        self._on_discard = on_discard
        self._on_refresh = on_refresh
        self._video_url_fn = video_url_fn
        self.status = "pending"

        self.title = ft.Text("Saved moments", size=26, color=TEXT_PRIMARY)
        self.close_btn = back_button(self._on_close)

        # ft.Tabs in this Flet version is a full TabBar/TabBarView pairing
        # (content + length, no simple "list of labelled tabs" shape) -
        # simple status filter pills instead, matching the pill pattern
        # already used elsewhere.
        self.status_buttons: dict[str, ft.Container] = {}
        pills = []
        for s in STATUSES:
            btn = ft.Container(
                height=36, padding=ft.Padding.symmetric(horizontal=16),
                on_click=lambda e, status=s: self._status_clicked(status),
                content=ft.Text(s.capitalize(), size=14, color=TEXT_SECONDARY),
            )
            self.status_buttons[s] = btn
            pills.append(btn)
        self.status_row = ft.Row(spacing=8, controls=pills)
        self._apply_tab_styles()

        self.list_view = ft.ListView(expand=True, spacing=8)

        self.control = ft.Container(
            visible=False, expand=True, padding=ft.Padding.symmetric(horizontal=28, vertical=24),
            content=ft.Column(expand=True, spacing=16, controls=[
                ft.Row(controls=[self.title, ft.Container(expand=True), self.close_btn]),
                ft.Container(height=1, bgcolor=BORDER_SUBTLE),
                self.status_row,
                self.list_view,
            ]),
        )

    def show(self) -> None:
        self.control.visible = True
        self.control.update()
        self._on_refresh(self.status)

    def hide(self) -> None:
        self.control.visible = False
        self.control.update()

    def _status_clicked(self, status: str) -> None:
        self.status = status
        self._apply_tab_styles()
        self.control.update()
        self._on_refresh(self.status)

    def _apply_tab_styles(self) -> None:
        for s, btn in self.status_buttons.items():
            active = s == self.status
            btn.bgcolor = BG_CHIP if active else None
            btn.border = None if active else ft.Border.all(1, BORDER_SUBTLE)
            btn.content.color = "#E8ECF1" if active else TEXT_SECONDARY

    def apply(self, clips: list[dict]) -> None:
        self.list_view.controls.clear()
        if not clips:
            self.list_view.controls.append(ft.Text("No clips here.", size=15, color=TEXT_SECONDARY))
            self.control.update()
            return
        for clip in clips:
            clip_id = clip.get("id")
            row_controls = [
                ft.Column(spacing=3, expand=True, controls=[
                    ft.Text(f"{clip.get('hazard_label', 'clip')} · #{clip_id}",
                            size=15, color=TEXT_PRIMARY),
                    ft.Text(f"{clip.get('created_at', '')} · {clip.get('status')}",
                            size=13, color=TEXT_SECONDARY),
                ]),
                ft.TextButton("Open", on_click=lambda e, cid=clip_id: self._open(cid)),
            ]
            if clip.get("status") == "pending":
                row_controls.append(ft.TextButton("Keep", on_click=lambda e, cid=clip_id: self._on_keep(cid)))
                row_controls.append(ft.TextButton("Discard", on_click=lambda e, cid=clip_id: self._on_discard(cid)))
            self.list_view.controls.append(
                ft.Container(bgcolor=BG_PANEL, padding=12, border=ft.Border.all(1, BORDER_SUBTLE),
                             content=ft.Row(controls=row_controls))
            )
        self.control.update()

    def show_error(self, message: str) -> None:
        self.list_view.controls.clear()
        self.list_view.controls.append(ft.Text(message, size=14, color="#FF8C8C"))
        self.control.update()

    def _open(self, clip_id: int) -> None:
        self.page.launch_url(self._video_url_fn(clip_id))
