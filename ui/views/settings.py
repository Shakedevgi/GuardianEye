"""settings.py - dev-mode toggle (CLAUDE.md decision 1's parent/dev split)
plus basic connection info.

Console direction rebuild: gives Settings a real place in the persistent
left rail's nav section (views/rail.py, "Event log / Saved clips /
Settings") instead of the old build's tiny grey icon in a far corner -
that was defect 1 in the rebuild brief, named verbatim by Shaked. The
dev-mode switch here is the "Settings toggle" half of the required switch
(the other half is the --dev CLI flag to boot straight into it - see
ui/app.py). It still lives one click into Settings, not on the main
screen, so a parent cannot flip it by accident.
"""

from __future__ import annotations

from typing import Callable

import flet as ft

from common import back_button

from colors import BORDER_SUBTLE, TEXT_DIM, TEXT_PRIMARY, TEXT_SECONDARY


class SettingsPanel:
    def __init__(self, page: ft.Page, on_close: Callable[[], None],
                 on_dev_mode_change: Callable[[bool], None],
                 base_url: str, dev_mode_default: bool, on_logout: Callable[[], None]):
        self.page = page
        self._on_close = on_close
        self._on_dev_mode_change = on_dev_mode_change
        self._on_logout = on_logout

        self.title = ft.Text("Settings", size=26, color=TEXT_PRIMARY)
        self.close_btn = back_button(self._on_close)

        self.dev_switch = ft.Switch(value=dev_mode_default, on_change=self._dev_changed)
        self.dev_row = ft.Row(vertical_alignment=ft.CrossAxisAlignment.CENTER, controls=[
            ft.Column(spacing=4, expand=True, controls=[
                ft.Text("Developer view", size=17, color=TEXT_PRIMARY),
                ft.Text("Shows fps, model, imgsz, conf, device, frame_seq, raw distances, "
                        "hazard ids/bboxes/origins, and ungrouped events with person ids.",
                        size=13, color=TEXT_SECONDARY),
            ]),
            self.dev_switch,
        ])

        self.backend_text = ft.Text(f"Backend: {base_url}", size=14, color=TEXT_SECONDARY,
                                     font_family="IBM Plex Mono")
        self.logout_btn = ft.TextButton("Forget this device's PIN", on_click=lambda e: self._on_logout())

        self.control = ft.Container(
            visible=False, expand=True, padding=ft.Padding.symmetric(horizontal=28, vertical=24),
            content=ft.Column(controls=[
                ft.Row(controls=[self.title, ft.Container(expand=True), self.close_btn]),
                ft.Container(height=1, bgcolor=BORDER_SUBTLE),
                ft.Container(height=24),
                self.dev_row,
                ft.Container(height=28),
                ft.Container(height=1, bgcolor=BORDER_SUBTLE),
                ft.Container(height=16),
                self.backend_text,
                ft.Container(height=8),
                self.logout_btn,
            ]),
        )

    def show(self) -> None:
        self.control.visible = True
        self.control.update()

    def hide(self) -> None:
        self.control.visible = False
        self.control.update()

    def _dev_changed(self, e) -> None:
        self._on_dev_mode_change(self.dev_switch.value)
