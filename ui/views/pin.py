"""pin.py - the PIN gate, first thing shown.

Phase 8 decision (2026-08-30): "No frame bytes may be pushed to a session
until it passes - enforce this in the Python session (do not start that
session's frame pushing), not by hiding a widget behind an overlay." This
view enforces exactly that: ui/app.py does not construct the MjpegClient
or start polling until on_success fires. The PIN itself is compared
locally in this Python session against the value ui/app.py loaded from a
CLI flag/env var - there is no backend PIN-check route (none of
backend/API.md's 13 routes do auth), so this is intentionally
session-local gating, not a server-enforced one.
"""

from __future__ import annotations

from typing import Callable

import flet as ft

from colors import BG_ROOT, TEXT_DIM, TEXT_PRIMARY, TEXT_SECONDARY


class PinView:
    def __init__(self, on_submit: Callable[[str], None]):
        self._on_submit = on_submit

        self.field = ft.TextField(
            label="PIN", password=True, can_reveal_password=True,
            width=260, autofocus=True, on_submit=self._submit,
            text_align=ft.TextAlign.CENTER,
        )
        self.error_text = ft.Text("", size=13, color="#FF8C8C")
        self.submit_btn = ft.Container(
            height=48, width=260, border_radius=24, bgcolor="#FF8C00",
            alignment=ft.Alignment.CENTER, on_click=self._submit,
            content=ft.Text("Unlock", size=16, weight=ft.FontWeight.W_600, color="#000000"),
        )

        self.control = ft.Container(
            expand=True, bgcolor=BG_ROOT, alignment=ft.Alignment.CENTER,
            content=ft.Column(
                horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=18,
                controls=[
                    ft.Text("GuardianEye", size=32, weight=ft.FontWeight.W_600, color=TEXT_PRIMARY),
                    ft.Text("Enter the PIN to view this room", size=15, color=TEXT_SECONDARY),
                    ft.Container(height=8),
                    self.field,
                    self.submit_btn,
                    self.error_text,
                ],
            ),
        )

    def _submit(self, e) -> None:
        self._on_submit(self.field.value or "")

    def show_error(self, message: str) -> None:
        self.error_text.value = message
        self.field.value = ""
        self.control.update()
