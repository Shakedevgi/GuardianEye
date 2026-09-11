"""common.py - shared view widgets.

Exists for one reason worth stating: Shaked, live 2026-08-30, "when I've
finished classifying there is literally no way to go back to the main screen
- we need ALWAYS a way to go back to the main screen." That was literally
true of the review panel (its only exit lived inside a body that got hidden
the moment the queue emptied), and the other panels were a softer version of
the same problem - a dim, unlabelled X in a corner, which is the same
complaint as "the settings are impossible to see."

So the way back is now one shared, labelled control used by every panel,
rather than four separately-styled icon buttons that can each drift or
disappear. A monitoring product must never strand a parent away from the
live view of the room.
"""

from __future__ import annotations

from typing import Callable

import flet as ft

from colors import BORDER_SUBTLE, TEXT_PRIMARY, TEXT_SECONDARY


def back_button(on_click: Callable[[], None], label: str = "Back to live view") -> ft.Container:
    return ft.Container(
        height=40,
        padding=ft.Padding.symmetric(horizontal=16),
        alignment=ft.Alignment.CENTER,
        border=ft.Border.all(1, BORDER_SUBTLE),
        on_click=lambda e: on_click(),
        tooltip=label,
        content=ft.Row(
            spacing=8, tight=True, vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Icon(ft.Icons.ARROW_BACK, size=16, color=TEXT_SECONDARY),
                ft.Text(label, size=14, color=TEXT_PRIMARY),
            ],
        ),
    )
