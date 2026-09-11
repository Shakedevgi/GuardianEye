"""log.py - event log panel: parent (grouped) view vs developer (raw) view.

Console direction: this panel replaces the main column's content while the
persistent left rail (views/rail.py) stays visible, so a parent never loses
the room's live state just for opening the log - consistent with the
rebuild brief's "lean into structure, dividers, section labels" note and
its defect list ("nothing communicated application").

Fetches /events only while this panel is visible (Phase 8 polling decision:
"/events and /clips on demand / only while their panel is visible" - not
polled continuously like /risk_status and /health).
"""

from __future__ import annotations

from typing import Callable

import flet as ft

from common import back_button

from colors import BG_PANEL, BORDER_SUBTLE, TEXT_DIM, TEXT_PRIMARY, TEXT_SECONDARY, ZONE_COLOR
from events import format_last_seen, group_events_by_hazard


class EventLogPanel:
    def __init__(self, page: ft.Page, on_close: Callable[[], None]):
        self.page = page
        self._on_close = on_close
        self.dev_mode = False

        self.title = ft.Text("Event log", size=26, color=TEXT_PRIMARY)
        self.close_btn = back_button(self._on_close)
        self.status_text = ft.Text("", size=13, color=TEXT_DIM, font_family="IBM Plex Mono")
        self.list_view = ft.ListView(expand=True, spacing=8)

        self.control = ft.Container(
            visible=False, expand=True, padding=ft.Padding.symmetric(horizontal=28, vertical=24),
            content=ft.Column(expand=True, spacing=16, controls=[
                ft.Row(controls=[self.title, ft.Container(expand=True), self.status_text, self.close_btn]),
                ft.Container(height=1, bgcolor=BORDER_SUBTLE),
                self.list_view,
            ]),
        )

    def show(self) -> None:
        self.control.visible = True
        self.control.update()

    def hide(self) -> None:
        self.control.visible = False
        self.control.update()

    def set_dev_mode(self, enabled: bool) -> None:
        self.dev_mode = enabled

    def apply(self, events: list[dict]) -> None:
        self.list_view.controls.clear()
        if self.dev_mode:
            self._render_raw(events)
        else:
            self._render_grouped(events)
        self.status_text.value = f"{len(events)} raw rows"
        self.control.update()

    def show_error(self, message: str) -> None:
        self.list_view.controls.clear()
        self.list_view.controls.append(ft.Text(message, size=14, color="#FF8C8C"))
        self.control.update()

    def _render_grouped(self, events: list[dict]) -> None:
        groups = group_events_by_hazard(events)
        if not groups:
            self.list_view.controls.append(ft.Text("No events yet.", size=15, color=TEXT_SECONDARY))
            return
        for g in groups:
            zone_color = ZONE_COLOR.get(g["peak_zone"], ZONE_COLOR["none"])
            duration_note = f" · {g['row_count']} sightings" if g["row_count"] > 1 else ""
            clip_note = " · clip saved" if g["clip_triggered"] else ""
            self.list_view.controls.append(
                ft.Container(
                    bgcolor=BG_PANEL, padding=14, border=ft.Border.all(1, BORDER_SUBTLE),
                    content=ft.Row(controls=[
                        ft.Container(width=6, height=40, bgcolor=zone_color),
                        ft.Column(spacing=3, expand=True, controls=[
                            ft.Text(f"Went near the {g['hazard_label']} · up to {g['peak_zone']}",
                                    size=15, color=TEXT_PRIMARY),
                            ft.Text(f"{format_last_seen(g['last_seen_at'])}{duration_note}{clip_note}",
                                    size=13, color=TEXT_SECONDARY),
                        ]),
                    ]),
                )
            )

    def _render_raw(self, events: list[dict]) -> None:
        if not events:
            self.list_view.controls.append(ft.Text("No events yet.", size=15, color=TEXT_SECONDARY))
            return
        for row in events:
            zone_color = ZONE_COLOR.get(row.get("peak_zone", "none"), ZONE_COLOR["none"])
            self.list_view.controls.append(
                ft.Container(
                    bgcolor=BG_PANEL, padding=12, border=ft.Border.all(1, BORDER_SUBTLE),
                    content=ft.Row(controls=[
                        ft.Container(width=6, height=36, bgcolor=zone_color),
                        ft.Column(spacing=3, expand=True, controls=[
                            ft.Text(
                                f"#{row.get('id')} person={row.get('person_id')} "
                                f"hazard={row.get('hazard_id')}({row.get('hazard_label')}) "
                                f"kind={row.get('kind')} zone={row.get('zone')}->peak={row.get('peak_zone')}",
                                size=13, color=TEXT_PRIMARY, font_family="IBM Plex Mono",
                            ),
                            ft.Text(
                                f"started {row.get('started_at')} · last_seen {row.get('last_seen_at')} "
                                f"· clip_triggered={row.get('clip_triggered')}",
                                size=12, color=TEXT_SECONDARY, font_family="IBM Plex Mono",
                            ),
                        ]),
                    ]),
                )
            )
