"""review.py - the Console review screen, translating
docs/design/phase-8/DirectionB_Review.dc.html.

This screen gets its OWN left rail (position-in-queue list), replacing the
persistent room rail (views/rail.py) while it is open - the mock draws
these as genuinely different layouts, and the two tasks ("watch the room"
vs "decide about this one spot") are different enough that reusing the
room rail here would just be clutter, not consistency.

Hard rules this view exists to enforce (see CLAUDE.md decisions 3/4 and
backend/API.md's /review section):

  - /review is strictly FIFO. Only `current_id` is actionable. The rail
    shows queue POSITION for every entry using /review's own `queue` list,
    but only the first row is interactive - the rest are visibly inert
    (opacity-dimmed, no on_click), matching the mock exactly ("position is
    shown; selection is not offered").
  - The card must say whether the item is ALREADY ALERTING (arrived after
    the first scan, unreviewed) or not (found in the first scan, awaiting
    review) - straight from `current.alerts_on_approach` / `is_first_scan`,
    never recomputed here.

Differences from the mock, noted rather than silently dropped:

  - The mock's queued rail items (rows 2, 3, ...) show real-looking labels
    ("Floor, by the sofa"). /review's `queue` field is IDs only - it does
    not carry labels for anything but `current`. Rather than fabricate
    names for entries this app cannot yet identify, labels for queued (not
    current) entries are resolved from the latest /risk_status `hazards`
    snapshot already being polled at 4 Hz elsewhere in the app (real data,
    reused) and fall back to "Item N" if that id isn't in the cached
    snapshot yet.
  - The mock's top strip shows "appeared 14:02 · after monitoring started".
    Nothing in backend/API.md's hazard shape carries a first-seen
    timestamp, so this build states the same fact in words only ("after
    monitoring started" / "found in the first scan") without inventing a
    clock time.
  - The mock's left-hand "what the camera saw" box is a static
    illustration. This view attempts a REAL crop of the latest video frame
    using the hazard's bbox (Pillow, already present transitively via
    Flet's dependency chain) and falls back to a plain placeholder if the
    crop fails for any reason - it never blocks the review flow on that.
"""

from __future__ import annotations

from typing import Callable, Optional

import flet as ft

from common import back_button

from colors import (
    BORDER_SUBTLE,
    HAZARD_CONFIRMED,
    HAZARD_DISMISSED,
    HAZARD_PENDING_FIRST_SCAN,
    HAZARD_PENDING_NEW,
    TEXT_DIM,
    TEXT_LABEL,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

RAIL_WIDTH = 260

_STATE_COLOR = {
    "confirmed": HAZARD_CONFIRMED,
    "dismissed": HAZARD_DISMISSED,
}


def _hazard_outline(hazard: Optional[dict]) -> str:
    if hazard is None:
        return HAZARD_PENDING_FIRST_SCAN
    state = hazard.get("state")
    if state in _STATE_COLOR:
        return _STATE_COLOR[state]
    return HAZARD_PENDING_NEW if hazard.get("alerts_on_approach") else HAZARD_PENDING_FIRST_SCAN


class ReviewPanel:
    def __init__(self, page: ft.Page, on_confirm: Callable[[int], None],
                 on_dismiss: Callable[[int], None], on_skip: Callable[[], None],
                 on_close: Callable[[], None]):
        self.page = page
        self._on_confirm = on_confirm
        self._on_dismiss = on_dismiss
        self._on_skip = on_skip
        self._on_close = on_close
        self._current_id: Optional[int] = None
        # Real camera width from /risk_status - see _try_crop for why the
        # crop cannot use the displayed frame's own width.
        self._camera_width: Optional[int] = None
        self.dev_mode = False

        # -- rail: FIFO queue position, current item only interactive ------
        self.queue_list = ft.Column(spacing=8, controls=[])
        self.footer_txt = ft.Text("parent view", size=11, color=TEXT_MUTED, font_family="IBM Plex Mono")
        self.rail = ft.Container(
            width=RAIL_WIDTH, bgcolor="#0E1217",
            border=ft.Border.only(right=ft.BorderSide(1, BORDER_SUBTLE)),
            padding=ft.Padding.symmetric(horizontal=18, vertical=20),
            content=ft.Column(spacing=20, horizontal_alignment=ft.CrossAxisAlignment.STRETCH, controls=[
                ft.Text("REVIEW QUEUE", size=11, color=TEXT_LABEL, style=ft.TextStyle(letter_spacing=1.1)),
                self.queue_list,
                ft.Text("You decide these one at a time, in order.", size=12, color=TEXT_MUTED),
                ft.Container(expand=True),
                self.footer_txt,
            ]),
        )

        # -- main content ----------------------------------------------
        self.badge_text = ft.Text("", size=11, color="#FFB04D", style=ft.TextStyle(letter_spacing=0.6))
        self.badge = ft.Container(
            height=26, padding=ft.Padding.symmetric(horizontal=11),
            bgcolor="#FF8C0022", border=ft.Border.all(1, "#FF8C00"),
            alignment=ft.Alignment.CENTER, content=self.badge_text,
        )
        self.origin_text = ft.Text("", size=12, color=TEXT_DIM, font_family="IBM Plex Mono")
        # The way back must NOT live inside card_body. It used to: card_body
        # is hidden the moment the queue empties, which took the only exit
        # with it and stranded the parent on a "Nothing to review" screen with
        # no control at all (Shaked, live 2026-08-30: "when I've finished
        # classifying there is literally no way to go back to the main
        # screen"). A dead end is bad in any UI; in a monitoring product it
        # means the live view of the room is unreachable until you restart the
        # app. This control is now a sibling of card_body and is never hidden.
        self.back_btn = back_button(self._on_close)
        top_row = ft.Row(vertical_alignment=ft.CrossAxisAlignment.CENTER, controls=[
            self.badge, self.origin_text, ft.Container(expand=True),
        ])

        self.thumb_caption = ft.Text("CROP AT DETECTION", size=10, color=TEXT_SECONDARY,
                                      font_family="IBM Plex Mono", style=ft.TextStyle(letter_spacing=0.5))
        self.thumb_image = ft.Container(expand=True, alignment=ft.Alignment.CENTER,
                                         content=ft.Text("no frame yet", size=12, color=TEXT_DIM))
        self.thumb = ft.Container(
            width=340, height=226, bgcolor="#191d23",
            border=ft.Border.all(3, HAZARD_PENDING_FIRST_SCAN),
            content=ft.Stack(expand=True, controls=[
                self.thumb_image,
                ft.Container(left=8, bottom=7, content=self.thumb_caption),
            ]),
        )

        self.headline = ft.Text("", size=28, color=TEXT_PRIMARY)
        self.description = ft.Text("", size=15, color=TEXT_SECONDARY)
        self.error_text = ft.Text("", size=13, color="#FF8C8C")

        card_body_right = ft.Column(expand=True, spacing=14, controls=[
            self.headline, self.description,
        ])
        card_row = ft.Row(spacing=20, vertical_alignment=ft.CrossAxisAlignment.START, controls=[
            self.thumb, card_body_right,
        ])

        self.confirm_btn = ft.Container(
            height=56, padding=ft.Padding.symmetric(horizontal=28), alignment=ft.Alignment.CENTER,
            bgcolor="#FF0000", on_click=self._confirm_clicked,
            content=ft.Text("Yes, dangerous", size=17, weight=ft.FontWeight.W_600, color="#FFFFFF"),
        )
        self.dismiss_btn = ft.Container(
            height=56, padding=ft.Padding.symmetric(horizontal=28), alignment=ft.Alignment.CENTER,
            border=ft.Border.all(1, BORDER_SUBTLE), on_click=self._dismiss_clicked,
            content=ft.Text("Not a hazard", size=17, color=TEXT_PRIMARY),
        )
        self.skip_btn = ft.Container(
            height=56, padding=ft.Padding.symmetric(horizontal=22), alignment=ft.Alignment.CENTER,
            on_click=self._skip_clicked,
            content=ft.Text("Skip all for now", size=16, color=TEXT_DIM),
        )
        actions_row = ft.Row(spacing=12, controls=[
            self.confirm_btn, self.dismiss_btn, ft.Container(expand=True), self.skip_btn,
        ])

        self.empty_done_btn = ft.Container(
            height=52, padding=ft.Padding.symmetric(horizontal=30),
            alignment=ft.Alignment.CENTER, bgcolor="#FFFFFF",
            on_click=lambda e: self._on_close(),
            content=ft.Text("Back to live view", size=17, weight=ft.FontWeight.W_600, color="#0B0E12"),
        )
        self.empty_state = ft.Column(
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.START, spacing=14,
            controls=[
                ft.Text("All caught up.", size=28, color=TEXT_PRIMARY),
                ft.Text("Nothing is waiting for your decision. The room is still being watched.",
                        size=15, color=TEXT_SECONDARY),
                self.empty_done_btn,
            ],
        )
        self.card_body = ft.Column(expand=True, spacing=18, controls=[
            top_row, card_row, ft.Container(expand=True), self.error_text, actions_row,
        ])

        self.main = ft.Container(
            expand=True, padding=ft.Padding.symmetric(horizontal=24, vertical=22),
            content=ft.Column(expand=True, spacing=14, controls=[
                ft.Row(controls=[self.back_btn, ft.Container(expand=True)]),
                ft.Container(expand=True,
                             content=ft.Stack(expand=True,
                                              controls=[self.card_body, self.empty_state])),
            ]),
        )

        self.control = ft.Row(
            visible=False, expand=True, spacing=0,
            controls=[self.rail, self.main],
        )

    def set_dev_mode(self, enabled: bool) -> None:
        self.dev_mode = enabled
        self.footer_txt.value = "developer view" if enabled else "parent view"

    def show(self) -> None:
        self.control.visible = True
        self.control.update()

    def hide(self) -> None:
        self.control.visible = False
        self.control.update()

    def apply(self, review_data: dict, latest_jpeg: Optional[bytes], latest_hazards: list[dict],
              camera_width: Optional[int] = None) -> None:
        if camera_width:
            self._camera_width = int(camera_width)
        length = review_data.get("length", 0)
        current = review_data.get("current")
        queue = review_data.get("queue", [])
        self._current_id = review_data.get("current_id")

        hazards_by_id = {h.get("id"): h for h in latest_hazards}
        self.queue_list.controls.clear()
        for pos, entry_id in enumerate(queue, start=1):
            is_current = entry_id == self._current_id
            hz = current if is_current and current else hazards_by_id.get(entry_id)
            label = (hz or {}).get("label", f"Item {entry_id}").replace("_", " ").title()
            color = _hazard_outline(hz)
            self.queue_list.controls.append(
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=11, vertical=10),
                    opacity=1.0 if is_current else 0.42,
                    border=ft.Border.all(1, "#FFFFFF") if is_current else None,
                    bgcolor="#161b22" if is_current else None,
                    content=ft.Row(spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER, controls=[
                        ft.Text(str(pos), size=12, color="#FFFFFF" if is_current else TEXT_MUTED,
                                font_family="IBM Plex Mono"),
                        ft.Container(expand=True, content=ft.Text(
                            label, size=14, color="#FFFFFF" if is_current else TEXT_SECONDARY)),
                        ft.Container(width=9, height=9, border=ft.Border.all(2, color)),
                    ]),
                )
            )

        has_item = length > 0 and current is not None
        self.card_body.visible = has_item
        self.empty_state.visible = not has_item
        if not has_item:
            self.control.update()
            return

        alerts_now = bool(current.get("alerts_on_approach"))
        is_first_scan = bool(current.get("is_first_scan"))
        if alerts_now:
            self.badge.bgcolor = "#FF8C0022"
            self.badge.border = ft.Border.all(1, "#FF8C00")
            self.badge_text.value = "ALERTING NOW"
            self.badge_text.color = "#FFB04D"
        else:
            self.badge.bgcolor = "#00FFFF18"
            self.badge.border = ft.Border.all(1, "#00FFFF")
            self.badge_text.value = "NOT ALERTING YET"
            self.badge_text.color = "#8FEFEF"

        self.origin_text.value = (
            "found in the first scan" if is_first_scan else "appeared after monitoring started"
        )

        # The periodic room scan is CLASS-AGNOSTIC by design (CLAUDE.md
        # decision 7: "find anything on a surface a child can reach, whether
        # or not we can name it... naming it is a bonus, never the
        # requirement"), and its generic label is literally the string
        # "object". So "looks like a object" was wrong twice over: the
        # article disagreed, and more seriously it CLAIMED AN IDENTIFICATION
        # THE SYSTEM NEVER MADE - directly contradicting the sentence
        # rendered right below it ("does not know for certain what it is").
        # Only the per-frame pass produces real class names (oven/microwave/
        # refrigerator, decision 2), and naming those is legitimate.
        raw_label = (current.get("label") or "").strip()
        named = raw_label.replace("_", " ") if raw_label and raw_label != "object" else None
        article = "an" if named and named[:1].lower() in "aeiou" else "a"
        if is_first_scan:
            self.headline.value = (
                f"There is {article} {named} that was already there when monitoring started"
                if named else
                "Something was already on a surface he can reach when monitoring started"
            )
        else:
            self.headline.value = (
                f"{article.capitalize()} {named} appeared where he can reach it"
                if named else
                "Something new appeared where he can reach it"
            )

        if alerts_now:
            self.description.value = (
                "GuardianEye does not know for certain what it is, and does not need to. "
                "Until you decide, it is treated as dangerous and will alert when he gets close."
            )
        else:
            self.description.value = (
                "This was already in view when monitoring started, so it will not alert until "
                "you review it - the room is being watched right now while you decide."
            )

        outline = HAZARD_PENDING_NEW if alerts_now else HAZARD_PENDING_FIRST_SCAN
        self.thumb.border = ft.Border.all(3, outline)
        crop = _try_crop(latest_jpeg, current.get("bbox"), self._camera_width)
        if crop is not None:
            self.thumb_image.content = ft.Image(src=crop, fit=ft.BoxFit.COVER, expand=True)
        else:
            self.thumb_image.content = ft.Text("no frame yet", size=12, color=TEXT_DIM)

        self.control.update()

    def show_error(self, message: str) -> None:
        self.error_text.value = message
        self.control.update()

    def _confirm_clicked(self, e) -> None:
        if self._current_id is not None:
            self.error_text.value = ""
            self._on_confirm(self._current_id)

    def _dismiss_clicked(self, e) -> None:
        if self._current_id is not None:
            self.error_text.value = ""
            self._on_dismiss(self._current_id)

    def _skip_clicked(self, e) -> None:
        self.error_text.value = ""
        self._on_skip()


def _try_crop(jpeg: Optional[bytes], bbox, camera_width: Optional[int] = None) -> Optional[bytes]:
    """Crop the hazard's bbox out of the latest displayed frame.

    `camera_width` is load-bearing and must be the CAMERA's real width from
    /risk_status, not the frame's. Bboxes arrive in full camera coordinates
    (1920 wide), but the frame this crops is whatever the UI is currently
    displaying - and on the WEB path that frame has been downscaled to 960px
    (see ui/video.py's MjpegClient). Cropping camera-space coordinates out of
    a downscaled image was a real, shipped bug (Shaked, live 2026-08-30:
    "most of the first detections are corrupted and we can't see them while
    classifying"): a bbox starting past 960 produced x2 <= x1 and no crop at
    all, and one straddling 960 produced a wrong, half-cut image. Both look
    like a broken review card, which is the one screen where the parent has
    to actually SEE the thing to judge it.
    """
    if not jpeg or not bbox or len(bbox) != 4:
        return None
    try:
        import io
        from PIL import Image

        img = Image.open(io.BytesIO(jpeg))
        scale = (img.width / float(camera_width)) if camera_width else 1.0
        x1, y1, x2, y2 = [max(0, int(v * scale)) for v in bbox]
        # A little context around the box reads far better than a tight crop
        # when the object is small - the parent is judging "what is that",
        # not admiring the detector's precision.
        pad = int(0.12 * max(x2 - x1, y2 - y1))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = x2 + pad, y2 + pad
        x2 = min(img.width, x2)
        y2 = min(img.height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img.crop((x1, y1, x2, y2))
        out = io.BytesIO()
        crop.save(out, format="JPEG", quality=80)
        return out.getvalue()
    except Exception:
        return None
