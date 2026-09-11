"""app.py - GuardianEye Flet shell (Phase 8, "Console" direction).

Read CLAUDE.md decision 1 and docs/decision-log.md's 2026-08-30 "Phase 8"
entries before changing anything here - this file's shape (PIN gate before
ANY frame push, one /risk_status poll driving the whole live view, no
direct ft.Image(src=<mjpeg url>)) follows measurements made this session,
not preference. Those constraints are unchanged from the earlier "Night
Watch" build; what changed in this rebuild is the SHELL - a persistent
left rail (views/rail.py) plus a main column whose content switches
between the live view, the review screen (its own rail - see
views/review.py), and the log/clips/settings panels, all instead of the
old build's bottom action bar and full-screen overlay panels.

Run as a desktop window:
    cd ui && ../.venv/bin/python3 app.py --backend http://127.0.0.1:8000 --pin 1234

Run as a local web app reachable from other devices on the WiFi:
    cd ui && ../.venv/bin/python3 app.py --mode web --web-host 0.0.0.0 \\
        --web-port 8551 --backend http://127.0.0.1:8000 --pin 1234

Develop against the synthetic camera harness instead of the real camera:
    .venv/bin/python3 docs/design/phase-8/fake_camera_server.py
    cd ui && ../.venv/bin/python3 app.py --backend http://127.0.0.1:8111 --pin 1234

--dev boots straight into the developer view (fps/model/imgsz/conf/device/
frame_seq/last_frame_age/raw distances/hazard ids-bboxes-origins, and
ungrouped /events rows with person_ids) instead of requiring a parent to
find the Settings toggle - CLAUDE.md decision 1's split, both switches
required, neither optional.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "views"))

import flet as ft
import flet_audio as fa

from api_client import ApiClient, ApiError, NotReady, ReviewCommandFailed, ReviewNotCurrent
from alert_audio import AlertSoundPolicy
from colors import BG_ROOT
from video import MjpegClient
import token_store
from views.pin import PinView
from views.rail import PersistentRail
from views.live import LiveView
from views.review import ReviewPanel
from views.log import EventLogPanel
from views.clips import ClipsPanel
from views.settings import SettingsPanel

# Web-path video budget. OFF by default (0 = no resize, no rate limit - see
# MjpegClient: max_width=0/max_fps=0 mean "pass the frame through
# untouched"). Only takes effect when explicitly requested via
# --web-max-width/--web-quality/--web-max-fps.
#
# HISTORY, because this got it wrong once already: 2026-08-30 this was tied
# to `page.web` - "any web session gets downscaled." That was too broad by
# exactly the amount that mattered: `page.web` is True for the normal Mac
# browser (127.0.0.1) too, not just a weak mobile link, so the SAME
# degradation that was meant for a struggling iPad over what looks like a
# personal-hotspot-class connection also silently hit the ordinary Mac
# browser session, which had been working fine at full quality and had no
# bandwidth problem at all (Shaked, 2026-09-11: "the live feed on the web
# app is now too small"). Tying image quality to "is this a browser" was
# the wrong signal - it conflates a transport with a network condition, and
# a Mac on loopback and an iPad on a phone hotspot are not the same
# situation just because both happen to be "web".
#
# So this is now OPT-IN only, per launch, via the flags below - never
# inferred from page.web again. The normal `--mode web` Mac-browser path
# gets full resolution/quality/fps unless someone explicitly asks
# otherwise. When testing a specific weak/mobile client (e.g. the
# `flet run --ios` path), pass the flags for THAT launch only - see
# SETUP.md and docs/decision-log.md, 2026-09-06 and 2026-09-11.
WEB_VIDEO_MAX_WIDTH = 0
WEB_VIDEO_QUALITY = 70
WEB_VIDEO_MAX_FPS = 0.0

RISK_STATUS_POLL_HZ = 4.0
HEALTH_POLL_HZ = 1.0
RECENT_EVENTS_POLL_SECONDS = 5.0

# How long to wait after unlock before the alert-sound Service control is
# trusted to actually play - measured directly against this Flet version
# (0.86.5): calling .play() on an Audio control within the same burst of
# code that constructed it reliably times out after 30s (RuntimeError:
# "Future not completed"), because the client has not finished mounting
# the newly-registered service yet. A ~1.5s pause after construction fixed
# it in every trial. See the comment where alert_audio is constructed for
# the full story and what this costs (a hazard in that exact ~1.5s window
# stays silent on THIS device only - the Mac's own speaker, driven
# independently by the detection loop, is unaffected either way).
ALERT_AUDIO_SETTLE_SECONDS = 1.5

PANEL_NAMES = ("live", "review", "log", "clips", "settings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GuardianEye Flet UI (Phase 8, Console direction)")
    parser.add_argument("--backend", default=os.environ.get("GUARDIANEYE_BACKEND", "http://127.0.0.1:8000"),
                         help="Base URL of backend/server.py's FastAPI app (stays bound to 127.0.0.1 "
                              "per CLAUDE.md's Phase 7 privacy decision - this UI is a client of it).")
    parser.add_argument("--pin", default=os.environ.get("GUARDIANEYE_PIN", "1234"),
                         help="PIN required to view this room. Configurable via flag/env, not hardcoded "
                              "(Phase 8 decision, 2026-08-30).")
    parser.add_argument("--mode", choices=["desktop", "web"], default="desktop")
    parser.add_argument("--web-host", default="127.0.0.1",
                         help="Defaults to loopback ONLY (Mac-only use). LAN/mobile access is "
                              "parked for now (Shaked, 2026-08-30 - see docs/decision-log.md) "
                              "while a better approach than certificates is found. Passing "
                              "'0.0.0.0' here is a real, known trap: Flet's own auto-opened "
                              "browser tab literally opens http://0.0.0.0:<port>, which does not "
                              "load - Flet only substitutes a working address when this value is "
                              "empty, and '0.0.0.0' is not empty. FastAPI stays on 127.0.0.1 "
                              "regardless of this flag either way (Phase 7 decision).")
    parser.add_argument("--web-port", type=int, default=8551)
    parser.add_argument("--web-max-width", type=int, default=WEB_VIDEO_MAX_WIDTH,
                         help="OFF by default (0 = full resolution, for every session "
                              "including the normal Mac browser). Set only when "
                              "deliberately testing a specific weak/mobile client - it "
                              "downscales frame width (px) for THAT launch. Lower = less "
                              "data, smoother over a weak link, blurrier picture. See "
                              "SETUP.md and docs/decision-log.md, 2026-09-11.")
    parser.add_argument("--web-quality", type=int, default=WEB_VIDEO_QUALITY,
                         help="JPEG quality (1-100), only used when --web-max-width > 0. "
                              "Lower = less data, more compression artifacts.")
    parser.add_argument("--web-max-fps", type=float, default=WEB_VIDEO_MAX_FPS,
                         help="OFF by default (0 = no cap, for every session). Set only "
                              "when deliberately testing a specific weak/mobile client. "
                              "Lower = less data and more time for a slow client to keep "
                              "up, choppier motion.")
    parser.add_argument("--dev", action="store_true",
                         help="Boot directly into the developer view.")
    return parser.parse_args()


def build_app(args: argparse.Namespace):
    api = ApiClient(args.backend)

    async def main(page: ft.Page) -> None:
        loop = asyncio.get_running_loop()
        page.title = "GuardianEye"
        page.bgcolor = BG_ROOT
        page.padding = 0
        page.window.width = 1180
        page.window.height = 820
        page.theme_mode = ft.ThemeMode.DARK

        state = {
            "dev_mode": args.dev, "mjpeg": None, "authed": False, "mode": "live",
            "latest_hazards": [], "red_alert_since": None,
        }

        # ---- alert sound: a Service control, not a visual one -
        # flet_audio.Audio.init() self-registers into the page's service
        # registry the moment it's constructed (confirmed by reading
        # flet/controls/services/service.py), so unlike ft.Image there is
        # no page.overlay.append()/page.services.append() to call. Files
        # live in ui/assets/audio/ - deliberately duplicated from cv/audio/
        # rather than fetched over HTTP, so this stays a client-side-only
        # addition with no new backend route.
        #
        # ALERT_AUDIO_SETTLE_SECONDS (defined above) is real and
        # load-bearing, not defensive padding - see its own comment for the
        # measured 30s-timeout bug it works around. Constructing with a
        # real `src` up front (not None) matches exactly what was tested -
        # not confirmed to matter, not worth re-testing to find out.
        alert_audio = fa.Audio(src="audio/hazard_detected.wav", autoplay=False)
        alert_sound_policy = AlertSoundPolicy()
        state["alert_audio_ready"] = False

        async def _arm_alert_audio() -> None:
            await asyncio.sleep(ALERT_AUDIO_SETTLE_SECONDS)
            state["alert_audio_ready"] = True

        async def _play_alert_clip(clip: str) -> None:
            # Silently skip rather than raise during the settle window -
            # the PRIMARY alert (this Mac's own speaker, from the real
            # detection loop) has no such delay and is unaffected either
            # way; a hazard in the first ~1.5s after unlock still alerts
            # normally there. Logged so a real miss during testing is
            # visible rather than silently invisible.
            if not state["alert_audio_ready"]:
                print(f"[alert_audio] skipped {clip!r} - still settling", flush=True)
                return
            try:
                alert_audio.src = f"audio/{clip}"
                await alert_audio.play()
                # Mirrors cv/risk_engine.py's own "ALERT:" print (its
                # docstring on speak() calls that line an evidence trail
                # live tests are checked against) - same idea here, for
                # this device's OWN, separate playback.
                print(f"[alert_audio] played {clip!r}", flush=True)
            except Exception as exc:
                print(f"[alert_audio] play failed for {clip!r}: {type(exc).__name__}: {exc}", flush=True)

        # ---- assemble views (video Image control is created here and is
        # never removed from the tree afterwards - see ui/video.py) -------

        def open_pending_clip() -> None:
            _fire(open_latest_pending_clip())

        rail = PersistentRail(page, on_nav=lambda name: show_panel(name))
        live = LiveView(page, on_open_clip=open_pending_clip)
        live.set_dev_mode(state["dev_mode"])

        def confirm_clicked(entry_id: int):
            _fire(review_action(api.review_confirm, entry_id))

        def dismiss_clicked(entry_id: int):
            _fire(review_action(api.review_dismiss, entry_id))

        def skip_clicked():
            _fire(review_action_no_id(api.review_skip))

        review_panel = ReviewPanel(
            page, on_confirm=confirm_clicked, on_dismiss=dismiss_clicked,
            on_skip=skip_clicked, on_close=lambda: show_panel("live"),
        )
        review_panel.set_dev_mode(state["dev_mode"])

        log_panel = EventLogPanel(page, on_close=lambda: show_panel("live"))
        log_panel.set_dev_mode(state["dev_mode"])

        def keep_clip(clip_id: int):
            _fire(clip_action(api.keep_clip, clip_id))

        def discard_clip(clip_id: int):
            _fire(clip_action(api.discard_clip, clip_id))

        def refresh_clips(status: str):
            _fire(load_clips(status))

        clips_panel = ClipsPanel(
            page, on_close=lambda: show_panel("live"), on_keep=keep_clip, on_discard=discard_clip,
            on_refresh=refresh_clips, video_url_fn=api.clip_video_url,
        )

        def dev_mode_changed(enabled: bool) -> None:
            state["dev_mode"] = enabled
            live.set_dev_mode(enabled)
            log_panel.set_dev_mode(enabled)
            review_panel.set_dev_mode(enabled)
            rail.set_dev_mode(enabled)

        def logout() -> None:
            token_store.clear_token_sync(page)
            state["authed"] = False
            show_pin_gate()

        settings_panel = SettingsPanel(
            page, on_close=lambda: show_panel("live"), on_dev_mode_change=dev_mode_changed,
            base_url=args.backend, dev_mode_default=state["dev_mode"], on_logout=logout,
        )
        rail.set_dev_mode(state["dev_mode"])

        # -- body: live/log/clips/settings share the persistent rail; only
        # one is visible at a time, matching CONTAINER structure of the
        # mock rather than a full-screen overlay covering the whole app --
        body_panels = {"live": live.control, "log": log_panel.control,
                        "clips": clips_panel.control, "settings": settings_panel.control}
        body_stack = ft.Stack(expand=True, controls=list(body_panels.values()))
        normal_shell = ft.Row(expand=True, spacing=0, controls=[rail.control, body_stack])

        # review gets its OWN rail (docs/design/phase-8/DirectionB_Review.dc.html)
        root_stack = ft.Stack(expand=True, controls=[normal_shell, review_panel.control])

        def show_panel(name: str) -> None:
            assert name in PANEL_NAMES, name
            state["mode"] = name
            for key, control in body_panels.items():
                control.visible = key == name
            review_panel.control.visible = name == "review"
            body_stack.visible = name != "review"
            normal_shell.visible = name != "review"
            root_stack.update()
            if name == "log":
                _fire(load_events())
            elif name == "clips":
                _fire(load_clips(clips_panel.status))
            elif name == "review":
                _fire(load_review())

        pin_view = PinView(on_submit=lambda pin: _fire(handle_pin_submit(pin)))

        root = ft.Container(expand=True, content=pin_view.control)
        page.add(root)

        # ---- async helpers -----------------------------------------------

        def _fire(coro) -> None:
            asyncio.run_coroutine_threadsafe(coro, loop)

        async def review_action(fn, entry_id: int) -> None:
            try:
                await asyncio.to_thread(fn, entry_id)
            except ReviewNotCurrent as exc:
                review_panel.show_error(f"That item moved on: {exc.reason}")
            except ReviewCommandFailed as exc:
                review_panel.show_error(f"Camera loop did not respond: {exc}")
            except ApiError as exc:
                review_panel.show_error(str(exc))

        async def review_action_no_id(fn) -> None:
            try:
                await asyncio.to_thread(fn)
            except ReviewCommandFailed as exc:
                review_panel.show_error(f"Camera loop did not respond: {exc}")
            except ApiError as exc:
                review_panel.show_error(str(exc))

        async def clip_action(fn, clip_id: int) -> None:
            try:
                await asyncio.to_thread(fn, clip_id)
            except ApiError as exc:
                clips_panel.show_error(str(exc))
            await load_clips(clips_panel.status)

        async def load_events() -> None:
            try:
                events = await asyncio.to_thread(api.events, 200)
                log_panel.apply(events)
            except ApiError as exc:
                log_panel.show_error(str(exc))

        async def load_clips(status: str) -> None:
            try:
                clip_rows = await asyncio.to_thread(api.clips, status)
                clips_panel.apply(clip_rows)
            except ApiError as exc:
                clips_panel.show_error(str(exc))

        async def load_review() -> None:
            try:
                review_data = await asyncio.to_thread(api.review)
                latest_jpeg = live.video_image.src if isinstance(
                    live.video_image.src, (bytes, bytearray)) else None
                review_panel.apply(review_data, latest_jpeg, state["latest_hazards"],
                                   camera_width=state.get("camera_width"))
            except NotReady:
                pass
            except ApiError as exc:
                review_panel.show_error(str(exc))

        async def open_latest_pending_clip() -> None:
            try:
                pending = await asyncio.to_thread(api.clips, "pending")
                if pending:
                    page.launch_url(api.clip_video_url(pending[0]["id"]))
            except ApiError:
                pass

        # ---- PIN gate: no frame bytes pushed to this session until this
        # passes (Phase 8 decision, 2026-08-30) ----------------------------

        def show_pin_gate() -> None:
            root.content = pin_view.control
            root.update()

        def show_main_shell() -> None:
            root.content = root_stack
            root.update()
            show_panel("live")

        async def handle_pin_submit(entered: str) -> None:
            if entered != args.pin:
                pin_view.show_error("Wrong PIN")
                return
            await token_store.save_token(page, entered)
            await unlock()

        async def unlock() -> None:
            state["authed"] = True
            show_main_shell()
            start_mjpeg()
            asyncio.create_task(_arm_alert_audio())
            asyncio.create_task(poll_risk_status())
            asyncio.create_task(poll_health())
            asyncio.create_task(poll_recent_events())

        def start_mjpeg() -> None:
            if state["mjpeg"] is not None:
                return
            # Full resolution/fps by default, for every session, no
            # exceptions - NOT gated on page.web. It used to be, and that
            # silently degraded the ordinary Mac browser along with the one
            # session that actually needed it (see the module-level comment
            # above). Only these explicit flags may reduce quality, and only
            # when someone deliberately passes them for a specific launch.
            client = MjpegClient(
                f"{args.backend}/video_feed",
                on_frame=live.push_frame,
                max_width=args.web_max_width,
                quality=args.web_quality,
                max_fps=args.web_max_fps,
            )
            client.start()
            state["mjpeg"] = client

            # Video diagnostics on stdout, once a second. Added 2026-08-30
            # because a black video pane is ambiguous from the outside: it
            # looks identical whether (a) the reader never connected, (b)
            # frames arrive in Python but never reach the browser, or (c)
            # the browser has them and is not painting. These three need
            # completely different fixes, and guessing between them wasted a
            # round already. `pushed` counts frames handed to the control;
            # if pushed climbs while the pane stays black, the problem is on
            # the browser side of the socket, not in this process.
            async def video_stats() -> None:
                last = -1
                while state["authed"]:
                    await asyncio.sleep(1.0)
                    c = state["mjpeg"]
                    if c is None:
                        continue
                    if c.frame_count != last:
                        last = c.frame_count
                        print(f"[video] pushed={c.frame_count} dropped={c.dropped_frames} "
                              f"web={bool(page.web)} err={c.last_error or '-'}", flush=True)
                    else:
                        print(f"[video] STALLED at pushed={c.frame_count} "
                              f"err={c.last_error or '-'}", flush=True)
                    # Always report, including (especially) when nothing is
                    # arriving - that is precisely the case the overlay is
                    # there to explain.
                    try:
                        live.report_video_state(c.frame_count, c.last_error)
                    except Exception:
                        pass

            asyncio.create_task(video_stats())

        # ---- polling loops (single /risk_status call drives the whole
        # live view + rail + review panel; /health separately; /events on
        # its own slow timer, per Phase 8 decision) ------------------------

        async def poll_risk_status() -> None:
            interval = 1.0 / RISK_STATUS_POLL_HZ
            while state["authed"]:
                try:
                    status = await asyncio.to_thread(api.risk_status)
                    live.apply_status(status)
                    rail.apply_status(status)
                    state["latest_hazards"] = status.get("hazards", []) or []
                    # Real camera width, needed to translate hazard bboxes
                    # (camera coordinates) into the displayed frame, which
                    # the web path downscales. See views/review.py _try_crop.
                    state["camera_width"] = (status.get("camera") or {}).get("width")
                    _track_alert_duration(status)
                    clip = alert_sound_policy.next_clip(status.get("alert"))
                    if clip:
                        await _play_alert_clip(clip)
                    if state["mode"] == "review":
                        await load_review()
                except NotReady:
                    live.set_booting()
                except ApiError:
                    pass
                await asyncio.sleep(interval)

        def _track_alert_duration(status: dict) -> None:
            risk = status.get("risk", {}) or {}
            alert = status.get("alert", {}) or {}
            is_red = risk.get("zone") == "red" and bool(alert.get("active"))
            if is_red:
                if state["red_alert_since"] is None:
                    state["red_alert_since"] = time.time()
                started = time.strftime("%H:%M:%S", time.localtime(state["red_alert_since"]))
                rail.set_in_play_duration(f"closing since {started}")
            else:
                state["red_alert_since"] = None

        async def poll_health() -> None:
            interval = 1.0 / HEALTH_POLL_HZ
            while state["authed"]:
                try:
                    health = await asyncio.to_thread(api.health)
                    rail.apply_health(health, reachable=True)
                except ApiError:
                    rail.apply_health(None, reachable=False)
                await asyncio.sleep(interval)

        async def poll_recent_events() -> None:
            # Deliberately separate from the 4 Hz /risk_status cadence -
            # the "recent" strip on the live screen is a nice-to-have, not
            # part of the risk-status contract, so it is refreshed slowly
            # and only feeds the live view's own summary.
            while state["authed"]:
                try:
                    events = await asyncio.to_thread(api.events, 50)
                    live.apply_events(events)
                except ApiError:
                    pass
                await asyncio.sleep(RECENT_EVENTS_POLL_SECONDS)

        # ---- try to skip the PIN screen if this client already has a
        # remembered token (best-effort - see ui/token_store.py's docstring
        # for the honest desktop-vs-web verification gap) ------------------

        remembered = await token_store.load_token(page)
        if remembered == args.pin:
            await unlock()

    return main


def _print_web_urls(port: int, host: str) -> None:
    """Print the address to open. `--web-host` now defaults to 127.0.0.1 -
    Mac-only - so there is normally exactly one address and no confusion.

    Kept simple on purpose. A prior version of this tried to make a plain-http
    LAN/tablet address usable and printed one; it never worked (browsers will
    not paint video on a non-loopback plain-http origin - measured
    2026-08-30) and an HTTPS-based fix was reverted at Shaked's request as
    over-complicated for what should be a simple thing to use (see
    docs/decision-log.md, 2026-08-30 "iPad/mobile access reverted"). Mobile
    access is parked pending a better approach - see that entry before
    reintroducing a --https or LAN flag here.
    """
    print(f"\nGuardianEye UI - open: http://{host}:{port}\n", flush=True)


def main() -> None:
    args = parse_args()
    app_main = build_app(args)
    if args.mode == "web":
        _print_web_urls(args.web_port, args.web_host)
        ft.run(app_main, view=ft.AppView.WEB_BROWSER, host=args.web_host,
               port=args.web_port, no_cdn=True)
    else:
        ft.run(app_main, no_cdn=True)


if __name__ == "__main__":
    main()
