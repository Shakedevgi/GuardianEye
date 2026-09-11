"""Focused desktop probe: does the NATIVE Flet window actually repaint, and
does each of the three video strategies render there?

A big TICK counter proves whether the self-screenshot is live or cached,
independently of anything to do with images.
"""
import asyncio, threading, time, urllib.request
import flet as ft

MJPEG_URL = "http://127.0.0.1:8111/video_feed"
STILL_URL = "http://127.0.0.1:8112/frame.jpg"


def read_mjpeg(on_frame, err):
    try:
        resp = urllib.request.urlopen(MJPEG_URL, timeout=10)
        buf = b""
        while True:
            while b"\r\n\r\n" not in buf:
                c = resp.read(4096)
                if not c:
                    return
                buf += c
            head, buf = buf.split(b"\r\n\r\n", 1)
            length = next((int(l.split(b":", 1)[1]) for l in head.split(b"\r\n")
                           if l.lower().startswith(b"content-length:")), None)
            if length is None:
                err.append("no content-length"); return
            while len(buf) < length:
                c = resp.read(length - len(buf))
                if not c:
                    return
                buf += c
            on_frame(buf[:length]); buf = buf[length:]
    except Exception as exc:
        err.append(f"{type(exc).__name__}: {exc}")


async def main(page: ft.Page):
    loop = asyncio.get_running_loop()
    page.title = "desktop probe 2"
    page.bgcolor = "#101014"
    page.window.width = 1250
    page.window.height = 760
    page.enable_screenshots = True

    tick_txt = ft.Text("TICK 0", size=34, color="#ffffff", weight=ft.FontWeight.BOLD)
    a_lbl = ft.Text("A: src=<mjpeg url>", size=12, color="#ffffff")
    b_lbl = ft.Text("B: src=<bytes>", size=12, color="#ffffff")
    c_lbl = ft.Text("C: src=<still>?t=n", size=12, color="#ffffff")

    seed = urllib.request.urlopen(STILL_URL, timeout=5).read()
    img_a = ft.Image(src=MJPEG_URL, width=360, height=203,
                     error_content=ft.Text("A: ERROR CONTENT", color="#ff5555", size=16))
    img_b = ft.Image(src=seed, width=360, height=203, gapless_playback=True)
    img_c = ft.Image(src=STILL_URL + "?t=0", width=360, height=203, gapless_playback=True)

    def cell(lbl, img):
        return ft.Container(content=ft.Column([lbl, img], spacing=6),
                            bgcolor="#1b1b22", padding=8, border_radius=8)

    page.add(tick_txt, ft.Row([cell(a_lbl, img_a), cell(b_lbl, img_b), cell(c_lbl, img_c)]))

    st = {"b": 0, "c": 0}
    err_b = []

    def on_frame(j):
        st["b"] += 1
        img_b.src = j
        page.update()

    threading.Thread(target=read_mjpeg, args=(on_frame, err_b), daemon=True).start()

    def poll_c():
        while True:
            st["c"] += 1
            img_c.src = f"{STILL_URL}?t={st['c']}"
            page.update()
            time.sleep(1 / 10)

    threading.Thread(target=poll_c, daemon=True).start()

    def ticker():
        n = 0
        while True:
            n += 1
            tick_txt.value = f"TICK {n}"
            a_lbl.value = f"A: src=<mjpeg url>  (no error event)"
            b_lbl.value = f"B: src=<bytes>  n={st['b']}  {err_b}"
            c_lbl.value = f"C: src=<still>  n={st['c']}"
            page.update()
            time.sleep(1)

    threading.Thread(target=ticker, daemon=True).start()

    log = open("probe2.log", "a", buffering=1)
    for i in range(3):
        await asyncio.sleep(18)
        try:
            png = await page.take_screenshot(pixel_ratio=1.5, delay=500)
            open(f"p2_shot_{i}.png", "wb").write(png)
            log.write(f"shot {i}: {len(png)} bytes, tick={tick_txt.value}, b={st['b']}, c={st['c']}\n")
        except Exception as exc:
            log.write(f"shot {i} FAILED {type(exc).__name__}: {exc}\n")


ft.run(main)
