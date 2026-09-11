"""Isolate the RSS growth seen when pushing JPEG bytes into ft.Image.

modes:
  read      - consume the MJPEG stream, never touch the UI   (control)
  page      - set img.src then page.update()   every frame
  ctrl      - set img.src then img.update()    every frame
  throttled - set img.src then img.update(), max ~10 fps, dropping frames
"""
import asyncio, os, sys, threading, time, urllib.request
import flet as ft

MODE = os.environ.get("LEAK_MODE", "read")
MJPEG_URL = "http://127.0.0.1:8111/video_feed"
STILL_URL = "http://127.0.0.1:8112/frame.jpg"


def read_mjpeg(on_frame):
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
        while len(buf) < length:
            c = resp.read(length - len(buf))
            if not c:
                return
            buf += c
        on_frame(buf[:length]); buf = buf[length:]


def rss_kb():
    return int(os.popen(f"ps -o rss= -p {os.getpid()}").read().strip() or 0)


def flet_client_rss_kb():
    """RSS of the native Flutter window process - strategy A's buffering would
    land here, not in the Python process."""
    out = os.popen("ps -o rss=,comm= -ax | grep 'Flet.app/Contents/MacOS/Flet' | grep -v grep").read()
    return sum(int(l.split()[0]) for l in out.strip().splitlines() if l.strip())


async def main(page: ft.Page):
    page.title = f"leak probe [{MODE}]"
    seed = urllib.request.urlopen(STILL_URL, timeout=5).read()
    if MODE == "urlimg":
        # strategy A alone: one ft.Image pointed at the never-ending MJPEG URL
        img = ft.Image(src=MJPEG_URL, width=480, height=270)
    else:
        img = ft.Image(src=seed, width=480, height=270, gapless_playback=True)
    page.add(ft.Text(f"mode={MODE}", size=20), img)

    st = {"n": 0, "last_push": 0.0}

    def on_frame(j):
        st["n"] += 1
        if MODE == "read":
            return
        if MODE == "throttled":
            now = time.monotonic()
            if now - st["last_push"] < 0.1:
                return          # drop the frame instead of queueing it
            st["last_push"] = now
        img.src = j
        if MODE == "page":
            page.update()
        else:
            img.update()

    if MODE != "urlimg":
        threading.Thread(target=read_mjpeg, args=(on_frame,), daemon=True).start()

    base = None
    for i in range(10):
        await asyncio.sleep(10)
        r = rss_kb()
        if base is None:
            base = r
        print(f"{MODE} t={(i+1)*10}s rss={r}KB delta={r-base}KB client={flet_client_rss_kb()}KB frames={st['n']}", flush=True)
    os._exit(0)


ft.run(main)
