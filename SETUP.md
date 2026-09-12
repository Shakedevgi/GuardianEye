# Running GuardianEye

The only file you need. If you forget how to start it, look here first —
don't ask in chat.

---

## Every time you want to use it

```bash
cd /Users/shakedivgi/Projects/GuardianEye
./start.sh
```

Wait for `PIN: 4821` to appear (~30-60 seconds — it's loading the AI models).

Then open in your browser:

```
http://127.0.0.1:8551
```

Enter PIN **4821**, click Unlock. Done.

**To stop:** go back to the terminal, press `Ctrl-C`. This stops the camera
and the app together — you never need to close them separately.

**Want the desktop app window too, at the same time?**
```bash
./start.sh --desktop
```

---

## Mobile / iPad / tablet

**Working, as of 2026-09-11** — video shows with almost no delay. Uses the
free "Flet" app from the App Store, no certificates, no extra setup on the
iPad beyond installing that one app.

**Steps:**

1. Make sure `./start.sh` is already running in one terminal (camera + backend).
2. **In a second terminal**, find this Mac's current WiFi address:
   ```bash
   ipconfig getifaddr en0
   ```
   (This can change between sessions — always check it fresh, don't reuse an
   old one written down somewhere.)
3. Start the preview server, using the address from step 2:
   ```bash
   cd /Users/shakedivgi/Projects/GuardianEye/ui
   ../.venv/bin/flet run --ios --host <address from step 2> -p 8552 app.py -- --pin 4821 --web-max-fps 2
   ```
   **`--web-max-fps 2` is required, not optional** — see "why 2" below.
4. A QR code appears in the terminal. On the iPad, open the **Flet** app →
   **+** → scan it (or type the URL the terminal prints next to
   `App is running on:`).
5. Enter the PIN, same as on the Mac.

**Ignore** any `GuardianEye UI - open: http://...` line this prints — that's
this app's own message and it's wrong in this mode; the actual address is
the one `flet run` shows you.

**To stop:** `Ctrl-C` in that second terminal. Camera/backend keep running —
you don't need to touch that terminal at all for this.

### Why `--web-max-fps 2`, and what it does and doesn't affect

The picture on the iPad updates about twice a second — noticeably choppier
than the smooth video on the Mac. **This is deliberate, not a bug to chase
away**, and it's worth understanding what it costs and what it doesn't:

- **The actual safety alerts are NOT slowed down by this.** The sound, the
  red warning, and saved clips all come from the real camera's full-speed
  detection loop and play from this Mac's own speaker — none of that
  depends on how fast the iPad's picture refreshes. A hazard would still
  alert at full speed even though the video looks stepped.
- **The risk banner and hazard counts on the iPad also update faster** than
  the video (about 4 times a second) — only the actual camera picture is
  capped at 2/sec.
- What you lose is smoothness of the *picture itself* on that screen — fine
  for "is someone near the stove right now," not great for watching fluid
  motion.

**The technical reason, briefly:** we found that Flet's own code (the
library this whole app is built on) holds every picture it needs to send in
one line, and refuses to skip old ones — it insists on delivering every
single one, in order, even after they're stale. Over Wi-Fi to a real
device, sending pictures faster than that line can be sent out just makes
the line longer and longer, and everything in it still gets delivered late
before anything current gets a turn. That's what looked like a "5-7 second
delay." Sending pictures less often keeps that line short enough that
nothing ever piles up.

**If you want to try for a smoother picture:** try `--web-max-fps 3` or
`--web-max-fps 4` instead of `2`. Higher usually means smoother right up
until it suddenly doesn't — the backlog either doesn't build or builds
without bound, there isn't much of an in-between. If it starts lagging
again, drop back to `2`.

---

## Alert sound on whichever screen is open

Whichever screen has the app open — Mac browser, desktop window, or iPad —
can now play the same alert sound your Mac's speaker plays. Nothing extra
to install; it's part of the app.

**Two separate things, worth keeping straight:**
- **Your Mac's own speaker** — always worked, always will, is not affected
  by anything below. It's driven directly by the camera's detection loop,
  independent of any screen being open at all.
- **A screen making its own sound** — the new part below. If this fails
  for some reason, the Mac's speaker still alerts normally regardless.

**One known, deliberate gap:** a screen's sound isn't armed for about the
first 1.5 seconds right after you unlock it — a real limitation in the
Flet library this app is built on, not a choice. If a hazard fires in that
exact instant, that one screen stays silent for it specifically; the Mac's
speaker is completely unaffected either way. Anything after those 1.5
seconds plays normally.

**It won't repeat itself.** One sound per genuinely new alert — it won't
replay while the same alert is still showing, and if two different alerts
happen in quick succession, there's a 3.5-second gap enforced between
sounds so it can't feel like spam.

**Only the most urgent sound ever plays.** If two alerts become true close
together — say "baby getting close" and "hazard detected," or either of
those with "immediate danger" — you'll only ever hear the more urgent one.
Priority, highest first: Immediate Danger, then Baby Getting Close, then
Hazard Detected. The one that loses isn't queued for later, but it's not
erased either — it's still in the event history, just silent.

**Two screens open at once will each make their own sound.** If you have,
say, the desktop window AND a Chrome tab open at the same time, you'll hear
the alert twice — once from each. That's expected, not a bug: each open
screen decides independently whether to play, the same way each has its own
picture of the video.

**What to check on each screen you use:**
- Does the sound play at all?
- Does it match what the Mac's speaker is saying?
- Does it stay quiet while one alert is ongoing, and not fire off rapidly
  for a fast run of different alerts?

**If testing on the iPad:** check the physical silent switch on the side
is OFF before you test — otherwise it may look like a bug when it's really
just the iPad being muted.

**If testing the Mac browser in Chrome specifically:** it may stay silent
there — Flet's own documentation says this doesn't work in Chrome or Edge,
only desktop, mobile apps, and Safari. That's expected, not a bug to chase;
the Mac's speaker covers the Mac either way.

---

## If something goes wrong

**"command not found" or nothing happens**
Make sure you're in the right folder first:
```bash
cd /Users/shakedivgi/Projects/GuardianEye
```

**Camera doesn't show up / times out waiting**
- Is the camera actually plugged in?
- Is anything else using it (another terminal, another app)?
- Run `./start.sh` again — leftover processes from a crashed previous run are
  cleaned up automatically at the start.

**"address already in use"**
Something is still running from before. `./start.sh` normally cleans this up
by itself; if it still happens, close your terminal entirely and reopen it,
then try again.

**Changed WiFi networks / it "used to work" and now doesn't**
Shouldn't matter — everything above uses `127.0.0.1`, which always means
"this computer," never a network address. If you're seeing something
different, tell whoever's helping you exactly what you typed and what you saw.

**Want a different PIN?**
```bash
GUARDIANEYE_PIN=1234 ./start.sh
```

---

## For whoever is developing this (more detail, skip if you're just using it)

`start.sh` does three things: starts `cv/risk_engine.py --serve` (camera +
detection + backend API), waits for it to report healthy, then starts
`ui/app.py --mode web` (the Flet interface). Ctrl-C tears both down.

Running the pieces by hand, if you need to:
```bash
# Terminal 1
cd cv && ../.venv/bin/python3 risk_engine.py --name Arducam --serve

# Terminal 2 (after Terminal 1 says "Serving on http://127.0.0.1:8000")
cd ui && ../.venv/bin/python3 app.py --mode web --pin 4821
```

`--web-host` defaults to `127.0.0.1` (Mac-only) on purpose — see
`docs/decision-log.md`, 2026-08-30, **"iPad/mobile access reverted"** for the
full story of why, and what was tried. Do not set it to `0.0.0.0` without
reading that entry first: it reintroduces a broken auto-opened link
(`http://0.0.0.0:PORT`, which doesn't load) and video that won't display
in a browser anyway.

Testing without the real camera: `docs/design/phase-8/fake_camera_server.py`
runs the real `backend/server.py` against a synthetic feed. See that file's
own docstring.

Tests: `cd ui && ../.venv/bin/python3 test_ui.py` (and the equivalents in
`cv/` and `backend/`). Plain-assert scripts — no pytest, don't add it.

---

*Last updated: 2026-09-12. This file gets updated as things change — if a
step here is wrong, that's a bug in this file, say so.*
