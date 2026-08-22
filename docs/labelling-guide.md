# Labelling guide

For labelling captures from `cv/detect_stream.py` with `cv/label_captures.py`
ready for fine-tuning. Read `docs/phase-3-collection-plan.md` first if you
haven't shot the images yet — this is only the "how to draw the boxes" part.

## Start it

```bash
source .venv/bin/activate
python cv/label_captures.py
```

That labels everything in `cv/captures/` by default (skipping anything
`triage_captures.py` moved into a `dupes/`/`duplicates/` folder — don't
label those, they were triaged out on purpose). A window opens showing the
first image, downscaled to fit your screen — the tool always saves in full
3840x2160 coordinates regardless of what you see on screen, so don't worry
about the window being smaller than the real photo.

Re-running the same command later **resumes** — it picks up wherever you
left off, and never throws away labels you already made.

## The two classes

Only these two. Nothing else gets a box.

| key | class | what goes in it |
|---|---|---|
| `1` | `sharp_object` | knives, scissors |
| `2` | `small_swallowable` | keys, coins, batteries, bottle caps, lighters, small toy parts |

Press `1` or `2` to set which class the *next* box you draw will get — the
active class name is shown top-left in its colour (red for `sharp_object`,
orange for `small_swallowable`), and every box drawn afterward uses that
colour, so you can always tell what you already labelled.

## Keys

| key | does |
|---|---|
| left-drag | draw a new box with the currently active class |
| `1` | switch active class to `sharp_object` |
| `2` | switch active class to `small_swallowable` |
| `u` | undo — removes the most recently drawn box on this image |
| `d` | delete the box currently under your mouse cursor (if boxes overlap, the smallest one under the cursor gets deleted — it's highlighted with a thicker border so you can see which one before you press it) |
| `n` | save this image, go to the next one |
| `p` | save this image, go back to the previous one |
| `q` | save this image, quit |

Saving happens automatically every time you press `n`, `p`, or `q` — you
never have to remember to save, and closing the window with the mouse also
saves before it exits.

## Rules that affect model quality

These matter more than they look — a fine-tune is only as good as the boxes
it learns from.

- **Box the whole object, including the handle.** A knife's box should cover
  blade *and* handle, not just the blade — the model needs to learn what the
  whole object looks like, not a fragment of it.
- **Partially-hidden objects: box only the visible part.** If a lighter is
  half tucked under a cushion, draw the box around what you can actually
  see, not where you think the rest of it is.
- **Skip anything too small or blurry to identify confidently.** If you
  genuinely can't tell whether that speck on the counter is a coin or a
  crumb, don't box it. A wrong label is worse than a missing one.
- **A frame with no hazard gets zero boxes — that's a real label, not a
  skipped image.** Just press `n` and move on. These "negative" frames are
  exactly what `docs/phase-3-collection-plan.md` asks for (~20% of the
  set) — they teach the model that not every frame contains a hazard, which
  is what keeps it from inventing false alarms.

## What comes out of this

- One `.txt` file per image, next to it in `cv/captures/` by default (same
  name, `.txt` instead of `.jpg`), in YOLO format — one line per box:
  `class_id cx cy w h`, all normalised 0–1. An unlabelled image has no
  `.txt` file yet; a labelled negative (no hazards) has an **empty**
  `.txt` file. The tool tells the two apart and shows which is which on
  screen ("LABELLED" vs "UNLABELLED").
- `cv/captures/data.yaml`, rewritten automatically every time you save,
  ready to hand to Ultralytics for fine-tuning later — with a couple of
  caveats written into the file itself (train/val aren't split yet, and the
  label-file layout needs a small restructure before training; both are
  training-time problems to solve when that phase starts, not labelling
  problems).

## What this tool does NOT do

It only labels. It doesn't shoot images (`cv/detect_stream.py`), doesn't
find near-duplicate/blurry frames (`cv/triage_captures.py` — run that
*before* labelling, not after), and doesn't train anything.
