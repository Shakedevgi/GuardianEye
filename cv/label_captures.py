"""
label_captures.py - Phase 3 bounding-box labelling tool.

Label Studio's dependency tree largely builds from source on this machine's
Python 3.14.4 and a previous attempt to install it hung - not worth fighting
for a two-class bounding-box task (see docs/phase-3-collection-plan.md and
the 2026-08 decision log). This is a small OpenCV-only labeller instead:
`opencv-python` and `numpy` are already installed and pinned for Phase
1/2/3's detection scripts, so this adds **zero new dependencies** and cannot
disturb the pinned torch/ultralytics environment.

Classes - exactly two, fixed order (the order IS the YOLO class id and must
never change - changing it after any labelling has happened silently
corrupts every existing label file):

    0 = sharp_object      knives, scissors
    1 = small_swallowable keys, coins, batteries, bottle caps, lighters,
                           small toy parts

See docs/phase-3-collection-plan.md for why these two classes and not
others (sockets/stairs/person are deliberately NOT labelled here).

Coordinate handling - the part most likely to silently corrupt a dataset:
captures are 3840x2160 (--4k in detect_stream.py) and do not fit any normal
screen, so the image is downscaled for the on-screen window. EVERY box is
stored, at the moment its drag completes, in full-resolution pixel space -
the display scale factor is applied once on the way in (display -> full-res)
and never touched again. See `display_to_full()` / `full_to_display()` and
DownscaleTransform below; nothing else in this file is allowed to store or
write a display-space coordinate.

Output format - Ultralytics YOLO text, one .txt per image, written directly
(no separate export step):

    class_id cx cy w h        (all normalised 0-1, space separated, one box
                                per line)

An image with zero hazards gets an EMPTY .txt file, not a missing one - see
docs/phase-3-collection-plan.md's ~20% negatives requirement. A missing file
means "not yet labelled"; an empty file means "labelled, no hazards here".
The tool relies on this distinction to resume correctly and to show progress
- see `is_labelled()`.

Resume: every navigation (n/p) and quit (q) saves the current image's boxes
before moving, so closing the window (or the process dying) never loses
work beyond the box currently mid-drag. Re-running the tool over the same
--dir/--labels-dir loads whatever .txt files already exist and continues
from there rather than starting over.

Usage:
    python cv/label_captures.py                       # labels cv/captures/
    python cv/label_captures.py --dir cv/captures --labels-dir cv/captures
    python cv/label_captures.py --start-index 30       # resume partway through

Keys (window must be focused):
    left-drag    draw a box in the active class
    1            active class -> sharp_object   (red boxes)
    2            active class -> small_swallowable (orange boxes)
    u            undo the most recently added box on THIS image
    d            delete the box under the mouse cursor (smallest box under
                 the cursor wins, if boxes overlap)
    n / p        save this image's labels, go to next / previous image
    q            save this image's labels, quit

Scope: labelling only. Does not collect (detect_stream.py's job), triage
(triage_captures.py's job), or train anything.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

CAPTURES_DIR = Path(__file__).resolve().parent / "captures"

WINDOW_NAME = "GuardianEye - label captures (Phase 3)"

# Fixed order - THIS ORDER IS THE YOLO CLASS ID. Do not reorder, insert, or
# remove without re-checking every already-written .txt file - see the
# module docstring.
CLASS_NAMES = ["sharp_object", "small_swallowable"]

# BGR, chosen to be visually distinct and to read as "danger" - red for the
# class that's dangerous by being sharp, orange for the class that's
# dangerous by being swallowable. Purely a labelling-UI convenience; has no
# effect on the written label data.
CLASS_COLORS = [
    (0, 0, 255),    # sharp_object      - red
    (0, 140, 255),  # small_swallowable - orange
]

# Directory name triage_captures.py's --move-duplicates conventionally moves
# near-duplicate frames into (the task's own example uses this). Any path
# component matching this, at any depth under --dir, is excluded from
# labelling - those frames were triaged out on purpose and must not be
# labelled. Matched case-sensitively against directory name components, not
# just the literal default "duplicates" triage_captures.py's --help
# suggests, so a locally renamed "dupes/" folder is still honoured per the
# task spec.
DUPES_DIR_NAMES = {"dupes", "duplicates"}

# The image doesn't fit most screens at full 3840x2160 - this caps the
# window's DISPLAY size only. Every stored/written coordinate is always
# full-resolution; see the module docstring's coordinate-handling section.
DISPLAY_MAX_WIDTH = 1600
DISPLAY_MAX_HEIGHT = 900

# A drag shorter than this many DISPLAY pixels (in either axis) is treated
# as an accidental click, not a box - avoids littering zero-area or
# near-zero boxes into the label file from a fumbled mouse-down/up.
MIN_DRAG_DISPLAY_PX = 4

OVERLAY_COLOR = (255, 255, 255)
OVERLAY_OUTLINE = (0, 0, 0)
OVERLAY_MARGIN = 10
OVERLAY_LINE_HEIGHT = 24

# Box under the cursor (candidate for 'd') is redrawn with this much extra
# line thickness so the operator can see, before pressing 'd', exactly which
# box would be deleted - deleting the wrong one silently is the failure mode
# this exists to prevent.
HOVER_THICKNESS_BONUS = 3
BOX_THICKNESS = 2


class DownscaleTransform:
    """The one place display<->full-resolution pixel conversion happens.

    scale = min(width_ratio, height_ratio, 1.0) - never upscale a smaller
    test image, only ever shrink a too-large one (today that's always the
    3840x2160 captures, but this stays correct if a differently-sized image
    ever lands in the same directory).
    """

    def __init__(self, full_width: int, full_height: int):
        self.full_width = full_width
        self.full_height = full_height
        self.scale = min(
            DISPLAY_MAX_WIDTH / full_width,
            DISPLAY_MAX_HEIGHT / full_height,
            1.0,
        )
        self.display_width = max(1, round(full_width * self.scale))
        self.display_height = max(1, round(full_height * self.scale))

    def to_display_image(self, full_image):
        if self.scale == 1.0:
            return full_image
        return cv2.resize(
            full_image,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA,
        )

    def display_to_full(self, x: int, y: int) -> tuple[int, int]:
        """Display-space pixel -> full-resolution pixel, clamped to the
        image bounds. This is the ONLY direction mouse events are converted
        - nothing downstream of this ever sees a display-space coordinate.
        """
        full_x = round(x / self.scale)
        full_y = round(y / self.scale)
        full_x = max(0, min(full_x, self.full_width - 1))
        full_y = max(0, min(full_y, self.full_height - 1))
        return full_x, full_y

    def full_to_display(self, x: int, y: int) -> tuple[int, int]:
        """Full-resolution pixel -> display-space pixel, for drawing
        existing (already full-res) boxes onto the downscaled canvas.
        """
        return round(x * self.scale), round(y * self.scale)


def discover_images(capture_dir: Path) -> list:
    """*_raw.jpg files under capture_dir, recursively, excluding anything
    under a directory named per DUPES_DIR_NAMES at any depth - see that
    constant's comment. Sorted for a stable, repeatable ordering across
    runs (resume depends on this).
    """
    images = []
    for path in capture_dir.rglob("*_raw.jpg"):
        relative_parts = path.relative_to(capture_dir).parts[:-1]
        if any(part in DUPES_DIR_NAMES for part in relative_parts):
            continue
        images.append(path)
    return sorted(images)


def label_path_for(image_path: Path, labels_dir: Path) -> Path:
    """The .txt this image's labels live at. Matches the image's stem
    exactly (only the extension changes) - see the module docstring's
    "Output format" section for why the stem is NOT further modified
    (Ultralytics-style same-stem pairing).
    """
    return labels_dir / (image_path.stem + ".txt")


def is_labelled(image_path: Path, labels_dir: Path) -> bool:
    """True iff a label file exists at all - including an EMPTY one. An
    empty file means "labelled, zero hazards" (a valid negative example);
    a missing file means "not yet visited". See module docstring.
    """
    return label_path_for(image_path, labels_dir).exists()


def read_labels(image_path: Path, labels_dir: Path, img_w: int, img_h: int) -> list:
    """Load existing boxes for this image, denormalising back to
    full-resolution pixel space. Returns [] both when the file doesn't
    exist yet (unlabelled) and when it exists but is empty (labelled
    negative) - is_labelled() is what tells those two cases apart, this
    function does not need to.
    """
    path = label_path_for(image_path, labels_dir)
    if not path.exists():
        return []

    boxes = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            print(f"Warning: {path.name} line {line_number} malformed, skipped: {line!r}")
            continue
        class_id, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        cx_px, cy_px = cx * img_w, cy * img_h
        w_px, h_px = w * img_w, h * img_h
        x1 = round(cx_px - w_px / 2)
        y1 = round(cy_px - h_px / 2)
        x2 = round(cx_px + w_px / 2)
        y2 = round(cy_px + h_px / 2)
        boxes.append(
            {
                "class_id": class_id,
                "x1": max(0, min(x1, img_w - 1)),
                "y1": max(0, min(y1, img_h - 1)),
                "x2": max(0, min(x2, img_w - 1)),
                "y2": max(0, min(y2, img_h - 1)),
            }
        )
    return boxes


def write_labels(image_path: Path, labels_dir: Path, boxes: list, img_w: int, img_h: int) -> None:
    """Write this image's boxes as normalised YOLO lines. Always writes the
    file - even with zero boxes, producing an empty file on purpose (see
    module docstring). This is the single source of truth for the
    full-res-pixels -> normalised-0..1 transform; read_labels() above is
    its exact inverse.
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    path = label_path_for(image_path, labels_dir)

    lines = []
    for box in boxes:
        w_px = box["x2"] - box["x1"]
        h_px = box["y2"] - box["y1"]
        cx_px = box["x1"] + w_px / 2
        cy_px = box["y1"] + h_px / 2
        cx, cy = cx_px / img_w, cy_px / img_h
        w, h = w_px / img_w, h_px / img_h
        lines.append(f"{box['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_data_yaml(labels_dir: Path, images_dir: Path) -> None:
    """Emit a data.yaml ready for Ultralytics - names in the fixed
    CLASS_NAMES order (see module docstring for why the order must never
    change).

    CAVEAT, flagged deliberately rather than discovered later at train time:
    Ultralytics finds each image's label file by substituting the literal
    path substring "/images/" with "/labels/" (see
    ultralytics.data.utils.img2label_paths in the installed 8.4.115). This
    tool keeps label .txt files beside the raw images (or wherever
    --labels-dir points) for labelling simplicity, which does NOT satisfy
    that convention unless the directory path happens to contain an
    "images" segment. Before actually training: either point --dir/
    --labels-dir at an images/ + labels/ pair of folders, or symlink/copy
    this flat layout into that shape. Nothing in this labelling tool trains
    anything, so this is left as a documented handoff note, not solved here.

    train/val both point at the same directory below because this tool has
    no train/val split logic - collection is still in progress (round 1
    target: 150 images/class, see docs/phase-3-collection-plan.md) and a
    real held-out split is a training-time decision, not a labelling-time
    one. Produce a genuine split before trusting any measured accuracy.
    """
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    content = f"""\
# Auto-generated by cv/label_captures.py - rewritten on every save, safe to
# regenerate, do not hand-edit (edits will be overwritten).
#
# CAVEAT (read before training) - Ultralytics discovers each image's label
# file by substituting the literal substring "/images/" in its path with
# "/labels/" (ultralytics.data.utils.img2label_paths). This dataset keeps
# labels beside the raw images for labelling simplicity, which will NOT
# auto-resolve unless the path below contains an "images" segment. Restructure
# (or symlink) into an images/ + labels/ pair before training - see
# write_data_yaml()'s docstring in cv/label_captures.py.
#
# CAVEAT #2 - train and val point at the SAME directory: there is no
# held-out split yet. Produce a real split before trusting any measured
# accuracy from this file as-is (see docs/phase-3-collection-plan.md).

path: {images_dir.resolve()}
train: .
val: .

names:
{names_block}
"""
    (labels_dir / "data.yaml").write_text(content)


class LabelState:
    """Mutable state the mouse callback reads/writes, plus the currently
    loaded image's boxes. One instance, reused across images (fields reset
    on navigation) rather than recreated, so the mouse callback registered
    once with cv2.setMouseCallback keeps a stable reference.
    """

    def __init__(self):
        self.active_class = 0
        self.boxes = []          # full-res pixel boxes for the current image
        self.transform = None    # DownscaleTransform for the current image
        self.dragging = False
        self.drag_start_display = None
        self.drag_current_display = None
        self.mouse_display = (0, 0)


def on_mouse(event, x, y, flags, state: LabelState) -> None:
    state.mouse_display = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        state.dragging = True
        state.drag_start_display = (x, y)
        state.drag_current_display = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if state.dragging:
            state.drag_current_display = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if state.dragging:
            state.dragging = False
            start_x, start_y = state.drag_start_display
            if abs(x - start_x) >= MIN_DRAG_DISPLAY_PX or abs(y - start_y) >= MIN_DRAG_DISPLAY_PX:
                fx1, fy1 = state.transform.display_to_full(start_x, start_y)
                fx2, fy2 = state.transform.display_to_full(x, y)
                box = {
                    "class_id": state.active_class,
                    "x1": min(fx1, fx2),
                    "y1": min(fy1, fy2),
                    "x2": max(fx1, fx2),
                    "y2": max(fy1, fy2),
                }
                state.boxes.append(box)
            state.drag_start_display = None
            state.drag_current_display = None


def box_area(box) -> int:
    return (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])


def find_box_under_cursor(state: LabelState):
    """The smallest box (by full-res pixel area) whose region contains the
    current mouse position - "smallest" so that a small box nested inside a
    bigger one is the one 'd' deletes, matching what a human looking at the
    overlap would expect. Returns None if the cursor isn't over any box.
    """
    if state.transform is None:
        return None
    fx, fy = state.transform.display_to_full(*state.mouse_display)
    candidates = [
        box for box in state.boxes
        if box["x1"] <= fx <= box["x2"] and box["y1"] <= fy <= box["y2"]
    ]
    if not candidates:
        return None
    return min(candidates, key=box_area)


def draw_overlay_line(image, text: str, line_number: int, color=OVERLAY_COLOR) -> None:
    origin = (OVERLAY_MARGIN, OVERLAY_MARGIN + line_number * OVERLAY_LINE_HEIGHT)
    for draw_color, thickness in ((OVERLAY_OUTLINE, 3), (color, 1)):
        cv2.putText(
            image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_color, thickness
        )


def build_canvas(full_image, state: LabelState, image_index: int, total_images: int,
                  image_path: Path, labels_dir: Path):
    """Compose the current display frame: downscaled image + all boxes
    (in display space, converted from the stored full-res boxes) + the
    in-progress drag rectangle + the status overlay. Rebuilt every loop
    iteration so dragging/hover feedback is live - nothing here writes to
    state.boxes, this function only reads it.
    """
    canvas = state.transform.to_display_image(full_image).copy()
    hover_box = find_box_under_cursor(state)

    for box in state.boxes:
        (dx1, dy1) = state.transform.full_to_display(box["x1"], box["y1"])
        (dx2, dy2) = state.transform.full_to_display(box["x2"], box["y2"])
        color = CLASS_COLORS[box["class_id"]]
        thickness = BOX_THICKNESS + (HOVER_THICKNESS_BONUS if box is hover_box else 0)
        cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), color, thickness)
        label = CLASS_NAMES[box["class_id"]]
        text_origin = (dx1, max(12, dy1 - 6))
        cv2.putText(canvas, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(canvas, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if state.dragging and state.drag_start_display and state.drag_current_display:
        cv2.rectangle(
            canvas, state.drag_start_display, state.drag_current_display,
            CLASS_COLORS[state.active_class], 1,
        )

    labelled = is_labelled(image_path, labels_dir)
    status = "LABELLED" if labelled else "UNLABELLED - not yet saved"
    draw_overlay_line(
        canvas,
        f"Image {image_index + 1}/{total_images}  boxes: {len(state.boxes)}  [{status}]",
        0,
    )
    draw_overlay_line(canvas, f"{image_path.name}", 1)
    active_name = CLASS_NAMES[state.active_class]
    draw_overlay_line(
        canvas,
        f"Active class [{state.active_class + 1}]: {active_name}  "
        f"(1=sharp_object 2=small_swallowable)",
        2,
        color=CLASS_COLORS[state.active_class],
    )
    draw_overlay_line(
        canvas,
        "drag=new box  u=undo  d=delete under cursor  n/p=next/prev  q=save+quit",
        3,
    )
    return canvas


def load_image_into_state(image_path: Path, labels_dir: Path, state: LabelState):
    full_image = cv2.imread(str(image_path))
    if full_image is None:
        raise ValueError(f"Could not read {image_path} as an image.")
    height, width = full_image.shape[:2]
    state.transform = DownscaleTransform(width, height)
    state.boxes = read_labels(image_path, labels_dir, width, height)
    state.dragging = False
    state.drag_start_display = None
    state.drag_current_display = None
    return full_image


def save_current(image_path: Path, labels_dir: Path, state: LabelState, full_image, images_dir: Path) -> None:
    height, width = full_image.shape[:2]
    write_labels(image_path, labels_dir, state.boxes, width, height)
    write_data_yaml(labels_dir, images_dir)


def print_startup_summary(image_paths: list, labels_dir: Path) -> None:
    labelled = [p for p in image_paths if is_labelled(p, labels_dir)]
    empty = [p for p in labelled if not read_labels(p, labels_dir, *_probe_size(p))]
    print(
        f"{len(image_paths)} image(s) to label in total: "
        f"{len(labelled)} already labelled ({len(empty)} of those with zero "
        f"boxes/negatives), {len(image_paths) - len(labelled)} unlabelled."
    )


def _probe_size(image_path: Path) -> tuple:
    """cv2.imread just for width/height, used only by the startup summary's
    empty-box count (read_labels needs the image dimensions to denormalise
    coordinates). Not on any hot path - called once per already-labelled
    image at startup only.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return (1, 1)
    h, w = image.shape[:2]
    return (w, h)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local OpenCV bounding-box labeller for GuardianEye's "
        "two Phase 3 hazard classes (sharp_object, small_swallowable). See "
        "docs/labelling-guide.md for the plain-language walkthrough."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=str(CAPTURES_DIR),
        help=f"Directory of *_raw.jpg captures to label (default: "
        f"{CAPTURES_DIR}). Searched recursively; any 'dupes'/'duplicates' "
        "subdirectory (triage_captures.py's --move-duplicates output) is "
        "excluded automatically.",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="Directory to read/write .txt label files (default: same as "
        "--dir, i.e. labels sit beside their images). See "
        "write_data_yaml()'s docstring for a caveat this implies for "
        "Ultralytics training later.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Image index (0-based, in the sorted file-listing order) to "
        "start from - handy for jumping back into the middle of a large "
        "set without clicking through everything already done.",
    )
    args = parser.parse_args()

    capture_dir = Path(args.dir)
    labels_dir = Path(args.labels_dir) if args.labels_dir else capture_dir

    image_paths = discover_images(capture_dir)
    if not image_paths:
        print(f"No *_raw.jpg files found under {capture_dir} (dupes/duplicates excluded) - nothing to label.")
        return

    if not (0 <= args.start_index < len(image_paths)):
        print(
            f"Error: --start-index {args.start_index} out of range for "
            f"{len(image_paths)} image(s) (valid: 0-{len(image_paths) - 1})."
        )
        return

    print_startup_summary(image_paths, labels_dir)

    cv2.namedWindow(WINDOW_NAME)
    state = LabelState()
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, param=state)

    index = args.start_index
    full_image = load_image_into_state(image_paths[index], labels_dir, state)

    try:
        while True:
            image_path = image_paths[index]
            canvas = build_canvas(full_image, state, index, len(image_paths), image_path, labels_dir)
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                save_current(image_path, labels_dir, state, full_image, capture_dir)
                print(f"Saved {label_path_for(image_path, labels_dir).name} ({len(state.boxes)} box(es)). Quit.")
                break

            elif key == ord("n"):
                save_current(image_path, labels_dir, state, full_image, capture_dir)
                if index + 1 >= len(image_paths):
                    print("Already at the last image - staying here (labels saved).")
                else:
                    index += 1
                    full_image = load_image_into_state(image_paths[index], labels_dir, state)

            elif key == ord("p"):
                save_current(image_path, labels_dir, state, full_image, capture_dir)
                if index == 0:
                    print("Already at the first image - staying here (labels saved).")
                else:
                    index -= 1
                    full_image = load_image_into_state(image_paths[index], labels_dir, state)

            elif key == ord("u"):
                if state.boxes:
                    removed = state.boxes.pop()
                    print(f"Undid last box ({CLASS_NAMES[removed['class_id']]}).")
                else:
                    print("Nothing to undo on this image.")

            elif key == ord("d"):
                target = find_box_under_cursor(state)
                if target is not None:
                    state.boxes.remove(target)
                    print(f"Deleted box under cursor ({CLASS_NAMES[target['class_id']]}).")
                else:
                    print("No box under the cursor to delete.")

            elif key == ord("1"):
                state.active_class = 0
            elif key == ord("2"):
                state.active_class = 1

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed - saving current image and exiting.")
                save_current(image_path, labels_dir, state, full_image, capture_dir)
                break

    except KeyboardInterrupt:
        print("Interrupted (Ctrl+C) - saving current image before exit.")
        save_current(image_paths[index], labels_dir, state, full_image, capture_dir)

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
