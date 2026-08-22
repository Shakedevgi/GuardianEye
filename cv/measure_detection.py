"""
measure_detection.py - Phase 3 Step 0: measure before collecting.

Phase 2's captures were all saved with the *live preview's* confidence
threshold burned in (conf=0.25 in every filename so far). That threshold
cannot tell the difference between two very different situations: "the model
emitted a weak box we discarded" vs. "the model emitted nothing at all." The
remedy for the first is threshold tuning; the remedy for the second is
fine-tuning a custom class. Phase 3 cannot be scoped correctly without
knowing which one is happening for each hazard class, so this script re-runs
stock YOLO26 over the already-saved raw frames at a near-zero confidence
floor (default 0.01) and records every box it finds, structured, instead of
eyeballing another screenshot.

This is an OFFLINE BATCH harness, not a live loop:
  - No camera access, no OpenCV window, no --index/--name (see camera.py).
  - Operates only on cv/captures/*_raw.jpg. The *_annotated.jpg siblings
    already have boxes burned into the pixels by detect_stream.py's
    results[0].plot() - running detection on those would mean re-detecting
    the model's own drawings, which is meaningless. Never point this script
    at an annotated image.
  - Sweeps model x imgsz x conf as a cross-product instead of single-shotting
    one config, because the whole point is comparing settings against each
    other on the SAME frames (Phase 2's captures were each taken at only one
    config, which is exactly why this measurement is needed).

Do NOT trust the filename for what config produced a detection here. The
filename encodes the config of the PREVIEW that captured the frame (e.g.
"..._yolo26x_imgsz1600_raw.jpg") - that says nothing about the model/imgsz/
conf THIS script chooses to re-run over it. Every CSV row records this
harness's own actual settings; the filename-derived config, if you want it
at all, is kept in a separately-labelled `capture_tag` column as provenance
only, never conflated with the measurement.

Scope (Phase 3 Step 0 only - see PHASE_PLAN.md and CLAUDE.md):
  - No data collection, no labelling, no training/fine-tuning here.
  - Does not modify detect_stream.py, camera.py, or stream_camera.py.
  - Nothing leaves this machine - reads local files, writes local files.

Usage:
    python measure_detection.py
    python measure_detection.py --models yolo26x.pt --imgsz 640,960,1280,1600 --conf 0.01
    python measure_detection.py --images "cv/captures/20260809-*_raw.jpg"
    python measure_detection.py --models yolo26l.pt,yolo26x.pt --imgsz 640,1600
"""

import argparse
import csv
import os
import re
import statistics
from glob import glob
from pathlib import Path

# Same reasoning as detect_stream.py: must be set before torch is imported
# anywhere (including transitively via ultralytics), or an unsupported MPS op
# hard-crashes instead of falling back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import torch
from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent / "models"
CAPTURES_DIR = Path(__file__).resolve().parent / "captures"
MEASUREMENTS_DIR = Path(__file__).resolve().parent / "measurements"

DEFAULT_IMAGES_GLOB = str(CAPTURES_DIR / "*_raw.jpg")
DEFAULT_MODELS = ["yolo26x.pt"]
DEFAULT_IMGSZ = [640, 960, 1280, 1600]

# This is the entire point of the script, not an arbitrary choice: 0.25 (the
# live-preview default) already discards exactly the weak-but-real boxes
# this measurement exists to find. 0.01 is close to "record everything the
# model produced at all" so a true absence and a suppressed weak box are
# distinguishable in the output.
DEFAULT_CONF = [0.01]

CSV_FIELDNAMES = [
    "source_image",
    "capture_tag",
    "model",
    "imgsz",
    "conf_floor",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "bbox_area_px",
    "bbox_area_frac",
]

# Matches the tag detect_stream.py's save_snapshot() writes, e.g.
# "20260809-083249_yolo26x_imgsz1600" out of
# "20260809-083249_yolo26x_imgsz1600_raw.jpg". Provenance only - see module
# docstring for why this must never be treated as this script's own config.
CAPTURE_TAG_RE = re.compile(r"^(.*)_raw$")


def parse_csv_list(raw: str):
    """Split a comma-separated CLI value into a stripped, non-empty list."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_device(requested: str) -> str:
    """Same fallback as detect_stream.py: prefer the requested device, but
    never crash a batch run over an unavailable MPS backend.
    """
    if requested == "mps" and not torch.backends.mps.is_available():
        print(
            "Warning: --device mps requested but MPS is not available on "
            "this machine. Falling back to CPU."
        )
        return "cpu"
    return requested


def load_model(model_name: str, device: str) -> YOLO:
    """Load from the predictable on-disk weights path, same convention as
    detect_stream.py, and load it exactly once per model name - the caller
    is responsible for reusing the returned model across every image and
    imgsz value rather than reloading per image.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = MODELS_DIR / model_name

    weights_already_present = weights_path.exists()
    if not weights_already_present:
        print(f"Weights not found at {weights_path} - downloading {model_name}...")

    model = YOLO(str(weights_path) if weights_already_present else model_name)

    if not weights_already_present:
        downloaded = Path(model_name)
        if downloaded.exists():
            downloaded.rename(weights_path)

    model.to(device)
    return model


def find_images(pattern: str) -> list:
    """Resolve the input glob, then refuse anything that isn't a raw capture.

    Only *_raw.jpg is a real photograph; *_annotated.jpg has boxes burned
    into the pixels by a previous run and would corrupt this measurement if
    re-detected. Rejecting rather than silently skipping so a bad --images
    value is loud, not a quietly smaller dataset.
    """
    matches = sorted(glob(pattern))
    if not matches:
        return []

    rejected = [path for path in matches if not path.endswith("_raw.jpg")]
    if rejected:
        raise ValueError(
            "Input glob matched non-raw image(s), which would corrupt the "
            "measurement (annotated frames already have boxes drawn on "
            f"them): {rejected}. Restrict --images to *_raw.jpg files."
        )

    return matches


def capture_tag_from_filename(image_path: str) -> str:
    """Extract the provenance tag from a *_raw.jpg filename, e.g.
    '20260809-083249_yolo26x_imgsz1600'. This is what config CAPTURED the
    frame, not what config this harness measured it with - see module
    docstring.
    """
    stem = Path(image_path).stem  # strips ".jpg"
    match = CAPTURE_TAG_RE.match(stem)
    return match.group(1) if match else stem


def measure(images, models, imgsz_values, conf_values, device):
    """Run the full model x imgsz x conf cross-product over every image and
    yield one dict per detection. Each model is loaded once and reused
    across every image/imgsz/conf combination for that model - imgsz and
    conf are inference-time arguments to predict(), not load-time ones, so
    there is no need to reload for them either.
    """
    rows = []
    for model_name in models:
        print(f"Loading {model_name} ...")
        model = load_model(model_name, device)
        print(f"{model_name} loaded.")

        for imgsz in imgsz_values:
            for conf in conf_values:
                for image_path in images:
                    frame = cv2.imread(image_path)
                    if frame is None:
                        print(f"  Warning: could not read {image_path}, skipping.")
                        continue

                    frame_height, frame_width = frame.shape[:2]
                    frame_area = float(frame_width * frame_height)

                    results = model.predict(
                        frame,
                        conf=conf,
                        imgsz=imgsz,
                        device=device,
                        verbose=False,
                    )
                    boxes = results[0].boxes
                    names = results[0].names
                    capture_tag = capture_tag_from_filename(image_path)

                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                        bbox_area_px = (x2 - x1) * (y2 - y1)

                        rows.append(
                            {
                                "source_image": Path(image_path).name,
                                "capture_tag": capture_tag,
                                "model": model_name,
                                "imgsz": imgsz,
                                "conf_floor": conf,
                                "class_name": names[class_id],
                                "confidence": round(confidence, 4),
                                "x1": round(x1, 1),
                                "y1": round(y1, 1),
                                "x2": round(x2, 1),
                                "y2": round(y2, 1),
                                "bbox_area_px": round(bbox_area_px, 1),
                                "bbox_area_frac": round(bbox_area_px / frame_area, 6),
                            }
                        )
        # Free the model before loading the next one in the sweep - these
        # are large weights and a multi-model sweep otherwise accumulates
        # all of them in memory/VRAM simultaneously for no benefit.
        del model

    return rows


def write_csv(rows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} detection rows to {output_path}")


def print_summary(rows) -> None:
    """Per class: how many distinct images it appeared in at all (across
    every model/imgsz/conf combo run), and the spread of confidence it was
    seen at. This is a sanity-check readout, not a substitute for querying
    the CSV directly for a specific model/imgsz/conf slice - the sweep mixes
    multiple configs together here on purpose to show the overall picture.
    """
    by_class = {}
    for row in rows:
        by_class.setdefault(row["class_name"], []).append(row)

    if not by_class:
        print("\nNo detections at all across the sweep.")
        return

    print("\nSummary (across all model/imgsz/conf combos run):")
    print(f"{'class':<15} {'images':>7} {'detections':>11} {'min':>7} {'median':>7} {'max':>7}")
    for class_name in sorted(by_class):
        class_rows = by_class[class_name]
        images_seen = {row["source_image"] for row in class_rows}
        confidences = [row["confidence"] for row in class_rows]
        print(
            f"{class_name:<15} {len(images_seen):>7} {len(class_rows):>11} "
            f"{min(confidences):>7.3f} {statistics.median(confidences):>7.3f} "
            f"{max(confidences):>7.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline batch measurement: sweep stock YOLO26 model x "
        "imgsz x conf over already-saved raw captures and emit structured "
        "per-detection numbers, instead of eyeballing another screenshot. "
        "Phase 3 Step 0 only - see PHASE_PLAN.md."
    )
    parser.add_argument(
        "--images",
        type=str,
        default=DEFAULT_IMAGES_GLOB,
        help=f"Glob of raw capture images to measure (default: {DEFAULT_IMAGES_GLOB}). "
        "Must match only *_raw.jpg - annotated frames already have boxes burned "
        "into the pixels and would corrupt the measurement.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated YOLO26 weights filenames to sweep (default: "
        f"{','.join(DEFAULT_MODELS)}). Each is loaded once from cv/models/ and "
        "reused across every image/imgsz/conf combination.",
    )
    parser.add_argument(
        "--imgsz",
        type=str,
        default=",".join(str(v) for v in DEFAULT_IMGSZ),
        help=f"Comma-separated inference resolutions to sweep (default: "
        f"{','.join(str(v) for v in DEFAULT_IMGSZ)}).",
    )
    parser.add_argument(
        "--conf",
        type=str,
        default=",".join(str(v) for v in DEFAULT_CONF),
        help=f"Comma-separated confidence floors to sweep (default: "
        f"{','.join(str(v) for v in DEFAULT_CONF)}). Deliberately near-zero by "
        "default - this harness exists specifically to distinguish 'weak box "
        "we would have discarded at conf 0.25' from 'the model emitted nothing "
        "at all,' which requires seeing everything the model produces.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Inference device: 'mps' (default, Apple GPU) or 'cpu'. Same "
        "escape hatch as detect_stream.py if MPS results look wrong.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(MEASUREMENTS_DIR / "detections.csv"),
        help=f"Per-detection CSV output path (default: {MEASUREMENTS_DIR / 'detections.csv'}).",
    )
    args = parser.parse_args()

    models = parse_csv_list(args.models)
    imgsz_values = [int(v) for v in parse_csv_list(args.imgsz)]
    conf_values = [float(v) for v in parse_csv_list(args.conf)]

    try:
        images = find_images(args.images)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    if not images:
        print(f"No images matched {args.images!r} - nothing to measure.")
        return

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(
        f"Sweeping {len(models)} model(s) x {len(imgsz_values)} imgsz x "
        f"{len(conf_values)} conf over {len(images)} image(s) "
        f"({len(models) * len(imgsz_values) * len(conf_values) * len(images)} "
        "predict() calls total)."
    )

    rows = measure(images, models, imgsz_values, conf_values, device)
    write_csv(rows, Path(args.output))
    print_summary(rows)


if __name__ == "__main__":
    main()
