"""
measure_openvocab.py - Phase 3 Step 0b: does open-vocabulary detection close
the gaps stock COCO measurably cannot, using only frames we already have?

Same shape and conventions as measure_detection.py: offline batch harness
over cv/captures/*_raw.jpg, conf floor 0.01 by default, sweeps
model x imgsz x image, records every box, writes one CSV row per detection
plus a printed summary. Read that script's docstring first - the reasoning
about conf floor, offline-sweep-over-raw-frames, and "never point this at an
*_annotated.jpg" all applies unchanged here.

What's different from measure_detection.py:
  - Stock YOLO has a fixed COCO class list baked into the weights. Open-vocab
    models (YOLO-World, YOLOE) instead take a *text prompt list* at runtime
    via model.set_classes(...) and detect only those classes. This script
    takes --prompts and calls set_classes() once per model, before the
    imgsz/image loop - the class list is not swept per-image, only the
    inference resolution and the model are.
  - Records a model_type column (yolo_world / yoloe) because the two use
    different set_classes() call shapes (see load_open_vocab_model below),
    and a downstream reader comparing rows needs to know which model family
    produced a given confidence number - the two are not the same technique.
  - Records wall-clock inference time per predict() call (inference_ms) so
    Phase 3 can answer "is this viable in the live pipeline or offline-only,"
    which measure_detection.py never had to answer because stock YOLO26 is
    already the live-pipeline model.

What this script confirmed about ultralytics==8.4.115 (the pinned version -
see cv/requirements.txt): both open-vocabulary families it ships are
available and loadable - YOLOWorld (CLIP-based, yolov8*-world*.pt /
yolov8*-worldv2.pt checkpoints) and YOLOE (yoloe-*-seg.pt checkpoints,
including a yoloe-26-seg family matching this project's YOLO26 stock models).
Both were verified to load and run set_classes() + predict() on this machine
before this script was written. See the Phase 3 open-vocab evaluation
write-up (delivered separately, not checked in as a file by this script) for
which one this repo recommends and why.

Privacy (CLAUDE.md decision 8 - no cloud, ever, for this data):
  - This script downloads model *weights* only (from ultralytics' GitHub
    release assets, same mechanism measure_detection.py already uses for
    yolo26*.pt). It never uploads, POSTs, or otherwise transmits any capture
    image anywhere. All inference is local torch/MPS/CPU.
  - Do not add any hosted/API detector (e.g. a cloud open-vocab endpoint) to
    this script. If open-vocab is adopted going forward it must stay a local
    weights file, exactly like the stock YOLO26 models already in cv/models/.

Scope (Phase 3 Step 0b only - see PHASE_PLAN.md and CLAUDE.md):
  - No data collection, no labelling, no training/fine-tuning here.
  - Does not modify detect_stream.py, camera.py, stream_camera.py, or
    measure_detection.py.

Usage:
    python measure_openvocab.py
    python measure_openvocab.py --models yoloe-26s-seg.pt --imgsz 640,1280,1600
    python measure_openvocab.py --models yoloe-26s-seg.pt,yolov8s-worldv2.pt \\
        --prompts "power strip,electrical outlet,car key,knife,person"
    python measure_openvocab.py --images "cv/captures/20260809-104604_raw.jpg"
"""

import argparse
import csv
import os
import re
import statistics
import time
from glob import glob
from pathlib import Path

# Same reasoning as detect_stream.py / measure_detection.py: must be set
# before torch is imported anywhere (including transitively via
# ultralytics), or an unsupported MPS op hard-crashes instead of falling
# back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import torch
from ultralytics import YOLO, YOLOE

MODELS_DIR = Path(__file__).resolve().parent / "models"
CAPTURES_DIR = Path(__file__).resolve().parent / "captures"
MEASUREMENTS_DIR = Path(__file__).resolve().parent / "measurements"

DEFAULT_IMAGES_GLOB = str(CAPTURES_DIR / "*_raw.jpg")

# Small/fast variants of each family, both already verified to load on this
# machine. Deliberately not the largest checkpoints - Step 0b is an offline
# feasibility screen (see "Speed" in the report requirements), not a final
# model choice; if the small variants show promise, a follow-up can spend
# the extra download/inference time on the larger ones for the specific
# frames that matter.
DEFAULT_MODELS = ["yoloe-26s-seg.pt", "yolov8s-worldv2.pt"]

DEFAULT_IMGSZ = [640, 1280, 1600]

# Same reasoning as measure_detection.py: 0.25 (a typical live-preview
# default) already discards the weak-but-real boxes this measurement exists
# to find. 0.01 is close to "record everything the model produced at all."
DEFAULT_CONF = [0.01]

# The gap classes named in docs/phase-3-step0-findings.md, plus `person` as
# the control (must not regress - CLAUDE.md decision 2 requires person/hazard
# detection in a single pass, so losing person to gain hazard prompts would
# be a net loss, not a win) and `scissors`/`knife` as the already-measured
# stock-COCO baseline to compare open-vocab against directly.
DEFAULT_PROMPTS = [
    "power strip",
    "electrical outlet",
    "wall socket",
    "power cable",
    "plug adapter",
    "car key",
    "keys",
    "cigarette lighter",
    "battery",
    "coin",
    "bottle cap",
    "knife",
    "kitchen knife",
    "scissors",
    "person",
]

CSV_FIELDNAMES = [
    "source_image",
    "capture_tag",
    "model",
    "model_type",
    "prompts",
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
    "inference_ms",
]

# Matches the tag detect_stream.py's save_snapshot() writes, e.g.
# "20260809-083249_yolo26x_imgsz1600" out of
# "20260809-083249_yolo26x_imgsz1600_raw.jpg". Provenance only - this is what
# config CAPTURED the frame, not what this harness measured it with.
CAPTURE_TAG_RE = re.compile(r"^(.*)_raw$")

# How to tell the two open-vocab families apart by checkpoint filename, so
# the right set_classes() call shape gets used. See load_open_vocab_model().
WORLD_NAME_MARKER = "world"


def parse_csv_list(raw: str):
    """Split a comma-separated CLI value into a stripped, non-empty list."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_device(requested: str) -> str:
    """Same fallback as measure_detection.py: prefer the requested device,
    but never crash a batch run over an unavailable MPS backend.
    """
    if requested == "mps" and not torch.backends.mps.is_available():
        print(
            "Warning: --device mps requested but MPS is not available on "
            "this machine. Falling back to CPU."
        )
        return "cpu"
    return requested


def model_type_for(model_name: str) -> str:
    """Classify a checkpoint filename into which open-vocab family it is.
    This determines the set_classes() call shape below - the two families
    are NOT interchangeable at that call site.
    """
    return "yolo_world" if WORLD_NAME_MARKER in model_name.lower() else "yoloe"


def load_open_vocab_model(model_name: str, device: str, prompts: list):
    """Load an open-vocab checkpoint from the predictable on-disk weights
    path (same convention as measure_detection.py's load_model), then bind
    the text prompt list exactly once, before any image is processed.

    The two families set their class list differently, and this is the one
    place that difference has to be handled explicitly:
      - YOLOWorld has its own CLIP text encoder built into set_classes() -
        model.set_classes(prompts) is the entire call.
      - YOLOE requires computing the text prompt embeddings first via
        get_text_pe(), then passing both the names AND the embeddings to
        set_classes(). Skipping get_text_pe() and passing only names would
        silently fail to bind the prompts correctly.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = MODELS_DIR / model_name

    weights_already_present = weights_path.exists()
    if not weights_already_present:
        print(f"Weights not found at {weights_path} - downloading {model_name}...")

    m_type = model_type_for(model_name)
    model_cls = YOLOE if m_type == "yoloe" else YOLO
    model = model_cls(str(weights_path) if weights_already_present else model_name)

    if not weights_already_present:
        downloaded = Path(model_name)
        if downloaded.exists():
            downloaded.rename(weights_path)

    model.to(device)

    if m_type == "yoloe":
        text_pe = model.get_text_pe(prompts)
        model.set_classes(prompts, text_pe)
    else:
        model.set_classes(prompts)

    return model, m_type


def find_images(pattern: str) -> list:
    """Resolve the input glob, then refuse anything that isn't a raw
    capture - identical reasoning to measure_detection.py's find_images:
    *_annotated.jpg already has boxes burned into the pixels and would
    corrupt the measurement if re-detected.
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
    """Extract the provenance tag from a *_raw.jpg filename - what config
    CAPTURED the frame, not what config this harness measured it with.
    """
    stem = Path(image_path).stem
    match = CAPTURE_TAG_RE.match(stem)
    return match.group(1) if match else stem


def measure(images, models, prompts, imgsz_values, conf_values, device):
    """Run the model x imgsz x conf x image cross-product and yield one dict
    per detection. Each model is loaded and bound to the full prompt list
    exactly once, then reused across every image/imgsz/conf combination -
    identical structure to measure_detection.py's measure(), with the
    prompt-binding step added and per-call timing recorded.
    """
    rows = []
    prompts_joined = "|".join(prompts)

    for model_name in models:
        print(f"Loading {model_name} ...")
        model, m_type = load_open_vocab_model(model_name, device, prompts)
        print(f"{model_name} loaded as {m_type}, bound to {len(prompts)} prompts.")

        for imgsz in imgsz_values:
            for conf in conf_values:
                for image_path in images:
                    frame = cv2.imread(image_path)
                    if frame is None:
                        print(f"  Warning: could not read {image_path}, skipping.")
                        continue

                    frame_height, frame_width = frame.shape[:2]
                    frame_area = float(frame_width * frame_height)

                    start = time.perf_counter()
                    results = model.predict(
                        frame,
                        conf=conf,
                        imgsz=imgsz,
                        device=device,
                        verbose=False,
                    )
                    inference_ms = (time.perf_counter() - start) * 1000.0

                    boxes = results[0].boxes
                    names = results[0].names
                    capture_tag = capture_tag_from_filename(image_path)

                    if len(boxes) == 0:
                        # Record a zero-box row so "the model ran and found
                        # nothing" is distinguishable from "this image/config
                        # combo was never measured" when reading the CSV -
                        # the row exists but has no class_name/confidence.
                        rows.append(
                            {
                                "source_image": Path(image_path).name,
                                "capture_tag": capture_tag,
                                "model": model_name,
                                "model_type": m_type,
                                "prompts": prompts_joined,
                                "imgsz": imgsz,
                                "conf_floor": conf,
                                "class_name": "",
                                "confidence": "",
                                "x1": "",
                                "y1": "",
                                "x2": "",
                                "y2": "",
                                "bbox_area_px": "",
                                "bbox_area_frac": "",
                                "inference_ms": round(inference_ms, 1),
                            }
                        )
                        continue

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
                                "model_type": m_type,
                                "prompts": prompts_joined,
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
                                "inference_ms": round(inference_ms, 1),
                            }
                        )
        del model

    return rows


def write_csv(rows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {output_path}")


def print_summary(rows) -> None:
    """Per (model, class): how many distinct images it appeared in, the
    confidence spread, and mean inference time. Detection rows only - the
    zero-box placeholder rows measure() adds are excluded here since they
    have no class_name, but they remain in the CSV for completeness.
    """
    detection_rows = [row for row in rows if row["class_name"]]
    by_model_class = {}
    for row in detection_rows:
        key = (row["model"], row["class_name"])
        by_model_class.setdefault(key, []).append(row)

    if not by_model_class:
        print("\nNo detections at all across the sweep.")
    else:
        print("\nSummary (across all imgsz/conf combos run, per model):")
        header = f"{'model':<20} {'class':<20} {'images':>7} {'dets':>6} {'min':>7} {'median':>7} {'max':>7}"
        print(header)
        for model_name, class_name in sorted(by_model_class):
            class_rows = by_model_class[(model_name, class_name)]
            images_seen = {row["source_image"] for row in class_rows}
            confidences = [row["confidence"] for row in class_rows]
            print(
                f"{model_name:<20} {class_name:<20} {len(images_seen):>7} "
                f"{len(class_rows):>6} {min(confidences):>7.3f} "
                f"{statistics.median(confidences):>7.3f} {max(confidences):>7.3f}"
            )

    print("\nMean inference time by model x imgsz (ms/frame):")
    by_model_imgsz = {}
    for row in rows:
        key = (row["model"], row["imgsz"])
        by_model_imgsz.setdefault(key, []).append(row["inference_ms"])
    for model_name, imgsz in sorted(by_model_imgsz):
        times = by_model_imgsz[(model_name, imgsz)]
        print(f"  {model_name:<20} imgsz={imgsz:<5} mean={statistics.mean(times):>8.1f} ms  n={len(times)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline batch measurement: sweep open-vocabulary "
        "detectors (YOLO-World and/or YOLOE) x imgsz over already-saved "
        "raw captures with a fixed text-prompt list, and emit structured "
        "per-detection numbers for direct comparison against stock YOLO26's "
        "measured baseline. Phase 3 Step 0b only - see PHASE_PLAN.md."
    )
    parser.add_argument(
        "--images",
        type=str,
        default=DEFAULT_IMAGES_GLOB,
        help=f"Glob of raw capture images to measure (default: {DEFAULT_IMAGES_GLOB}). "
        "Must match only *_raw.jpg.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated open-vocab weights filenames to sweep (default: "
        f"{','.join(DEFAULT_MODELS)}). Each is loaded once from cv/models/ and "
        "classified as yoloe or yolo_world by filename (must contain 'world' "
        "for YOLO-World checkpoints).",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=",".join(DEFAULT_PROMPTS),
        help="Comma-separated text class prompts to bind via set_classes() "
        f"(default: the Phase 3 gap-class list, {len(DEFAULT_PROMPTS)} prompts).",
    )
    parser.add_argument(
        "--imgsz",
        type=str,
        default=",".join(str(v) for v in DEFAULT_IMGSZ),
        help=f"Comma-separated inference resolutions to sweep (default: "
        f"{','.join(str(v) for v in DEFAULT_IMGSZ)}). Do not go above 1600 - "
        "Step 0 measured that 2560 degrades nearly everything for stock "
        "YOLO26, and there is no reason to expect open-vocab models trained "
        "the same way to behave differently; this script does not enforce "
        "the cap, but exceeding it should be a deliberate choice, not a "
        "default.",
    )
    parser.add_argument(
        "--conf",
        type=str,
        default=",".join(str(v) for v in DEFAULT_CONF),
        help=f"Comma-separated confidence floors to sweep (default: "
        f"{','.join(str(v) for v in DEFAULT_CONF)}). Deliberately near-zero, "
        "same reasoning as measure_detection.py.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Inference device: 'mps' (default, Apple GPU) or 'cpu'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(MEASUREMENTS_DIR / "openvocab.csv"),
        help=f"Per-detection CSV output path (default: {MEASUREMENTS_DIR / 'openvocab.csv'}).",
    )
    args = parser.parse_args()

    models = parse_csv_list(args.models)
    prompts = parse_csv_list(args.prompts)
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
    print(f"Prompts ({len(prompts)}): {prompts}")
    print(
        f"Sweeping {len(models)} model(s) x {len(imgsz_values)} imgsz x "
        f"{len(conf_values)} conf over {len(images)} image(s) "
        f"({len(models) * len(imgsz_values) * len(conf_values) * len(images)} "
        "predict() calls total)."
    )

    rows = measure(images, models, prompts, imgsz_values, conf_values, device)
    write_csv(rows, Path(args.output))
    print_summary(rows)


if __name__ == "__main__":
    main()
