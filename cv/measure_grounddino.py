"""
measure_grounddino.py - Phase 4 Part 1: commit Grounding DINO (IDEA-Research,
via HuggingFace `transformers`) as a real, rerunnable tool in this repo.

Why this file exists: docs/phase-4-detection-research.md Part 3 measured
Grounding DINO as the strongest hazard-recall result anywhere in this
project, but the script that produced those numbers only ever existed in a
throwaway scratch venv (`gdino_venv`) from that research session and was
never committed - a real, named "Tier 2" evidence gap in that doc's own
"Where the evidence actually lives" section. This script closes that gap:
same model, same prompt-formatting approach, same quirks worked out, now
committed and runnable from this repo's own `.venv` with
`cv/requirements.txt`'s pinned `transformers==5.15.0` + `torch==2.13.0`
(verified together, on real MPS hardware, in THIS repo's `.venv` - not just
copied from the research session's numbers - before being pinned).

Same shape and conventions as measure_openvocab.py (its closest sibling):
offline batch harness over `cv/captures/*_raw.jpg`, `--images`/`--conf`/
`--output` CLI shape, one CSV row per detection plus a printed summary.
What's different, because Grounding DINO is a different kind of model:

  - No `imgsz` sweep. Grounding DINO's HuggingFace processor resizes
    internally from its own config; there is no equivalent dial to
    measure_detection.py's/measure_openvocab.py's `--imgsz` here.
  - Two confidence-shaped arguments, not one: `--conf` (mapped onto
    `post_process_grounded_object_detection()`'s `threshold` argument - see
    the quirk below) and `--text-threshold` (mapped onto that same call's
    `text_threshold` argument, gating how confidently a text token must
    match before it's kept in a box's label at all).
  - The prompt list is formatted as Grounding DINO's documented
    period-separated phrase-grounding query (`"knife. scissors. ..."`), not
    bound via `set_classes()` the way YOLO-World/YOLOE take it.
  - Predicted labels can be COMPOUND (see label_contains_any_keyword below)
    - multiple adjacent prompt phrases fused into one string on a single
      box. Scoring accounts for this explicitly with substring/keyword
      matching rather than an exact class-name lookup.

Two real API quirks, discovered in the research session and reconfirmed
directly in THIS repo's own `.venv` before this script was written (not
assumed from memory):

  1. `post_process_grounded_object_detection()`'s confidence-floor argument
     is named `threshold`, not `box_threshold` - that name changed at some
     point between transformers versions, and code snippets found online
     for an older version will crash on this exact mismatch against
     `transformers==5.15.0`.
  2. Multiple prompt phrases matching the same detected span come back
     fused into one predicted string (e.g. `"blade knife kitchen knife"`),
     not the clean one-class-per-box output YOLO-World/YOLOE give. A caller
     has to do substring/keyword matching against the returned text, not an
     exact class-name equality check - see label_contains_any_keyword().

Privacy (CLAUDE.md decision 8 - no cloud, ever, for this data): this script
downloads model *weights* only, once, from the HuggingFace Hub (a public
model repository, not a live per-frame API call) - identical reasoning to
measure_openvocab.py's own privacy note. All inference after the one-time
weights download is local torch/MPS/CPU. Never add a hosted/API detector
here.

Scope (Phase 4 Part 1 only - see docs/phase-4-detection-research.md and the
Phase 4 task that produced this file):
  - Does not modify measure_openvocab.py, measure_detection.py,
    evaluate_home_frames.py, or measure_change_detection.py.
  - Scoring reuses evaluate_home_frames.py's ground-truth loading and IoU
    matching (find_home_frames, load_ground_truth, iou) UNCHANGED, imported
    directly rather than reimplemented, so this script is measured against
    the exact same 32-frame held-out ground truth by the exact same IoU>=0.5
    rule the rest of this project's detection numbers use - only the
    "does this prediction count toward this concept" step differs (label
    substring matching instead of class-id equality), because Grounding
    DINO's output shape requires it.
  - This script is also the source risk_engine.py imports from for Part 3's
    live change-detection-confirmation step (load_grounding_dino_model,
    format_phrase_grounding_query, label_contains_any_keyword,
    parse_grounded_detections) - same reuse pattern risk_engine.py already
    uses for measure_openvocab.py's load_open_vocab_model() and
    measure_change_detection.py's detection functions.

Usage:
    python measure_grounddino.py
    python measure_grounddino.py --model grounding-dino-tiny,grounding-dino-base
    python measure_grounddino.py --images "cv/captures/*home*_raw.jpg" \\
        --conf 0.20 --text-threshold 0.15 \\
        --output cv/measurements/gdino_tiny_phase4.csv
"""

import argparse
import csv
import re
import statistics
import time
from glob import glob
from pathlib import Path

import cv2

from evaluate_home_frames import CLASS_NAMES, find_home_frames, iou, load_ground_truth

CAPTURES_DIR = Path(__file__).resolve().parent / "captures"
MEASUREMENTS_DIR = Path(__file__).resolve().parent / "measurements"

DEFAULT_IMAGES_GLOB = str(CAPTURES_DIR / "*home*_raw.jpg")

# HuggingFace model ids, not local weights filenames (unlike
# measure_openvocab.py's cv/models/*.pt convention) - `transformers` manages
# its own on-disk cache (~/.cache/huggingface) rather than this project's
# cv/models/ directory. "tiny" is the default: the research doc measured it
# as BOTH faster AND higher-recall than "base" on this project's held-out
# set (0.790 vs 0.548 sharp_object recall at conf>=0.20) - a genuine,
# non-obvious result (bigger was not better here), not an oversight.
MODEL_ID_BY_SHORT_NAME = {
    "grounding-dino-tiny": "IDEA-Research/grounding-dino-tiny",
    "grounding-dino-base": "IDEA-Research/grounding-dino-base",
}
DEFAULT_MODELS = ["grounding-dino-tiny"]

# Same broadened prompt list docs/phase-4-detection-research.md Part 3 used
# ("the same broadened prompt list as Part 2, minus medication bottle/
# medicine bottle, merged for brevity") - "person" kept as the same
# must-not-regress control Part 2 used it for (CLAUDE.md decision 2).
DEFAULT_PROMPTS = [
    "small object",
    "small item",
    "tiny object",
    "small object on the floor",
    "sharp object",
    "blade",
    "knife",
    "kitchen knife",
    "scissors",
    "cable",
    "power cord",
    "electrical cord",
    "charging cable",
    "wire",
    "bottle",
    "plastic bottle",
    "medicine bottle",
    "pill bottle",
    "toy",
    "small toy",
    "toy block",
    "cabinet handle",
    "drawer handle",
    "cabinet knob",
    "handle",
    "person",
]

# The two prompt families scored against the 32-frame ground truth (only
# concepts that ground truth actually exists for - see
# docs/labelling-guide.md - matching docs/phase-4-detection-research.md
# Part 2's "Which individual prompt phrasing actually did the work" section
# exactly). Every other prompt above (cable/bottle/handle) has no ground
# truth in this project and is intentionally NOT scored here - the research
# doc's qualitative-only treatment of those concepts stands; this script
# does not invent numbers for them.
CONCEPT_PROMPT_FAMILIES = {
    "sharp_object": ["sharp object", "blade", "knife", "kitchen knife", "scissors"],
    "small_swallowable": [
        "small object",
        "small item",
        "tiny object",
        "small object on the floor",
        "toy",
        "small toy",
        "toy block",
    ],
}
CONCEPT_CLASS_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

DEFAULT_CONF = 0.20  # `threshold` arg - see the module docstring's quirk #1.
DEFAULT_TEXT_THRESHOLD = 0.15
DEFAULT_SCORE_CONF_THRESHOLDS = [0.30, 0.20]  # matches the research doc's tables
IOU_MATCH_THRESHOLD = 0.5  # same rule as evaluate_home_frames.py, unchanged

CSV_FIELDNAMES = [
    "source_image",
    "capture_tag",
    "model",
    "prompts",
    "conf_floor",
    "text_threshold",
    "label",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "bbox_area_px",
    "bbox_area_frac",
    "inference_ms",
]

# Matches the tag detect_stream.py's save_snapshot() writes - identical
# regex/reasoning to measure_openvocab.py's own CAPTURE_TAG_RE.
CAPTURE_TAG_RE = re.compile(r"^(.*)_raw$")


def parse_csv_list(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_device(requested: str) -> str:
    """Same fallback as measure_openvocab.py/measure_detection.py: prefer
    the requested device, never crash a batch run over unavailable MPS.
    """
    import torch

    if requested == "mps" and not torch.backends.mps.is_available():
        print(
            "Warning: --device mps requested but MPS is not available on "
            "this machine. Falling back to CPU."
        )
        return "cpu"
    return requested


def format_phrase_grounding_query(prompts: list) -> str:
    """Grounding DINO's documented phrase-grounding query format: phrases
    lowercased, period-separated, with a trailing period
    (`"small object. knife. scissors. person."`). Deliberately a pure
    string function with no model/tensor dependency, so risk_engine.py's
    Part 3 crop-confirmation step (and this file's own tests, if any are
    added later) can call it without loading a model.
    """
    return ". ".join(p.strip().lower() for p in prompts if p.strip()) + "."


def label_contains_any_keyword(label: str, keywords: list) -> bool:
    """Does a (possibly COMPOUND - see the module docstring's quirk #2)
    Grounding DINO label contain ANY of `keywords` as a substring, matched
    case-insensitively on whole extracted words? A compound label like
    `"blade knife kitchen knife"` must match `"knife"`; a label like
    `"kitchen knifelike"` must NOT match `"knife"` on a bare substring check
    (a real, if unlikely, false-match risk with plain `in`), so this
    tokenizes both sides and checks token-sequence containment instead of
    raw substring search. Pure string logic, no model/image dependency -
    the piece both this script's scorer and risk_engine.py's Part 3
    crop-confirmation policy share and that cv/test_risk_engine.py exercises
    headlessly.
    """
    if not label:
        return False
    label_tokens = re.findall(r"[a-z0-9]+", label.lower())
    for keyword in keywords:
        keyword_tokens = re.findall(r"[a-z0-9]+", keyword.lower())
        if not keyword_tokens:
            continue
        n = len(keyword_tokens)
        for i in range(len(label_tokens) - n + 1):
            if label_tokens[i : i + n] == keyword_tokens:
                return True
    return False


def load_grounding_dino_model(model_id: str, device: str):
    """Load a Grounding DINO checkpoint + its processor. `model_id` is
    either a short name (a key of MODEL_ID_BY_SHORT_NAME) or a full
    HuggingFace repo id, so callers (this script's --model flag,
    risk_engine.py's --gdino-model flag) can use either.
    """
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    full_id = MODEL_ID_BY_SHORT_NAME.get(model_id, model_id)
    processor = AutoProcessor.from_pretrained(full_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(full_id).to(device)
    model.eval()
    return processor, model


def run_grounding_dino(
    processor,
    model,
    device: str,
    frame_bgr,
    prompts: list,
    conf: float,
    text_threshold: float,
):
    """Run one Grounding DINO inference pass over a single BGR (OpenCV-
    convention) frame/crop and return (detections, inference_ms), where each
    detection is {"label": str, "confidence": float, "x1","y1","x2","y2"}
    in the ORIGINAL frame/crop's pixel coordinates (post_process_...'s
    target_sizes argument handles the internal-resize-to-original-size
    conversion, so callers never have to rescale boxes by hand).

    Shared by this script's offline sweep and risk_engine.py's live Part 3
    crop-confirmation step - one implementation of "call Grounding DINO
    correctly, handling the threshold-argument rename," not two.
    """
    import torch
    from PIL import Image

    query = format_phrase_grounding_query(prompts)
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    start = time.perf_counter()
    inputs = processor(images=image, text=query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Quirk #1 (see module docstring): this argument is named `threshold`,
    # NOT `box_threshold` - that renamed at some point between transformers
    # versions and the first attempt at this in the research session
    # crashed on the old name.
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=conf,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )
    inference_ms = (time.perf_counter() - start) * 1000.0

    detections = []
    result = results[0]
    # `text_labels` (human-readable, possibly compound - quirk #2) is what
    # this project scores against; `labels` in this transformers version is
    # identical in practice but `text_labels` is the documented field to
    # rely on for the actual matched phrase(s).
    for label, score, box in zip(result["text_labels"], result["scores"], result["boxes"]):
        x1, y1, x2, y2 = (float(v) for v in box.tolist())
        detections.append(
            {
                "label": str(label),
                "confidence": float(score),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return detections, inference_ms


def find_images(pattern: str) -> list:
    """Same *_raw.jpg-only guard as measure_openvocab.py's find_images -
    duplicated rather than imported, matching this project's existing
    convention of small generic glob/CSV helpers living per-script rather
    than a shared utils module (see measure_openvocab.py/measure_detection.py,
    neither of which imports the other's identical-shaped helper either).
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
    stem = Path(image_path).stem
    match = CAPTURE_TAG_RE.match(stem)
    return match.group(1) if match else stem


def measure(images, model_ids, prompts, conf, text_threshold, device):
    """Model x image sweep (no imgsz axis - see module docstring). Each
    model is loaded once and reused across every image, same structure as
    measure_openvocab.py's measure().
    """
    rows = []
    prompts_joined = "|".join(prompts)

    for model_id in model_ids:
        print(f"Loading {model_id} ({MODEL_ID_BY_SHORT_NAME.get(model_id, model_id)}) ...")
        processor, model = load_grounding_dino_model(model_id, device)
        print(f"{model_id} loaded.")

        for image_path in images:
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"  Warning: could not read {image_path}, skipping.")
                continue
            frame_h, frame_w = frame.shape[:2]
            frame_area = float(frame_w * frame_h)
            capture_tag = capture_tag_from_filename(image_path)

            detections, inference_ms = run_grounding_dino(
                processor, model, device, frame, prompts, conf, text_threshold
            )

            if not detections:
                rows.append(
                    {
                        "source_image": Path(image_path).name,
                        "capture_tag": capture_tag,
                        "model": model_id,
                        "prompts": prompts_joined,
                        "conf_floor": conf,
                        "text_threshold": text_threshold,
                        "label": "",
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

            for det in detections:
                bbox_area_px = max(0.0, det["x2"] - det["x1"]) * max(0.0, det["y2"] - det["y1"])
                rows.append(
                    {
                        "source_image": Path(image_path).name,
                        "capture_tag": capture_tag,
                        "model": model_id,
                        "prompts": prompts_joined,
                        "conf_floor": conf,
                        "text_threshold": text_threshold,
                        "label": det["label"],
                        "confidence": round(det["confidence"], 4),
                        "x1": round(det["x1"], 1),
                        "y1": round(det["y1"], 1),
                        "x2": round(det["x2"], 1),
                        "y2": round(det["y2"], 1),
                        "bbox_area_px": round(bbox_area_px, 1),
                        "bbox_area_frac": round(bbox_area_px / frame_area, 6),
                        "inference_ms": round(inference_ms, 1),
                    }
                )
        del model, processor

    return rows


def write_csv(rows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {output_path}")


def print_speed_summary(rows) -> None:
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row["inference_ms"])
    print("\nMean inference time by model (ms/frame):")
    for model_id, times in sorted(by_model.items()):
        print(f"  {model_id:<24} mean={statistics.mean(times):>8.1f} ms  n={len(times)}")


def score_against_ground_truth(rows, model_id: str, conf_thresholds: list) -> None:
    """IoU>=0.5 precision/recall of this model's rows against the 32-frame
    held-out home set, per CONCEPT_PROMPT_FAMILIES concept - same matching
    RULE as evaluate_home_frames.py (imported find_home_frames/
    load_ground_truth/iou unchanged), with one necessary difference:
    Grounding DINO has no class_id, so "does this prediction belong to
    concept X" is decided by label_contains_any_keyword() against that
    concept's prompt family instead of a class-id equality check. This
    reproduces docs/phase-4-detection-research.md Part 3's table shape
    exactly (per concept, per conf threshold: TP/FP/FN/precision/recall),
    so a reader can directly compare this script's output against that
    doc's numbers.
    """
    frames = find_home_frames(CAPTURES_DIR)
    if not frames:
        print("\nNo labelled home frames found - skipping ground-truth scoring.")
        return

    model_rows = [r for r in rows if r["model"] == model_id and r["label"]]
    rows_by_image = {}
    for row in model_rows:
        rows_by_image.setdefault(row["source_image"], []).append(row)

    print(f"\n=== {model_id} vs 32-frame held-out ground truth (IoU>=0.5) ===")
    print(f"{'concept':<20} {'conf':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'precision':>10} {'recall':>8}")

    for concept, keywords in CONCEPT_PROMPT_FAMILIES.items():
        class_id = CONCEPT_CLASS_ID[concept]
        for conf in sorted(conf_thresholds, reverse=True):
            tp = fp = fn = 0
            for image_path, label_path in frames:
                img = cv2.imread(str(image_path))
                h, w = img.shape[:2]
                gt = [
                    box
                    for box in load_ground_truth(label_path, w, h)
                    if box["class_id"] == class_id
                ]
                preds = [
                    {"x1": r["x1"], "y1": r["y1"], "x2": r["x2"], "y2": r["y2"]}
                    for r in rows_by_image.get(image_path.name, [])
                    if float(r["confidence"]) >= conf
                    and label_contains_any_keyword(r["label"], keywords)
                ]
                matched_gt = set()
                for pred in preds:
                    best_iou, best_idx = 0.0, None
                    for idx, gt_box in enumerate(gt):
                        if idx in matched_gt:
                            continue
                        candidate_iou = iou(pred, gt_box)
                        if candidate_iou > best_iou:
                            best_iou, best_idx = candidate_iou, idx
                    if best_idx is not None and best_iou >= IOU_MATCH_THRESHOLD:
                        matched_gt.add(best_idx)
                        tp += 1
                    else:
                        fp += 1
                fn += len(gt) - len(matched_gt)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            print(
                f"{concept:<20} {conf:>6.2f} {tp:>4} {fp:>4} {fn:>4} "
                f"{precision:>10.3f} {recall:>8.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline batch measurement: run Grounding DINO "
        "(IDEA-Research, via HuggingFace transformers) over already-saved "
        "raw captures with a fixed phrase-grounding prompt list, emit "
        "structured per-detection numbers, and score against the 32-frame "
        "held-out home ground truth. Phase 4 Part 1 - see "
        "docs/phase-4-detection-research.md."
    )
    parser.add_argument(
        "--images",
        type=str,
        default=DEFAULT_IMAGES_GLOB,
        help=f"Glob of raw capture images to measure (default: {DEFAULT_IMAGES_GLOB}). "
        "Must match only *_raw.jpg.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated model short names or HuggingFace repo ids "
        f"(default: {','.join(DEFAULT_MODELS)}). Short names: "
        f"{', '.join(MODEL_ID_BY_SHORT_NAME)}.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=",".join(DEFAULT_PROMPTS),
        help=f"Comma-separated phrase-grounding prompts (default: the Phase 4 "
        f"broadened hazard list, {len(DEFAULT_PROMPTS)} prompts).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"Confidence floor - maps onto post_process_grounded_object_"
        f"detection()'s `threshold` argument, NOT `box_threshold` (see "
        f"the module docstring's quirk #1) (default: {DEFAULT_CONF}).",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=DEFAULT_TEXT_THRESHOLD,
        help=f"post_process_grounded_object_detection()'s `text_threshold` "
        f"argument (default: {DEFAULT_TEXT_THRESHOLD}).",
    )
    parser.add_argument(
        "--score-conf",
        type=str,
        default=",".join(str(v) for v in DEFAULT_SCORE_CONF_THRESHOLDS),
        help="Comma-separated confidence thresholds to score precision/"
        f"recall at, reusing the SAME raw detections from --conf (default: "
        f"{','.join(str(v) for v in DEFAULT_SCORE_CONF_THRESHOLDS)}, matching "
        "the research doc's tables). Must each be >= --conf, or a threshold "
        "here would ask for boxes that were never even recorded.",
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
        default=str(MEASUREMENTS_DIR / "gdino_tiny_phase4.csv"),
        help="Per-detection CSV output path (default: "
        f"{MEASUREMENTS_DIR / 'gdino_tiny_phase4.csv'}).",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Skip ground-truth precision/recall scoring (still writes the "
        "CSV) - useful when sweeping prompts/models that have no ground "
        "truth to score against.",
    )
    args = parser.parse_args()

    model_ids = parse_csv_list(args.model)
    prompts = parse_csv_list(args.prompts)
    score_conf_thresholds = [float(v) for v in parse_csv_list(args.score_conf)]

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
    print(f"Sweeping {len(model_ids)} model(s) over {len(images)} image(s).")

    rows = measure(images, model_ids, prompts, args.conf, args.text_threshold, device)

    output_path = Path(args.output)
    if len(model_ids) == 1:
        write_csv(rows, output_path)
    else:
        # One CSV per model when sweeping more than one, named
        # <output-stem>_<model>.csv - avoids silently interleaving two
        # models' rows into a filename that implies a single model (the
        # default --output name literally says "tiny").
        for model_id in model_ids:
            model_rows = [r for r in rows if r["model"] == model_id]
            per_model_path = output_path.with_name(f"{output_path.stem}_{model_id}{output_path.suffix}")
            write_csv(model_rows, per_model_path)

    print_speed_summary(rows)

    if not args.no_score:
        for model_id in model_ids:
            score_against_ground_truth(rows, model_id, score_conf_thresholds)


if __name__ == "__main__":
    main()
