"""
evaluate_home_frames.py - Phase 3: IoU-0.5 precision/recall of a fine-tuned
weights file against the 32 held-out *home*_raw.jpg frames.

This is the eval harness round 1's numbers (precision 0.246 / recall 0.147 at
conf 0.25 - see docs/phase-3-step0-findings.md, "Round 1 fine-tune: measured
generalisation failure") were produced with, reconstructed here as a
committed, rerunnable script rather than ad hoc code, specifically so round 2
can be compared on identical methodology and this doesn't quietly drift.

Matching rule (standard single-IoU-threshold detection eval, no confidence
interpolation / no mAP curve - deliberately simple and explainable per
CLAUDE.md's "prefer simple, explainable logic" instruction):

  1. Only ground-truth frames with a label file are scored. The 32 home
     frames are exactly `cv/captures/*home*_raw.jpg` with a matching `.txt`.
  2. Predictions are filtered to the requested confidence threshold first.
  3. Per image, per class: predictions are matched to ground-truth boxes
     greedily in descending confidence order. A prediction matches the
     highest-IoU unmatched ground-truth box of the SAME class if IoU >= 0.5;
     each ground-truth box can be matched at most once (no double-counting
     one box as two true positives).
  4. TP = matched predictions. FP = unmatched predictions. FN = unmatched
     ground-truth boxes. Precision = TP/(TP+FP), recall = TP/(TP+FN), pooled
     across BOTH classes together (matching how round 1's headline number was
     reported) - per-class TP/FP/FN are also printed for diagnosis.

This script never touches cv/captures/ label files - read-only.
"""

import argparse
from pathlib import Path

import cv2

CAPTURES_DIR = Path(__file__).resolve().parent / "captures"
CLASS_NAMES = ["sharp_object", "small_swallowable"]

DEFAULT_CONF_THRESHOLDS = [0.25, 0.15, 0.05]
IOU_MATCH_THRESHOLD = 0.5


def find_home_frames(captures_dir: Path) -> list:
    """The 32-frame held-out test set: *home*_raw.jpg with a label file.
    Frames without a label file are skipped with a warning rather than
    silently dropped - a missing label on a frame the caller expected to be
    part of the test set is exactly the kind of quiet contamination bug the
    task warned about.
    """
    candidates = sorted(captures_dir.glob("*home*_raw.jpg"))
    labelled = []
    for image_path in candidates:
        label_path = image_path.with_suffix(".txt")
        if label_path.exists():
            labelled.append((image_path, label_path))
        else:
            print(f"Warning: {image_path.name} has no label file - excluded from eval.")
    return labelled


def load_ground_truth(label_path: Path, img_w: int, img_h: int) -> list:
    """YOLO-format label file -> list of {class_id, x1, y1, x2, y2} in
    full-resolution pixel space. Empty file (a real negative) yields [].
    """
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        class_id, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        cx_px, cy_px, w_px, h_px = cx * img_w, cy * img_h, w * img_w, h * img_h
        boxes.append({
            "class_id": class_id,
            "x1": cx_px - w_px / 2,
            "y1": cy_px - h_px / 2,
            "x2": cx_px + w_px / 2,
            "y2": cy_px + h_px / 2,
        })
    return boxes


def iou(a: dict, b: dict) -> float:
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_frame(predictions: list, ground_truth: list) -> tuple:
    """Greedy match, descending confidence, class-restricted, IoU >= 0.5.
    Returns (tp, fp, fn, per_class_counts) where per_class_counts is
    {class_id: {"tp": n, "fp": n, "fn": n}}.
    """
    per_class = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in range(len(CLASS_NAMES))}
    matched_gt = set()

    preds_sorted = sorted(predictions, key=lambda p: -p["confidence"])
    for pred in preds_sorted:
        best_iou, best_idx = 0.0, None
        for idx, gt in enumerate(ground_truth):
            if idx in matched_gt or gt["class_id"] != pred["class_id"]:
                continue
            candidate_iou = iou(pred, gt)
            if candidate_iou > best_iou:
                best_iou, best_idx = candidate_iou, idx
        if best_idx is not None and best_iou >= IOU_MATCH_THRESHOLD:
            matched_gt.add(best_idx)
            per_class[pred["class_id"]]["tp"] += 1
        else:
            per_class[pred["class_id"]]["fp"] += 1

    for idx, gt in enumerate(ground_truth):
        if idx not in matched_gt:
            per_class[gt["class_id"]]["fn"] += 1

    tp = sum(c["tp"] for c in per_class.values())
    fp = sum(c["fp"] for c in per_class.values())
    fn = sum(c["fn"] for c in per_class.values())
    return tp, fp, fn, per_class


def run_predictions(model, image_path: Path, conf: float, imgsz: int, device: str) -> list:
    frame = cv2.imread(str(image_path))
    results = model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)
    boxes = results[0].boxes
    predictions = []
    for box in boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        predictions.append({
            "class_id": int(box.cls[0]),
            "confidence": float(box.conf[0]),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    return predictions


def evaluate(weights_path: str, captures_dir: Path, conf_thresholds: list,
             imgsz: int, device: str) -> dict:
    from ultralytics import YOLO

    frames = find_home_frames(captures_dir)
    if not frames:
        raise SystemExit(f"No labelled *home*_raw.jpg frames found under {captures_dir}.")
    print(f"Evaluating {weights_path} against {len(frames)} home frame(s).")

    model = YOLO(weights_path)

    # Predict once at the lowest requested confidence (a superset of every
    # higher threshold's boxes), then re-threshold in Python for each
    # requested conf - avoids re-running inference per threshold, and
    # guarantees the three thresholds are evaluated on IDENTICAL raw
    # detections (no re-inference jitter between thresholds).
    lowest_conf = min(conf_thresholds)
    raw_predictions_by_frame = {}
    ground_truth_by_frame = {}
    for image_path, label_path in frames:
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        ground_truth_by_frame[image_path] = load_ground_truth(label_path, w, h)
        raw_predictions_by_frame[image_path] = run_predictions(
            model, image_path, lowest_conf, imgsz, device
        )

    results_by_threshold = {}
    for conf in conf_thresholds:
        total_tp = total_fp = total_fn = 0
        per_class_totals = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in range(len(CLASS_NAMES))}
        for image_path, label_path in frames:
            preds = [p for p in raw_predictions_by_frame[image_path] if p["confidence"] >= conf]
            gt = ground_truth_by_frame[image_path]
            tp, fp, fn, per_class = match_frame(preds, gt)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            for cid, counts in per_class.items():
                for key in ("tp", "fp", "fn"):
                    per_class_totals[cid][key] += counts[key]

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        results_by_threshold[conf] = {
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": precision, "recall": recall,
            "per_class": per_class_totals,
        }
    return results_by_threshold


def print_results(weights_path: str, results_by_threshold: dict) -> None:
    print(f"\n=== {weights_path} — IoU>=0.5 match, pooled across both classes ===")
    print(f"{'conf':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'precision':>10} {'recall':>8}")
    for conf in sorted(results_by_threshold, reverse=True):
        r = results_by_threshold[conf]
        print(f"{conf:>6} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
              f"{r['precision']:>10.3f} {r['recall']:>8.3f}")

    print("\nPer-class breakdown:")
    for conf in sorted(results_by_threshold, reverse=True):
        print(f"  conf={conf}:")
        for cid, name in enumerate(CLASS_NAMES):
            c = results_by_threshold[conf]["per_class"][cid]
            tp, fp, fn = c["tp"], c["fp"], c["fn"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            print(f"    {name:<18} TP={tp:>3} FP={fp:>3} FN={fn:>3} "
                  f"precision={prec:.3f} recall={rec:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IoU-0.5 precision/recall of a YOLO weights file against "
        "the 32 held-out home frames, at the same conf thresholds used to "
        "measure round 1 (0.25/0.15/0.05)."
    )
    parser.add_argument("weights", type=str, help="Path to a .pt weights file.")
    parser.add_argument("--captures-dir", type=str, default=str(CAPTURES_DIR))
    parser.add_argument("--conf", type=str, default=",".join(str(v) for v in DEFAULT_CONF_THRESHOLDS))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    conf_thresholds = [float(v) for v in args.conf.split(",")]
    results = evaluate(args.weights, Path(args.captures_dir), conf_thresholds, args.imgsz, args.device)
    print_results(args.weights, results)


if __name__ == "__main__":
    main()
