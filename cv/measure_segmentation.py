"""
measure_segmentation.py - Phase 3: does class-agnostic segmentation solve the
"hazard already present at startup" case that classical single-frame methods
(colour clustering recall 0.09, local-texture objectness recall 0.79 /
~0.33 eyeballed precision) and the round-3 fine-tuned classifier
(recall 0.216) could not?

Read cv/../docs/phase-3-step0-findings.md and CLAUDE.md decision 7 (amended
2026-08-10) before touching this file - it explains why this exists and what
"solved" already means for the other half of the problem (change detection).

What this measures, in order:
  1. Raw segmentation recall against the 32 labelled *home*_raw.jpg frames -
     everything FastSAM finds, no filter, matched against ground truth boxes
     at IoU >= 0.5 (class-agnostic: a prediction counts if it covers ANY
     labelled box, regardless of sharp_object vs small_swallowable - naming
     is not the goal here).
  2. Filtered recall - same matching, after the size / person-overlap /
     extent filter below.
  3. A crop of every filtered-in detection, written to the scratchpad (never
     the repo - CLAUDE.md privacy scope aside, these are throwaway evidence
     for eyeballing precision, not something to check in) for manual
     precision-by-eye review, since ground truth only covers two classes and
     scoring a hit on an unlabelled real object as a false positive would be
     wrong (see the task's precision caveat).
  4. The same pipeline, unfiltered counts only, on a sample of the
     unlabelled *room2*_raw.jpg frames (darker, artificial light) - no
     recall number is possible there, just a box count and crops for the
     same by-eye check.
  5. Wall-clock seconds/frame for the segmentation call itself and for
     person-detection + segmentation combined, since this runs at startup
     and on a hazard-map-reset trigger, not every frame - seconds are
     acceptable, but the number needs to be real, not assumed.

Filter design (the actual deliverable - segmentation alone finds everything,
including every floor tile, cabinet edge and the person):
  - SIZE: bbox area fraction of the frame within [MIN_AREA_FRAC,
    MAX_AREA_FRAC]. Too small is segmentation noise (a shadow edge, a tile
    grout line); too large is a surface itself (floor, wall, table top) or a
    piece of furniture, not an object resting on one.
  - NOT PERSON: reject any segment whose bbox overlaps a YOLO person box
    (reused from cv/models/yolo26l.pt at imgsz 640 - CLAUDE.md decision 2's
    single-pass detector already produces this every frame; this script
    does not re-implement person detection, just consumes the box) above
    PERSON_OVERLAP_THRESHOLD of the segment's own area.
  - NOT STRUCTURAL (extent): reject low "fill ratio" segments - mask pixel
    count divided by its own bounding-box area. A compact object (a knife,
    a bottle, a toy) mostly fills its bbox. A drawer handle, a door frame
    edge, or a shadow line is long and thin relative to its bbox and fills
    only a sliver of it. This is the one candidate signal the classical
    texture method (round-3 findings) did not have available - it worked on
    raw pixel texture, not on a segmented region's own geometry - so it is
    the main thing being tested here, not a restatement of size.
  - ASPECT RATIO (--min-aspect-ratio, NOT part of the default filter, added
    2026-08-12 per docs/phase-4-detection-research.md Part 5): bbox
    min-side/max-side ratio, a genuinely different signal from extent -
    extent is fill ratio (does the mask fill its own bbox), which does not
    penalise a long thin shape that fills its own tight bbox well (a
    wall-corner edge or a straight furniture seam scores HIGH extent despite
    being obviously not a compact object). Measured to help against
    elongated false positives (a wall-corner edge, aspect_ratio 0.16) but
    NOT against round/compact texture blobs (a cushion-fabric blob,
    aspect_ratio 0.77 - indistinguishable from a real compact object on this
    signal alone). See Part 5 for the full measurement.
  - CONTAINMENT is NOT implemented as a hard filter. Identifying "sits on a
    flatter enclosing region" requires either a second segmentation pass to
    find surface regions or a floor/table-plane heuristic (e.g. "reachable"
    = lower N% of frame). That heuristic assumes a fixed camera-to-floor
    relationship the project has already rejected once (CLAUDE.md decision
    5's resolution-independence rule, and the fact that this project tests
    camera height 60cm vs 190cm precisely because it varies). It is
    reported as an optional, clearly-labelled experiment
    (--y-containment-frac) rather than folded into the default filter, so
    its effect can be seen in isolation rather than silently baked in.

Scope:
  - No training, no fine-tuning.
  - Does not modify detect_stream.py, camera.py, measure_detection.py,
    measure_openvocab.py, or evaluate_home_frames.py. Ground-truth reading
    and IoU here are re-implemented standalone (not imported) so this file
    has zero coupling to those scripts' internals changing later.
  - Never writes into cv/captures/ - only reads *_raw.jpg and *_raw.txt.
    All crops/visual evidence go to --crop-dir, which defaults to a
    scratchpad path, never a repo path.

Usage:
    python cv/measure_segmentation.py
    python cv/measure_segmentation.py --model FastSAM-s.pt --imgsz 1024
    python cv/measure_segmentation.py --room2-sample 15
"""

import argparse
import csv
import os
import random
import time
from glob import glob
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import torch
from ultralytics import YOLO, FastSAM

CV_DIR = Path(__file__).resolve().parent
MODELS_DIR = CV_DIR / "models"
CAPTURES_DIR = CV_DIR / "captures"
MEASUREMENTS_DIR = CV_DIR / "measurements"
DEFAULT_CROP_DIR = Path(
    "/private/tmp/claude-501/-Users-shakedivgi-Projects-GuardianEye/"
    "4ff5b567-409f-4edc-84a1-57104903c919/scratchpad/segcrops"
)

PERSON_MODEL = "yolo26l.pt"
PERSON_IMGSZ = 640  # per docs/phase-3-step0-findings.md Finding 1

DEFAULT_SEG_MODEL = "FastSAM-s.pt"
DEFAULT_SEG_IMGSZ = 1024
SEG_CONF = 0.25
SEG_IOU = 0.9  # FastSAM's own NMS between overlapping proposals, not our match IoU

MATCH_IOU_THRESHOLD = 0.5

# Filter defaults - see module docstring for the reasoning behind each.
MIN_AREA_FRAC = 0.0003
# MAX_AREA_FRAC tightened 0.05 -> 0.01 on 2026-08-12 (Part 5,
# docs/phase-4-detection-research.md): measured on the 32 home + room2-sample-30
# spread, this loses ZERO recall (28/116 filtered recall unchanged at either
# value - the 32 home frames' 116 ground-truth boxes only reach area_frac
# 0.019 at the max, p90 0.013, so nothing above 0.01 was ever a real hazard
# in this test set) while cutting per-frame candidate volume ~17-19% (home
# mean 33.8->27.9, room2 mean 64.6->52.0). A real, free, measured
# improvement - but not close to solving the volume problem alone; see Part 5
# for the honest "still not walkable" conclusion and why a tighter value
# starts costing recall past this point.
MAX_AREA_FRAC = 0.01
PERSON_OVERLAP_THRESHOLD = 0.3
MIN_EXTENT = 0.35


def resolve_device(requested: str) -> str:
    if requested == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS unavailable, falling back to CPU.")
        return "cpu"
    return requested


def load_weights(model_cls, model_name: str, device: str):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = MODELS_DIR / model_name
    already_present = weights_path.exists()
    if not already_present:
        print(f"Downloading {model_name} ...")
    model = model_cls(str(weights_path) if already_present else model_name)
    if not already_present:
        downloaded = Path(model_name)
        if downloaded.exists():
            downloaded.rename(weights_path)
    model.to(device)
    return model


def read_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list:
    """YOLO-format label file -> list of {x1,y1,x2,y2} pixel boxes.
    Class id is intentionally dropped - matching here is class-agnostic.
    """
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        _cls, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx) * img_w, float(cy) * img_h, float(w) * img_w, float(h) * img_h
        boxes.append({"x1": cx - w / 2, "y1": cy - h / 2, "x2": cx + w / 2, "y2": cy + h / 2})
    return boxes


def iou(a, b) -> float:
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


def overlap_frac_of_a(a, b) -> float:
    """Fraction of box a's own area covered by intersection with b - used
    for the person filter, where we want 'is most of this segment inside
    the person box', not symmetric IoU (the person box is usually far
    bigger than a small hazard segment, so IoU would always look small even
    when the segment is entirely inside the person).
    """
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    return inter / area_a if area_a > 0 else 0.0


def segment_frame(seg_model, frame, imgsz, device):
    """One FastSAM 'everything' pass. retina_masks=False deliberately - see
    module docstring: masks are only used for their own fill ratio (extent),
    which is scale-invariant, so the far smaller processing-resolution mask
    is used instead of upsampling every mask to full frame resolution
    (which would be gigabytes across a 32-frame x several-config sweep).
    """
    results = seg_model.predict(
        frame, device=device, retina_masks=False, imgsz=imgsz,
        conf=SEG_CONF, iou=SEG_IOU, verbose=False,
    )
    r = results[0]
    frame_h, frame_w = frame.shape[:2]
    frame_area = float(frame_w * frame_h)
    segments = []
    if r.boxes is None or len(r.boxes) == 0:
        return segments
    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    masks = r.masks.data if r.masks is not None else None
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = (float(v) for v in boxes_xyxy[i])
        bbox_area_px = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        if masks is not None:
            mask_i = masks[i]
            mask_h, mask_w = mask_i.shape[-2:]
            mask_fill = float(mask_i.sum().item())
            mask_bbox_area_frac_of_own_res = (
                (x2 - x1) / frame_w * mask_w * (y2 - y1) / frame_h * mask_h
            )
            extent = (
                mask_fill / mask_bbox_area_frac_of_own_res
                if mask_bbox_area_frac_of_own_res > 0 else 0.0
            )
        else:
            extent = None
        bbox_w, bbox_h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        # aspect_ratio: bbox min-side/max-side, in (0, 1] - 1.0 is a square
        # bbox, near 0 is a long thin rectangle. Deliberately a DIFFERENT
        # signal from extent: extent is fill-ratio (does the mask fill its
        # own bbox), which does not penalise a long thin shape that fills
        # its own tight bbox well (a rod-shaped object, a wall-corner edge,
        # a straight furniture seam all score high extent). aspect_ratio
        # penalises the bbox's own elongation regardless of how well the
        # mask fills it. See docs/phase-4-detection-research.md Part 5 for
        # why this alone does not solve the round/compact texture-blob false
        # positive case (a round cushion-fabric blob has aspect_ratio close
        # to 1, same as a real compact object - this signal only catches the
        # elongated false positives, not the round ones).
        aspect_ratio = (
            min(bbox_w, bbox_h) / max(bbox_w, bbox_h) if max(bbox_w, bbox_h) > 0 else 0.0
        )
        segments.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": float(confs[i]),
            "bbox_area_px": bbox_area_px,
            "area_frac": bbox_area_px / frame_area,
            "extent": extent,
            "aspect_ratio": aspect_ratio,
            "cy_frac": ((y1 + y2) / 2.0) / frame_h,
        })
    return segments


def apply_filter(segments, person_boxes, min_area, max_area,
                  person_overlap_thresh, min_extent, y_containment_frac,
                  min_aspect_ratio=None):
    kept = []
    for seg in segments:
        if not (min_area <= seg["area_frac"] <= max_area):
            continue
        if seg["extent"] is not None and seg["extent"] < min_extent:
            continue
        if min_aspect_ratio is not None and seg["aspect_ratio"] < min_aspect_ratio:
            continue
        is_person = any(
            overlap_frac_of_a(seg, pbox) > person_overlap_thresh
            for pbox in person_boxes
        )
        if is_person:
            continue
        if y_containment_frac is not None and seg["cy_frac"] < y_containment_frac:
            continue
        kept.append(seg)
    return kept


def match_recall(predictions, ground_truth):
    """Class-agnostic: a ground-truth box counts as found if ANY prediction
    covers it at IoU >= MATCH_IOU_THRESHOLD. Returns (found, total).
    """
    found = 0
    for gt in ground_truth:
        if any(iou(gt, pred) >= MATCH_IOU_THRESHOLD for pred in predictions):
            found += 1
    return found, len(ground_truth)


def save_crop(frame, seg, out_dir: Path, frame_stem: str, idx: int):
    x1, y1, x2, y2 = (int(max(0, v)) for v in (seg["x1"], seg["y1"], seg["x2"], seg["y2"]))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    out_path = out_dir / (
        f"{frame_stem}_seg{idx:03d}_area{seg['area_frac']:.4f}"
        f"_ext{seg['extent']:.2f}_asp{seg['aspect_ratio']:.2f}.jpg"
    )
    cv2.imwrite(str(out_path), crop)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_SEG_MODEL)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_SEG_IMGSZ)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--min-area-frac", type=float, default=MIN_AREA_FRAC)
    parser.add_argument("--max-area-frac", type=float, default=MAX_AREA_FRAC)
    parser.add_argument("--person-overlap", type=float, default=PERSON_OVERLAP_THRESHOLD)
    parser.add_argument("--min-extent", type=float, default=MIN_EXTENT)
    parser.add_argument("--min-aspect-ratio", type=float, default=None,
                         help="Optional: reject segments whose bbox min-side/max-side "
                              "ratio is below this (0=very elongated, 1=square). NOT "
                              "part of the default filter - see module docstring / "
                              "docs/phase-4-detection-research.md Part 5.")
    parser.add_argument("--y-containment-frac", type=float, default=None,
                         help="Optional: reject segments whose center is above this "
                              "fraction of frame height (0=top,1=bottom). NOT part of "
                              "the default filter - see module docstring.")
    parser.add_argument("--room2-sample", type=int, default=15)
    parser.add_argument("--crop-sample-per-frame", type=int, default=4,
                         help="Max filtered-in crops saved per frame, to keep the "
                              "eyeball sample a manageable, roughly-uniform size "
                              "rather than dumping every detection.")
    parser.add_argument("--crop-dir", default=str(DEFAULT_CROP_DIR))
    parser.add_argument("--output", default=str(MEASUREMENTS_DIR / "segmentation.csv"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    crop_dir = Path(args.crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading person model {PERSON_MODEL} ...")
    person_model = load_weights(YOLO, PERSON_MODEL, device)
    print(f"Loading segmentation model {args.model} ...")
    seg_model = load_weights(FastSAM, args.model, device)

    home_images = sorted(glob(str(CAPTURES_DIR / "*home*_raw.jpg")))
    room2_images = sorted(glob(str(CAPTURES_DIR / "*room2*_raw.jpg")))
    if args.room2_sample and len(room2_images) > args.room2_sample:
        room2_images = sorted(random.sample(room2_images, args.room2_sample))

    print(f"{len(home_images)} home frames, {len(room2_images)} room2 frames sampled.")

    csv_rows = []
    total_found_raw = total_found_filt = total_gt = 0
    person_ms_list, seg_ms_list = [], []
    all_filtered_segments = []  # (frame_path, frame, seg) for crop sampling

    for image_path in home_images:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"  Warning: could not read {image_path}")
            continue
        label_path = Path(image_path).with_suffix(".txt")
        gt_boxes = read_yolo_labels(label_path, frame.shape[1], frame.shape[0])

        t0 = time.time()
        person_result = person_model.predict(
            frame, conf=0.25, imgsz=PERSON_IMGSZ, device=device, verbose=False
        )[0]
        t1 = time.time()
        person_boxes = []
        names = person_result.names
        for box in person_result.boxes:
            if names[int(box.cls[0])] == "person":
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                person_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

        segments = segment_frame(seg_model, frame, args.imgsz, device)
        t2 = time.time()
        person_ms_list.append((t1 - t0) * 1000)
        seg_ms_list.append((t2 - t1) * 1000)

        filtered = apply_filter(
            segments, person_boxes, args.min_area_frac, args.max_area_frac,
            args.person_overlap, args.min_extent, args.y_containment_frac,
            args.min_aspect_ratio,
        )

        raw_found, gt_total = match_recall(segments, gt_boxes)
        filt_found, _ = match_recall(filtered, gt_boxes)
        total_found_raw += raw_found
        total_found_filt += filt_found
        total_gt += gt_total

        csv_rows.append({
            "image": Path(image_path).name, "split": "home",
            "num_gt": gt_total, "num_raw_segments": len(segments),
            "num_filtered_segments": len(filtered),
            "gt_found_raw": raw_found, "gt_found_filtered": filt_found,
            "person_ms": round((t1 - t0) * 1000, 1),
            "seg_ms": round((t2 - t1) * 1000, 1),
        })

        for seg in filtered:
            all_filtered_segments.append((image_path, frame, seg))

    room2_raw_count = room2_filt_count = 0
    for image_path in room2_images:
        frame = cv2.imread(image_path)
        if frame is None:
            continue
        t0 = time.time()
        person_result = person_model.predict(
            frame, conf=0.25, imgsz=PERSON_IMGSZ, device=device, verbose=False
        )[0]
        t1 = time.time()
        person_boxes = []
        names = person_result.names
        for box in person_result.boxes:
            if names[int(box.cls[0])] == "person":
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                person_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        segments = segment_frame(seg_model, frame, args.imgsz, device)
        t2 = time.time()
        person_ms_list.append((t1 - t0) * 1000)
        seg_ms_list.append((t2 - t1) * 1000)
        filtered = apply_filter(
            segments, person_boxes, args.min_area_frac, args.max_area_frac,
            args.person_overlap, args.min_extent, args.y_containment_frac,
            args.min_aspect_ratio,
        )
        room2_raw_count += len(segments)
        room2_filt_count += len(filtered)
        csv_rows.append({
            "image": Path(image_path).name, "split": "room2",
            "num_gt": "", "num_raw_segments": len(segments),
            "num_filtered_segments": len(filtered),
            "gt_found_raw": "", "gt_found_filtered": "",
            "person_ms": round((t1 - t0) * 1000, 1),
            "seg_ms": round((t2 - t1) * 1000, 1),
        })
        for seg in filtered:
            all_filtered_segments.append((image_path, frame, seg))

    # Write CSV
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "split", "num_gt", "num_raw_segments", "num_filtered_segments",
                  "gt_found_raw", "gt_found_filtered", "person_ms", "seg_ms"]
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nWrote {len(csv_rows)} rows to {args.output}")

    # Crop sample for precision-by-eye: cap per-frame to keep sample manageable.
    by_frame = {}
    for image_path, frame, seg in all_filtered_segments:
        by_frame.setdefault(image_path, []).append((frame, seg))
    saved = 0
    for image_path, items in by_frame.items():
        random.shuffle(items)
        stem = Path(image_path).stem
        for i, (frame, seg) in enumerate(items[:args.crop_sample_per_frame]):
            save_crop(frame, seg, crop_dir, stem, i)
            saved += 1
    print(f"Saved {saved} crops to {crop_dir}")

    print("\n=== SUMMARY ===")
    print(f"Home frames: {len(home_images)}, ground-truth boxes: {total_gt}")
    print(f"Recall (raw, unfiltered):    {total_found_raw}/{total_gt} = "
          f"{total_found_raw/total_gt:.3f}" if total_gt else "n/a")
    print(f"Recall (after filter):       {total_found_filt}/{total_gt} = "
          f"{total_found_filt/total_gt:.3f}" if total_gt else "n/a")
    print(f"Room2 frames sampled: {len(room2_images)}, "
          f"raw segments: {room2_raw_count}, filtered: {room2_filt_count}")
    if person_ms_list:
        print(f"\nperson-detect ms/frame: mean={sum(person_ms_list)/len(person_ms_list):.1f}")
    if seg_ms_list:
        print(f"segmentation ms/frame:  mean={sum(seg_ms_list)/len(seg_ms_list):.1f} "
              f"min={min(seg_ms_list):.1f} max={max(seg_ms_list):.1f}")


if __name__ == "__main__":
    main()
