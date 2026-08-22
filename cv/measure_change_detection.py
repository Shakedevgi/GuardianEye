"""
measure_change_detection.py - re-measure frame-to-frame change detection,
now with a PERSISTENCE-based decision instead of single-pair person overlap.

Why this rewrite exists: the previous version of this script (single-pair
person-overlap suppression only) was measured in
docs/phase-3-step0-findings.md, "Change detection: re-measured with real
person-suppression, 2026-08-11", and that measurement surfaced three
confirmed problems:

  1. Suppressing any blob that overlaps a person box also suppresses a real
     hazard being SET DOWN, because while it is in an open palm its own
     motion region overlaps the hand/arm silhouette. This is why the
     previous version reported the car-key/lighter claim as refuted - not
     because the object didn't move, but because "ignore the person" also
     means "ignore anything touching the person," which includes the exact
     moment a hazard is placed.
  2. Floor speckle/grout/furniture-edge noise fires as a change blob on a
     single frame pair and is indistinguishable, on that one pair alone,
     from a real object.
  3. The old test bursts (home-60cm / room1-60cm / room1-190cm) are
     static-scene labelling bursts, not a continuous placement sequence, so
     they structurally cannot exercise "object appears and is left."

This version replaces single-pair suppression with a PERSISTENCE check:

  For a blob found by diffing frame[N-1] against frame[N] (the "primary
  pair"), look ahead to frame[N-1+K] for K in {2, 3} (i.e. two and three
  hops past the reference frame N-1) and re-run the SAME classical diff -
  detect_change_blobs(frame[N-1], frame[N-1+K]) - against the ORIGINAL
  reference frame N-1, not against frame N. This directly answers "is this
  location still different from the pre-change background," which is the
  actual definition of "an object was placed here and is still here":

    - A real placed hazard: the object stays in frame N-1+K, so it is still
      different from frame N-1's background there. Blob still shows up in
      roughly the same place. If a person was in the way in frame N-1+K
      (e.g. still crouched by it), that check fails at that K and the next K
      is tried; once the person has stepped away, the region is confirmed.
    - A hand caught mid-placement: the hand itself moves on to somewhere
      else within a couple of hops, so the location it occupied in frame N
      goes back to matching the pre-change background - diffing frame[N-1]
      against frame[N-1+K] at that spot shows NO blob any more. Persistence
      fails and the detection is correctly discarded, instead of being
      discarded for the wrong reason (person overlap) as before.
    - Floor speckle / shadow flicker: a one-frame lighting or JPEG-noise
      blip. Comparing frame[N-1] to frame[N-1+K] two or three hops later,
      the flicker is gone and the location matches the original background
      again - no blob, persistence fails, correctly discarded.

  A blob is KEPT (counted as a real detection) if EITHER the K=2 or the K=3
  lookahead check confirms it (a blob still exists in roughly the same
  location, per POSITION_TOLERANCE_FRAC below, AND is not covered by a
  person box at that later frame, per OVERLAP_FRAC_THRESHOLD) - "OR", per
  the task's literal instruction ("check ... at frame N+2 or N+3 ... if yes,
  keep it"). If neither K=2 nor K=3 has a frame available within
  --max-lookahead-gap-seconds (e.g. the primary pair is near the end of a
  burst), the blob cannot be confirmed and is discarded - noted explicitly
  in the CSV (`lookahead_checks_attempted` = 0) rather than silently treated
  the same as an actively-failed check.

Exact thresholds (every one stated here, per the task's instruction not to
bury them in a function body):
  - POSITION_TOLERANCE_FRAC = 0.04: two blobs (from two different diff
    computations) are considered "the same location" if their centroids are
    within 4% of the frame diagonal of each other. At 4K (diagonal ~4406px)
    this is ~176px - loose enough to absorb the few-pixel jitter contour
    dilation introduces between two independent diff computations of the
    same object, tight enough that a different object elsewhere in a large
    frame will not accidentally match.
  - LOOKAHEAD_STEPS = (2, 3): matches the task's "frame N+2 or N+3"
    instruction exactly.
  - OVERLAP_FRAC_THRESHOLD = 0.5 (unchanged from the previous version): a
    lookahead blob is still treated as person-covered, and that K rejected,
    if >=50% of ITS OWN area is covered by the union of person boxes at that
    later frame.
  - --max-lookahead-gap-seconds (default 40.0): a K=2/K=3 lookahead is only
    attempted if the elapsed time from the reference frame is within this
    budget. Chosen as roughly 3x the default --max-pair-gap-seconds (15s),
    which comfortably covers K=3 on both the ~1s-apart changetest burst and
    the ~3s-apart labelled bursts, while still excluding a lookahead frame
    that is minutes away and would make "same location, still there" claims
    meaningless.
  - STABILITY_MAX_CHANGE_FRAC = 0.3: see "Location-only persistence is not
    enough" below - a second, necessary gate added after the first
    implementation was measured and found to under-filter floor noise.

Location-only persistence is NOT enough - a real measured failure, fixed
before this became the shipped version. The first implementation of this
script matched blobs by LOCATION only (does a blob still show up near the
same spot when comparing frame[N-1] to frame[N-1+K]). Measured directly on
changetest's first pair (20260811-021847 -> 021848, a period with nobody in
frame and nothing placed - a true negative control), that version kept 57 of
69 raw blobs as "persisted," every one of them floor grout, whiteboard-line
glare or jacket-plastic-wrap sheen (`cv/measurements/verify_20260811-021847
..._021848....jpg`, checked by eye). The reason: grout seams and whiteboard
lines are the highest-local-contrast edges in the frame, so ANY tiny
lighting/exposure/JPEG-recompression difference between two exposures lights
them up as a diff blob, at THE SAME location, every single time one exposure
is compared against another - because the location is fixed by the tile
grout pattern, not by anything that moved. Location persistence alone cannot
tell "the same object is still sitting there" apart from "the same physical
edge relights slightly differently on every comparison."

The fix: require the candidate region to also be VISUALLY STABLE between
frame N (when it was first seen) and frame N+K (the lookahead frame) - i.e.
crop frame[N] and frame[N+K] to the candidate's own bounding box (valid
because the camera is static - the two crops are the same physical patch)
and re-run the same grayscale-blur-absdiff-threshold pipeline restricted to
that crop. A genuinely placed, motionless object looks nearly the same in
both crops (low changed-pixel fraction). A relighting grout line or glare
line does NOT look the same from one exposure to the next (that is
literally what "flicker" means) - the crop-to-crop diff stays high. A blob
must pass BOTH the location check (still differs from the pre-placement
background, roughly the same place) AND the stability check (looks like
itself two frames later) to be confirmed. Re-measured after this fix: same
static-scene control pair (021847->021848) drops from 57/69 kept to a
number reported in the write-up alongside real precision - see the results
document for what actually held up.

What is UNCHANGED from the previous version (see its own rationale, still
accurate): the classical diff pipeline itself (blur + absdiff + threshold 25
+ dilate + contour + area-fraction filter 0.0003), person detection via
yolo26l.pt at imgsz 640/conf 0.25, and ground truth / IoU>=0.5 matching
reused from evaluate_home_frames.py (import only, never edited).

Honest limitation this script still cannot fix: home-60cm / room1-60cm /
room1-190cm remain static-scene labelling bursts. Persistence can only help
where an object's SIZE and NEW-ness are true (something appearing/moving
between two of the sampled moments); it cannot invent a diff where a
labelled object was equally present, unmoving, in every sampled frame.

Scope:
  - No training, no fine-tuning.
  - Does not modify camera.py, detect_stream.py, measure_detection.py,
    measure_segmentation.py, or evaluate_home_frames.py (imported from,
    never edited).
  - Never writes into cv/captures/. Crops go to --crop-dir (default
    cv/measurements/, gitignored).
  - Nothing leaves this machine.

Usage:
    python measure_change_detection.py
    python measure_change_detection.py --session-tag changetest
    python measure_change_detection.py --max-lookahead-gap-seconds 60
"""

import argparse
import csv
import math
import os
import re
import statistics
import time
from datetime import datetime
from glob import glob
from pathlib import Path

# Same reasoning as detect_stream.py / measure_detection.py: must be set
# before torch is imported anywhere (including transitively via
# ultralytics), or an unsupported MPS op hard-crashes instead of falling
# back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from evaluate_home_frames import CLASS_NAMES, iou, load_ground_truth

MODELS_DIR = Path(__file__).resolve().parent / "models"
CAPTURES_DIR = Path(__file__).resolve().parent / "captures"
MEASUREMENTS_DIR = Path(__file__).resolve().parent / "measurements"

DEFAULT_IMAGES_GLOB = str(CAPTURES_DIR / "*_raw.jpg")

# Change-detection constants - see module docstring for why each value was
# picked. Kept as named constants (not buried in function bodies) so a
# reader can see and challenge every threshold in one place, same convention
# as measure_detection.py's DEFAULT_CONF comment.
BLUR_KERNEL = (5, 5)
DIFF_THRESHOLD = 25
DILATE_KERNEL = np.ones((5, 5), np.uint8)
DILATE_ITERATIONS = 2
MIN_AREA_FRAC = 0.0003

PERSON_MODEL = "yolo26l.pt"
PERSON_IMGSZ = 640
PERSON_CONF = 0.25
PERSON_CLASS_NAME = "person"
OVERLAP_FRAC_THRESHOLD = 0.5

MAX_PAIR_GAP_SECONDS = 15.0

# Persistence-check constants - see module docstring "Exact thresholds"
# section for the full reasoning behind each of these.
LOOKAHEAD_STEPS = (2, 3)
POSITION_TOLERANCE_FRAC = 0.04
STABILITY_MAX_CHANGE_FRAC = 0.3
DEFAULT_MAX_LOOKAHEAD_GAP_SECONDS = 40.0

# Filename convention written by detect_stream.py's save_snapshot():
# "{ts}_{optional-session-tag}_{model}_imgsz{N}_raw.jpg". The session tag is
# optional (session 1 / early session 2 captures predate the tagging
# convention) - non-greedy tag group tries to match nothing first, and only
# backtracks into consuming a tag if the model+imgsz suffix doesn't match
# without it.
FILENAME_RE = re.compile(
    r"^(?P<ts>\d{8}-\d{6})_(?:(?P<tag>.+?)_)?"
    r"(?P<model>yolo\S+?)_imgsz(?P<imgsz>\d+)_raw$"
)

CSV_FIELDNAMES = [
    "session_tag",
    "frame_a",
    "frame_b",
    "gap_seconds",
    "blob_x1",
    "blob_y1",
    "blob_x2",
    "blob_y2",
    "blob_area_px",
    "blob_area_frac",
    "person_overlap_frac",
    "suppressed_as_person_single_pair",
    "lookahead_checks_attempted",
    "persisted_frames",
    "persisted_confirmed",
    "first_seen_frame",
    "matched_gt",
    "matched_gt_class",
    "match_iou",
    "diff_ms",
    "person_detect_ms",
]


def parse_filename(path: str):
    """Extract (timestamp, session_tag_or_None) from a *_raw.jpg filename.
    Returns None if the filename doesn't match the expected convention at
    all (caller should skip it loudly, not silently).
    """
    stem = Path(path).stem
    match = FILENAME_RE.match(stem)
    if not match:
        return None
    ts = datetime.strptime(match.group("ts"), "%Y%m%d-%H%M%S")
    tag = match.group("tag")
    return ts, tag


def group_frames_by_tag(pattern: str, session_tag_filter: str = None):
    """Group raw captures by session tag (untagged frames form their own
    'untagged' bucket), sort each group by timestamp. Returns
    (frames_by_tag, unparsed) where frames_by_tag is tag -> sorted
    [(ts, path), ...]. This is the shared sequence view both the primary
    pair loop and the persistence lookahead need - previously discover_pairs
    only exposed pairs, which is not enough context for "look 2-3 frames
    ahead in this same sequence."
    """
    by_tag = {}
    unparsed = []
    for path in sorted(glob(pattern)):
        parsed = parse_filename(path)
        if parsed is None:
            unparsed.append(path)
            continue
        ts, tag = parsed
        tag = tag or "untagged"
        by_tag.setdefault(tag, []).append((ts, path))

    for tag in by_tag:
        by_tag[tag].sort(key=lambda item: item[0])

    if session_tag_filter:
        by_tag = {tag: frames for tag, frames in by_tag.items() if tag == session_tag_filter}

    return by_tag, unparsed


def build_skip_report(frames_by_tag: dict, max_pair_gap_seconds: float) -> dict:
    """Same accounting the old discover_pairs() printed - how many adjacent
    frame pairs exist per tag vs. how many are within the gap budget kept as
    primary pairs - reported so a reader can see what's excluded and why,
    same as before.
    """
    report = {}
    for tag, frames in sorted(frames_by_tag.items()):
        total_adjacent = max(0, len(frames) - 1)
        kept = 0
        max_gap = 0.0
        for (ts_a, _), (ts_b, _) in zip(frames, frames[1:]):
            gap = (ts_b - ts_a).total_seconds()
            max_gap = max(max_gap, gap)
            if gap <= max_pair_gap_seconds:
                kept += 1
        report[tag] = {
            "total_adjacent": total_adjacent,
            "kept": kept,
            "max_gap": max_gap,
            "num_frames": len(frames),
        }
    return report


def resolve_device(requested: str) -> str:
    """Same fallback as measure_detection.py."""
    if requested == "mps" and not torch.backends.mps.is_available():
        print(
            "Warning: --device mps requested but MPS is not available on "
            "this machine. Falling back to CPU."
        )
        return "cpu"
    return requested


def load_person_model(device: str) -> YOLO:
    weights_path = MODELS_DIR / PERSON_MODEL
    if not weights_path.exists():
        raise FileNotFoundError(
            f"{weights_path} not found. This script expects the model "
            "already downloaded (per the task: GPU is free, weights already "
            "in cv/models/)."
        )
    model = YOLO(str(weights_path))
    model.to(device)
    return model


def detect_person_boxes(model: YOLO, frame, device: str):
    """Return a list of (x1, y1, x2, y2) person boxes at PERSON_IMGSZ/CONF."""
    results = model.predict(
        frame, conf=PERSON_CONF, imgsz=PERSON_IMGSZ, device=device, verbose=False
    )
    boxes = results[0].boxes
    names = results[0].names
    out = []
    for box in boxes:
        class_id = int(box.cls[0])
        if names[class_id] != PERSON_CLASS_NAME:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        out.append((x1, y1, x2, y2))
    return out


def blob_person_overlap_frac(blob, person_boxes):
    """Fraction of the blob's own area covered by the UNION of person boxes.
    Approximated as the max single-box coverage rather than a true union
    (person boxes rarely overlap each other in these single-child captures,
    and exact polygon union isn't worth the complexity for a suppression
    heuristic) - documented here so it isn't mistaken for exact union math.
    """
    bx1, by1, bx2, by2 = blob
    blob_area = max(0.0, (bx2 - bx1) * (by2 - by1))
    if blob_area <= 0 or not person_boxes:
        return 0.0
    best = 0.0
    for (px1, py1, px2, py2) in person_boxes:
        ix1, iy1 = max(bx1, px1), max(by1, py1)
        ix2, iy2 = min(bx2, px2), min(by2, py2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        best = max(best, inter / blob_area)
    return best


def detect_change_blobs(frame_a, frame_b):
    """Classical frame differencing - see module docstring for each step's
    rationale. Returns (blobs, diff_ms) where blobs is a list of
    (x1, y1, x2, y2, area_px) in frame_b's pixel space and diff_ms is the
    wall-clock cost of JUST this function (no person model).
    """
    start = time.perf_counter()

    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    blur_a = cv2.GaussianBlur(gray_a, BLUR_KERNEL, 0)
    blur_b = cv2.GaussianBlur(gray_b, BLUR_KERNEL, 0)

    diff = cv2.absdiff(blur_a, blur_b)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, DILATE_KERNEL, iterations=DILATE_ITERATIONS)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_h, frame_w = frame_b.shape[:2]
    frame_area = float(frame_w * frame_h)
    min_area_px = MIN_AREA_FRAC * frame_area

    blobs = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_px = w * h
        if area_px < min_area_px:
            continue
        blobs.append((float(x), float(y), float(x + w), float(y + h), float(area_px)))

    diff_ms = (time.perf_counter() - start) * 1000.0
    return blobs, diff_ms


def blob_centroid(blob):
    x1, y1, x2, y2 = blob[:4]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def find_matching_blob(candidate_blob, blob_list, frame_diag, tolerance_frac=POSITION_TOLERANCE_FRAC):
    """Return the blob in blob_list whose centroid is closest to
    candidate_blob's centroid, IF that distance is within
    tolerance_frac * frame_diag - else None. See module docstring
    "POSITION_TOLERANCE_FRAC" for why centroid distance (not IoU) is the
    match criterion: two independent diff computations of the same object
    a few frames apart can produce a slightly different box shape (contour
    dilation is not pixel-deterministic across different neighbouring
    content), but the object's centroid does not move if the object itself
    hasn't moved.
    """
    cand_cx, cand_cy = blob_centroid(candidate_blob)
    max_dist = tolerance_frac * frame_diag
    best_blob, best_dist = None, None
    for blob in blob_list:
        cx, cy = blob_centroid(blob)
        dist = math.hypot(cx - cand_cx, cy - cand_cy)
        if dist <= max_dist and (best_dist is None or dist < best_dist):
            best_blob, best_dist = blob, dist
    return best_blob


def region_change_frac(frame_x, frame_y, bbox):
    """Fraction of pixels inside bbox that changed (>= DIFF_THRESHOLD) between
    frame_x and frame_y, restricted to that crop. Valid only because the
    camera is static within one session - the same bbox coordinates address
    the same physical patch in both frames. Returns 1.0 (maximally unstable)
    for a degenerate/empty crop rather than silently treating "no pixels" as
    "stable," which would wrongly pass edge-of-frame blobs.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    h_x, w_x = frame_x.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w_x, x2), min(h_x, y2)
    if x2c <= x1c or y2c <= y1c:
        return 1.0
    crop_x = frame_x[y1c:y2c, x1c:x2c]
    h_y, w_y = frame_y.shape[:2]
    x2y, y2y = min(w_y, x2c), min(h_y, y2c)
    if x2y <= x1c or y2y <= y1c:
        return 1.0
    crop_y = frame_y[y1c:y2y, x1c:x2y]
    if crop_x.shape != crop_y.shape:
        crop_y = cv2.resize(crop_y, (crop_x.shape[1], crop_x.shape[0]))
    gray_x = cv2.GaussianBlur(cv2.cvtColor(crop_x, cv2.COLOR_BGR2GRAY), BLUR_KERNEL, 0)
    gray_y = cv2.GaussianBlur(cv2.cvtColor(crop_y, cv2.COLOR_BGR2GRAY), BLUR_KERNEL, 0)
    diff = cv2.absdiff(gray_x, gray_y)
    total = diff.size
    if total == 0:
        return 1.0
    changed = int((diff >= DIFF_THRESHOLD).sum())
    return changed / total


class PersistenceChecker:
    """Owns the caches (decoded images, per-frame person boxes, per-(a,b)
    lookahead diff results) so persistence checks for every blob in one
    primary pair reuse the same lookahead diff computation and the same
    person-detection call, instead of recomputing per blob. Also
    accumulates real timing so the script can report honest, cache-aware
    costs rather than assuming caching away.
    """

    def __init__(self, person_model, device, max_lookahead_gap_seconds):
        self.person_model = person_model
        self.device = device
        self.max_lookahead_gap_seconds = max_lookahead_gap_seconds
        self.image_cache = {}
        self.person_boxes_cache = {}
        self.lookahead_diff_cache = {}
        self.person_call_ms = []
        self.person_cache_hits = 0
        self.lookahead_diff_ms = []
        self.lookahead_cache_hits = 0

    def get_image(self, path):
        if path not in self.image_cache:
            self.image_cache[path] = cv2.imread(path)
        return self.image_cache[path]

    def get_person_boxes(self, path):
        if path in self.person_boxes_cache:
            self.person_cache_hits += 1
            return self.person_boxes_cache[path]
        frame = self.get_image(path)
        start = time.perf_counter()
        boxes = detect_person_boxes(self.person_model, frame, self.device)
        ms = (time.perf_counter() - start) * 1000.0
        self.person_call_ms.append(ms)
        self.person_boxes_cache[path] = boxes
        return boxes

    def get_lookahead_blobs(self, ref_path, ref_frame, path_k):
        key = (ref_path, path_k)
        if key in self.lookahead_diff_cache:
            self.lookahead_cache_hits += 1
            return self.lookahead_diff_cache[key]
        frame_k = self.get_image(path_k)
        if frame_k is None or ref_frame is None:
            self.lookahead_diff_cache[key] = []
            return []
        if ref_frame.shape != frame_k.shape:
            frame_k_resized = cv2.resize(frame_k, (ref_frame.shape[1], ref_frame.shape[0]))
        else:
            frame_k_resized = frame_k
        blobs, ms = detect_change_blobs(ref_frame, frame_k_resized)
        self.lookahead_diff_ms.append(ms)
        self.lookahead_diff_cache[key] = blobs
        return blobs

    def check(self, ref_ts, ref_path, ref_frame, frame_at_candidate, candidate_blob,
              frames, ref_index, frame_diag):
        """Run the K=2 / K=3 lookahead checks for one candidate blob. Each K
        must pass THREE gates - see module docstring "Location-only
        persistence is NOT enough" for why the third gate exists:
          1. Location: a blob still shows up near the same place when
             comparing the pre-change reference frame to frame N+K.
          2. Not person-covered: that later blob isn't mostly covered by a
             person box at frame N+K.
          3. Stability: the candidate's own bbox, cropped from frame N and
             from frame N+K, looks like itself (low internal change) -
             rules out relighting edges/grout seams that satisfy gate 1 on
             location alone but never look visually stable frame to frame.
        Returns dict with:
          attempted: number of K in LOOKAHEAD_STEPS whose lookahead frame
            existed and was within the gap budget.
          passed: number of those K that passed all three gates.
          confirmed: True iff passed > 0 and attempted > 0 (OR semantics,
            per the task's literal "at frame N+2 OR N+3").
        """
        attempted = 0
        passed = 0
        for k in LOOKAHEAD_STEPS:
            idx_k = ref_index + k
            if idx_k >= len(frames):
                continue
            ts_k, path_k = frames[idx_k]
            gap = (ts_k - ref_ts).total_seconds()
            if gap > self.max_lookahead_gap_seconds:
                continue
            attempted += 1
            lookahead_blobs = self.get_lookahead_blobs(ref_path, ref_frame, path_k)
            matched = find_matching_blob(candidate_blob, lookahead_blobs, frame_diag)
            if matched is None:
                continue
            person_boxes_k = self.get_person_boxes(path_k)
            overlap = blob_person_overlap_frac(matched[:4], person_boxes_k)
            if overlap >= OVERLAP_FRAC_THRESHOLD:
                continue
            frame_k = self.get_image(path_k)
            if frame_k is None:
                continue
            stability = region_change_frac(frame_at_candidate, frame_k, candidate_blob)
            if stability <= STABILITY_MAX_CHANGE_FRAC:
                passed += 1
        return {
            "attempted": attempted,
            "passed": passed,
            "confirmed": attempted > 0 and passed > 0,
        }


def match_against_ground_truth(blob_boxes_confirmed, ground_truth):
    """Greedy IoU>=0.5 match, reusing evaluate_home_frames.iou(). Only
    persistence-confirmed blobs are matched - matching a discarded blob
    would overstate what the detector actually outputs. Returns a dict
    blob_index -> (matched_bool, matched_class_id_or_None, best_iou).
    """
    results = {}
    matched_gt = set()
    for idx, blob in blob_boxes_confirmed:
        best_iou, best_gt_idx, best_class = 0.0, None, None
        blob_box = {"x1": blob[0], "y1": blob[1], "x2": blob[2], "y2": blob[3]}
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            candidate_iou = iou(blob_box, gt)
            if candidate_iou > best_iou:
                best_iou, best_gt_idx, best_class = candidate_iou, gt_idx, gt["class_id"]
        if best_gt_idx is not None and best_iou >= 0.5:
            matched_gt.add(best_gt_idx)
            results[idx] = (True, best_class, round(best_iou, 3))
        else:
            results[idx] = (False, None, round(best_iou, 3))
    return results


def label_path_for(image_path: str) -> Path:
    return Path(image_path).with_suffix(".txt")


def measure(frames_by_tag, max_pair_gap_seconds, checker: PersistenceChecker,
            crop_dir: Path, verify_frames: set):
    rows = []
    recall_stats = {}  # tag -> {"gt_total": n, "gt_found": n}
    total_diff_ms = []  # primary-pair diffs only, same metric as before

    for tag, frames in sorted(frames_by_tag.items()):
        n = len(frames)
        for i in range(1, n):
            ts_a, path_a = frames[i - 1]
            ts_b, path_b = frames[i]
            gap = (ts_b - ts_a).total_seconds()
            if gap > max_pair_gap_seconds:
                continue

            frame_a = checker.get_image(path_a)
            frame_b = checker.get_image(path_b)
            if frame_a is None or frame_b is None:
                print(f"  Warning: could not read {path_a} or {path_b}, skipping pair.")
                continue
            if frame_a.shape != frame_b.shape:
                frame_a_resized = cv2.resize(frame_a, (frame_b.shape[1], frame_b.shape[0]))
            else:
                frame_a_resized = frame_a

            blobs, diff_ms = detect_change_blobs(frame_a_resized, frame_b)
            total_diff_ms.append(diff_ms)

            person_boxes_a = checker.get_person_boxes(path_a)
            person_boxes_b = checker.get_person_boxes(path_b)
            person_boxes_union = person_boxes_a + person_boxes_b

            frame_h, frame_w = frame_b.shape[:2]
            frame_diag = math.hypot(frame_w, frame_h)

            label_path = label_path_for(path_b)
            ground_truth = []
            if label_path.exists():
                ground_truth = load_ground_truth(label_path, frame_w, frame_h)

            confirmed_indexed = []
            persistence_results = {}
            overlap_fracs = {}
            for idx, blob in enumerate(blobs):
                overlap_frac = blob_person_overlap_frac(blob[:4], person_boxes_union)
                overlap_fracs[idx] = overlap_frac

                persistence = checker.check(
                    ref_ts=ts_a, ref_path=path_a, ref_frame=frame_a_resized,
                    frame_at_candidate=frame_b, candidate_blob=blob[:4],
                    frames=frames, ref_index=i - 1, frame_diag=frame_diag,
                )
                persistence_results[idx] = persistence
                if persistence["confirmed"]:
                    confirmed_indexed.append((idx, blob[:4]))

            match_results = {}
            if ground_truth:
                match_results = match_against_ground_truth(confirmed_indexed, ground_truth)
                stats = recall_stats.setdefault(tag, {"gt_total": 0, "gt_found": 0})
                stats["gt_total"] += len(ground_truth)
                stats["gt_found"] += sum(1 for m in match_results.values() if m[0])

            for idx, blob in enumerate(blobs):
                matched, matched_class, match_iou = match_results.get(idx, (False, None, 0.0))
                persistence = persistence_results[idx]
                rows.append({
                    "session_tag": tag,
                    "frame_a": Path(path_a).name,
                    "frame_b": Path(path_b).name,
                    "gap_seconds": round(gap, 1),
                    "blob_x1": round(blob[0], 1),
                    "blob_y1": round(blob[1], 1),
                    "blob_x2": round(blob[2], 1),
                    "blob_y2": round(blob[3], 1),
                    "blob_area_px": round(blob[4], 1),
                    "blob_area_frac": round(blob[4] / (frame_h * frame_w), 6),
                    "person_overlap_frac": round(overlap_fracs[idx], 3),
                    "suppressed_as_person_single_pair": overlap_fracs[idx] >= OVERLAP_FRAC_THRESHOLD,
                    "lookahead_checks_attempted": persistence["attempted"],
                    "persisted_frames": persistence["passed"],
                    "persisted_confirmed": persistence["confirmed"],
                    "first_seen_frame": Path(path_a).name,
                    "matched_gt": matched,
                    "matched_gt_class": CLASS_NAMES[matched_class] if matched_class is not None else "",
                    "match_iou": match_iou,
                    "diff_ms": round(diff_ms, 3),
                    "person_detect_ms": "",  # per-call cost reported in aggregate now; see checker
                })

            frame_b_stem = Path(path_b).stem
            if any(tag_str in frame_b_stem for tag_str in verify_frames):
                kept_flags = {idx: persistence_results[idx]["confirmed"] for idx in range(len(blobs))}
                save_verification_crop(frame_a, frame_b, blobs, kept_flags, path_a, path_b, crop_dir)

    return rows, recall_stats, total_diff_ms


def save_verification_crop(frame_a, frame_b, blobs, kept_flags, path_a, path_b, crop_dir: Path):
    """Draw every detected blob on frame_b (green = kept after persistence
    confirmation, red = discarded) and save it next to a copy of frame_a,
    side by side. Saved under crop_dir (default cv/measurements/, already
    gitignored - these are photographs of a real building).
    """
    crop_dir.mkdir(parents=True, exist_ok=True)
    annotated_b = frame_b.copy()
    for idx, blob in enumerate(blobs):
        x1, y1, x2, y2 = (int(v) for v in blob[:4])
        color = (0, 255, 0) if kept_flags.get(idx) else (0, 0, 255)
        cv2.rectangle(annotated_b, (x1, y1), (x2, y2), color, 3)

    h = min(frame_a.shape[0], annotated_b.shape[0])
    frame_a_resized = cv2.resize(frame_a, (int(frame_a.shape[1] * h / frame_a.shape[0]), h))
    annotated_b_resized = cv2.resize(annotated_b, (int(annotated_b.shape[1] * h / annotated_b.shape[0]), h))
    side_by_side = np.hstack([frame_a_resized, annotated_b_resized])

    out_name = f"verify_{Path(path_a).stem}__{Path(path_b).stem}.jpg"
    out_path = crop_dir / out_name
    cv2.imwrite(str(out_path), side_by_side)
    print(f"  Saved verification crop: {out_path} (green=persisted/kept, red=discarded)")


def write_csv(rows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} blob rows to {output_path}")


def print_summary(rows, recall_stats, total_diff_ms, checker: PersistenceChecker, skipped_report):
    print("\n=== Burst grouping / gap filtering (primary pairs) ===")
    print(f"{'session_tag':<20} {'frames':>7} {'adjacent':>9} {'pairs_kept':>11} {'max_gap_s':>10}")
    for tag, info in sorted(skipped_report.items()):
        print(
            f"{tag:<20} {info['num_frames']:>7} {info['total_adjacent']:>9} "
            f"{info['kept']:>11} {info['max_gap']:>10.1f}"
        )

    if not rows:
        print("\nNo change-blobs detected across any kept pair.")
    else:
        by_tag = {}
        for row in rows:
            by_tag.setdefault(row["session_tag"], []).append(row)

        print("\n=== Blobs per session (persisted/kept vs discarded) ===")
        print(f"{'session_tag':<20} {'total_blobs':>11} {'kept':>6} {'discarded':>10} {'no_lookahead':>13}")
        for tag, tag_rows in sorted(by_tag.items()):
            kept = sum(1 for r in tag_rows if r["persisted_confirmed"])
            no_lookahead = sum(1 for r in tag_rows if r["lookahead_checks_attempted"] == 0)
            print(f"{tag:<20} {len(tag_rows):>11} {kept:>6} {len(tag_rows) - kept:>10} {no_lookahead:>13}")

    print("\n=== Recall against labelled ground truth (IoU>=0.5, persistence-confirmed blobs only) ===")
    if not recall_stats:
        print("No pairs had a ground-truth label file for the newer frame.")
    else:
        print(f"{'session_tag':<20} {'gt_total':>9} {'gt_found':>9} {'recall':>8}")
        grand_total, grand_found = 0, 0
        for tag, stats in sorted(recall_stats.items()):
            recall = stats["gt_found"] / stats["gt_total"] if stats["gt_total"] else 0.0
            print(f"{tag:<20} {stats['gt_total']:>9} {stats['gt_found']:>9} {recall:>8.3f}")
            grand_total += stats["gt_total"]
            grand_found += stats["gt_found"]
        pooled_recall = grand_found / grand_total if grand_total else 0.0
        print(f"{'POOLED':<20} {grand_total:>9} {grand_found:>9} {pooled_recall:>8.3f}")

    if total_diff_ms:
        print("\n=== Timing (real, measured on this machine) ===")
        print(
            f"Primary-pair frame-diff:  mean {statistics.mean(total_diff_ms):.3f}ms  "
            f"median {statistics.median(total_diff_ms):.3f}ms  "
            f"(n={len(total_diff_ms)} primary pairs)"
        )
        if checker.lookahead_diff_ms:
            print(
                f"Lookahead frame-diff (K=2/K=3, cache-deduplicated): "
                f"mean {statistics.mean(checker.lookahead_diff_ms):.3f}ms  "
                f"(n={len(checker.lookahead_diff_ms)} unique lookahead diffs, "
                f"{checker.lookahead_cache_hits} cache hits avoided recompute)"
            )
        else:
            print("Lookahead frame-diff: none computed.")
        if checker.person_call_ms:
            print(
                f"Person-detection YOLO calls (cache-deduplicated): "
                f"mean {statistics.mean(checker.person_call_ms):.3f}ms  "
                f"(n={len(checker.person_call_ms)} unique frames detected, "
                f"{checker.person_cache_hits} cache hits avoided recompute)"
            )
        total_diff_all = sum(total_diff_ms) + sum(checker.lookahead_diff_ms)
        total_person_all = sum(checker.person_call_ms)
        n_primary = len(total_diff_ms)
        if n_primary:
            print(
                f"\nAmortized real cost per primary pair (all diff work + all "
                f"person-detection work / number of primary pairs, i.e. the "
                f"true incremental system cost once caching from a live "
                f"sequential frame stream is accounted for): "
                f"{(total_diff_all + total_person_all) / n_primary:.3f}ms"
            )
            print(
                "This is NOT '43ms/pair' from the old single-pair measurement - "
                "persistence tracking touches more frames per primary pair, so "
                "it costs more per pair even with caching. See the exact figure "
                "above and the write-up for the honest number."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-measure classical frame-differencing change detection "
        "with a persistence-based decision (frame N+2/N+3 lookahead against "
        "the pre-change reference frame) replacing single-pair person-overlap "
        "suppression, scored against labelled ground truth where available."
    )
    parser.add_argument("--images", type=str, default=DEFAULT_IMAGES_GLOB)
    parser.add_argument("--session-tag", type=str, default=None,
                         help="Only process this session tag (e.g. changetest). "
                         "Default: process every tag found, automatically.")
    parser.add_argument("--max-pair-gap-seconds", type=float, default=MAX_PAIR_GAP_SECONDS,
                         help=f"Consecutive frames further apart than this are not "
                         f"paired as a primary pair (default {MAX_PAIR_GAP_SECONDS}s).")
    parser.add_argument("--max-lookahead-gap-seconds", type=float,
                         default=DEFAULT_MAX_LOOKAHEAD_GAP_SECONDS,
                         help=f"A K=2/K=3 lookahead frame further than this from the "
                         f"reference frame is not used (default "
                         f"{DEFAULT_MAX_LOOKAHEAD_GAP_SECONDS}s).")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output", type=str, default=str(MEASUREMENTS_DIR / "change_detection.csv"))
    parser.add_argument("--crop-dir", type=str, default=str(MEASUREMENTS_DIR),
                         help="Where verification crops are saved. Gitignored by "
                         "default, same as the rest of cv/measurements/.")
    parser.add_argument("--verify-frames", type=str,
                         default="20260811-021855,20260811-021858,20260811-021900",
                         help="Comma-separated frame-timestamp substrings whose pair "
                         "gets an annotated before/after proof crop saved. Defaults "
                         "to the changetest scissors-drop sequence: mid-drop, "
                         "settled-with-person-nearby, settled-alone.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    frames_by_tag, unparsed = group_frames_by_tag(args.images, args.session_tag)
    if unparsed:
        print(
            f"Warning: {len(unparsed)} file(s) did not match the expected "
            f"filename convention and were skipped: {unparsed[:5]}"
            + (" ..." if len(unparsed) > 5 else "")
        )
    skipped_report = build_skip_report(frames_by_tag, args.max_pair_gap_seconds)
    total_pairs = sum(info["kept"] for info in skipped_report.values())
    print(f"\nFound {total_pairs} usable primary consecutive-frame pair(s) across "
          f"{len(frames_by_tag)} session group(s) (gap <= {args.max_pair_gap_seconds}s).")
    if not total_pairs:
        print("Nothing to measure.")
        return

    print(f"Loading {PERSON_MODEL} for person-suppression ...")
    person_model = load_person_model(device)
    print(f"{PERSON_MODEL} loaded.")

    verify_frames = {t.strip() for t in args.verify_frames.split(",") if t.strip()}
    checker = PersistenceChecker(person_model, device, args.max_lookahead_gap_seconds)

    rows, recall_stats, total_diff_ms = measure(
        frames_by_tag, args.max_pair_gap_seconds, checker, Path(args.crop_dir), verify_frames
    )
    write_csv(rows, Path(args.output))
    print_summary(rows, recall_stats, total_diff_ms, checker, skipped_report)


if __name__ == "__main__":
    main()
