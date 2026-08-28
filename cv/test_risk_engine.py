"""
test_risk_engine.py - headless self-test for risk_engine.py's pure logic.

No camera, no saved photos, no model weights required - this exercises the
arithmetic/data-structure layer against synthetic in-memory bounding boxes
and small synthetic in-memory images (plain numpy arrays built in-test,
never a real photograph or camera frame): distance normalization, rolling-
window smoothing, zone classification, person tracking/staleness,
--seed-hazard parsing, HazardMap's three-state model (propose/confirm/
dismiss), the periodic room scan's arrived/removed diffing (including the
two-consecutive-scans jitter guard and duplicate-candidate dedup), the
dismissal re-raise check, the live ReviewQueue, and CLAUDE.md decision 4's
proximity-alert eligibility table.

Phase 5 additions (same no-camera-required style): AlertManager's
exit-hysteresis state machine (open/escalate/close, per-(person,hazard)
dedup), should_trigger_clip's red-only/once-per-event/global-cooldown gate,
audio_for_signal/banner_text_for_signal's pure mapping, RollingBuffer's
bounded length and snapshot-is-a-copy behaviour, and ClipRecorder's tail-
window timing. write_clip() and AudioPlayer's actual afplay/VideoWriter I/O
are exercised for real (real cv2, real temp files, no mocking, matching this
suite's existing style) EXCEPT for a real audio device play - that would
make the suite noisy/slow on every run, so AudioPlayer's non-blocking
property is covered by direct measurement instead (see
docs/decision-log.md's 2026-08-22 Phase 5 entry), and only its pure
missing-file-handling logic is unit-tested here.

Plain asserts, no pytest dependency - run directly:

    python cv/test_risk_engine.py

Exits non-zero (via AssertionError propagating) on first failure, prints
"ALL TESTS PASSED" on success.
"""

import json
import math
import os
import tempfile
import threading
from collections import defaultdict, deque

import cv2
import numpy as np

import risk_engine as re


# --- geometry ----------------------------------------------------------------


def test_bbox_center():
    assert re.bbox_center((0, 0, 10, 20)) == (5.0, 10.0)
    assert re.bbox_center((10, 10, 10, 10)) == (10.0, 10.0)


def test_bbox_iou():
    assert math.isclose(re.bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
    assert re.bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # box A: 0-10 x 0-10 (area 100); box B: 5-15 x 0-10 (area 100)
    # intersection: 5-10 x 0-10 = area 50; union = 100+100-50 = 150
    assert math.isclose(re.bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)), 50 / 150)


def test_normalized_center_distance():
    frame_diagonal = math.hypot(100, 100)
    box_a = (0, 0, 0, 0)
    box_b = (100, 100, 100, 100)
    dist = re.normalized_center_distance(box_a, box_b, frame_diagonal)
    assert math.isclose(dist, 1.0)  # spans the full diagonal

    frame_diagonal_big = math.hypot(1000, 1000)
    dist_big_frame = re.normalized_center_distance(box_a, box_b, frame_diagonal_big)
    assert dist_big_frame < dist  # same real distance, bigger frame -> smaller normalized value

    assert re.normalized_center_distance(box_a, box_b, 0.0) == 0.0  # no div-by-zero


def test_classify_zone_thresholds():
    eps = 1e-9
    assert re.classify_zone(0.0) == "red"
    assert re.classify_zone(re.RISK_ZONE_RED_MAX - eps) == "red"
    assert re.classify_zone(re.RISK_ZONE_RED_MAX) == "orange"
    assert re.classify_zone(re.RISK_ZONE_ORANGE_MAX - eps) == "orange"
    assert re.classify_zone(re.RISK_ZONE_ORANGE_MAX) == "yellow"
    assert re.classify_zone(re.RISK_ZONE_YELLOW_MAX - eps) == "yellow"
    assert re.classify_zone(re.RISK_ZONE_YELLOW_MAX) == "none"
    assert re.classify_zone(1.0) == "none"


def test_zone_rank_ordering():
    assert re.zone_rank("none") < re.zone_rank("yellow")
    assert re.zone_rank("yellow") < re.zone_rank("orange")
    assert re.zone_rank("orange") < re.zone_rank("red")


def test_find_match_iou_and_center_distance_paths():
    frame_diagonal = math.hypot(1000, 1000)

    class Fake:
        def __init__(self, id, label, bbox):
            self.id = id
            self.label = label
            self.bbox = bbox

    entries = [Fake(1, "chair", (100, 100, 200, 200))]

    match = re.find_match((102, 102, 202, 202), entries, frame_diagonal, iou_threshold=0.3, center_dist_frac=0.0)
    assert match is not None and match.id == 1

    match = re.find_match((145, 145, 155, 155), entries, frame_diagonal, iou_threshold=0.9, center_dist_frac=0.05)
    assert match is not None and match.id == 1  # low IoU but close center

    match = re.find_match((900, 900, 950, 950), entries, frame_diagonal, iou_threshold=0.3, center_dist_frac=0.05)
    assert match is None  # far away, no match on either path

    match = re.find_match(
        (102, 102, 202, 202), entries, frame_diagonal, iou_threshold=0.3, center_dist_frac=0.05, label="oven_microwave"
    )
    assert match is None  # same geometry, wrong label


def test_dedupe_candidates_drops_overlapping_boxes():
    candidates = [
        (100, 100, 200, 200),
        (105, 103, 205, 203),  # near-duplicate of the first -> dropped
        (900, 900, 950, 950),  # unrelated -> kept
    ]
    kept = re.dedupe_candidates(candidates, iou_threshold=0.4)
    assert kept == [(100, 100, 200, 200), (900, 900, 950, 950)]


def test_dedupe_candidates_keeps_genuinely_separate_boxes():
    candidates = [(0, 0, 10, 10), (500, 500, 510, 510)]
    assert re.dedupe_candidates(candidates) == candidates


def test_person_coverage_frac():
    assert re.person_coverage_frac([], 1000, 1000) == 0.0
    # One person covering a quarter of the frame.
    frac = re.person_coverage_frac([(0, 0, 500, 500)], 1000, 1000)
    assert math.isclose(frac, 0.25)
    # Coverage is capped at 1.0 even if summed boxes exceed the frame area
    # (overlap is not corrected for - a documented simplification).
    frac_capped = re.person_coverage_frac([(0, 0, 1000, 1000), (0, 0, 1000, 1000)], 1000, 1000)
    assert frac_capped == 1.0


# --- dismissal fingerprint re-raise (Step 4) ---------------------------------


def _solid_frame(color, shape=(60, 60, 3)):
    frame = np.zeros(shape, dtype=np.uint8)
    frame[:, :] = color
    return frame


def test_fingerprint_changed_true_for_materially_different_region():
    fingerprint = _solid_frame((10, 10, 10))
    later_frame = _solid_frame((250, 250, 250))
    assert re.fingerprint_changed(fingerprint, later_frame, (0, 0, 60, 60)) is True


def test_fingerprint_changed_false_when_region_is_stable():
    fingerprint = _solid_frame((10, 10, 10))
    later_frame = _solid_frame((10, 10, 10))
    assert re.fingerprint_changed(fingerprint, later_frame, (0, 0, 60, 60)) is False


def test_fingerprint_changed_true_when_fingerprint_missing():
    # Err toward re-asking rather than trusting a comparison that couldn't
    # actually run (Shaked's "better safe than sorry" ruling).
    later_frame = _solid_frame((10, 10, 10))
    assert re.fingerprint_changed(None, later_frame, (0, 0, 60, 60)) is True


# --- HazardMap: named/seed proposals -----------------------------------------


def test_hazard_map_add_seed_is_confirmed_immediately():
    hm = re.HazardMap()
    entry = hm.add_seed((10, 10, 50, 50), "stove")
    assert len(hm.entries) == 1
    assert entry.origin == re.HAZARD_ORIGIN_SEED
    assert entry.state == re.HAZARD_STATE_CONFIRMED
    assert entry.label == "stove"


def test_hazard_map_propose_named_merges_same_object():
    frame_diagonal = math.hypot(1920, 1080)
    hm = re.HazardMap()

    e1, created1 = hm.propose_named((100, 100, 300, 300), "oven_microwave", frame_diagonal, is_first_scan=True)
    assert created1 is True
    assert e1.state == re.HAZARD_STATE_PENDING
    assert e1.origin == re.HAZARD_ORIGIN_NAMED
    assert len(hm.entries) == 1

    # Same physical object, slightly shifted next frame -> merges, not a
    # second entry - state is untouched by a re-detection.
    e2, created2 = hm.propose_named((105, 103, 305, 303), "oven_microwave", frame_diagonal, is_first_scan=True)
    assert created2 is False
    assert e2.id == e1.id
    assert e2.hits == 2
    assert e2.bbox == (105, 103, 305, 303)
    assert len(hm.entries) == 1

    # Far away, same label -> different physical object -> new entry.
    e3, created3 = hm.propose_named((1500, 800, 1700, 1000), "oven_microwave", frame_diagonal, is_first_scan=False)
    assert created3 is True
    assert e3.id != e1.id
    assert len(hm.entries) == 2

    # Same location, different label -> new entry (label-scoped matching).
    e4, created4 = hm.propose_named((100, 100, 300, 300), "refrigerator", frame_diagonal, is_first_scan=True)
    assert created4 is True
    assert len(hm.entries) == 3


def test_hazard_map_propose_named_confirmed_state_untouched_by_match():
    """A human-confirmed named entry re-detected next frame stays CONFIRMED
    (position refresh only) - propose_named never downgrades state.
    """
    frame_diagonal = math.hypot(1920, 1080)
    hm = re.HazardMap()
    e1, _ = hm.propose_named((0, 0, 100, 100), "refrigerator", frame_diagonal, is_first_scan=True)
    hm.confirm(e1.id)
    e2, created = hm.propose_named((2, 1, 102, 101), "refrigerator", frame_diagonal, is_first_scan=True)
    assert created is False
    assert e2.id == e1.id
    assert e2.state == re.HAZARD_STATE_CONFIRMED


# --- HazardMap: periodic room scan diffing (Step 3) --------------------------


def test_apply_scan_candidates_two_consecutive_scans_required_to_arrive():
    frame_diagonal = math.hypot(1920, 1080)
    frame = _solid_frame((100, 100, 100))
    hm = re.HazardMap()

    # First-ever sighting: not yet promoted to a real entry.
    diff1 = hm.apply_scan_candidates([(100, 100, 200, 200)], frame, frame_diagonal, is_first_scan_cycle=True)
    assert diff1.arrived == []
    assert len(hm.entries) == 0

    # Same spot again, next scan -> now it counts as "arrived."
    diff2 = hm.apply_scan_candidates([(102, 101, 202, 201)], frame, frame_diagonal, is_first_scan_cycle=False)
    assert len(diff2.arrived) == 1
    assert len(hm.entries) == 1
    assert hm.entries[0].origin == re.HAZARD_ORIGIN_SCAN
    assert hm.entries[0].state == re.HAZARD_STATE_PENDING
    # The is_first_scan flag reflects whether the SPOT was already present
    # at scan cycle 0, not which cycle actually created the entry.
    assert hm.entries[0].is_first_scan is True


def test_apply_scan_candidates_single_sighting_does_not_arrive_and_does_not_repeat():
    """A candidate seen once, then never again, must not be promoted - and
    must not silently linger forever as a phantom provisional record.
    """
    frame_diagonal = math.hypot(1920, 1080)
    frame = _solid_frame((100, 100, 100))
    hm = re.HazardMap()

    hm.apply_scan_candidates([(100, 100, 200, 200)], frame, frame_diagonal, is_first_scan_cycle=True)
    # A totally different spot the next cycle - the first sighting is noise
    # and must not be carried forward.
    diff2 = hm.apply_scan_candidates([(900, 900, 950, 950)], frame, frame_diagonal, is_first_scan_cycle=False)
    assert diff2.arrived == []
    diff3 = hm.apply_scan_candidates([(100, 100, 200, 200)], frame, frame_diagonal, is_first_scan_cycle=False)
    # The ORIGINAL spot reappearing on cycle 3 must be treated as a fresh
    # first sighting (cycle 1 didn't carry it forward), not an instant arrival.
    assert diff3.arrived == []


def test_apply_scan_candidates_arrived_later_is_not_first_scan():
    frame_diagonal = math.hypot(1920, 1080)
    frame = _solid_frame((100, 100, 100))
    hm = re.HazardMap()

    # Nothing in the first scan.
    hm.apply_scan_candidates([], frame, frame_diagonal, is_first_scan_cycle=True)
    hm.apply_scan_candidates([(300, 300, 400, 400)], frame, frame_diagonal, is_first_scan_cycle=False)
    diff = hm.apply_scan_candidates([(302, 301, 402, 401)], frame, frame_diagonal, is_first_scan_cycle=False)
    assert len(diff.arrived) == 1
    assert diff.arrived[0].is_first_scan is False


def test_apply_scan_candidates_removed_after_two_consecutive_absences():
    frame_diagonal = math.hypot(1920, 1080)
    frame = _solid_frame((100, 100, 100))
    hm = re.HazardMap()

    hm.apply_scan_candidates([(100, 100, 200, 200)], frame, frame_diagonal, is_first_scan_cycle=True)
    hm.apply_scan_candidates([(100, 100, 200, 200)], frame, frame_diagonal, is_first_scan_cycle=False)
    assert len(hm.entries) == 1  # arrived

    # One missed scan is not enough to remove it.
    diff_miss1 = hm.apply_scan_candidates([], frame, frame_diagonal, is_first_scan_cycle=False)
    assert diff_miss1.removed == []
    assert len(hm.entries) == 1

    # A second consecutive miss clears it.
    diff_miss2 = hm.apply_scan_candidates([], frame, frame_diagonal, is_first_scan_cycle=False)
    assert len(diff_miss2.removed) == 1
    assert len(hm.entries) == 0


def test_apply_scan_candidates_reappearing_before_second_miss_resets_absence():
    frame_diagonal = math.hypot(1920, 1080)
    frame = _solid_frame((100, 100, 100))
    hm = re.HazardMap()
    hm.apply_scan_candidates([(100, 100, 200, 200)], frame, frame_diagonal, is_first_scan_cycle=True)
    hm.apply_scan_candidates([(100, 100, 200, 200)], frame, frame_diagonal, is_first_scan_cycle=False)

    hm.apply_scan_candidates([], frame, frame_diagonal, is_first_scan_cycle=False)  # 1 miss
    diff = hm.apply_scan_candidates([(100, 100, 200, 200)], frame, frame_diagonal, is_first_scan_cycle=False)  # reappears
    assert diff.removed == []
    assert len(hm.entries) == 1
    assert hm.entries[0].absent_scans == 0

    # Now it can survive another single miss without being removed.
    diff_after = hm.apply_scan_candidates([], frame, frame_diagonal, is_first_scan_cycle=False)
    assert diff_after.removed == []
    assert len(hm.entries) == 1


def test_apply_scan_candidates_duplicate_within_one_pass_is_deduped():
    frame_diagonal = math.hypot(1920, 1080)
    frame = _solid_frame((100, 100, 100))
    hm = re.HazardMap()

    # Two overlapping boxes for the same physical object, in the SAME scan
    # pass, on two consecutive cycles - must resolve to exactly one entry,
    # never two.
    hm.apply_scan_candidates(
        [(100, 100, 200, 200), (103, 101, 202, 199)], frame, frame_diagonal, is_first_scan_cycle=True
    )
    diff = hm.apply_scan_candidates(
        [(101, 100, 201, 200), (104, 102, 203, 198)], frame, frame_diagonal, is_first_scan_cycle=False
    )
    assert len(diff.arrived) == 1
    assert len(hm.entries) == 1


def test_apply_scan_candidates_dismissed_entry_reraised_after_two_consecutive_changed_scans():
    # Updated 2026-08-26: a single changed scan is no longer sufficient -
    # see DISMISS_REAPPEAR_CONSECUTIVE_SCANS_REQUIRED.
    frame_diagonal = math.hypot(1920, 1080)
    quiet_frame = _solid_frame((100, 100, 100))
    hm = re.HazardMap()

    hm.apply_scan_candidates([(0, 0, 60, 60)], quiet_frame, frame_diagonal, is_first_scan_cycle=True)
    hm.apply_scan_candidates([(0, 0, 60, 60)], quiet_frame, frame_diagonal, is_first_scan_cycle=False)
    entry = hm.entries[0]
    hm.dismiss(entry.id, quiet_frame)
    assert entry.state == re.HAZARD_STATE_DISMISSED
    assert entry.fingerprint is not None

    # A later scan finds the SAME spot still occupied but visually
    # different (e.g. a new object placed where the dismissed one was).
    # First changed scan: bumps the counter, does NOT re-raise yet.
    changed_frame = _solid_frame((250, 250, 250))
    diff = hm.apply_scan_candidates([(0, 0, 60, 60)], changed_frame, frame_diagonal, is_first_scan_cycle=False)
    assert diff.reraised == []
    assert entry.state == re.HAZARD_STATE_DISMISSED
    assert entry.changed_scans == 1

    # Second CONSECUTIVE changed scan: now it re-raises.
    diff = hm.apply_scan_candidates([(0, 0, 60, 60)], changed_frame, frame_diagonal, is_first_scan_cycle=False)
    assert len(diff.reraised) == 1
    assert entry.state == re.HAZARD_STATE_PENDING
    assert entry.changed_scans == 0
    # A re-raise is never treated as part of the original room state, even
    # if the FIRST dismissal happened to be from the first scan.
    assert entry.is_first_scan is False


def test_apply_scan_candidates_dismissed_entry_stays_dismissed_if_unchanged():
    frame_diagonal = math.hypot(1920, 1080)
    quiet_frame = _solid_frame((100, 100, 100))
    hm = re.HazardMap()

    hm.apply_scan_candidates([(0, 0, 60, 60)], quiet_frame, frame_diagonal, is_first_scan_cycle=True)
    hm.apply_scan_candidates([(0, 0, 60, 60)], quiet_frame, frame_diagonal, is_first_scan_cycle=False)
    entry = hm.entries[0]
    hm.dismiss(entry.id, quiet_frame)

    diff = hm.apply_scan_candidates([(0, 0, 60, 60)], quiet_frame, frame_diagonal, is_first_scan_cycle=False)
    assert diff.reraised == []
    assert entry.state == re.HAZARD_STATE_DISMISSED
    assert entry.changed_scans == 0


def test_apply_scan_candidates_dismissed_entry_non_consecutive_changes_never_reraise():
    # The exact live scenario this fix targets: scan-to-scan noise, not a
    # real persistent change. changed, unchanged, changed - never two IN A
    # ROW - must never re-raise, no matter how many total "changed" scans
    # accumulate over time.
    frame_diagonal = math.hypot(1920, 1080)
    quiet_frame = _solid_frame((100, 100, 100))
    changed_frame = _solid_frame((250, 250, 250))
    hm = re.HazardMap()

    hm.apply_scan_candidates([(0, 0, 60, 60)], quiet_frame, frame_diagonal, is_first_scan_cycle=True)
    hm.apply_scan_candidates([(0, 0, 60, 60)], quiet_frame, frame_diagonal, is_first_scan_cycle=False)
    entry = hm.entries[0]
    hm.dismiss(entry.id, quiet_frame)

    for frame in (changed_frame, quiet_frame, changed_frame, quiet_frame):
        diff = hm.apply_scan_candidates([(0, 0, 60, 60)], frame, frame_diagonal, is_first_scan_cycle=False)
        assert diff.reraised == []
        assert entry.state == re.HAZARD_STATE_DISMISSED


def test_dismiss_resets_changed_scans_counter():
    frame_diagonal = math.hypot(1920, 1080)
    quiet_frame = _solid_frame((100, 100, 100))
    changed_frame = _solid_frame((250, 250, 250))
    hm = re.HazardMap()

    hm.apply_scan_candidates([(0, 0, 60, 60)], quiet_frame, frame_diagonal, is_first_scan_cycle=True)
    hm.apply_scan_candidates([(0, 0, 60, 60)], quiet_frame, frame_diagonal, is_first_scan_cycle=False)
    entry = hm.entries[0]
    hm.dismiss(entry.id, quiet_frame)
    hm.apply_scan_candidates([(0, 0, 60, 60)], changed_frame, frame_diagonal, is_first_scan_cycle=False)
    assert entry.changed_scans == 1

    hm.confirm(entry.id)  # e.g. a human re-decides via some other path
    hm.dismiss(entry.id, quiet_frame)  # fresh dismiss - must not inherit the old counter
    assert entry.changed_scans == 0


def _stripe_frame(size=200, stripe_width=4):
    # Fine, high-contrast vertical stripes - not a flat color or a smooth
    # gradient, either of which turn out too shift-tolerant after
    # region_change_frac's Gaussian blur to actually reproduce the bug.
    # Stripes stand in for a real fine-textured object (a ribbed surface, a
    # cable) - directly verified (2026-08-26) that a 2-4px horizontal shift
    # of the crop region trips fingerprint_changed's OLD candidate-bbox
    # comparison on EVERY shift tried, matching the live evidence (a
    # dismissed entry re-raising on nearly every single scan, not
    # occasionally) - this is a faithful reproduction, not a guess.
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        val = 255 if (i // stripe_width) % 2 == 0 else 0
        frame[:, i] = (val, val, val)
    return frame


def test_apply_scan_candidates_dismissed_entry_box_jitter_alone_does_not_false_trigger():
    frame_diagonal = math.hypot(1920, 1080)
    stripes = _stripe_frame()
    hm = re.HazardMap()

    hm.apply_scan_candidates([(50, 50, 110, 110)], stripes, frame_diagonal, is_first_scan_cycle=True)
    hm.apply_scan_candidates([(50, 50, 110, 110)], stripes, frame_diagonal, is_first_scan_cycle=False)
    entry = hm.entries[0]
    hm.dismiss(entry.id, stripes)
    assert entry.fingerprint_bbox == (50, 50, 110, 110)

    # Later scans propose a slightly SHIFTED box each time (segmentation
    # jitter) over the SAME static, unmoved scene - nothing actually
    # changed. Comparing against each scan's own shifted box reads
    # "changed" every time on this pattern (verified directly); pinning to
    # fingerprint_bbox must not.
    for dx in (2, -3, 4, -2, 3, -4):
        jittered_bbox = (50 + dx, 50, 110 + dx, 110)
        diff = hm.apply_scan_candidates([jittered_bbox], stripes, frame_diagonal, is_first_scan_cycle=False)
        assert diff.reraised == [], f"false re-raise from box jitter alone (dx={dx})"
    assert entry.state == re.HAZARD_STATE_DISMISSED


def test_apply_scan_candidates_dismissed_entry_still_reraises_on_real_change_with_jittering_boxes():
    # The fix must not weaken true-positive detection: a GENUINE change
    # (not just box jitter) must still re-raise, even while the candidate
    # box also jitters scan to scan.
    frame_diagonal = math.hypot(1920, 1080)
    stripes = _stripe_frame()
    solid = _solid_frame((250, 250, 250), shape=(200, 200, 3))
    hm = re.HazardMap()

    hm.apply_scan_candidates([(50, 50, 110, 110)], stripes, frame_diagonal, is_first_scan_cycle=True)
    hm.apply_scan_candidates([(50, 50, 110, 110)], stripes, frame_diagonal, is_first_scan_cycle=False)
    entry = hm.entries[0]
    hm.dismiss(entry.id, stripes)

    diff = hm.apply_scan_candidates([(52, 50, 112, 110)], solid, frame_diagonal, is_first_scan_cycle=False)
    assert diff.reraised == []  # first changed scan - not yet 2 consecutive
    diff = hm.apply_scan_candidates([(48, 50, 108, 110)], solid, frame_diagonal, is_first_scan_cycle=False)
    assert len(diff.reraised) == 1
    assert entry.state == re.HAZARD_STATE_PENDING


# --- proximity-alert eligibility (Step 5 / CLAUDE.md decision 4) ------------


def _entry(state, is_first_scan, origin=re.HAZARD_ORIGIN_SCAN):
    return re.HazardEntry(id=1, label="object", bbox=(0, 0, 10, 10), origin=origin, state=state, is_first_scan=is_first_scan)


def test_hazard_alerts_on_approach_confirmed_always_alerts():
    assert re.hazard_alerts_on_approach(_entry(re.HAZARD_STATE_CONFIRMED, is_first_scan=True)) is True
    assert re.hazard_alerts_on_approach(_entry(re.HAZARD_STATE_CONFIRMED, is_first_scan=False)) is True


def test_hazard_alerts_on_approach_pending_first_scan_does_not_alert():
    assert re.hazard_alerts_on_approach(_entry(re.HAZARD_STATE_PENDING, is_first_scan=True)) is False


def test_hazard_alerts_on_approach_pending_arrived_later_does_alert():
    assert re.hazard_alerts_on_approach(_entry(re.HAZARD_STATE_PENDING, is_first_scan=False)) is True


def test_hazard_alerts_on_approach_dismissed_never_alerts():
    assert re.hazard_alerts_on_approach(_entry(re.HAZARD_STATE_DISMISSED, is_first_scan=False)) is False
    assert re.hazard_alerts_on_approach(_entry(re.HAZARD_STATE_DISMISSED, is_first_scan=True)) is False


# --- ReviewQueue --------------------------------------------------------------


def test_review_queue_fifo_confirm_advances():
    q = re.ReviewQueue()
    q.enqueue(1)
    q.enqueue(2)
    assert q.current_id() == 1
    q.advance()
    assert q.current_id() == 2
    q.advance()
    assert q.current_id() is None
    assert len(q) == 0


def test_review_queue_skip_remaining_clears_only_current_queue():
    q = re.ReviewQueue()
    q.enqueue(1)
    q.enqueue(2)
    q.skip_remaining()
    assert len(q) == 0
    assert q.current_id() is None
    # New arrivals after a skip still enqueue normally.
    q.enqueue(3)
    assert q.current_id() == 3


def test_review_queue_discard_removes_specific_id_anywhere_in_queue():
    q = re.ReviewQueue()
    q.enqueue(1)
    q.enqueue(2)
    q.enqueue(3)
    q.discard(2)
    assert list(q._ids) == [1, 3]
    q.discard(99)  # not present -> no-op, no error
    assert list(q._ids) == [1, 3]


# --- person tracking -----------------------------------------------------------


def test_person_tracker_persists_id_across_frames():
    frame_diagonal = math.hypot(1920, 1080)
    tracker = re.PersonTracker()

    live1 = tracker.update([(100, 100, 200, 400)], frame_diagonal)
    person_id = live1[0].id

    live2 = tracker.update([(110, 105, 210, 405)], frame_diagonal)
    assert live2[0].id == person_id

    live3 = tracker.update([(110, 105, 210, 405), (1500, 100, 1600, 400)], frame_diagonal)
    assert len(live3) == 2
    ids = {p.id for p in live3}
    assert person_id in ids
    assert len(ids) == 2


def test_person_tracker_drops_stale_entries():
    frame_diagonal = math.hypot(1920, 1080)
    tracker = re.PersonTracker()
    live = tracker.update([(100, 100, 200, 400)], frame_diagonal)
    person_id = live[0].id

    for entry in tracker.entries:
        if entry.id == person_id:
            entry.last_seen -= re.PERSON_STALE_SECONDS + 1.0

    live_after = tracker.update([], frame_diagonal)
    assert live_after == []
    assert all(e.id != person_id for e in tracker.entries)


# --- Layer B: rolling window + score_frame ------------------------------------


def test_rolling_window_smooths_flicker():
    frame_diagonal = math.hypot(1920, 1080)
    rolling_windows = defaultdict(lambda: deque(maxlen=re.ROLLING_WINDOW_SIZE))

    person = re.PersonEntry(id=1, bbox=(500, 500, 600, 700), last_seen=0.0)
    hazard = re.HazardEntry(
        id=1, label="chair", bbox=(500, 500, 600, 700), origin=re.HAZARD_ORIGIN_NAMED,
        state=re.HAZARD_STATE_CONFIRMED, last_seen=0.0,
    )
    for _ in range(re.ROLLING_WINDOW_SIZE + 3):
        per_person, frame_risk = re.score_frame([person], [hazard], frame_diagonal, rolling_windows)

    window = rolling_windows[(1, 1)]
    assert len(window) == re.ROLLING_WINDOW_SIZE
    hazard_match, smoothed, zone = per_person[1]
    assert hazard_match.id == 1
    assert math.isclose(smoothed, 0.0, abs_tol=1e-9)
    assert zone == "red"
    assert frame_risk["zone"] == "red"
    assert frame_risk["person_id"] == 1
    assert frame_risk["hazard_label"] == "chair"


def test_rolling_window_flicker_resistance():
    rolling_windows = defaultdict(lambda: deque(maxlen=re.ROLLING_WINDOW_SIZE))
    key = (1, 1)
    for _ in range(re.ROLLING_WINDOW_SIZE):
        rolling_windows[key].append(0.01)
    smoothed_before = sum(rolling_windows[key]) / len(rolling_windows[key])
    assert re.classify_zone(smoothed_before) == "red"

    rolling_windows[key].append(0.9)
    smoothed_after = sum(rolling_windows[key]) / len(rolling_windows[key])

    assert smoothed_after > smoothed_before
    assert smoothed_after < 0.9
    assert re.classify_zone(smoothed_after) != "none"


def test_score_frame_picks_nearest_hazard_per_person():
    frame_diagonal = math.hypot(1920, 1080)
    rolling_windows = defaultdict(lambda: deque(maxlen=re.ROLLING_WINDOW_SIZE))

    person = re.PersonEntry(id=1, bbox=(0, 0, 10, 10), last_seen=0.0)
    near_hazard = re.HazardEntry(
        id=1, label="chair", bbox=(20, 0, 30, 10), origin=re.HAZARD_ORIGIN_NAMED,
        state=re.HAZARD_STATE_CONFIRMED, last_seen=0.0,
    )
    far_hazard = re.HazardEntry(
        id=2, label="oven_microwave", bbox=(1900, 1070, 1920, 1080), origin=re.HAZARD_ORIGIN_NAMED,
        state=re.HAZARD_STATE_CONFIRMED, last_seen=0.0,
    )

    per_person, frame_risk = re.score_frame([person], [near_hazard, far_hazard], frame_diagonal, rolling_windows)
    hazard_match, _smoothed, _zone = per_person[1]
    assert hazard_match.id == near_hazard.id
    assert frame_risk["hazard_label"] == "chair"
    assert frame_risk["hazard_id"] == near_hazard.id
    assert frame_risk["hazard_bbox"] == near_hazard.bbox


def test_score_frame_no_hazards_yields_no_pairing():
    frame_diagonal = math.hypot(1920, 1080)
    rolling_windows = defaultdict(lambda: deque(maxlen=re.ROLLING_WINDOW_SIZE))
    person = re.PersonEntry(id=1, bbox=(0, 0, 10, 10), last_seen=0.0)

    per_person, frame_risk = re.score_frame([person], [], frame_diagonal, rolling_windows)
    assert per_person == {}
    assert frame_risk["zone"] == re.RISK_ZONE_NONE
    assert frame_risk["person_id"] is None
    assert frame_risk["hazard_id"] is None
    assert frame_risk["hazard_bbox"] is None


# --- --seed-hazard parsing -----------------------------------------------------


def test_parse_seed_hazard_valid():
    x1, y1, x2, y2, label = re.parse_seed_hazard("200,400,300,150,stove")
    assert (x1, y1) == (200.0, 400.0)
    assert (x2, y2) == (200.0 + 300.0, 400.0 + 150.0)
    assert label == "stove"


def test_parse_seed_hazard_label_can_contain_spaces_not_stray_commas():
    x1, y1, x2, y2, label = re.parse_seed_hazard("0,0,10,10,low shelf edge")
    assert label == "low shelf edge"


def test_parse_seed_hazard_rejects_malformed_input():
    import argparse

    bad_inputs = [
        "1,2,3,4",
        "1,2,3",
        "a,2,3,4,label",
        "1,2,-3,4,label",
        "1,2,3,0,label",
        "1,2,3,4,",
        "1,2,3,4,   ",
    ]
    for bad in bad_inputs:
        try:
            re.parse_seed_hazard(bad)
        except argparse.ArgumentTypeError:
            continue
        raise AssertionError(f"expected parse_seed_hazard({bad!r}) to raise ArgumentTypeError")


# --- overlay helpers (pure string/lookup logic, no cv2 drawing needed) -------


def test_hazard_box_color_distinguishes_pending_first_scan_vs_arrived():
    first_scan = _entry(re.HAZARD_STATE_PENDING, is_first_scan=True)
    arrived = _entry(re.HAZARD_STATE_PENDING, is_first_scan=False)
    assert re.hazard_box_color(first_scan) != re.hazard_box_color(arrived)


def test_hazard_state_tag_pending_reflects_alert_eligibility():
    assert re.hazard_state_tag(_entry(re.HAZARD_STATE_PENDING, is_first_scan=True)) == "PENDING"
    assert re.hazard_state_tag(_entry(re.HAZARD_STATE_PENDING, is_first_scan=False)) == "NEW-UNREVIEWED"
    assert re.hazard_state_tag(_entry(re.HAZARD_STATE_CONFIRMED, is_first_scan=True)) == "HAZARD"
    assert re.hazard_state_tag(_entry(re.HAZARD_STATE_DISMISSED, is_first_scan=True)) == "DISMISSED"


# --- Phase 5 (2026-08-26 addendum #2): label/outline edge clipping ------------
#
# A live test found hazard boxes near the TOP of the frame had visibly
# missing top borders (the review-outline's -2px offset pushed it off-
# canvas) and invisible state/REVIEW labels (cv2.putText's origin is the
# text BASELINE, so a label positioned "above" a box near y=0 has its own
# origin, and therefore its glyphs, off-canvas). See label_anchor_y and
# draw_hazard_box's clamped outline coordinates.


def test_label_anchor_y_places_label_above_when_room_exists():
    # Anchor comfortably far from the top edge - normal "above" placement.
    assert re.label_anchor_y(100, frame_height=480, above_offset=8, below_offset=16) == 92


def test_label_anchor_y_flips_below_when_too_close_to_top():
    # Anchor at y=5: "above" (5-8=-3) is invalid - must flip below.
    result = re.label_anchor_y(5, frame_height=480, above_offset=8, below_offset=16)
    assert result == 21  # 5 + 16, not clamped to some invalid position like 0
    assert result > 5  # actually below the anchor, on-canvas


def test_label_anchor_y_flipped_position_clamped_to_frame_bottom():
    # Anchor near the very bottom AND too close to the top simultaneously
    # can't happen for a real box, but the flip target itself must still
    # never exceed the frame - guards the arithmetic, not a real scenario.
    result = re.label_anchor_y(2, frame_height=10, above_offset=8, below_offset=50)
    assert result == 9  # frame_height - 1, not 52


def test_draw_hazard_box_review_outline_does_not_crash_near_frame_edges():
    # Regression test for the exact live bug (2026-08-26): a review-
    # candidate box whose top-left corner sits at the very edge of the
    # frame must not raise, and the outline must actually be drawn -
    # NOT silently vanish because -2/+2 pushed every coordinate off-canvas.
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    entry = re.HazardEntry(
        id=1, label="object", bbox=(0, 0, 30, 40), origin=re.HAZARD_ORIGIN_SCAN,
        state=re.HAZARD_STATE_PENDING, is_first_scan=False,
    )
    re.draw_hazard_box(frame, entry, is_review_candidate=True)
    assert frame.max() > 0  # something was actually drawn, not an all-black no-op
    # The top row of the frame must contain outline pixels (the black halo,
    # OVERLAY_OUTLINE) - the exact symptom reported live was a missing top
    # border because the unclamped rectangle was drawn at y=-2, off-canvas.
    assert frame[0].max() > 0


def test_review_candidate_label_includes_position_and_total():
    assert re.review_candidate_label(1, 3) == "REVIEW 1/3 - h=hazard n=not s=skip queued"
    assert "REVIEW 0/" not in re.review_candidate_label(1, 5)


# --- Phase 5: AlertManager (proximity hysteresis) ----------------------------


def _hazard(id=1, label="object", bbox=(0, 0, 10, 10)):
    return re.HazardEntry(id=id, label=label, bbox=bbox, origin=re.HAZARD_ORIGIN_SCAN, state=re.HAZARD_STATE_CONFIRMED)


def test_alert_manager_opens_event_on_first_scored_zone():
    mgr = re.AlertManager()
    hazard = _hazard()
    signals = mgr.update_proximity({1: (hazard, 0.4, "yellow")}, now=0.0)
    assert len(signals) == 1
    assert signals[0].kind == "opened"
    assert signals[0].event.zone == "yellow"
    assert signals[0].event.peak_zone == "yellow"
    assert signals[0].event.person_id == 1
    assert signals[0].event.hazard_id == hazard.id


def test_alert_manager_no_signal_on_unchanged_or_lower_zone_resight():
    mgr = re.AlertManager()
    hazard = _hazard()
    mgr.update_proximity({1: (hazard, 0.4, "yellow")}, now=0.0)
    same = mgr.update_proximity({1: (hazard, 0.4, "yellow")}, now=0.1)
    assert same == []
    lower = mgr.update_proximity({1: (hazard, 0.45, "yellow")}, now=0.2)
    assert lower == []


def test_alert_manager_escalation_signals_again():
    mgr = re.AlertManager()
    hazard = _hazard()
    mgr.update_proximity({1: (hazard, 0.4, "yellow")}, now=0.0)
    signals = mgr.update_proximity({1: (hazard, 0.2, "orange")}, now=0.1)
    assert len(signals) == 1
    assert signals[0].kind == "escalated"
    assert signals[0].event.peak_zone == "orange"

    signals = mgr.update_proximity({1: (hazard, 0.1, "red")}, now=0.2)
    assert len(signals) == 1
    assert signals[0].kind == "escalated"
    assert signals[0].event.peak_zone == "red"


def test_alert_manager_deescalation_within_hold_does_not_signal():
    # De-escalating (red -> orange) while the pair is still being seen every
    # frame must NOT re-signal - only an escalation past the event's own
    # peak, or a fresh open after the hold window elapses, should.
    mgr = re.AlertManager()
    hazard = _hazard()
    mgr.update_proximity({1: (hazard, 0.1, "red")}, now=0.0)
    signals = mgr.update_proximity({1: (hazard, 0.25, "orange")}, now=0.1)
    assert signals == []


def test_alert_manager_holds_open_through_brief_dip():
    # Regression test modeling the real flicker measured in
    # cv/captures/Screen Recording 2026-08-22 at 13.34.16.mov (see
    # ALERT_HOLD_SECONDS' comment): RED dipping to ORANGE for well under
    # ALERT_HOLD_SECONDS must not close the event or open a second one.
    mgr = re.AlertManager()
    hazard = _hazard()
    mgr.update_proximity({1: (hazard, 0.1, "red")}, now=0.0)
    mgr.update_proximity({}, now=0.5)  # brief gap, e.g. a missed detection
    signals = mgr.update_proximity({1: (hazard, 0.12, "red")}, now=1.0)
    assert signals == []  # same event, no re-open, no re-escalate
    assert len(mgr._proximity_events) == 1


def test_alert_manager_closes_after_hold_timeout():
    mgr = re.AlertManager()
    hazard = _hazard()
    mgr.update_proximity({1: (hazard, 0.1, "red")}, now=0.0)
    signals = mgr.update_proximity({}, now=0.0 + re.ALERT_HOLD_SECONDS + 0.01)
    assert len(signals) == 1
    assert signals[0].kind == "closed"
    assert len(mgr._proximity_events) == 0


def test_alert_manager_reopens_as_new_event_after_close():
    mgr = re.AlertManager()
    hazard = _hazard()
    mgr.update_proximity({1: (hazard, 0.1, "red")}, now=0.0)
    mgr.update_proximity({}, now=re.ALERT_HOLD_SECONDS + 0.01)  # closes
    signals = mgr.update_proximity({1: (hazard, 0.1, "red")}, now=re.ALERT_HOLD_SECONDS + 0.5)
    assert len(signals) == 1
    assert signals[0].kind == "opened"


def test_alert_manager_separate_events_per_person_hazard_pair():
    mgr = re.AlertManager()
    hazard_a, hazard_b = _hazard(id=1, label="a"), _hazard(id=2, label="b")
    signals = mgr.update_proximity(
        {1: (hazard_a, 0.1, "red"), 2: (hazard_b, 0.1, "red")}, now=0.0
    )
    assert len(signals) == 2
    assert {s.event.hazard_id for s in signals} == {1, 2}
    assert len(mgr._proximity_events) == 2


def test_alert_manager_zone_none_never_opens_an_event():
    mgr = re.AlertManager()
    hazard = _hazard()
    signals = mgr.update_proximity({1: (hazard, 0.9, re.RISK_ZONE_NONE)}, now=0.0)
    assert signals == []
    assert len(mgr._proximity_events) == 0


def test_open_new_object_creates_one_shot_opened_signal():
    mgr = re.AlertManager()
    entry = _hazard(label="knife")
    signal = mgr.open_new_object(entry, "room scan", now=5.0)
    assert signal.kind == "opened"
    assert signal.event.kind == re.ALERT_KIND_NEW_OBJECT
    assert signal.event.reason == "room scan"
    assert signal.event.person_id is None
    assert signal.event.hazard_label == "knife"


# --- Phase 5: should_trigger_clip (decision 6 gate) ---------------------------


def test_should_trigger_clip_true_on_red_peak_once():
    mgr = re.AlertManager()
    hazard = _hazard()
    signals = mgr.update_proximity({1: (hazard, 0.1, "red")}, now=0.0)
    assert mgr.should_trigger_clip(signals[0], now=0.0) is True
    # A second call for the SAME signal object must not re-trigger.
    assert mgr.should_trigger_clip(signals[0], now=0.0) is False


def test_should_trigger_clip_false_for_yellow_or_orange():
    mgr = re.AlertManager()
    hazard = _hazard()
    signals = mgr.update_proximity({1: (hazard, 0.4, "yellow")}, now=0.0)
    assert mgr.should_trigger_clip(signals[0], now=0.0) is False


def test_should_trigger_clip_false_after_first_red_in_same_event():
    # RED -> ORANGE -> RED within one event (the exact flicker
    # ALERT_HOLD_SECONDS was measured against) must produce only one clip,
    # not one per RED escalation.
    mgr = re.AlertManager()
    hazard = _hazard()
    opened = mgr.update_proximity({1: (hazard, 0.1, "red")}, now=0.0)
    assert mgr.should_trigger_clip(opened[0], now=0.0) is True
    mgr.update_proximity({1: (hazard, 0.25, "orange")}, now=0.1)
    escalated = mgr.update_proximity({1: (hazard, 0.1, "red")}, now=0.2)
    # peak_zone was already "red" so this re-sighting doesn't even escalate
    # (no rank increase past the existing peak) - no signal, nothing to gate.
    assert escalated == []


def test_should_trigger_clip_never_for_new_object():
    mgr = re.AlertManager()
    entry = _hazard()
    signal = mgr.open_new_object(entry, "room scan", now=0.0)
    assert mgr.should_trigger_clip(signal, now=0.0) is False


def test_should_trigger_clip_respects_global_cooldown_across_events():
    mgr = re.AlertManager()
    hazard_a, hazard_b = _hazard(id=1), _hazard(id=2)
    signals_a = mgr.update_proximity({1: (hazard_a, 0.1, "red")}, now=0.0)
    assert mgr.should_trigger_clip(signals_a[0], now=0.0) is True
    # A DIFFERENT hazard reaching red 0.5s later must still be blocked by
    # the global disk-safety cooldown (CLIP_MIN_INTERVAL_SECONDS = 30s).
    signals_b = mgr.update_proximity({2: (hazard_b, 0.1, "red")}, now=0.5)
    assert mgr.should_trigger_clip(signals_b[0], now=0.5) is False


def test_should_trigger_clip_allows_new_event_after_cooldown_elapses():
    mgr = re.AlertManager()
    hazard_a, hazard_b = _hazard(id=1), _hazard(id=2)
    signals_a = mgr.update_proximity({1: (hazard_a, 0.1, "red")}, now=0.0)
    assert mgr.should_trigger_clip(signals_a[0], now=0.0) is True
    later = re.CLIP_MIN_INTERVAL_SECONDS + 1.0
    signals_b = mgr.update_proximity({2: (hazard_b, 0.1, "red")}, now=later)
    assert mgr.should_trigger_clip(signals_b[0], now=later) is True


# --- Phase 5 (2026-08-26 addendum): alert_priority / AlertArbiter -------------
#
# Added after a live test (Shaked, 2026-08-26): a dismissed hazard-map spot
# re-raised four times in a row (same hazard_id, HazardMap's own "better
# safe than sorry" fingerprint re-raise being deliberately sensitive), each
# re-raise firing its own unthrottled new-object alert, landing in the same
# few seconds as a real proximity escalation into RED - six alarms at once.
# AlertManager's per-pair hysteresis (tested above) was never the gap; there
# was simply no global pacing across DIFFERENT signals at all.


def _proximity_signal(zone, kind="opened", hazard_id=1, person_id=1):
    event = re.AlertEvent(
        id=hazard_id, kind=re.ALERT_KIND_PROXIMITY, person_id=person_id, hazard_id=hazard_id,
        hazard_label="object", hazard_bbox=(0, 0, 10, 10), zone=zone, peak_zone=zone,
        started_at=0.0, last_seen_at=0.0,
    )
    return re.AlertSignal(event=event, kind=kind)


def _new_object_signal(hazard_id=1, reason="room scan"):
    event = re.AlertEvent(
        id=hazard_id, kind=re.ALERT_KIND_NEW_OBJECT, person_id=None, hazard_id=hazard_id,
        hazard_label="object", hazard_bbox=(0, 0, 10, 10), zone=re.RISK_ZONE_NONE,
        peak_zone=re.RISK_ZONE_NONE, started_at=0.0, last_seen_at=0.0, reason=reason,
    )
    return re.AlertSignal(event=event, kind="opened")


def test_alert_priority_ranks_red_above_getting_close_above_new_object():
    assert re.alert_priority(_proximity_signal("red")) == re.ALERT_PRIORITY_RED
    assert re.alert_priority(_proximity_signal("orange")) == re.ALERT_PRIORITY_GETTING_CLOSE
    assert re.alert_priority(_proximity_signal("yellow")) == re.ALERT_PRIORITY_GETTING_CLOSE
    assert re.alert_priority(_new_object_signal()) == re.ALERT_PRIORITY_NEW_OBJECT
    assert re.ALERT_PRIORITY_RED > re.ALERT_PRIORITY_GETTING_CLOSE > re.ALERT_PRIORITY_NEW_OBJECT


def test_alert_arbiter_first_signal_voices_immediately():
    arb = re.AlertArbiter()
    signal = _new_object_signal()
    assert arb.offer(signal, now=0.0) is signal


def test_alert_arbiter_second_signal_within_window_is_held_not_voiced():
    arb = re.AlertArbiter()
    arb.offer(_new_object_signal(hazard_id=1), now=0.0)
    second = _new_object_signal(hazard_id=2)
    assert arb.offer(second, now=0.5) is None


def test_alert_arbiter_held_signal_released_by_poll_after_window():
    arb = re.AlertArbiter()
    arb.offer(_new_object_signal(hazard_id=1), now=0.0)
    held = _new_object_signal(hazard_id=2)
    arb.offer(held, now=0.5)
    assert arb.poll(now=1.0) is None  # window (2.5s) not elapsed yet
    released = arb.poll(now=re.GLOBAL_ALERT_MIN_INTERVAL_SECONDS + 0.1)
    assert released is held


def test_alert_arbiter_higher_priority_signal_wins_while_held():
    # Regression test for the exact 2026-08-26 scenario: several low-
    # priority new-object pulses arrive, then a real getting-close signal -
    # the more important one must be what eventually gets voiced.
    arb = re.AlertArbiter()
    arb.offer(_new_object_signal(hazard_id=1), now=0.0)  # voiced immediately
    arb.offer(_new_object_signal(hazard_id=2), now=0.3)  # held
    getting_close = _proximity_signal("orange", hazard_id=3)
    arb.offer(getting_close, now=0.6)  # higher priority - should replace held
    released = arb.poll(now=re.GLOBAL_ALERT_MIN_INTERVAL_SECONDS + 0.1)
    assert released is getting_close


def test_alert_arbiter_lower_priority_signal_does_not_replace_held():
    arb = re.AlertArbiter()
    arb.offer(_proximity_signal("orange", hazard_id=1), now=0.0)  # voiced
    getting_close = _proximity_signal("orange", hazard_id=2)
    arb.offer(getting_close, now=0.3)  # held
    arb.offer(_new_object_signal(hazard_id=3), now=0.6)  # lower priority - must not replace
    released = arb.poll(now=re.GLOBAL_ALERT_MIN_INTERVAL_SECONDS + 0.1)
    assert released is getting_close


def test_alert_arbiter_red_always_bypasses_the_window():
    arb = re.AlertArbiter()
    arb.offer(_new_object_signal(hazard_id=1), now=0.0)  # voiced, starts the window
    red = _proximity_signal("red", hazard_id=2)
    assert arb.offer(red, now=0.2) is red  # bypasses immediately, well inside the window


def test_alert_arbiter_red_clears_anything_being_held():
    arb = re.AlertArbiter()
    arb.offer(_new_object_signal(hazard_id=1), now=0.0)
    arb.offer(_new_object_signal(hazard_id=2), now=0.3)  # held
    arb.offer(_proximity_signal("red", hazard_id=3), now=0.6)  # bypasses AND clears held
    assert arb.poll(now=re.GLOBAL_ALERT_MIN_INTERVAL_SECONDS + 1.0) is None


def test_alert_arbiter_repeated_same_hazard_bursts_collapse_to_one_voiced():
    # The literal scenario from the 2026-08-26 recording: hazard #15
    # dismissed and re-raised four times in a handful of seconds. Only the
    # first should be voiced; the rest are held/superseded, never all four.
    arb = re.AlertArbiter()
    voiced = []
    v = arb.offer(_new_object_signal(hazard_id=15), now=0.0)
    if v is not None:
        voiced.append(v)
    for t in (0.4, 0.9, 1.6):
        v = arb.offer(_new_object_signal(hazard_id=15), now=t)
        if v is not None:
            voiced.append(v)
    assert len(voiced) == 1


# --- Phase 5: audio_for_signal / banner_text_for_signal -----------------------


def test_audio_for_signal_new_object_is_hazard_detected():
    mgr = re.AlertManager()
    signal = mgr.open_new_object(_hazard(), "room scan", now=0.0)
    assert re.audio_for_signal(signal) == re.AUDIO_HAZARD_DETECTED


def test_audio_for_signal_yellow_and_orange_are_baby_getting_close():
    mgr = re.AlertManager()
    hazard = _hazard()
    for zone in ("yellow", "orange"):
        signals = mgr.update_proximity({1: (hazard, 0.3, zone)}, now=0.0)
        if signals:
            assert re.audio_for_signal(signals[0]) == re.AUDIO_BABY_GETTING_CLOSE
        mgr = re.AlertManager()  # reset between zones


def test_audio_for_signal_red_is_immediate_danger():
    mgr = re.AlertManager()
    hazard = _hazard()
    signals = mgr.update_proximity({1: (hazard, 0.1, "red")}, now=0.0)
    assert re.audio_for_signal(signals[0]) == re.AUDIO_IMMEDIATE_DANGER


def test_audio_for_signal_closed_is_silent():
    mgr = re.AlertManager()
    hazard = _hazard()
    mgr.update_proximity({1: (hazard, 0.1, "red")}, now=0.0)
    closed = mgr.update_proximity({}, now=re.ALERT_HOLD_SECONDS + 0.01)
    assert closed[0].kind == "closed"
    assert re.audio_for_signal(closed[0]) is None


def test_banner_text_for_signal_new_object_names_reason_and_label():
    mgr = re.AlertManager()
    signal = mgr.open_new_object(_hazard(label="lighter"), "wall socket", now=0.0)
    text = re.banner_text_for_signal(signal)
    assert "wall socket" in text
    assert "lighter" in text


def test_banner_text_for_signal_proximity_names_zone_and_person():
    mgr = re.AlertManager()
    signals = mgr.update_proximity({7: (_hazard(label="stove"), 0.1, "red")}, now=0.0)
    text = re.banner_text_for_signal(signals[0])
    assert "RED" in text
    assert "stove" in text
    assert "#7" in text


# --- Phase 5: RollingBuffer ----------------------------------------------------


def _tiny_frame(color=(10, 10, 10)):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def test_rolling_buffer_bounded_length():
    buf = re.RollingBuffer(fps=10.0, seconds=2.0)  # maxlen = 20
    for i in range(30):
        buf.append(_tiny_frame(), timestamp=float(i))
    assert len(buf) == 20


def test_rolling_buffer_append_returns_encoded_bytes():
    buf = re.RollingBuffer(fps=10.0, seconds=1.0)
    encoded = buf.append(_tiny_frame(), timestamp=0.0)
    assert encoded is not None
    assert encoded.size > 0


def test_rolling_buffer_snapshot_is_a_copy_not_a_live_view():
    buf = re.RollingBuffer(fps=10.0, seconds=1.0)  # maxlen = 10
    buf.append(_tiny_frame(), timestamp=0.0)
    snapshot = buf.snapshot()
    assert len(snapshot) == 1
    for i in range(1, 15):
        buf.append(_tiny_frame(), timestamp=float(i))
    assert len(snapshot) == 1  # unaffected by further appends after the fact


# --- Phase 5: ClipRecorder (tail-window timing, no real thread) ---------------


def test_clip_recorder_add_tail_frame_ignores_after_deadline():
    with tempfile.TemporaryDirectory() as tmp:
        recorder = re.ClipRecorder(tmp, fps=10.0, tail_seconds=2.0)
        event = re.AlertEvent(
            id=1, kind=re.ALERT_KIND_PROXIMITY, person_id=1, hazard_id=1, hazard_label="object",
            hazard_bbox=(0, 0, 10, 10), zone="red", peak_zone="red", started_at=0.0, last_seen_at=0.0,
        )
        recorder.trigger(buffer_snapshot=[], event=event, now=0.0)  # deadline = 2.0
        recorder.add_tail_frame(b"in-window", timestamp=1.5)
        recorder.add_tail_frame(b"too-late", timestamp=2.5)
        assert recorder._pending[0]["frames"] == [(1.5, b"in-window")]


def test_clip_recorder_poll_only_finalizes_after_deadline():
    with tempfile.TemporaryDirectory() as tmp:
        recorder = re.ClipRecorder(tmp, fps=10.0, tail_seconds=2.0)
        event = re.AlertEvent(
            id=1, kind=re.ALERT_KIND_PROXIMITY, person_id=1, hazard_id=1, hazard_label="object",
            hazard_bbox=(0, 0, 10, 10), zone="red", peak_zone="red", started_at=0.0, last_seen_at=0.0,
        )
        ok, jpeg = cv2.imencode(".jpg", _tiny_frame())
        assert ok
        recorder.trigger(buffer_snapshot=[(0.0, jpeg)], event=event, now=0.0)
        assert recorder.poll(now=1.0) == []  # tail not elapsed yet
        results = recorder.poll(now=2.1)  # tail elapsed - spawns a write thread
        assert len(results) == 1
        path, returned_event = results[0]
        assert path.endswith("_event1_object.mp4")
        assert returned_event is event  # Phase 6: poll() now returns (path, event) pairs
        assert recorder._pending == []


def test_clip_recorder_poll_calls_on_write_complete_with_path_and_event():
    # Phase 6 (backend-agent): ClipRecorder's optional on_write_complete hook
    # fires once write_clip() actually finishes, with (path, event, success,
    # frame_count) - real thread, real file, matching this suite's no-mocking
    # style (like test_write_clip_produces_a_real_playable_file below).
    with tempfile.TemporaryDirectory() as tmp:
        calls = []
        recorder = re.ClipRecorder(
            tmp, fps=10.0, tail_seconds=2.0,
            on_write_complete=lambda path, event, success, frame_count: calls.append(
                (path, event, success, frame_count)
            ),
        )
        event = re.AlertEvent(
            id=7, kind=re.ALERT_KIND_PROXIMITY, person_id=1, hazard_id=1, hazard_label="object",
            hazard_bbox=(0, 0, 10, 10), zone="red", peak_zone="red", started_at=0.0, last_seen_at=0.0,
        )
        ok, jpeg = cv2.imencode(".jpg", _tiny_frame())
        assert ok
        recorder.trigger(buffer_snapshot=[(0.0, jpeg)], event=event, now=0.0)
        results = recorder.poll(now=2.1)
        path, _ = results[0]
        thread_list = threading.enumerate()
        for t in thread_list:
            if t.name != threading.main_thread().name:
                t.join(timeout=5.0)
        assert len(calls) == 1
        called_path, called_event, success, frame_count = calls[0]
        assert called_path == path
        assert called_event is event
        assert success is True
        assert frame_count == 1


def test_write_clip_produces_a_real_playable_file():
    # Called directly and synchronously (not via a thread) - see write_clip's
    # docstring for why it's a free function rather than hidden inside
    # ClipRecorder. Real cv2 VideoWriter, real temp file, matching this
    # suite's no-mocking style.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "clip.mp4")
        frames = []
        for color in ((10, 10, 10), (200, 200, 200), (50, 100, 150)):
            ok, jpeg = cv2.imencode(".jpg", _tiny_frame(color))
            assert ok
            frames.append(jpeg)
        result = re.write_clip(path, frames, fps=10.0)
        assert result is True
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


def test_write_clip_returns_false_for_empty_frame_list():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "clip.mp4")
        assert re.write_clip(path, [], fps=10.0) is False
        assert not os.path.isfile(path)


def test_write_clip_still_works_with_no_on_complete_argument():
    # Phase 6 added an optional on_complete parameter - must remain callable
    # exactly as before with no callback (existing callers/tests do this).
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "clip.mp4")
        ok, jpeg = cv2.imencode(".jpg", _tiny_frame())
        assert ok
        assert re.write_clip(path, [jpeg], fps=10.0) is True
        assert os.path.isfile(path)


def test_write_clip_calls_on_complete_with_success_and_frame_count():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "clip.mp4")
        frames = []
        for color in ((10, 10, 10), (200, 200, 200), (50, 100, 150)):
            ok, jpeg = cv2.imencode(".jpg", _tiny_frame(color))
            assert ok
            frames.append(jpeg)
        calls = []
        result = re.write_clip(path, frames, fps=10.0, on_complete=lambda success, count: calls.append((success, count)))
        assert result is True
        assert calls == [(True, 3)]


def test_write_clip_calls_on_complete_on_failure_paths_too():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "clip.mp4")
        calls = []
        result = re.write_clip(path, [], fps=10.0, on_complete=lambda success, count: calls.append((success, count)))
        assert result is False
        assert calls == [(False, 0)]


# --- Phase 5: AudioPlayer (pure missing-file logic only - see module docstring) --


def test_audio_player_missing_file_does_not_raise_and_warns_once():
    with tempfile.TemporaryDirectory() as tmp:
        player = re.AudioPlayer(tmp)
        player.play("does_not_exist.wav")  # must not raise
        assert "does_not_exist.wav" in player._warned


# --- Phase 7: build_risk_status (/risk_status JSON payload) -----------------


def _empty_status_kwargs():
    """Shared base kwargs for build_risk_status - a fully empty session (no
    persons tracked, no hazards proposed yet, nothing scored), matching what
    a freshly-started run's very first --serve frame would produce.
    """
    return dict(
        camera_index=0,
        camera_name="Arducam",
        width=1920,
        height=1080,
        frame_risk={
            "zone": re.RISK_ZONE_NONE, "value": None, "person_id": None,
            "hazard_label": None, "hazard_id": None, "hazard_bbox": None,
        },
        live_persons=[],
        per_person={},
        hazard_map=re.HazardMap(),
        review_queue=re.ReviewQueue(),
        alert_text=None,
        alert_active=False,
        smoothed_fps=None,
        model_name="yolo26l.pt",
        imgsz=640,
        conf=0.35,
        device="mps",
        scan_enabled=True,
        socket_detect_enabled=True,
        persistence_enabled=True,
    )


def test_build_risk_status_empty_case():
    status = re.build_risk_status(**_empty_status_kwargs())

    assert status["camera"] == {"index": 0, "name": "Arducam", "width": 1920, "height": 1080}
    assert status["risk"]["zone"] == re.RISK_ZONE_NONE
    assert status["risk"]["hazard_bbox"] is None
    assert status["persons"] == []
    assert status["hazards"] == []
    assert status["hazard_counts"] == {
        "total": 0, "confirmed": 0, "pending_first_scan": 0, "pending_new": 0, "dismissed": 0,
    }
    assert status["review_queue"] == {"length": 0, "current_id": None, "queue": []}
    assert status["alert"] == {"text": None, "active": False}
    assert status["diagnostics"]["fps"] is None
    assert status["diagnostics"]["model"] == "yolo26l.pt"
    assert status["diagnostics"]["imgsz"] == 640
    assert status["diagnostics"]["conf"] == 0.35
    assert status["diagnostics"]["device"] == "mps"
    assert status["diagnostics"]["scan_enabled"] is True
    assert status["diagnostics"]["socket_detect_enabled"] is True
    assert status["diagnostics"]["persistence_enabled"] is True
    # A fresh wall-clock ISO8601 read, per persistence.now_iso() - not a
    # time.monotonic() value (see build_risk_status's docstring).
    assert "T" in status["timestamp"]


def test_build_risk_status_populated_case():
    hazard_map = re.HazardMap()
    confirmed = hazard_map._new_entry("chair", (20.0, 0.0, 30.0, 10.0), re.HAZARD_ORIGIN_NAMED, is_first_scan=True, state=re.HAZARD_STATE_CONFIRMED)
    pending_first = hazard_map._new_entry("object", (100.0, 100.0, 110.0, 110.0), re.HAZARD_ORIGIN_SCAN, is_first_scan=True)
    pending_new = hazard_map._new_entry("object", (200.0, 200.0, 210.0, 210.0), re.HAZARD_ORIGIN_SCAN, is_first_scan=False)
    dismissed = hazard_map._new_entry("object", (300.0, 300.0, 310.0, 310.0), re.HAZARD_ORIGIN_SCAN, is_first_scan=False, state=re.HAZARD_STATE_DISMISSED)

    review_queue = re.ReviewQueue()
    review_queue.enqueue(pending_new.id)

    person = re.PersonEntry(id=1, bbox=(0.0, 0.0, 10.0, 10.0), last_seen=0.0)
    per_person = {1: (confirmed, 0.05, "red")}
    frame_risk = {
        "zone": "red", "value": 0.05, "person_id": 1, "hazard_label": "chair",
        "hazard_id": confirmed.id, "hazard_bbox": confirmed.bbox,
    }

    kwargs = _empty_status_kwargs()
    kwargs.update(
        hazard_map=hazard_map,
        review_queue=review_queue,
        live_persons=[person],
        per_person=per_person,
        frame_risk=frame_risk,
        alert_text="RED - person #1 near chair",
        alert_active=True,
        smoothed_fps=14.756,
    )
    status = re.build_risk_status(**kwargs)

    assert status["risk"]["zone"] == "red"
    assert status["risk"]["value"] == 0.05
    assert status["risk"]["person_id"] == 1
    assert status["risk"]["hazard_id"] == confirmed.id
    assert status["risk"]["hazard_bbox"] == [20.0, 0.0, 30.0, 10.0]

    assert len(status["persons"]) == 1
    p = status["persons"][0]
    assert p["id"] == 1
    assert p["bbox"] == [0.0, 0.0, 10.0, 10.0]
    assert p["nearest_hazard_id"] == confirmed.id
    assert p["distance"] == 0.05
    assert p["zone"] == "red"

    assert len(status["hazards"]) == 4
    by_id = {h["id"]: h for h in status["hazards"]}
    assert by_id[confirmed.id]["state"] == re.HAZARD_STATE_CONFIRMED
    assert by_id[confirmed.id]["alerts_on_approach"] is True
    assert by_id[pending_first.id]["state"] == re.HAZARD_STATE_PENDING
    assert by_id[pending_first.id]["is_first_scan"] is True
    assert by_id[pending_first.id]["alerts_on_approach"] is False
    assert by_id[pending_new.id]["alerts_on_approach"] is True
    assert by_id[dismissed.id]["state"] == re.HAZARD_STATE_DISMISSED
    assert by_id[dismissed.id]["alerts_on_approach"] is False

    assert status["hazard_counts"] == re.hazard_counts(hazard_map)
    assert status["review_queue"] == {
        "length": 1, "current_id": pending_new.id, "queue": [pending_new.id],
    }
    assert status["alert"] == {"text": "RED - person #1 near chair", "active": True}
    assert status["diagnostics"]["fps"] == 14.756


def test_build_risk_status_counts_match_draw_risk_readout_helper():
    # hazard_counts() is the single shared implementation draw_risk_readout
    # and build_risk_status both read from - this guards against the two
    # representations of the same numbers drifting apart (Phase 7 brief's
    # explicit ask).
    hazard_map = re.HazardMap()
    hazard_map._new_entry("chair", (0, 0, 1, 1), re.HAZARD_ORIGIN_NAMED, is_first_scan=True, state=re.HAZARD_STATE_CONFIRMED)
    hazard_map._new_entry("object", (0, 0, 1, 1), re.HAZARD_ORIGIN_SCAN, is_first_scan=True)
    hazard_map._new_entry("object", (0, 0, 1, 1), re.HAZARD_ORIGIN_SCAN, is_first_scan=False)
    hazard_map._new_entry("object", (0, 0, 1, 1), re.HAZARD_ORIGIN_SCAN, is_first_scan=False, state=re.HAZARD_STATE_DISMISSED)

    kwargs = _empty_status_kwargs()
    kwargs.update(hazard_map=hazard_map)
    status = re.build_risk_status(**kwargs)

    assert status["hazard_counts"] == re.hazard_counts(hazard_map)
    assert status["hazard_counts"] == {
        "total": 4, "confirmed": 1, "pending_first_scan": 1, "pending_new": 1, "dismissed": 1,
    }


def test_build_risk_status_survives_json_dumps():
    # The test that actually catches numpy leakage: a bbox built from real
    # box.xyxy[0]-style numpy float32 scalars (exactly what main() reads off
    # a YOLO Boxes object before its own `tuple(float(v) for v in ...)`
    # conversion) must still come out the other side of build_risk_status as
    # plain, json.dumps()-safe Python types - see _bbox_to_list's docstring.
    np_box = tuple(np.float32(v) for v in (20.0, 0.0, 30.0, 10.0))

    hazard_map = re.HazardMap()
    hazard = hazard_map._new_entry("chair", np_box, re.HAZARD_ORIGIN_NAMED, is_first_scan=True, state=re.HAZARD_STATE_CONFIRMED)

    person = re.PersonEntry(id=np.int64(1), bbox=tuple(np.float32(v) for v in (0.0, 0.0, 10.0, 10.0)), last_seen=0.0)
    per_person = {1: (hazard, np.float64(0.12), "orange")}
    frame_risk = {
        "zone": "orange", "value": np.float64(0.12), "person_id": np.int64(1), "hazard_label": "chair",
        "hazard_id": hazard.id, "hazard_bbox": np_box,
    }

    kwargs = _empty_status_kwargs()
    kwargs.update(
        hazard_map=hazard_map, live_persons=[person], per_person=per_person, frame_risk=frame_risk,
        smoothed_fps=np.float32(14.8),
    )
    status = re.build_risk_status(**kwargs)

    # json.dumps with NO custom encoder - this is what actually 500s at
    # serialization time on the HTTP thread if a numpy scalar leaked through.
    dumped = json.dumps(status)
    assert isinstance(dumped, str)
    reloaded = json.loads(dumped)
    assert reloaded["persons"][0]["bbox"] == [0.0, 0.0, 10.0, 10.0]
    assert reloaded["risk"]["hazard_bbox"] == [20.0, 0.0, 30.0, 10.0]


def test_build_risk_status_queue_is_real_membership_not_pending_hazards():
    """The published review queue must be ReviewQueue's ACTUAL FIFO
    membership, never something a client could re-derive from `hazards`.

    Regression test for a real defect found during Phase 7 review: /review
    briefly derived its `queue` as "every hazard whose state is pending",
    which is a different set. This test pins the two cases where they
    diverge, both of which a parent-facing Phase 8 UI would render wrongly:

      1. After skip_remaining(), the queue is empty while every one of those
         entries is still PENDING - the derived version reported `length: 0`
         next to a non-empty queue in the same payload, offering a parent
         items that confirm/dismiss would refuse (the loop-side handlers
         fail closed on anything that is not the current candidate).
      2. The queue is FIFO; `hazards` is id-ordered. Enqueueing out of id
         order must survive into the payload, or "next up" names the wrong
         entry.
    """
    hazard_map = re.HazardMap()
    a = hazard_map._new_entry("object", (0, 0, 1, 1), re.HAZARD_ORIGIN_SCAN, is_first_scan=False)
    b = hazard_map._new_entry("object", (2, 2, 3, 3), re.HAZARD_ORIGIN_SCAN, is_first_scan=False)
    c = hazard_map._new_entry("object", (4, 4, 5, 5), re.HAZARD_ORIGIN_SCAN, is_first_scan=False)

    # Case 2 first: enqueue deliberately out of id order.
    review_queue = re.ReviewQueue()
    for entry in (c, a, b):
        review_queue.enqueue(entry.id)

    kwargs = _empty_status_kwargs()
    kwargs["hazard_map"] = hazard_map
    kwargs["review_queue"] = review_queue
    status = re.build_risk_status(**kwargs)

    assert status["review_queue"]["queue"] == [c.id, a.id, b.id], "FIFO order must survive"
    assert status["review_queue"]["current_id"] == c.id
    assert status["review_queue"]["length"] == 3

    # Case 1: skip clears the queue but leaves all three entries PENDING.
    review_queue.skip_remaining()
    status = re.build_risk_status(**kwargs)

    assert all(e.state == re.HAZARD_STATE_PENDING for e in hazard_map.entries)
    assert status["hazard_counts"]["pending_new"] == 3
    assert status["review_queue"] == {"length": 0, "current_id": None, "queue": []}


# --- Phase 7 (2026-08-28): AlertDispatcher and the speak-implies-record
# invariant -------------------------------------------------------------------
#
# These are the tests docs/phase-writeups/phase-6.md said were possible but
# did not exist. Phase 6's data-loss bug (roughly half of one live session's
# voiced alerts never reaching the database) lived in main()'s closures, so no
# unit test could reach it. The stated reason for adding none at the time was
# that a test mirroring main()'s call sequence would duplicate the wiring
# rather than test it - correct, and the reason these tests do NOT do that.
# They exercise the REAL AlertArbiter through AlertDispatcher and assert the
# invariant directly, with no camera and no copy of main()'s call order.


class _SpyRecorder:
    """Stands in for backend/persistence.py's EventWriter - same .record()
    surface, no SQLite. Deliberately not a Mock: this project has no mocking
    library and the existing suites all use real objects or tiny hand-written
    stand-ins (see backend/test_persistence.py's FakeEvent).
    """

    def __init__(self):
        self.recorded = []

    def record(self, signal):
        self.recorded.append(signal)
        return len(self.recorded)


def _dispatcher_with_spies(recorder=None):
    """A real AlertArbiter plus a spy recorder, with speak() wrapped so the
    test can see exactly what was voiced. Wrapping the bound method on the
    INSTANCE (rather than subclassing) means offer()/poll() still reach it
    through normal attribute lookup - i.e. we observe the real code path
    rather than a reimplementation of it.
    """
    recorder = recorder if recorder is not None else _SpyRecorder()
    dispatcher = re.AlertDispatcher(re.AlertArbiter(), audio_player=None, event_recorder=recorder)
    spoken = []
    original_speak = dispatcher.speak

    def spy_speak(signal):
        spoken.append(signal)
        return original_speak(signal)

    dispatcher.speak = spy_speak
    return dispatcher, recorder, spoken


def test_alert_dispatcher_every_voiced_signal_is_also_recorded_including_held_release():
    """THE invariant. This is the exact failure Phase 6 shipped and fixed:
    AlertArbiter.offer() holds a non-RED signal back when the
    GLOBAL_ALERT_MIN_INTERVAL_SECONDS pacing window is closed, and a separate
    poll() call releases it later. Before the fix, the release path spoke to
    the parent without persisting anything.

    The test deliberately forces at least one signal down the held-then-
    released path, then asserts speak() and record() saw the same set, in the
    same order.
    """
    dispatcher, recorder, spoken = _dispatcher_with_spies()

    # t=0: first signal is voiced immediately (window has never been used).
    first = dispatcher.offer(_new_object_signal(hazard_id=1), now=0.0)
    assert first is not None
    assert len(spoken) == 1

    # t=0.5: inside the 2.5s pacing window, so this one is HELD, not voiced.
    second = dispatcher.offer(_new_object_signal(hazard_id=2), now=0.5)
    assert second is None, "should have been held by the pacing window"
    assert len(spoken) == 1, "a held signal must not be voiced yet"
    assert len(recorder.recorded) == 1, "and must not be persisted yet either"

    # A poll before the window reopens releases nothing.
    assert dispatcher.poll(now=1.0) is None
    assert len(spoken) == 1

    # t=3.0: window has reopened - poll() releases the held signal. THIS is
    # the path that used to speak without recording.
    released = dispatcher.poll(now=3.0)
    assert released is not None
    assert released.event.hazard_id == 2
    assert len(spoken) == 2

    # The invariant itself, stated as directly as it can be:
    assert recorder.recorded == spoken, "every voiced signal must also be recorded"


def test_alert_dispatcher_records_red_signals_that_bypass_pacing():
    """RED bypasses the arbiter's pacing entirely (Shaked, 2026-08-26:
    immediate danger must never wait its turn). That is a THIRD way into
    speak(), and the whole point of putting persistence inside speak() is that
    a new path gets it for free rather than having to remember it.
    """
    dispatcher, recorder, spoken = _dispatcher_with_spies()

    dispatcher.offer(_new_object_signal(hazard_id=1), now=0.0)
    # Immediately inside the pacing window - a non-RED signal would be held.
    red = dispatcher.offer(_proximity_signal("red", hazard_id=9), now=0.1)

    assert red is not None, "RED must never be held by the pacing window"
    assert len(spoken) == 2
    assert recorder.recorded == spoken


def test_alert_dispatcher_with_persistence_disabled_still_voices():
    """--disable-persistence passes event_recorder=None. The parent must
    still be told; only the database write is skipped. Guards against a future
    refactor making persistence load-bearing for voicing.
    """
    dispatcher = re.AlertDispatcher(re.AlertArbiter(), audio_player=None, event_recorder=None)
    voiced = dispatcher.offer(_new_object_signal(), now=0.0)
    assert voiced is not None
    assert dispatcher.banner_text is not None


def test_alert_dispatcher_banner_expires_on_its_own_clock():
    """The banner state main() used to hold as alert_text/alert_until locals.
    Both the cv2.imshow overlay and /risk_status's `alert` field now read it
    through banner(), so the pixels and the JSON cannot disagree.
    """
    fake_now = [100.0]
    dispatcher = re.AlertDispatcher(
        re.AlertArbiter(), audio_player=None, event_recorder=None,
        clock=lambda: fake_now[0],
    )
    assert dispatcher.banner(now=100.0) == (None, False)

    dispatcher.offer(_new_object_signal(), now=0.0)
    text, active = dispatcher.banner(now=100.0)
    assert text is not None and active is True

    # Just before expiry, still active; just after, text is retained but the
    # banner reports inactive (matching the old `now < alert_until` check).
    _, active = dispatcher.banner(now=100.0 + re.ALERT_BANNER_SECONDS - 0.01)
    assert active is True
    text, active = dispatcher.banner(now=100.0 + re.ALERT_BANNER_SECONDS + 0.01)
    assert active is False
    assert text is not None, "text is retained; only `active` goes false"


def test_alert_dispatcher_closed_signals_never_reach_it():
    """Sanity check on the scope boundary: 'closed' signals are filtered by
    main()'s handle_alert_signal BEFORE the dispatcher, and audio_for_signal
    returns None for them. There is deliberately no persisted close time
    (backend/db.py: no ended_at column, by design) - if a 'closed' signal ever
    did reach speak(), it would create a row that decision 3 says should not
    exist. Pinned so that stays a deliberate choice.
    """
    assert re.audio_for_signal(_proximity_signal("red", kind="closed")) is None


def run_all():
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"ALL TESTS PASSED ({len(tests)} tests)")


if __name__ == "__main__":
    run_all()
