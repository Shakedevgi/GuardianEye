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

Plain asserts, no pytest dependency - run directly:

    python cv/test_risk_engine.py

Exits non-zero (via AssertionError propagating) on first failure, prints
"ALL TESTS PASSED" on success.
"""

import math
from collections import defaultdict, deque

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


def test_apply_scan_candidates_dismissed_entry_reraised_on_material_change():
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
    changed_frame = _solid_frame((250, 250, 250))
    diff = hm.apply_scan_candidates([(0, 0, 60, 60)], changed_frame, frame_diagonal, is_first_scan_cycle=False)
    assert len(diff.reraised) == 1
    assert entry.state == re.HAZARD_STATE_PENDING
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


def test_review_candidate_label_includes_position_and_total():
    assert re.review_candidate_label(1, 3) == "REVIEW 1/3 - h=hazard n=not s=skip queued"
    assert "REVIEW 0/" not in re.review_candidate_label(1, 5)


def run_all():
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"ALL TESTS PASSED ({len(tests)} tests)")


if __name__ == "__main__":
    run_all()
