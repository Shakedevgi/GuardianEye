"""
triage_captures.py - Phase 3 dataset hygiene tool.

`detect_stream.py --interval` (see its module docstring) makes bulk capture
easy, which creates a new problem: an interval timer has no idea whether the
scene changed between two saves, so a stationary subject or a slow moment
will inevitably produce runs of frames that are visually identical. Near
duplicates don't add information to a training set - they add overfitting
risk and wasted labelling time. This script is meant to run once per
collection session, before anything gets anywhere near a labelling tool.

Two independent checks, both report-only unless you opt into an action:

1. **Near-duplicate detection (the main event).** A perceptual hash (dHash,
   implemented directly with numpy/cv2 - no new dependency) is computed for
   every `*_raw.jpg` in a directory. Frames whose hashes are within a
   configurable Hamming distance of each other are grouped. Grouping is
   transitive (A~B and B~C group A,B,C together even if A and C individually
   exceed the threshold) - this is a deliberate simplification: for the runs
   this tool exists to catch (a static or slowly-drifting scene during
   interval capture) transitive chaining is the right behaviour, but it does
   mean a sufficiently long, sufficiently gradual pan could chain unrelated
   frames into one group. Inspect a report before trusting --move-duplicates
   blindly on unfamiliar footage.

2. **Blur detection.** Variance of the Laplacian per frame (a standard, cheap
   focus-quality proxy: a sharp image has more high-frequency edge content,
   so its Laplacian has higher variance; a blurred image's edges are
   smoothed out, lowering it). The lowest-scoring frames are reported so
   obviously motion-blurred captures can be reviewed and dropped by hand.

`--move-duplicates DIR` relocates (never deletes - shutil.move, always
recoverable) all but one representative frame per duplicate group. The kept
representative is the sharpest frame in the group (highest Laplacian
variance) - not simply "the first one alphabetically" - since sharper is a
better label candidate whenever the group's frames aren't already identical.
Both the `_raw.jpg` and its `_annotated.jpg` sibling (if present) move
together, so a pair produced by one `detect_stream.py` save never gets split
across two directories.

Usage:
    python triage_captures.py                                  # report only
    python triage_captures.py --dir cv/captures --hamming 5
    python triage_captures.py --move-duplicates cv/captures/duplicates
    python triage_captures.py --blur-count 10

Scope: report/triage tooling only. Does not collect, label, or train
anything. Nothing leaves this machine. Does not touch detect_stream.py,
camera.py, or stream_camera.py.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

CAPTURES_DIR = Path(__file__).resolve().parent / "captures"

# dHash parameters. hash_size=8 -> a hash_size x (hash_size+1) grayscale
# thumbnail -> 64 bits -> a Python int. This is the standard dHash
# construction (Neal Krawetz, "kind of like that"): resize small and coarse
# on purpose - the hash is meant to capture the frame's coarse gradient
# structure (is x lighter than its right neighbour), not fine pixel detail,
# so genuinely-near-duplicate frames (same scene, same framing, sub-second
# apart) hash identically or near-identically despite JPEG noise, minor
# exposure drift, or a slightly moving foot.
DHASH_SIZE = 8

# Default Hamming distance (out of 64 bits) below which two frames are
# considered near-duplicates. Chosen per the task spec as a starting point,
# not independently re-derived here - tune with --hamming against your own
# footage if it's grouping too aggressively or not aggressively enough.
DEFAULT_HAMMING_THRESHOLD = 5

# How many lowest-blur-score frames to report by default - enough to spot a
# pattern (e.g. "every shot from the crouching block is soft") without
# dumping the whole directory's scores unread.
DEFAULT_BLUR_REPORT_COUNT = 10


def compute_dhash(image_path: Path, hash_size: int = DHASH_SIZE) -> int:
    """Compute a dHash for one image, returned as a single Python int (its
    bits are the hash). Grayscale + a tiny resize before hashing, per the
    dHash construction - see DHASH_SIZE's comment for why coarse is the
    point, not a shortcut.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read {image_path} as an image.")

    # hash_size+1 wide so there's exactly one "right neighbour" comparison
    # per column in the final hash_size-wide grid.
    resized = cv2.resize(
        image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA
    )
    diff = resized[:, 1:] > resized[:, :-1]

    hash_value = 0
    for bit in diff.flatten():
        hash_value = (hash_value << 1) | int(bit)
    return hash_value


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def laplacian_variance(image_path: Path) -> float:
    """Variance of the Laplacian - a standard cheap sharpness/focus proxy.
    Lower means blurrier. Computed on grayscale so colour has no influence.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read {image_path} as an image.")
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


class UnionFind:
    """Minimal union-find for grouping frames transitively by Hamming
    distance - see the module docstring for why transitivity is a
    deliberate, documented simplification here rather than an oversight.
    """

    def __init__(self, items):
        self._parent = {item: item for item in items}

    def find(self, item):
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_a] = root_b

    def groups(self):
        members = {}
        for item in self._parent:
            members.setdefault(self.find(item), []).append(item)
        return list(members.values())


def find_duplicate_groups(hashes: dict, threshold: int) -> list:
    """hashes: {path: dhash_int}. Returns a list of groups (each a list of
    paths), one entry per group that has more than one member - singletons
    (a frame with no near-duplicate) are not "groups" and are omitted, since
    there's nothing to triage about them.

    O(n^2) pairwise comparison - fine at the scale this tool is meant for
    (a few hundred frames from one collection session, not a standing
    archive); if that stops being true, this is the place to swap in an
    indexed nearest-neighbour approach instead.
    """
    paths = list(hashes.keys())
    uf = UnionFind(paths)
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if hamming_distance(hashes[paths[i]], hashes[paths[j]]) <= threshold:
                uf.union(paths[i], paths[j])

    return [sorted(group) for group in uf.groups() if len(group) > 1]


def annotated_sibling(raw_path: Path) -> Path | None:
    """The *_annotated.jpg saved alongside this *_raw.jpg by
    detect_stream.py's save_snapshot(), if it still exists.
    """
    if not raw_path.name.endswith("_raw.jpg"):
        return None
    sibling = raw_path.with_name(raw_path.name[: -len("_raw.jpg")] + "_annotated.jpg")
    return sibling if sibling.exists() else None


def move_duplicates(groups: list, sharpness: dict, destination: Path) -> list:
    """For each group, keep the sharpest frame (highest Laplacian variance)
    in place and move every other member - plus its *_annotated.jpg sibling,
    if present - to `destination`. Always a move (shutil.move), never a
    delete: a wrongly-binned frame must be recoverable by moving it back.

    Returns the list of raw paths that were moved (not including the kept
    representative of each group), for the summary printout.
    """
    destination.mkdir(parents=True, exist_ok=True)
    moved = []

    for group in groups:
        keeper = max(group, key=lambda path: sharpness[path])
        for path in group:
            if path == keeper:
                continue

            target = destination / path.name
            shutil.move(str(path), str(target))
            moved.append(path)

            sibling = annotated_sibling(path)
            if sibling is not None:
                shutil.move(str(sibling), str(destination / sibling.name))

    return moved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset hygiene: find near-duplicate and blurry frames "
        "in a directory of detect_stream.py captures, before they reach a "
        "labelling tool. Report-only by default."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=str(CAPTURES_DIR),
        help=f"Directory of captures to triage (default: {CAPTURES_DIR}). "
        "Only *_raw.jpg files are hashed/scored - *_annotated.jpg siblings "
        "are moved alongside their raw frame when --move-duplicates is used, "
        "but never independently hashed (they have detection boxes burned "
        "into the pixels, which would corrupt both the duplicate hash and "
        "the blur score).",
    )
    parser.add_argument(
        "--hamming",
        type=int,
        default=DEFAULT_HAMMING_THRESHOLD,
        help=f"Maximum dHash Hamming distance (out of 64 bits) for two "
        f"frames to be grouped as near-duplicates (default: "
        f"{DEFAULT_HAMMING_THRESHOLD}). Lower = stricter (fewer, tighter "
        "groups); higher = looser.",
    )
    parser.add_argument(
        "--blur-count",
        type=int,
        default=DEFAULT_BLUR_REPORT_COUNT,
        help="How many of the lowest-scoring (blurriest) frames to list in "
        f"the report (default: {DEFAULT_BLUR_REPORT_COUNT}).",
    )
    parser.add_argument(
        "--move-duplicates",
        type=str,
        default=None,
        help="If given, move (never delete) all but the sharpest frame per "
        "duplicate group into this directory, along with each moved frame's "
        "*_annotated.jpg sibling if present. Absent by default - the tool "
        "only reports otherwise.",
    )
    args = parser.parse_args()

    capture_dir = Path(args.dir)
    raw_paths = sorted(capture_dir.glob("*_raw.jpg"))

    if not raw_paths:
        print(f"No *_raw.jpg files found in {capture_dir} - nothing to triage.")
        return

    print(f"Triaging {len(raw_paths)} raw frame(s) in {capture_dir} ...")

    hashes = {}
    sharpness = {}
    unreadable = []
    for path in raw_paths:
        try:
            hashes[path] = compute_dhash(path)
            sharpness[path] = laplacian_variance(path)
        except ValueError as exc:
            unreadable.append((path, str(exc)))

    if unreadable:
        print(f"\nWarning: {len(unreadable)} file(s) could not be read, skipped:")
        for path, reason in unreadable:
            print(f"  {path.name}: {reason}")

    groups = find_duplicate_groups(hashes, args.hamming)
    frames_to_remove = sum(len(group) - 1 for group in groups)
    frames_remaining = len(hashes) - frames_to_remove

    print(
        f"\nNear-duplicate groups (Hamming distance <= {args.hamming}): "
        f"{len(groups)}"
    )
    for i, group in enumerate(groups, start=1):
        keeper = max(group, key=lambda path: sharpness[path])
        print(f"\n  Group {i} ({len(group)} frames):")
        for path in group:
            marker = "KEEP (sharpest)" if path == keeper else "duplicate"
            print(
                f"    {path.name:<55} sharpness={sharpness[path]:>9.1f}  {marker}"
            )

    print("\nBlurriest frames (lowest Laplacian variance):")
    blurriest = sorted(sharpness.items(), key=lambda kv: kv[1])[: args.blur_count]
    for path, score in blurriest:
        print(f"  {score:>9.1f}  {path.name}")

    print(
        f"\nSummary: {len(hashes)} frame(s) triaged, {len(groups)} duplicate "
        f"group(s) found, {frames_to_remove} frame(s) would be removed by "
        f"--move-duplicates, {frames_remaining} would remain."
    )

    if args.move_duplicates:
        destination = Path(args.move_duplicates)
        moved = move_duplicates(groups, sharpness, destination)
        print(
            f"\nMoved {len(moved)} duplicate frame(s) (+ their *_annotated.jpg "
            f"siblings, where present) to {destination}. Nothing was deleted; "
            f"move them back to undo."
        )


if __name__ == "__main__":
    main()
