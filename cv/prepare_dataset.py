"""
prepare_dataset.py - Phase 3: build a train/val YOLO dataset from
cv/captures's labelled frames.

Two problems this solves, both real and both already hit:

1. **Ultralytics needs an images/ + labels/ directory pair, not the flat
   layout `label_captures.py` uses.** Ultralytics discovers each image's
   label file by substituting the literal path substring "/images/" with
   "/labels/" (`ultralytics.data.utils.img2label_paths`) - see
   `label_captures.py`'s `write_data_yaml()` docstring, which documents this
   and explicitly defers solving it to training time. This script is that
   training-time step.

2. **13 of `triage_captures.py`'s near-duplicate groups (41 of 92 frames)
   were never moved out before labelling, so they're now part of the
   dataset.** Deleting them throws away legitimate extra signal; a naive
   random split would put near-identical frames on both sides of train/val
   and make the validation number meaningless (the model would effectively
   be "validated" against frames it already saw, wearing a different JPEG
   compression artifact). The fix is neither delete nor ignore: split by
   DUPLICATE GROUP, never by individual frame, so every frame in a group
   lands on the same side of the split. This script reuses
   `triage_captures.py`'s dHash implementation directly rather than
   reimplementing it - see that module for why dHash and why transitive
   grouping is the right simplification here.

What this script does NOT do: it does not label anything (`label_captures.py`
does that), does not collect anything (`detect_stream.py`), and does not
train anything (see the `Part 2` fine-tune command documented in
`docs/phase-writeups/phase-3-step1.md` once written - this script only
prepares the input training expects).

Output layout (Ultralytics-required, see img2label_paths above):

    <dataset-dir>/
      images/train/*.jpg   (symlinks into cv/captures/ by default)
      images/val/*.jpg
      labels/train/*.txt   (symlinks into cv/captures/, empty files carried
                             through unchanged for negatives)
      labels/val/*.txt
      data.yaml

Symlinks are preferred over copies so `cv/captures/` stays the single source
of truth and the dataset is cheaply rebuildable (no gigabytes duplicated).
This script does not just assume symlinks work with Ultralytics' loader -
`verify_dataset()` below actually instantiates Ultralytics' own
`YOLODataset` against the built layout and falls back to real copies if that
fails for any reason (e.g. a filesystem that doesn't support symlinks).

Determinism: the group order is sorted (not filesystem/glob order, which can
vary) before being shuffled with a seeded `random.Random`, so the same
--seed always produces the same split - required for the "actual resulting
counts" this script reports to mean anything across reruns.

Usage:
    python cv/prepare_dataset.py
    python cv/prepare_dataset.py --seed 7 --val-fraction 0.2
    python cv/prepare_dataset.py --force-copy   # skip the symlink attempt
"""

import argparse
import random
import shutil
from pathlib import Path

from label_captures import CLASS_NAMES, discover_images, is_labelled, label_path_for
from triage_captures import DEFAULT_HAMMING_THRESHOLD, compute_dhash, find_duplicate_groups

CAPTURES_DIR = Path(__file__).resolve().parent / "captures"

# Gitignored (see .gitignore's cv/datasets/ entry, added alongside this
# script) - this is a derived build artifact, rebuildable at any time from
# cv/captures/, not source of record.
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "datasets" / "phase3_v1"

DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SEED = 3


def find_labelled_images(capture_dir: Path, labels_dir: Path) -> list:
    """*_raw.jpg files under capture_dir that already have a label file
    (present, possibly empty - see label_captures.py's is_labelled()).
    Frames still unlabelled (no .txt at all) are excluded, not treated as
    negatives - a missing file means "not yet visited," not "no hazard."
    """
    return [p for p in discover_images(capture_dir) if is_labelled(p, labels_dir)]


def build_duplicate_groups(images: list, hamming_threshold: int) -> list:
    """Every labelled image assigned to exactly one group: real duplicate
    groups from triage_captures.find_duplicate_groups() (>1 member) plus a
    singleton group of size 1 for every ungrouped image (that function omits
    singletons - see its docstring - because it has nothing to report for a
    triage run, but this script needs every frame accounted for so the
    train/val split has nowhere for a frame to fall through).
    """
    hashes = {path: compute_dhash(path) for path in images}
    multi_groups = find_duplicate_groups(hashes, hamming_threshold)

    grouped = {path for group in multi_groups for path in group}
    singleton_groups = [[path] for path in images if path not in grouped]

    return multi_groups + singleton_groups


def split_groups(groups: list, val_fraction: float, seed: int) -> tuple:
    """Shuffle groups deterministically (sorted first, so the shuffle result
    doesn't depend on filesystem/glob ordering) and split by GROUP COUNT,
    never by frame - every frame in a group must land on the same side. This
    will not land exactly on the target fraction once whole groups are kept
    intact (a 5-frame group tips the balance further than a singleton) -
    that's expected, and the caller reports the real resulting numbers
    rather than pretending it hit the target.
    """
    ordered = sorted(groups, key=lambda group: tuple(str(p) for p in sorted(group)))
    rng = random.Random(seed)
    rng.shuffle(ordered)

    val_group_count = round(len(ordered) * val_fraction)
    val_groups = ordered[:val_group_count]
    train_groups = ordered[val_group_count:]
    return train_groups, val_groups


def count_boxes(label_path: Path) -> dict:
    """{class_id: count} for one label file. An empty/missing file yields
    an empty dict - both are valid negatives (see find_labelled_images()).
    """
    counts = {}
    if not label_path.exists():
        return counts
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        class_id = int(line.split()[0])
        counts[class_id] = counts.get(class_id, 0) + 1
    return counts


def clear_dataset_dir(dataset_dir: Path) -> None:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)


def populate_split(image_paths: list, labels_dir: Path, dataset_dir: Path,
                    split_name: str, use_symlinks: bool) -> None:
    images_out = dataset_dir / "images" / split_name
    labels_out = dataset_dir / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        label_src = label_path_for(image_path, labels_dir)
        image_dst = images_out / image_path.name
        label_dst = labels_out / label_src.name

        if use_symlinks:
            # Relative symlinks so the dataset directory can be moved (or
            # the whole repo relocated) as a unit without breaking - the
            # target is resolved relative to the symlink's own directory,
            # not the cwd this script happened to run from.
            import os
            image_dst.symlink_to(os.path.relpath(image_path.resolve(), images_out))
            # An empty label file still needs a real target to symlink to -
            # label_captures.py always writes the .txt (even empty, see its
            # docstring), so label_src is guaranteed to exist for every
            # image find_labelled_images() returned.
            label_dst.symlink_to(os.path.relpath(label_src.resolve(), labels_out))
        else:
            shutil.copy2(image_path, image_dst)
            shutil.copy2(label_src, label_dst)


def write_data_yaml(dataset_dir: Path) -> Path:
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    content = f"""\
# Auto-generated by cv/prepare_dataset.py - rewritten on every run, safe to
# regenerate, do not hand-edit (edits will be overwritten). Do not commit -
# this whole directory is derived from cv/captures/ (see .gitignore).

path: {dataset_dir.resolve()}
train: images/train
val: images/val

names:
{names_block}
"""
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(content)
    return data_yaml


def verify_dataset(data_yaml: Path) -> tuple:
    """Actually instantiate Ultralytics' own YOLODataset against the built
    layout for both splits, rather than assuming img2label_paths' string
    substitution resolves correctly through a symlinked images/ directory.
    Returns (ok: bool, message: str). Any exception here is treated as
    "symlinks did not work" and the caller falls back to real copies -
    silently trusting a symlink layout Ultralytics has not actually proven
    it can load would be worse than the extra disk space a copy costs.
    """
    try:
        import yaml as yaml_lib
        from ultralytics.data.dataset import YOLODataset

        data_cfg = yaml_lib.safe_load(data_yaml.read_text())
        dataset_root = Path(data_cfg["path"])

        for split_key in ("train", "val"):
            images_dir = dataset_root / data_cfg[split_key]
            dataset = YOLODataset(
                img_path=str(images_dir),
                data={"names": {i: n for i, n in enumerate(CLASS_NAMES)}, "channels": 3},
                task="detect",
                augment=False,
            )
            if len(dataset) == 0:
                return False, f"YOLODataset loaded zero images for split '{split_key}'."
            # Force at least one full sample through the loader (image read
            # + label parse), not just path discovery - a symlink pointing
            # nowhere would fail here, not at construction time.
            _ = dataset[0]
    except Exception as exc:  # noqa: BLE001 - any failure means "fall back to copies"
        return False, f"{type(exc).__name__}: {exc}"

    return True, "Ultralytics YOLODataset loaded both splits successfully."


def report_split(name: str, image_paths: list, labels_dir: Path) -> None:
    class_totals = {i: 0 for i in range(len(CLASS_NAMES))}
    negatives = 0
    for image_path in image_paths:
        counts = count_boxes(label_path_for(image_path, labels_dir))
        if not counts:
            negatives += 1
        for class_id, count in counts.items():
            class_totals[class_id] += count

    per_class = ", ".join(
        f"{CLASS_NAMES[i]}={class_totals[i]}" for i in range(len(CLASS_NAMES))
    )
    print(
        f"  {name}: {len(image_paths)} frame(s), {negatives} negative(s), "
        f"boxes: {per_class}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an Ultralytics-ready images/+labels/ train/val "
        "dataset from cv/captures/'s labelled frames, splitting by "
        "near-duplicate group (not by individual frame) so validation "
        "isn't contaminated by near-identical frames on both sides."
    )
    parser.add_argument("--captures-dir", type=str, default=str(CAPTURES_DIR))
    parser.add_argument("--labels-dir", type=str, default=None,
                         help="Default: same as --captures-dir (flat layout, "
                         "matching label_captures.py's default).")
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--hamming", type=int, default=DEFAULT_HAMMING_THRESHOLD)
    parser.add_argument("--force-copy", action="store_true",
                         help="Skip the symlink attempt and copy files "
                         "directly - use if symlinks are known not to work "
                         "on this filesystem.")
    args = parser.parse_args()

    capture_dir = Path(args.captures_dir)
    labels_dir = Path(args.labels_dir) if args.labels_dir else capture_dir
    dataset_dir = Path(args.dataset_dir)

    images = find_labelled_images(capture_dir, labels_dir)
    if not images:
        print(f"No labelled frames found under {capture_dir} - nothing to build.")
        return

    print(f"{len(images)} labelled frame(s) found under {capture_dir}.")

    groups = build_duplicate_groups(images, args.hamming)
    multi_group_count = sum(1 for g in groups if len(g) > 1)
    print(
        f"{len(groups)} group(s) total: {multi_group_count} near-duplicate "
        f"group(s) (Hamming <= {args.hamming}, {sum(len(g) for g in groups if len(g) > 1)} "
        f"frame(s)), {len(groups) - multi_group_count} singleton frame(s)."
    )

    train_groups, val_groups = split_groups(groups, args.val_fraction, args.seed)
    train_images = sorted(p for g in train_groups for p in g)
    val_images = sorted(p for g in val_groups for p in g)

    print(
        f"\nSplit by group (target val fraction {args.val_fraction}, seed "
        f"{args.seed}): {len(train_groups)} train group(s), "
        f"{len(val_groups)} val group(s)."
    )
    print("Actual resulting counts (not exactly the target - see module docstring):")
    report_split("train", train_images, labels_dir)
    report_split("val", val_images, labels_dir)

    clear_dataset_dir(dataset_dir)

    print(f"\nBuilding dataset at {dataset_dir} via symlinks...")
    use_symlinks = not args.force_copy
    if use_symlinks:
        try:
            populate_split(train_images, labels_dir, dataset_dir, "train", use_symlinks=True)
            populate_split(val_images, labels_dir, dataset_dir, "val", use_symlinks=True)
        except OSError as exc:
            print(f"Symlink creation failed ({exc}) - falling back to copies.")
            use_symlinks = False
            clear_dataset_dir(dataset_dir)

    data_yaml = write_data_yaml(dataset_dir)

    if use_symlinks:
        ok, message = verify_dataset(data_yaml)
        print(f"Symlink verification (Ultralytics YOLODataset load): {message}")
        if not ok:
            print("Falling back to real copies instead.")
            clear_dataset_dir(dataset_dir)
            populate_split(train_images, labels_dir, dataset_dir, "train", use_symlinks=False)
            populate_split(val_images, labels_dir, dataset_dir, "val", use_symlinks=False)
            data_yaml = write_data_yaml(dataset_dir)
            ok, message = verify_dataset(data_yaml)
            print(f"Copy verification (Ultralytics YOLODataset load): {message}")
            if not ok:
                print("Warning: dataset still failed to load even with real copies - "
                      "investigate before training.")

    print(
        f"\nDataset ready: {data_yaml} "
        f"({'symlinks' if use_symlinks else 'copies'})."
    )


if __name__ == "__main__":
    main()
