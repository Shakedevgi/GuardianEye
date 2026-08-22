#!/bin/bash
# train_mixed.sh - Phase 3 round 3 ("mixed") training driver.
#
# Single training run from yolo26l.pt (NOT round 2's stage-1 weights) on the
# combined dataset built by cv/build_mixed_dataset.py: 1,078 public
# sharp_object-only images + our 79 train frames (both classes) repeated 14x
# to sit near parity with the public count, so class 1 (small_swallowable)
# is supervised in every epoch instead of being absent for an entire first
# stage the way round 2's sequential approach left it.
#
# Round 1 weights:  runs/detect/cv/runs/phase3_v2/weights/  (not touched)
# Round 2 weights:  cv/runs/phase3_v3_stage2/weights/       (not touched)
# This run writes to: cv/runs/phase3_v4_mixed/weights/
#
# CRITICAL: project= is an ABSOLUTE path. A relative project= path
# (project=cv/runs) is what nested round 2's stage 1 under
# runs/detect/cv/runs/... instead of cv/runs/... and broke the stage 1 ->
# stage 2 handoff after 3.5 hours of training - see
# docs/phase-3-step0-findings.md and cv/train_two_stage.sh's header. This
# script hardcodes the absolute repo path for that reason; do not
# "simplify" it back to a relative path.
#
# Usage (run detached, from repo root):
#   source .venv/bin/activate
#   python cv/build_mixed_dataset.py          # regenerate train.txt/data.yaml first
#   nohup bash cv/train_mixed.sh > cv/runs/mixed_driver.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

DATA="cv/datasets/phase3_v4_mixed/data.yaml"
PROJECT="/Users/shakedivgi/Projects/GuardianEye/cv/runs"
NAME="phase3_v4_mixed"

if [ ! -f "$DATA" ]; then
    echo "ERROR: $DATA not found - run cv/build_mixed_dataset.py first."
    exit 1
fi

echo "=== $(date) : mixed training starting (fresh yolo26l.pt, combined public+ours data) ==="
yolo detect train \
    model=cv/models/yolo26l.pt \
    data="$DATA" \
    epochs=100 \
    patience=15 \
    batch=16 \
    imgsz=640 \
    device=mps \
    project="$PROJECT" \
    name="$NAME" \
    exist_ok=true \
    seed=0 \
    deterministic=true

echo "=== $(date) : mixed training done. Weights: $PROJECT/$NAME/weights/best.pt ==="
