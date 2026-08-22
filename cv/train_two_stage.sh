#!/bin/bash
# train_two_stage.sh - Phase 3 two-stage training experiment driver.
#
# Runs stage 1 (public Open Images sharp_object data, visual diversity) then
# chains directly into stage 2 (our own labelled office frames, deployment
# context - gets the "last word" per the task) IN THE SAME PROCESS, so a
# single `nohup ... &` covers both stages and survives the launching shell
# session ending. This exists because an earlier run in this project was
# killed mid-training when an agent session closed - see
# docs/phase-3-step0-findings.md and the task that produced this script.
#
# Usage (run detached, from repo root):
#   source .venv/bin/activate
#   nohup bash cv/train_two_stage.sh > cv/runs/two_stage_driver.log 2>&1 &
#
# Does NOT touch runs/detect/cv/runs/phase3_v2 (round 1's weights) - writes
# to a new project name (phase3_v3_stage1 / phase3_v3_stage2) so round 1
# stays intact for comparison, per the task's explicit constraint.
#
# The 32 *home*_raw.jpg frames are NOT part of either stage's data.yaml -
# stage 1 is data/openimages_public (no home frames, no cv/ frames at all),
# stage 2 is cv/datasets/phase3_v1 (built 2026-08-09, before the home frames
# existed on disk - verified by directory listing before this script was
# written, not just assumed).

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

STAGE1_DATA="data/openimages_public/data_stage1.yaml"
STAGE1_PROJECT="cv/runs"
STAGE1_NAME="phase3_v3_stage1"

STAGE2_DATA="cv/datasets/phase3_v1/data.yaml"
STAGE2_PROJECT="cv/runs"
STAGE2_NAME="phase3_v3_stage2"

echo "=== $(date) : stage 1 starting (public Open Images, nc=2 head, sharp_object-only data) ==="
yolo detect train \
    model=cv/models/yolo26l.pt \
    data="$STAGE1_DATA" \
    epochs=100 \
    patience=15 \
    batch=16 \
    imgsz=640 \
    device=mps \
    project="$STAGE1_PROJECT" \
    name="$STAGE1_NAME" \
    exist_ok=true \
    seed=0 \
    deterministic=true

STAGE1_WEIGHTS="$STAGE1_PROJECT/$STAGE1_NAME/weights/best.pt"
echo "=== $(date) : stage 1 done. Weights: $STAGE1_WEIGHTS ==="

if [ ! -f "$STAGE1_WEIGHTS" ]; then
    echo "ERROR: stage 1 weights not found at $STAGE1_WEIGHTS - aborting before stage 2."
    exit 1
fi

echo "=== $(date) : stage 2 starting (our office frames, fine-tune from stage 1 weights) ==="
# Same hyperparameters as round 1 (runs/detect/cv/runs/phase3_v2/args.yaml)
# except the starting weights, so stage-1-vs-round-1 is the only variable
# that changed - epochs/patience/batch/imgsz/seed all match round 1 exactly.
yolo detect train \
    model="$STAGE1_WEIGHTS" \
    data="$STAGE2_DATA" \
    epochs=150 \
    patience=30 \
    batch=8 \
    imgsz=640 \
    device=mps \
    project="$STAGE2_PROJECT" \
    name="$STAGE2_NAME" \
    exist_ok=true \
    seed=0 \
    deterministic=true

echo "=== $(date) : stage 2 done. Weights: $STAGE2_PROJECT/$STAGE2_NAME/weights/best.pt ==="
echo "=== $(date) : two-stage training pipeline complete ==="
