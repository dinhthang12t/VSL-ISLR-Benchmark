#!/usr/bin/env bash
# ==============================================================================
# Reproduce Table 4: Pose-based TGCN (MediaPipe, MMPose, ViTPose)
# Protocol 1: Original 1,182 classes with signer overlap
# Keypoints: 55 whole-body joints, T=64 frames
# ==============================================================================

set -euo pipefail

CSV_PATH="${CSV_PATH:-data/splits_original_1182/ISLR_promax.csv}"
POSE_BASE_DIR="${POSE_BASE_DIR:-data/pose_features}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/table4_tgcn}"
SEED="${SEED:-42}"

echo "=================================================================="
echo "Training TGCN on 55-Keypoint Pose Data (1,182 Classes, Seed: ${SEED})"
echo "=================================================================="

BACKENDS=("mediapipe" "mmpose" "vitpose")

for backend in "${BACKENDS[@]}"; do
    echo "--- Training TGCN with ${backend} features ---"
    python src/training/train_tgcn.py \
        --csv_path "${CSV_PATH}" \
        --pose_dir "${POSE_BASE_DIR}/${backend}" \
        --save_dir "${OUTPUT_DIR}/${backend}" \
        --target_len 64 \
        --num_nodes 55 \
        --batch_size 32 \
        --epochs 100 \
        --lr 1e-3 \
        --weight_decay 1e-4 \
        --dropout 0.3 \
        --augment \
        --seed "${SEED}"
done

echo "Finished reproducing Table 4 (Pose-based TGCN)."
