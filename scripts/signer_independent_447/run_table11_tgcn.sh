#!/usr/bin/env bash
# ==============================================================================
# Reproduce Table 11: Pose-based TGCN on Signer-Independent 447 Split
# Protocol 2: 447 classes, 4,252 videos across 50 disjoint signers
# Backends: MediaPipe, MMPose, ViTPose (55 whole-body joints)
# ==============================================================================

set -euo pipefail

CSV_PATH="${CSV_PATH:-data/splits_signer_indep_447/ISLR_447_benchmark.csv}"
POSE_BASE_DIR="${POSE_BASE_DIR:-data/pose_features_447}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/table11_tgcn}"
SEED="${SEED:-42}"

echo "=================================================================="
echo "Training TGCN on 447 Signer-Independent Classes (Seed: ${SEED})"
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
        --dropout 0.3 \
        --augment \
        --seed "${SEED}"
done

echo "Finished Table 11 (Pose TGCN)."
