#!/usr/bin/env bash
# ==============================================================================
# Reproduce Table 11: 3D CNNs on Signer-Independent 447 Split
# Protocol 2: 447 classes, 4,252 videos across 50 disjoint signers
# Models: X3D-M, R3D-18, I3D, R(2+1)D-18
# ==============================================================================

set -euo pipefail

CSV_PATH="${CSV_PATH:-data/splits_signer_indep_447/ISLR_447_benchmark.csv}"
VIDEO_DIR="${VIDEO_DIR:-/path/to/videos}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/table11_3d_cnn}"
SEED="${SEED:-42}"

echo "=================================================================="
echo "Training 3D CNNs on 447 Signer-Independent Classes (Seed: ${SEED})"
echo "=================================================================="

MODELS=("x3d_m" "r3d_18" "i3d" "r2plus1d_18")

for m in "${MODELS[@]}"; do
    echo "--- Training 3D Backbone: ${m} ---"
    python src/training/train_3d.py \
        --csv_path "${CSV_PATH}" \
        --video_dir "${VIDEO_DIR}" \
        --model_name "${m}" \
        --save_dir "${OUTPUT_DIR}/${m}" \
        --frames 64 \
        --batch_size 8 \
        --epochs 35 \
        --freeze_early \
        --seed "${SEED}"
done

echo "Finished Table 11 (3D CNNs)."
