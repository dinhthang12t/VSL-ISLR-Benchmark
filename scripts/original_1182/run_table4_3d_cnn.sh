#!/usr/bin/env bash
# ==============================================================================
# Reproduce Table 4: 3D CNNs (X3D-M, R3D-18, I3D, C3D, R(2+1)D-18)
# Protocol 1: Original 1,182 classes with signer overlap
# Frame sampling: T=64 uniform, Kinetics-400 normalization
# ==============================================================================

set -euo pipefail

CSV_PATH="${CSV_PATH:-data/splits_original_1182/ISLR_promax.csv}"
VIDEO_DIR="${VIDEO_DIR:-/path/to/videos}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/table4_3d_cnn}"
SEED="${SEED:-42}"

echo "=================================================================="
echo "Training 3D CNNs on 1,182 Classes (T=64, Seed: ${SEED})"
echo "=================================================================="

MODELS=("x3d_m" "r3d_18" "i3d" "c3d" "r2plus1d_18")

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
        --lr_backbone 1e-4 \
        --lr_head 1e-3 \
        --weight_decay 1e-4 \
        --label_smoothing 0.05 \
        --freeze_early \
        --seed "${SEED}"
done

echo "Finished reproducing Table 4 (3D CNNs)."
