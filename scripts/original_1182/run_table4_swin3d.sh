#!/usr/bin/env bash
# ==============================================================================
# Reproduce Table 4: Swin3D-Tiny
# Protocol 1: Original 1,182 classes with signer overlap
# Frame sampling: T=64 uniform, Kinetics-400 normalization
# ==============================================================================

set -euo pipefail

CSV_PATH="${CSV_PATH:-data/splits_original_1182/ISLR_promax.csv}"
VIDEO_DIR="${VIDEO_DIR:-/path/to/videos}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/table4_swin3d}"
SEED="${SEED:-42}"

echo "=================================================================="
echo "Training Swin3D-Tiny on 1,182 Classes (T=64, Seed: ${SEED})"
echo "=================================================================="

python src/training/train_transformer.py \
    --csv_path "${CSV_PATH}" \
    --video_dir "${VIDEO_DIR}" \
    --model_name "swin3d_t" \
    --save_dir "${OUTPUT_DIR}" \
    --frames 64 \
    --batch_size 4 \
    --epochs 35 \
    --lr_backbone 5e-5 \
    --lr_head 5e-4 \
    --weight_decay 1e-4 \
    --label_smoothing 0.05 \
    --freeze_early \
    --seed "${SEED}"

echo "Finished reproducing Swin3D-Tiny."
