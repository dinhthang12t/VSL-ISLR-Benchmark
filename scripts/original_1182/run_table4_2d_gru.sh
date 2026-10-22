#!/usr/bin/env bash
# ==============================================================================
# Reproduce Table 4: 2D CNN + Temporal Models (Mean-pooling & GRU)
# Protocol 1: Original 1,182 classes with signer overlap
# ==============================================================================

set -euo pipefail

CSV_PATH="${CSV_PATH:-data/splits_original_1182/ISLR_promax.csv}"
FEATURE_DIR="${FEATURE_DIR:-features/original_1182}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/table4_2d_temporal}"
SEED="${SEED:-42}"

echo "=================================================================="
echo "Training 2D CNN + Temporal Models on 1,182 Classes (Seed: ${SEED})"
echo "=================================================================="

BACKBONES=("resnet18" "resnet50" "efficientnet_b0" "convnext_tiny")

for bb in "${BACKBONES[@]}"; do
    # 1. Mean-pooling
    echo "--- Training ${bb} + Mean ---"
    python src/training/train_2d_temporal.py \
        --csv_path "${CSV_PATH}" \
        --feature_dir "${FEATURE_DIR}/${bb}" \
        --save_dir "${OUTPUT_DIR}/${bb}_mean" \
        --temporal_type mean \
        --backbone "${bb}" \
        --batch_size 64 \
        --epochs 50 \
        --lr 1e-3 \
        --weight_decay 1e-4 \
        --seed "${SEED}"

    # 2. GRU (2 layers, hidden 256, dropout 0.3)
    echo "--- Training ${bb} + GRU ---"
    python src/training/train_2d_temporal.py \
        --csv_path "${CSV_PATH}" \
        --feature_dir "${FEATURE_DIR}/${bb}" \
        --save_dir "${OUTPUT_DIR}/${bb}_gru" \
        --temporal_type gru \
        --backbone "${bb}" \
        --hidden_dim 256 \
        --num_layers 2 \
        --dropout 0.3 \
        --head_dropout 0.3 \
        --batch_size 64 \
        --epochs 50 \
        --lr 1e-3 \
        --weight_decay 1e-4 \
        --seed "${SEED}"
done

echo "Finished reproducing Table 4 (2D CNN + Temporal)."
