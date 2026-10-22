#!/usr/bin/env bash
# ==============================================================================
# Reproduce Table 11: 2D CNN + Temporal on Signer-Independent 447 Split
# Protocol 2: 447 classes, 4,252 videos across 50 disjoint signers
# ==============================================================================

set -euo pipefail

CSV_PATH="${CSV_PATH:-data/splits_signer_indep_447/ISLR_447_benchmark.csv}"
FEATURE_DIR="${FEATURE_DIR:-features/signer_indep_447}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/table11_2d_temporal}"
SEED="${SEED:-42}"

echo "=================================================================="
echo "Training 2D CNNs on 447 Signer-Independent Classes (Seed: ${SEED})"
echo "=================================================================="

BACKBONES=("resnet18" "resnet50" "efficientnet_b0" "convnext_tiny")

for bb in "${BACKBONES[@]}"; do
    # Mean pooling
    echo "--- Training ${bb} + Mean (447 classes) ---"
    python src/training/train_2d_temporal.py \
        --csv_path "${CSV_PATH}" \
        --feature_dir "${FEATURE_DIR}/${bb}" \
        --save_dir "${OUTPUT_DIR}/${bb}_mean" \
        --temporal_type mean \
        --backbone "${bb}" \
        --seed "${SEED}"

    # GRU
    echo "--- Training ${bb} + GRU (447 classes) ---"
    python src/training/train_2d_temporal.py \
        --csv_path "${CSV_PATH}" \
        --feature_dir "${FEATURE_DIR}/${bb}" \
        --save_dir "${OUTPUT_DIR}/${bb}_gru" \
        --temporal_type gru \
        --backbone "${bb}" \
        --seed "${SEED}"
done

echo "Finished Table 11 (2D temporal models)."
