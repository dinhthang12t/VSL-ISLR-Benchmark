#!/usr/bin/env bash
# ==============================================================================
# Reproduce Table 11: Video Transformers on Signer-Independent 447 Split
# Protocol 2: 447 classes, 4,252 videos across 50 disjoint signers
# Models: Swin3D-Tiny, TimeSformer-Base, ViViT-B
# ==============================================================================

set -euo pipefail

CSV_PATH="${CSV_PATH:-data/splits_signer_indep_447/ISLR_447_benchmark.csv}"
VIDEO_DIR="${VIDEO_DIR:-/path/to/videos}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/table11_transformers}"
SEED="${SEED:-42}"

echo "=================================================================="
echo "Training Video Transformers on 447 Classes (Seed: ${SEED})"
echo "=================================================================="

TRANSFORMERS=("swin3d_t" "timesformer_b" "vivit_b")

for tf in "${TRANSFORMERS[@]}"; do
    echo "--- Training Transformer: ${tf} ---"
    python src/training/train_transformer.py \
        --csv_path "${CSV_PATH}" \
        --video_dir "${VIDEO_DIR}" \
        --model_name "${tf}" \
        --save_dir "${OUTPUT_DIR}/${tf}" \
        --frames 64 \
        --batch_size 4 \
        --epochs 35 \
        --freeze_early \
        --seed "${SEED}"
done

echo "Finished Table 11 (Video Transformers)."
