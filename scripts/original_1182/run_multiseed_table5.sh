#!/usr/bin/env bash
# ==============================================================================
# Reproduce Table 5: Multi-Seed Stability Benchmark across 7 Seeds
# Seeds: [42, 123, 2024, 7, 19, 88, 4096]
# Selected Models: Swin3D, X3D-M, ResNet-50+Mean, EfficientNet-B0+GRU, ViTPose+TGCN
# ==============================================================================

set -euo pipefail

CSV_PATH="${CSV_PATH:-data/splits_original_1182/ISLR_promax.csv}"
VIDEO_DIR="${VIDEO_DIR:-/path/to/videos}"
FEATURE_DIR="${FEATURE_DIR:-features/original_1182}"
POSE_DIR="${POSE_DIR:-data/pose_features/vitpose}"
OUTPUT_BASE="${OUTPUT_BASE:-checkpoints/table5_multiseed}"

SEEDS=(42 123 2024 7 19 88 4096)

echo "=================================================================="
echo "Running Multi-Seed Stability Experiments across 7 Seeds"
echo "=================================================================="

for s in "${SEEDS[@]}"; do
    echo "============================================="
    echo ">>> Running Seed: ${s} <<<"
    echo "============================================="

    # 1. ResNet-50 + Mean
    python src/training/train_2d_temporal.py \
        --csv_path "${CSV_PATH}" \
        --feature_dir "${FEATURE_DIR}/resnet50" \
        --save_dir "${OUTPUT_BASE}/resnet50_mean_seed${s}" \
        --temporal_type mean \
        --backbone resnet50 \
        --seed "${s}"

    # 2. EfficientNet-B0 + GRU
    python src/training/train_2d_temporal.py \
        --csv_path "${CSV_PATH}" \
        --feature_dir "${FEATURE_DIR}/efficientnet_b0" \
        --save_dir "${OUTPUT_BASE}/efficientnet_b0_gru_seed${s}" \
        --temporal_type gru \
        --backbone efficientnet_b0 \
        --seed "${s}"

    # 3. ViTPose + TGCN
    python src/training/train_tgcn.py \
        --csv_path "${CSV_PATH}" \
        --pose_dir "${POSE_DIR}" \
        --save_dir "${OUTPUT_BASE}/vitpose_tgcn_seed${s}" \
        --seed "${s}"

    # 4. X3D-M
    python src/training/train_3d.py \
        --csv_path "${CSV_PATH}" \
        --video_dir "${VIDEO_DIR}" \
        --model_name x3d_m \
        --save_dir "${OUTPUT_BASE}/x3d_seed${s}" \
        --seed "${s}"

    # 5. Swin3D-Tiny
    python src/training/train_transformer.py \
        --csv_path "${CSV_PATH}" \
        --video_dir "${VIDEO_DIR}" \
        --model_name swin3d_t \
        --save_dir "${OUTPUT_BASE}/swin3d_seed${s}" \
        --seed "${s}"
done

echo "Aggregating Table 5 results..."
python src/evaluation/compute_table5_metrics.py --results_dir "${OUTPUT_BASE}" --output_csv "${OUTPUT_BASE}/table5_summary.csv"
