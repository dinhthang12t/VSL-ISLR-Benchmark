#!/usr/bin/env bash
# ==============================================================================
# Evaluate Table 11 Metrics: Head / Medium / Tail + Macro-F1
# Groups:
#   - Head: >= 6 train videos (182 classes, 426 test videos)
#   - Medium: 3-5 train videos (135 classes, 268 test videos)
#   - Tail: 1-2 train videos (130 classes, 193 test videos)
# ==============================================================================

set -euo pipefail

CSV_PATH="${CSV_PATH:-data/splits_signer_indep_447/ISLR_447_benchmark.csv}"
PREDS_FILE="${1:-checkpoints/table11_preds.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-checkpoints/table11_metrics_summary.csv}"

echo "=================================================================="
echo "Computing Table 11 Breakdown Metrics (Head/Med/Tail, Macro-F1)"
echo "Predictions file: ${PREDS_FILE}"
echo "=================================================================="

python src/evaluation/metrics_table11.py \
    --csv_path "${CSV_PATH}" \
    --preds_file "${PREDS_FILE}" \
    --output_csv "${OUTPUT_CSV}"

echo "Table 11 evaluation complete."
