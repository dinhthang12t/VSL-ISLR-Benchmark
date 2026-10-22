#!/usr/bin/env bash
# ==============================================================================
# Reproduce Table 6: Exact Paired Permutation Test with Holm Correction
# Compares:
#   1. Swin3D vs X3D
#   2. ViTPose + TGCN vs ResNet-50 + Mean
# ==============================================================================

set -euo pipefail

OUTPUT_CSV="${OUTPUT_CSV:-checkpoints/table6_significance.csv}"

echo "=================================================================="
echo "Computing Exact Paired Permutation Test for Table 6"
echo "=================================================================="

python src/evaluation/permutation_test.py --output_csv "${OUTPUT_CSV}"

echo "Table 6 evaluation complete."
