"""
Evaluation Metrics for Table 11: Signer-Independent 447-Class Benchmark.
Computes:
  - Top-1, Top-5, Top-10 Accuracies (%)
  - Macro-F1 across all 447 classes (%)
  - Groupwise Top-1 for Head (>=6 train samples, 182 classes, 426 test samples),
    Medium (3-5 train samples, 135 classes, 268 test samples), and
    Tail (1-2 train samples, 130 classes, 193 test samples).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def get_head_med_tail_classes(train_df: pd.DataFrame, label_col: str = "Meaning"):
    """
    Groups classes by training-set frequencies:
      - Head:   >= 6 videos (182 classes)
      - Medium: 3-5 videos  (135 classes)
      - Tail:   1-2 videos  (130 classes)
    """
    counts = train_df[label_col].value_counts()
    head_cls = set(counts[counts >= 6].index)
    med_cls = set(counts[(counts >= 3) & (counts <= 5)].index)
    tail_cls = set(counts[counts <= 2].index)
    return head_cls, med_cls, tail_cls


def compute_table11_metrics(
    y_true: np.ndarray,
    y_pred_topk: np.ndarray,
    test_meanings: List[str],
    head_classes: set,
    med_classes: set,
    tail_classes: set,
    num_classes: int = 447,
) -> Dict[str, float]:
    """
    Computes all Table 11 benchmark metrics.
    y_true: [N] array of integer labels
    y_pred_topk: [N, K] array of top predicted integer labels (K >= 10)
    test_meanings: [N] list of string class names corresponding to ground truth
    """
    n = len(y_true)
    top1_correct = (y_pred_topk[:, 0] == y_true)
    top5_correct = np.any(y_pred_topk[:, :5] == y_true[:, None], axis=1)
    top10_correct = np.any(y_pred_topk[:, :10] == y_true[:, None], axis=1)

    top1_acc = 100.0 * np.mean(top1_correct)
    top5_acc = 100.0 * np.mean(top5_correct)
    top10_acc = 100.0 * np.mean(top10_correct)

    # Macro-F1 across all 447 classes
    macro_f1 = 100.0 * f1_score(y_true, y_pred_topk[:, 0], average="macro", zero_division=0)

    # Groupwise Top-1 Accuracy
    head_mask = np.array([m in head_classes for m in test_meanings])
    med_mask = np.array([m in med_classes for m in test_meanings])
    tail_mask = np.array([m in tail_classes for m in test_meanings])

    head_top1 = 100.0 * np.mean(top1_correct[head_mask]) if np.sum(head_mask) > 0 else 0.0
    med_top1 = 100.0 * np.mean(top1_correct[med_mask]) if np.sum(med_mask) > 0 else 0.0
    tail_top1 = 100.0 * np.mean(top1_correct[tail_mask]) if np.sum(tail_mask) > 0 else 0.0

    return {
        "Top-1": top1_acc,
        "Top-5": top5_acc,
        "Top-10": top10_acc,
        "Macro-F1": macro_f1,
        "Head Top-1": head_top1,
        "Medium Top-1": med_top1,
        "Tail Top-1": tail_top1,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Table 11 Metrics (Signer-Independent)")
    parser.add_argument("--csv_path", type=str,
                        default="data/splits_signer_indep_447/ISLR_447_benchmark.csv",
                        help="Path to ISLR_447_benchmark.csv")
    parser.add_argument("--preds_file", type=str, required=True,
                        help="Path to CSV containing cols: ID_video, y_true, y_pred or topk preds")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional output summary path")
    args = parser.parse_args()

    df_manifest = pd.read_csv(args.csv_path)
    train_df = df_manifest[df_manifest["split"] == "train"]
    head_cls, med_cls, tail_cls = get_head_med_tail_classes(train_df)

    preds_df = pd.read_csv(args.preds_file)
    y_true = preds_df["y_true"].values

    # Check if multiple pred columns exist (pred_0, pred_1, ... or just y_pred)
    pred_cols = [c for c in preds_df.columns if c.startswith("pred_") or c.startswith("top_")]
    if pred_cols:
        y_pred_topk = preds_df[pred_cols].values
    else:
        y_pred_topk = preds_df[["y_pred"]].values

    test_meanings = preds_df["Meaning"].tolist() if "Meaning" in preds_df.columns else []

    metrics = compute_table11_metrics(y_true, y_pred_topk, test_meanings, head_cls, med_cls, tail_cls)

    print("\n" + "=" * 70)
    print("TABLE 11: Signer-Independent 447-Class Benchmark Results")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"{k:<15}: {v:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
