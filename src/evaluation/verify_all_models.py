"""
Full Benchmark Audit & Verification Suite for Reviewers.
Runs evaluation across all individual models for Table 4 (Protocol 1) and Table 11 (Protocol 2),
outputs sample-level JSON predictions for every model, and generates a side-by-side
audit table against published values in 'VISLR.v15_new.md'.

Usage:
    python src/evaluation/verify_all_models.py --table 4
    python src/evaluation/verify_all_models.py --table 11
    python src/evaluation/verify_all_models.py --table all
"""

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd
import numpy as np

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent.parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.evaluation.reproduce_tables import run_evaluation, PUBLISHED_TABLE_4, PUBLISHED_TABLE_11, PUBLISHED_TABLE_8


# Known local paths for checkpoints and prediction artifacts
LOCAL_CHECKPOINTS_TABLE_4 = {
    "resnet18_mean": {
        "checkpoint": r"d:\projects\kltn\kltn_reproduce\ckpt\2D_CNN\resnet18_trial1\last_checkpoint.pth",
        "data_dir": r"d:\projects\kltn\data\rgb_resnet18_64_iter",
    },
    "resnet50_mean": {
        "checkpoint": r"d:\projects\kltn\kltn_reproduce\ckpt\2D_CNN\resnet50_iter_42\last_checkpoint.pth",
        "data_dir": r"d:\projects\kltn\data\rgb_resnet50_64_iter",
    },
    "efficientnet_b0_mean": {
        "checkpoint": r"d:\projects\kltn\kltn_reproduce\ckpt\2D_CNN\efficient_trial1\last_checkpoint.pth",
        "data_dir": r"d:\projects\kltn\data\rgb_efficientnet_b0_64_iter",
    },
    "convnext_tiny_mean": {
        "checkpoint": r"d:\projects\kltn\kltn_reproduce\ckpt\2D_CNN\convnext_tiny\last_checkpoint.pth",
        "data_dir": r"d:\projects\kltn\data\rgb_convnext_tiny_64_iter",
    },
    "resnet18_gru": {
        "checkpoint": r"d:\projects\kltn\kltn_reproduce\ckpt\GRU\resnet18_trial5\last_checkpoint.pth",
        "data_dir": r"d:\projects\kltn\data\rgb_resnet18_64_iter",
    },
    "resnet50_gru": {
        "checkpoint": r"d:\projects\kltn\kltn_reproduce\ckpt\GRU\resnet50_trial5\last_checkpoint.pth",
        "data_dir": r"d:\projects\kltn\data\rgb_resnet50_64_iter",
    },
    "efficientnet_b0_gru": {
        "checkpoint": r"d:\projects\kltn\kltn_reproduce\ckpt\GRU\efficient_trial5\last_checkpoint.pth",
        "data_dir": r"d:\projects\kltn\data\rgb_efficientnet_b0_64_iter",
    },
    "convnext_tiny_gru": {
        "checkpoint": r"d:\projects\kltn\kltn_reproduce\ckpt\GRU\convnext_tiny_trial5\last_checkpoint.pth",
        "data_dir": r"d:\projects\kltn\data\rgb_convnext_tiny_64_iter",
    },
    "resnet50_lstm": {
        "checkpoint": r"d:\projects\kltn\kltn_reproduce\ckpt\LSTM\resnet50\last_checkpoint.pth",
        "data_dir": r"d:\projects\kltn\data\rgb_resnet50_64_iter",
    },
}

LOCAL_PREDICTIONS_TABLE_4 = {
    "r3d_18": r"d:\projects\kltn\3D\logs\r3d\R3D-18_test_36373.txt",
    "i3d": r"d:\projects\kltn\kltn_reproduce\results\test\3D_CNN\logs_35182\I3D_last_checkpoint.txt",
    "x3d_m": r"d:\projects\kltn\kltn_reproduce\results\test\3D_CNN\logs_35182\X3D_best_model.txt",
    "c3d": r"d:\projects\kltn\kltn_reproduce\results\test\3D_CNN\logs_35182\C3D_best_model.txt",
    "swin3d_t": r"d:\projects\kltn\transformer\logs\test2\swin\swin3d_b\test_swint_(ckpt_job13302)_15525_out.txt",
    "mediapipe_tgcn": r"d:\projects\kltn\reproduce_slurm\logs\TGCN\mediapipe\test_tgcn3_Mediapipe_iter_64_16634.txt",
    "mmpose_tgcn": r"d:\projects\kltn\kltn_reproduce\results\test\Pose_based\logs_35181\mmpose_last_checkpoint.txt",
    "vitpose_tgcn": r"d:\projects\kltn\kltn_reproduce\results\test\Pose_based\logs_35181\vitpose_last_checkpoint.txt",
}

LOCAL_PREDICTIONS_TABLE_11 = {
    "resnet18_mean": r"d:\projects\kltn\hpc_reproduce\checkpoints\2d_mean\resnet18_seed42_augsimple_protosi_reg_v1\test_predictions.npz",
    "resnet50_mean": r"d:\projects\kltn\hpc_reproduce\checkpoints\2d_mean\resnet50_seed42_augsimple_protosi_reg_v1\test_predictions.npz",
    "efficientnet_b0_mean": r"d:\projects\kltn\hpc_reproduce\checkpoints\2d_mean\efficientnet_b0_seed42_augsimple_protosi_reg_v1\test_predictions.npz",
    "convnext_tiny_mean": r"d:\projects\kltn\hpc_reproduce\checkpoints\2d_mean\convnext_tiny_seed42_augsimple_protosi_reg_v1\test_predictions.npz",
    "resnet18_gru": r"d:\projects\kltn\hpc_reproduce\checkpoints\gru\resnet18_seed42_augsimple_protosi_reg_v1\test_predictions.npz",
    "resnet50_gru": r"d:\projects\kltn\hpc_reproduce\checkpoints\gru\resnet50_seed42_augsimple_protosi_reg_v1\test_predictions.npz",
    "efficientnet_b0_gru": r"d:\projects\kltn\hpc_reproduce\checkpoints\gru\efficientnet_b0_seed42_augsimple_protosi_reg_v1\test_predictions.npz",
    "convnext_tiny_gru": r"d:\projects\kltn\hpc_reproduce\checkpoints\gru\convnext_tiny_seed42_augsimple_protosi_reg_v1\test_predictions.npz",
    "r3d_18": r"d:\projects\kltn\new_split\checkpoints\r3d18\development\seed42\test_predictions.npz",
    "i3d": r"d:\projects\kltn\new_split\checkpoints\i3d\final\seed42\test_predictions.npz",
    "x3d_m": r"d:\projects\kltn\new_split\checkpoints\x3d_m\final\seed42\test_predictions.npz",
    "r2plus1d_r50": r"d:\projects\kltn\hpc_reproduce\checkpoints\3d\r2plus1d_r50\r2plus1d_r50_seed42_augsimple_protosi_reg_v1\test_predictions.npz",
    "swin3d_t": r"d:\projects\kltn\new_split\checkpoints\swin3d_tiny\development\seed42\test_predictions.npz",
    # TimeSformer and ViViT are pending
    "mediapipe_tgcn": r"d:\projects\kltn\hpc_reproduce\checkpoints\pose55_tgcn\mediapipe_seed42_protopaper_exact_v1\test_split_predictions.csv",
    "mmpose_tgcn": r"d:\projects\kltn\hpc_reproduce\checkpoints\pose55_tgcn\mmpose_seed42_protopaper_exact_v1\test_split_predictions.csv",
    "vitpose_tgcn": r"d:\projects\kltn\hpc_reproduce\checkpoints\pose55_tgcn\vitpose_seed42_protopaper_exact_v1\test_split_predictions.csv",
}


def verify_table_11(results_dir: Path):
    print("\n" + "=" * 80)
    print("VERIFYING PROTOCOL 2: TABLE 11 (447 Classes, Signer-Independent)")
    print("=" * 80)
    summary_rows = []

    for model_name, pred_path in LOCAL_PREDICTIONS_TABLE_11.items():
        if not os.path.exists(pred_path):
            print(f"Skipping {model_name}: artifact not found at {pred_path}")
            continue

        out_json = results_dir / f"table11_{model_name}.json"
        metrics, status = run_evaluation(
            table=11,
            model_name=model_name,
            from_predictions=pred_path,
            output_json=str(out_json),
        )
        pub = PUBLISHED_TABLE_11.get(model_name, {})
        summary_rows.append({
            "model": model_name,
            "top1_comp": metrics.get("top1"),
            "top1_pub": pub.get("top1"),
            "top5_comp": metrics.get("top5"),
            "top5_pub": pub.get("top5"),
            "f1_comp": metrics.get("macro_f1"),
            "f1_pub": pub.get("macro_f1"),
            "head_comp": metrics.get("head_top1"),
            "head_pub": pub.get("head_top1"),
            "med_comp": metrics.get("medium_top1"),
            "med_pub": pub.get("medium_top1"),
            "tail_comp": metrics.get("tail_top1"),
            "tail_pub": pub.get("tail_top1"),
            "status": status,
        })

    # Print summary table
    print("\n" + "=" * 96)
    print("TABLE 11 FINAL AUDIT SUMMARY (COMPUTED vs PUBLISHED IN VISLR.v15_new.md)")
    print("=" * 96)
    print(f"{'Model':<22} | {'Top-1':^13} | {'Top-5':^13} | {'Macro-F1':^13} | {'Head':^11} | {'Med':^11} | {'Status'}")
    print(f"{'':<22} | {'C / P':^13} | {'C / P':^13} | {'C / P':^13} | {'C / P':^11} | {'C / P':^11} |")
    print("-" * 96)
    for r in summary_rows:
        t1_str = f"{r['top1_comp']:.2f}/{r['top1_pub']:.2f}"
        t5_str = f"{r['top5_comp']:.2f}/{r['top5_pub']:.2f}"
        f1_str = f"{r['f1_comp']:.2f}/{r['f1_pub']:.2f}"
        head_str = f"{r['head_comp']:.2f}/{r['head_pub']:.2f}"
        med_str = f"{r['med_comp']:.2f}/{r['med_pub']:.2f}"
        print(f"{r['model']:<22} | {t1_str:^13} | {t5_str:^13} | {f1_str:^13} | {head_str:^11} | {med_str:^11} | {r['status']}")
    print("=" * 96)


def verify_table_4(results_dir: Path):
    print("\n" + "=" * 80)
    print("VERIFYING PROTOCOL 1: TABLE 4 (1182 Classes, Class-Stratified Overlap)")
    print("=" * 80)
    summary_rows = []

    # 1. Feature-based / checkpoint models (live forward inference)
    for model_name, cfg in LOCAL_CHECKPOINTS_TABLE_4.items():
        ckpt_p = cfg["checkpoint"]
        data_p = cfg["data_dir"]
        if not os.path.exists(ckpt_p) or not os.path.exists(data_p):
            print(f"Skipping {model_name}: checkpoint or data not found.")
            continue

        out_json = results_dir / f"table4_{model_name}.json"
        table_num = 8 if model_name == "resnet50_lstm" else 4
        metrics, status = run_evaluation(
            table=table_num,
            model_name=model_name,
            checkpoint=ckpt_p,
            data_dir=data_p,
            output_json=str(out_json),
        )
        pub_db = PUBLISHED_TABLE_8 if table_num == 8 else PUBLISHED_TABLE_4
        pub = pub_db.get(model_name, {})
        summary_rows.append({
            "model": model_name,
            "top1_comp": metrics.get("top1"),
            "top1_pub": pub.get("top1"),
            "top5_comp": metrics.get("top5"),
            "top5_pub": pub.get("top5"),
            "top10_comp": metrics.get("top10"),
            "top10_pub": pub.get("top10"),
            "status": status,
        })

    # 2. 3D CNN, Transformer, and Pose-based models (from verified prediction / test artifacts)
    for model_name, pred_path in LOCAL_PREDICTIONS_TABLE_4.items():
        if not os.path.exists(pred_path):
            print(f"Skipping {model_name}: artifact not found at {pred_path}")
            continue

        out_json = results_dir / f"table4_{model_name}.json"
        metrics, status = run_evaluation(
            table=4,
            model_name=model_name,
            from_predictions=pred_path,
            output_json=str(out_json),
        )
        pub = PUBLISHED_TABLE_4.get(model_name, {})
        summary_rows.append({
            "model": model_name,
            "top1_comp": metrics.get("top1"),
            "top1_pub": pub.get("top1"),
            "top5_comp": metrics.get("top5"),
            "top5_pub": pub.get("top5"),
            "top10_comp": metrics.get("top10"),
            "top10_pub": pub.get("top10"),
            "status": status,
        })

    print("\n" + "=" * 80)
    print("TABLE 4/8 FINAL AUDIT SUMMARY (COMPUTED vs PUBLISHED IN VISLR.v15_new.md)")
    print("=" * 80)
    print(f"{'Model':<24} | {'Top-1 (C / P)':^18} | {'Top-5 (C / P)':^18} | {'Status'}")
    print("-" * 80)
    for r in summary_rows:
        t1_str = f"{r['top1_comp']:.2f}% / {r['top1_pub']:.2f}%"
        t5_str = f"{r['top5_comp']:.2f}% / {r['top5_pub']:.2f}%" if r.get("top5_pub") else f"{r['top5_comp']:.2f}% / N/A"
        print(f"{r['model']:<24} | {t1_str:^18} | {t5_str:^18} | {r['status']}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run complete benchmark audit suite.")
    parser.add_argument("--table", type=str, default="11", choices=["4", "11", "all"],
                        help="Table to audit: 4, 11, or all")
    parser.add_argument("--results_dir", type=str, default=str(BENCHMARK_ROOT / "results"),
                        help="Directory to store audit prediction JSON files")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.table in ["11", "all"]:
        verify_table_11(results_dir)

    if args.table in ["4", "all"]:
        verify_table_4(results_dir)


if __name__ == "__main__":
    main()
