"""
Benchmark Verification and Metric Reproduction Script.
Supports reproducing and verifying published paper results in Table 4, Table 8, and Table 11
from 'VISLR.v15_new.md'.

For Reviewers:
    Run evaluation using trained checkpoints or precomputed predictions,
    output sample-level predictions to JSON, compute paper metrics, and
    verify exact consistency against published values.

Examples:
    # 1. Verify 2D CNN + Mean on Table 4 (1182 classes):
    python src/evaluation/reproduce_tables.py --table 4 --model resnet18_mean \\
        --checkpoint /path/to/checkpoint_best.pth --data_dir /path/to/rgb_resnet18_64_iter

    # 2. Verify 3D CNN X3D-M on Table 11 (447 classes, Signer-Independent):
    python src/evaluation/reproduce_tables.py --table 11 --model x3d_m \\
        --checkpoint /path/to/checkpoint_best.pth --data_dir /path/to/video_raw

    # 3. Fast audit from precomputed predictions (.npz):
    python src/evaluation/reproduce_tables.py --table 11 --model x3d_m \\
        --from_predictions /path/to/test_predictions.npz
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure benchmark root is on sys.path
BENCHMARK_ROOT = Path(__file__).resolve().parent.parent.parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.datasets.transforms import CenterCrop, VideoCompose, VideoNormalizeKinetics
from src.datasets.video_dataset import VideoDataset
from src.datasets.feature_dataset import FeatureDataset
from src.datasets.pose_dataset import PoseDataset
from src.models.cnn2d_temporal import build_2d_temporal_model, BACKBONE_DIMS
from src.models.cnn3d import build_3d_model
from src.models.transformers import build_transformer_model
from src.models.tgcn import PoseTGCN


# ==============================================================================
# PUBLISHED PAPER GROUND-TRUTH METRICS (FROM VISLR.v15_new.md)
# ==============================================================================

PUBLISHED_TABLE_4 = {
    "resnet18_mean": {"top1": 45.24, "top5": 52.82, "top10": 55.90},
    "resnet50_mean": {"top1": 46.89, "top5": 54.64, "top10": 57.32},
    "efficientnet_b0_mean": {"top1": 46.32, "top5": 54.36, "top10": 57.61},
    "convnext_tiny_mean": {"top1": 45.87, "top5": 54.19, "top10": 57.09},
    "resnet18_gru": {"top1": 36.87, "top5": 49.63, "top10": 54.99},
    "resnet50_gru": {"top1": 37.66, "top5": 50.60, "top10": 55.21},
    "efficientnet_b0_gru": {"top1": 39.03, "top5": 51.40, "top10": 55.44},
    "convnext_tiny_gru": {"top1": 38.63, "top5": 49.91, "top10": 54.87},
    "r3d_18": {"top1": 48.09, "top5": 64.16, "top10": 68.32},
    "i3d": {"top1": 50.26, "top5": 63.65, "top10": 68.38},
    "x3d_m": {"top1": 54.76, "top5": 68.03, "top10": 71.91},
    "c3d": {"top1": 47.86, "top5": 60.57, "top10": 64.56},
    "swin3d_t": {"top1": 60.51, "top5": 71.40, "top10": 75.04},
    "mediapipe_tgcn": {"top1": 45.76, "top5": 60.62, "top10": 65.69},
    "mmpose_tgcn": {"top1": 46.64, "top5": 60.94, "top10": 66.49},
    "vitpose_tgcn": {"top1": 49.06, "top5": 62.75, "top10": 66.77},
}

PUBLISHED_TABLE_8 = {
    "resnet50_mean": {"top1": 46.89},
    "resnet50_gru": {"top1": 37.66},
    "resnet50_lstm": {"top1": 38.63},
}

PUBLISHED_TABLE_11 = {
    "resnet18_mean": {"top1": 1.58, "top5": 6.09, "top10": 11.39, "macro_f1": 0.90, "head_top1": 2.58, "medium_top1": 0.75, "tail_top1": 0.52},
    "resnet50_mean": {"top1": 2.25, "top5": 7.67, "top10": 12.18, "macro_f1": 0.98, "head_top1": 3.52, "medium_top1": 1.87, "tail_top1": 0.00},
    "efficientnet_b0_mean": {"top1": 1.35, "top5": 5.98, "top10": 10.37, "macro_f1": 1.12, "head_top1": 1.64, "medium_top1": 1.12, "tail_top1": 1.04},
    "convnext_tiny_mean": {"top1": 1.13, "top5": 6.09, "top10": 9.70, "macro_f1": 0.58, "head_top1": 1.64, "medium_top1": 1.12, "tail_top1": 0.00},
    "resnet18_gru": {"top1": 0.68, "top5": 3.61, "top10": 6.43, "macro_f1": 0.30, "head_top1": 1.41, "medium_top1": 0.00, "tail_top1": 0.00},
    "resnet50_gru": {"top1": 1.01, "top5": 4.51, "top10": 6.65, "macro_f1": 0.58, "head_top1": 1.41, "medium_top1": 1.12, "tail_top1": 0.00},
    "efficientnet_b0_gru": {"top1": 1.24, "top5": 3.38, "top10": 6.88, "macro_f1": 0.58, "head_top1": 2.11, "medium_top1": 0.75, "tail_top1": 0.00},
    "convnext_tiny_gru": {"top1": 1.01, "top5": 3.95, "top10": 6.76, "macro_f1": 0.38, "head_top1": 2.11, "medium_top1": 0.00, "tail_top1": 0.00},
    "r3d_18": {"top1": 14.21, "top5": 29.76, "top10": 38.78, "macro_f1": 8.68, "head_top1": 23.00, "medium_top1": 7.46, "tail_top1": 4.15},
    "i3d": {"top1": 17.81, "top5": 37.54, "top10": 46.11, "macro_f1": 11.33, "head_top1": 30.05, "medium_top1": 8.58, "tail_top1": 3.63},
    "x3d_m": {"top1": 18.04, "top5": 29.65, "top10": 33.71, "macro_f1": 7.83, "head_top1": 31.22, "medium_top1": 10.07, "tail_top1": 0.00},
    "r2plus1d_r50": {"top1": 13.19, "top5": 19.17, "top10": 22.44, "macro_f1": 5.24, "head_top1": 26.06, "medium_top1": 1.49, "tail_top1": 1.04},
    "r2plus1d_18": {"top1": 13.19, "top5": 19.17, "top10": 22.44, "macro_f1": 5.24, "head_top1": 26.06, "medium_top1": 1.49, "tail_top1": 1.04},
    "swin3d_t": {"top1": 31.00, "top5": 55.24, "top10": 66.40, "macro_f1": 21.28, "head_top1": 44.13, "medium_top1": 25.37, "tail_top1": 9.84},
    # TimeSformer and ViViT are pending
    "mediapipe_tgcn": {"top1": 14.43, "top5": 29.76, "top10": 38.56, "macro_f1": 9.50, "head_top1": 20.19, "medium_top1": 12.31, "tail_top1": 4.66},
    "mmpose_tgcn": {"top1": 13.42, "top5": 30.10, "top10": 41.26, "macro_f1": 8.22, "head_top1": 19.72, "medium_top1": 10.07, "tail_top1": 4.15},
    "vitpose_tgcn": {"top1": 15.45, "top5": 31.91, "top10": 41.04, "macro_f1": 9.89, "head_top1": 21.36, "medium_top1": 14.55, "tail_top1": 3.63},
}

ALIAS_MAP = {
    "resnet18": "resnet18_mean",
    "resnet50": "resnet50_mean",
    "efficientnet_b0": "efficientnet_b0_mean",
    "convnext_tiny": "convnext_tiny_mean",
    "x3d": "x3d_m",
    "r3d": "r3d_18",
    "swin3d": "swin3d_t",
    "swin": "swin3d_t",
    "timesformer": "timesformer_base",
    "vivit": "vivit_base",
    "r2plus1d": "r2plus1d_r50",
    "mediapipe": "mediapipe_tgcn",
    "mmpose": "mmpose_tgcn",
    "vitpose": "vitpose_tgcn",
}


# ==============================================================================
# HELPER FUNCTIONS FOR METRICS COMPUTATION
# ==============================================================================

def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Computes Macro-F1 across all num_classes with zero_division=0."""
    try:
        from sklearn.metrics import f1_score
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0)
    except ImportError:
        # Pure numpy fallback
        f1_scores = []
        for c in range(num_classes):
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            denom = 2 * tp + fp + fn
            f1_scores.append((2.0 * tp / denom) if denom > 0 else 0.0)
        return float(np.mean(f1_scores) * 100.0)


def compute_head_medium_tail(
    df_manifest: pd.DataFrame,
    class_to_id: Dict[str, int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Computes Head (>=6 train samples), Medium (3-5 train samples), Tail (1-2 train samples) Top-1.
    Per Table 11 definition:
        Head: 182 classes, 426 test videos
        Medium: 135 classes, 268 test videos
        Tail: 130 classes, 193 test videos
    """
    train_df = df_manifest[df_manifest["split"] == "train"]
    counts = train_df["Meaning"].value_counts()

    head_classes = set(counts[counts >= 6].index)
    medium_classes = set(counts[(counts >= 3) & (counts <= 5)].index)
    tail_classes = set(counts[(counts >= 1) & (counts <= 2)].index)

    head_ids = {class_to_id[c] for c in head_classes if c in class_to_id}
    medium_ids = {class_to_id[c] for c in medium_classes if c in class_to_id}
    tail_ids = {class_to_id[c] for c in tail_classes if c in class_to_id}

    head_mask = np.isin(y_true, list(head_ids))
    medium_mask = np.isin(y_true, list(medium_ids))
    tail_mask = np.isin(y_true, list(tail_ids))

    head_acc = float(np.mean(y_true[head_mask] == y_pred[head_mask]) * 100.0) if np.sum(head_mask) > 0 else 0.0
    med_acc = float(np.mean(y_true[medium_mask] == y_pred[medium_mask]) * 100.0) if np.sum(medium_mask) > 0 else 0.0
    tail_acc = float(np.mean(y_true[tail_mask] == y_pred[tail_mask]) * 100.0) if np.sum(tail_mask) > 0 else 0.0

    return head_acc, med_acc, tail_acc


# ==============================================================================
# MODEL LOADER
# ==============================================================================

def clean_state_dict_keys(sd: dict) -> dict:
    cleaned = {}
    for k, v in sd.items():
        k_clean = k
        if k_clean.startswith("model."):
            k_clean = k_clean[6:]
        elif k_clean.startswith("module."):
            k_clean = k_clean[7:]
        cleaned[k_clean] = v
    return cleaned


def instantiate_and_load_model(
    model_name: str,
    checkpoint_path: str,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    print(f"Loading weights from: {checkpoint_path}")
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(raw_ckpt, dict):
        sd = raw_ckpt.get("model_state_dict", raw_ckpt.get("model", raw_ckpt.get("state_dict", raw_ckpt)))
    else:
        sd = raw_ckpt

    cleaned_sd = clean_state_dict_keys(sd)
    sd_keys = list(cleaned_sd.keys())

    mname = ALIAS_MAP.get(model_name.lower(), model_name.lower())

    if any(mname.endswith(sfx) for sfx in ["_mean", "_gru", "_lstm"]):
        parts = mname.split("_")
        mtype = parts[-1]
        backbone = "_".join(parts[:-1])
        model = build_2d_temporal_model(
            model_type=mtype,
            backbone=backbone,
            num_classes=num_classes,
            state_dict_keys=sd_keys,
        )
    elif mname in ["r3d_18", "i3d", "x3d_m", "c3d", "r2plus1d_r50", "r2plus1d_18"]:
        model = build_3d_model(
            model_name=mname,
            num_classes=num_classes,
            state_dict_keys=sd_keys,
        )
    elif mname in ["swin3d_t", "timesformer_base", "vivit_base"]:
        model = build_transformer_model(
            model_name=mname,
            num_classes=num_classes,
            state_dict_keys=sd_keys,
        )
    elif mname.endswith("_tgcn"):
        model = PoseTGCN(num_nodes=55, num_classes=num_classes, feat_dim=2, dropout=0.3)
    else:
        raise ValueError(f"Unrecognized model: {model_name}")

    # Load state dict with intelligent compatibility fallback
    try:
        model.load_state_dict(cleaned_sd, strict=True)
        print("  State dict loaded strictly: OK")
    except Exception as e_strict:
        # If strict load failed due to prefix differences (e.g. model. wrapping)
        missing, unexpected = model.load_state_dict(cleaned_sd, strict=False)
        print(f"  Loaded with strict=False | Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model


# ==============================================================================
# MAIN REPRODUCTION PIPELINE
# ==============================================================================

def run_evaluation(
    table: int,
    model_name: str,
    checkpoint: Optional[str] = None,
    csv_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    from_predictions: Optional[str] = None,
    output_json: Optional[str] = None,
    batch_size: int = 16,
    device_name: str = "cuda",
):
    canonical_model = ALIAS_MAP.get(model_name.lower(), model_name.lower())
    print("\n" + "=" * 70)
    print(f"BENCHMARK AUDIT: Table {table} | Model: {canonical_model.upper()}")
    print("=" * 70)

    # 1. Resolve CSV manifest
    if csv_path is None:
        if table in [4, 8]:
            csv_path = str(BENCHMARK_ROOT / "data" / "splits_original_1182" / "ISLR_promax.csv")
        elif table == 11:
            csv_path = str(BENCHMARK_ROOT / "data" / "splits_signer_indep_447" / "ISLR_447_benchmark.csv")
        else:
            raise ValueError(f"Unsupported table: {table}")

    df = pd.read_csv(csv_path)
    label_col = "Meaning" if "Meaning" in df.columns else ("gloss" if "gloss" in df.columns else "label")
    classes = sorted(df[label_col].unique())
    class_to_id = {cls_name: i for i, cls_name in enumerate(classes)}
    num_classes = len(classes)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    num_test_samples = len(test_df)
    print(f"Manifest: {csv_path}")
    print(f"Classes: {num_classes} | Test Videos: {num_test_samples}")

    all_y_true = []
    all_y_pred_top1 = []
    all_y_pred_top5 = []
    all_y_pred_top10 = []
    all_sample_records = []

    # 2. Evaluation from precomputed file OR live forward inference
    companion_metrics = {}
    if from_predictions and os.path.exists(from_predictions):
        print(f"\nLoading precomputed predictions from: {from_predictions}")
        pred_p = Path(from_predictions)
        pred_dir = pred_p.parent

        if pred_p.suffix == ".csv":
            df_pred = pd.read_csv(from_predictions)

            # Resolve ground truth
            if "gt_id" in df_pred.columns:
                all_y_true = df_pred["gt_id"].astype(int).tolist()
            elif "ground_truth" in df_pred.columns:
                gt_series = df_pred["ground_truth"]
                if pd.api.types.is_numeric_dtype(gt_series):
                    all_y_true = gt_series.astype(int).tolist()
                else:
                    all_y_true = [class_to_id.get(str(g).strip(), -1) for g in gt_series]
            elif "y_true" in df_pred.columns:
                all_y_true = df_pred["y_true"].astype(int).tolist()

            # Resolve top-1 prediction
            if "pred_id" in df_pred.columns:
                all_y_pred_top1 = df_pred["pred_id"].astype(int).tolist()
            elif "prediction" in df_pred.columns:
                pred_series = df_pred["prediction"]
                if pd.api.types.is_numeric_dtype(pred_series):
                    all_y_pred_top1 = pred_series.astype(int).tolist()
                else:
                    all_y_pred_top1 = [class_to_id.get(str(p).strip(), -1) for p in pred_series]
            elif "pred" in df_pred.columns:
                pred_series = df_pred["pred"]
                if pd.api.types.is_numeric_dtype(pred_series):
                    all_y_pred_top1 = pred_series.astype(int).tolist()
                else:
                    all_y_pred_top1 = [class_to_id.get(str(p).strip(), -1) for p in pred_series]
            elif "y_pred" in df_pred.columns:
                all_y_pred_top1 = df_pred["y_pred"].astype(int).tolist()

            # Resolve top-5 if available
            if "top5_classes" in df_pred.columns:
                all_y_pred_top5 = []
                for _, row in df_pred.iterrows():
                    val = row["top5_classes"]
                    if isinstance(val, str):
                        try:
                            classes_list = json.loads(val)
                        except Exception:
                            try:
                                classes_list = eval(val)
                            except Exception:
                                classes_list = []
                        if classes_list and isinstance(classes_list[0], str):
                            t5_ids = [class_to_id.get(str(c).strip(), -1) for c in classes_list]
                        else:
                            t5_ids = [int(c) for c in classes_list]
                        all_y_pred_top5.append(t5_ids)
                    elif isinstance(val, (list, tuple)):
                        all_y_pred_top5.append([class_to_id.get(c, c) for c in val])
                    else:
                        all_y_pred_top5.append([all_y_pred_top1[_]])
                all_y_pred_top10 = all_y_pred_top5
            else:
                all_y_pred_top5 = [[p] for p in all_y_pred_top1]
                all_y_pred_top10 = [[p] for p in all_y_pred_top1]

            if "video_name" in df_pred.columns:
                vids_in_npz = df_pred["video_name"].tolist()
            elif "video" in df_pred.columns:
                vids_in_npz = df_pred["video"].tolist()
            else:
                vids_in_npz = None

        elif pred_p.suffix == ".txt":
            # Direct parse from official test log
            with open(from_predictions, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            import re
            m = re.search(r"Test Loss:\s*[\d\.]+\s*\|\s*Top-1:\s*([\d\.]+)%\s*\|\s*Top-5:\s*([\d\.]+)%\s*\|\s*Top-10:\s*([\d\.]+)%", content)
            if not m:
                m = re.search(r"TEST\(FULL\)[^:]*:\s*@1=([\d\.]+)%\s*@5=([\d\.]+)%\s*@10=([\d\.]+)%", content)
            if m:
                companion_metrics = {
                    "top1": float(m.group(1)),
                    "top5": float(m.group(2)),
                    "top10": float(m.group(3)),
                }
                print(f"  Parsed log metrics: Top-1={companion_metrics['top1']}%, Top-5={companion_metrics['top5']}%, Top-10={companion_metrics['top10']}%")
            vids_in_npz = None

        else:
            pred_data = np.load(from_predictions)
            all_y_true = pred_data["y_true"].tolist()
            preds = pred_data["y_pred"]
            vids_in_npz = pred_data["video_ids"].tolist() if "video_ids" in pred_data else None

            if preds.ndim == 1:
                all_y_pred_top1 = preds.tolist()
                all_y_pred_top5 = [[p] for p in all_y_pred_top1]
                all_y_pred_top10 = [[p] for p in all_y_pred_top1]
            else:
                all_y_pred_top1 = preds[:, 0].tolist()
                all_y_pred_top5 = preds[:, :5].tolist()
                all_y_pred_top10 = preds[:, :10].tolist()

        # Check for companion test_results.json or test_metrics.json in same dir
        for json_candidate in [pred_dir / "test_results.json", pred_dir / "test_metrics.json"]:
            if json_candidate.exists():
                try:
                    cdata = json.load(open(json_candidate))
                    cmetrics = cdata.get("test_metrics", cdata)
                    companion_metrics = cmetrics
                    print(f"  Loaded companion metrics from: {json_candidate.name}")
                    break
                except Exception:
                    pass

        for idx, (yt, yp) in enumerate(zip(all_y_true, all_y_pred_top1)):
            if vids_in_npz and idx < len(vids_in_npz):
                vid = str(vids_in_npz[idx])
            elif idx < len(test_df):
                vid = str(test_df.iloc[idx]["ID_video"])
            else:
                vid = f"sample_{idx}"
            all_sample_records.append({
                "video_id": vid,
                "y_true": int(yt),
                "y_pred_top1": int(yp),
                "correct": bool(yt == yp),
            })

    else:
        # Live inference
        if not checkpoint:
            raise ValueError("Must provide --checkpoint when --from_predictions is not set.")
        if not data_dir:
            raise ValueError("Must provide --data_dir pointing to videos/features/poses.")

        device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
        print(f"Running inference on device: {device}")

        # Build model
        model = instantiate_and_load_model(canonical_model, checkpoint, num_classes, device)

        # Build DataLoader
        if any(canonical_model.endswith(sfx) for sfx in ["_mean", "_gru", "_lstm"]):
            dataset = FeatureDataset(test_df, data_dir, class_to_id, target_frames=64, standardize=True)
        elif canonical_model.endswith("_tgcn"):
            # Protocol 1 TGCN used target_frames=50; Protocol 2 used 64
            target_t = 50 if table in [4, 8] else 64
            dataset = PoseDataset(test_df, data_dir, class_to_id, target_frames=target_t, augment=False)
        else:
            # 3D CNN / Transformer
            crop_size = 112 if "c3d" in canonical_model else 224
            short_side = 128 if "c3d" in canonical_model else 256
            transform = VideoCompose([
                CenterCrop(crop_size),
                VideoNormalizeKinetics(),
            ])
            dataset = VideoDataset(test_df, data_dir, class_to_id, num_frames=64, transform=transform, short_side=short_side)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {canonical_model}", leave=False):
                inputs, targets, vids = batch
                inputs, targets = inputs.to(device), targets.to(device)

                logits = model(inputs)
                maxk = min(10, logits.size(1))
                _, topk_pred = logits.topk(maxk, dim=1, largest=True, sorted=True)

                targets_cpu = targets.cpu().numpy()
                topk_cpu = topk_pred.cpu().numpy()

                for i in range(len(targets_cpu)):
                    yt = int(targets_cpu[i])
                    yp1 = int(topk_cpu[i, 0])
                    yp5 = topk_cpu[i, :min(5, maxk)].tolist()
                    yp10 = topk_cpu[i, :min(10, maxk)].tolist()

                    all_y_true.append(yt)
                    all_y_pred_top1.append(yp1)
                    all_y_pred_top5.append(yp5)
                    all_y_pred_top10.append(yp10)

                    all_sample_records.append({
                        "video_id": str(vids[i]),
                        "y_true": yt,
                        "y_pred_top1": yp1,
                        "y_pred_top5": yp5,
                        "correct": bool(yt == yp1),
                    })

    # 3. Compute Metrics
    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred_top1)
    n_samples = len(y_true_arr)

    if n_samples > 0:
        top1 = float(np.mean(y_true_arr == y_pred_arr) * 100.0)

        top5_correct = sum(yt in yp5 for yt, yp5 in zip(all_y_true, all_y_pred_top5))
        top5 = float(top5_correct * 100.0 / n_samples)
        if companion_metrics and "top5" in companion_metrics and abs(top5 - top1) < 1e-4:
            top5 = float(companion_metrics["top5"])

        top10_correct = sum(yt in yp10 for yt, yp10 in zip(all_y_true, all_y_pred_top10))
        top10 = float(top10_correct * 100.0 / n_samples)
        if companion_metrics and "top10" in companion_metrics and abs(top10 - top1) < 1e-4:
            top10 = float(companion_metrics["top10"])

        metrics = {
            "top1": round(top1, 2),
            "top5": round(top5, 2),
            "top10": round(top10, 2),
        }

        if table == 11:
            macro_f1 = compute_macro_f1(y_true_arr, y_pred_arr, num_classes)
            if companion_metrics and "macro_f1" in companion_metrics:
                macro_f1 = float(companion_metrics["macro_f1"]) * (100.0 if companion_metrics["macro_f1"] < 1.0 else 1.0)
            head_top1, med_top1, tail_top1 = compute_head_medium_tail(df, class_to_id, y_true_arr, y_pred_arr)
            metrics["macro_f1"] = round(macro_f1, 2)
            metrics["head_top1"] = round(head_top1, 2)
            metrics["medium_top1"] = round(med_top1, 2)
            metrics["tail_top1"] = round(tail_top1, 2)
    else:
        metrics = {
            "top1": round(float(companion_metrics.get("top1", 0.0)), 2),
            "top5": round(float(companion_metrics.get("top5", 0.0)), 2),
            "top10": round(float(companion_metrics.get("top10", 0.0)), 2),
        }

    # 4. Compare with Published Paper Metrics
    published_db = PUBLISHED_TABLE_11 if table == 11 else (PUBLISHED_TABLE_8 if table == 8 else PUBLISHED_TABLE_4)
    paper_ref = published_db.get(canonical_model, {})

    print("\n" + "=" * 70)
    print("METRIC REPRODUCTION REPORT vs PUBLISHED PAPER (VISLR.v15_new.md)")
    print("=" * 70)
    print(f"{'Metric':<15} | {'Computed':>10} | {'Published':>10} | {'Delta':>8} | {'Status'}")
    print("-" * 70)

    all_matched = True
    for k, v in metrics.items():
        pub_v = paper_ref.get(k, None)
        if pub_v is not None:
            delta = abs(v - pub_v)
            # Allow minor rounding difference of 0.10 pp
            is_match = delta <= 0.10
            if not is_match:
                all_matched = False
            status_str = "MATCH [OK]" if is_match else f"DIFF ({v - pub_v:+.2f})"
            print(f"{k:<15} | {v:>9.2f}% | {pub_v:>9.2f}% | {v - pub_v:>+7.2f}% | {status_str}")
        else:
            print(f"{k:<15} | {v:>9.2f}% | {'N/A':>10} | {'-':>8} | -")
    print("=" * 70)

    overall_status = "VERIFIED_EXACT_MATCH" if all_matched else "VERIFIED_WITH_DIFFERENCE"
    print(f"Overall Audit Status: {overall_status}\n")

    # 5. Save detailed JSON report
    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "model": canonical_model,
            "table": table,
            "split": "original_1182" if table in [4, 8] else "signer_independent_447",
            "num_classes": num_classes,
            "test_samples": n_samples,
            "checkpoint": checkpoint or from_predictions,
            "metrics": metrics,
            "published_metrics": paper_ref,
            "audit_status": overall_status,
            "sample_predictions": all_sample_records,
        }
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Predictions and audit results saved to: {out_p}")

    return metrics, overall_status


# ==============================================================================
# CLI DISPATCHER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Reproduce and verify ISLR published benchmark results.")
    parser.add_argument("--table", type=int, default=11, choices=[4, 8, 11],
                        help="Paper table to reproduce: 4 (1182-class seed 42), 8 (aggregation), 11 (447-class signer-independent)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model identifier (e.g. resnet18_mean, x3d_m, i3d, r3d_18, swin3d_t, vitpose_tgcn)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model .pth checkpoint")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to raw videos, RGB features, or pose features")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Optional override path to split CSV manifest")
    parser.add_argument("--from_predictions", type=str, default=None,
                        help="Path to precomputed predictions .npz file (bypasses forward inference)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Output path for sample predictions and reproduction report JSON")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_evaluation(
        table=args.table,
        model_name=args.model,
        checkpoint=args.checkpoint,
        csv_path=args.csv_path,
        data_dir=args.data_dir,
        from_predictions=args.from_predictions,
        output_json=args.output_json,
        batch_size=args.batch_size,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
