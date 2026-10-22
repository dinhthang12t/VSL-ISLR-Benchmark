"""
Universal Evaluation Script for all ISLR Models.
Supports 2D CNN + Temporal heads, 3D CNNs, Video Transformers, and Pose TGCN.
Evaluates Top-1, Top-5, and Top-10 accuracies on test set.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.datasets.transforms import CenterCrop, VideoCompose, VideoNormalizeKinetics
from src.datasets.video_dataset import VideoDataset
from src.datasets.feature_dataset import FeatureDataset
from src.datasets.pose_dataset import PoseDataset
from src.models.cnn2d_temporal import MeanPoolingClassifier, SignLanguageGRU, SignLanguageLSTM
from src.models.cnn3d import build_3d_model
from src.models.transformers import build_transformer_model
from src.models.tgcn import PoseTGCN


BACKBONE_DIMS = {
    "resnet18": 512,
    "resnet50": 2048,
    "efficientnet_b0": 1280,
    "convnext_tiny": 768,
}


def load_model(args, num_classes, device):
    mtype = args.model_type
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    # Clean keys if saved with wrapper prefixes (e.g. 'model.' or 'module.')
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned_state_dict[k[6:]] = v
        elif k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    if mtype in ["2d_mean", "2d_gru", "2d_lstm"]:
        feat_dim = BACKBONE_DIMS.get(args.backbone, 2048)
        if mtype == "2d_mean":
            model = MeanPoolingClassifier(feat_dim=feat_dim, num_classes=num_classes, dropout=0.3)
        elif mtype == "2d_gru":
            model = SignLanguageGRU(feat_dim=feat_dim, hidden_dim=256, num_classes=num_classes,
                                    num_layers=2, dropout=0.3, head_dropout=0.3)
        elif mtype == "2d_lstm":
            model = SignLanguageLSTM(feat_dim=feat_dim, hidden_dim=256, num_classes=num_classes,
                                     num_layers=2, dropout=0.3, head_dropout=0.3)
    elif mtype == "3d":
        model = build_3d_model(model_name=args.model_name, num_classes=num_classes,
                               freeze_early=False, pretrained=False)
    elif mtype == "transformer":
        model = build_transformer_model(model_name=args.model_name, num_classes=num_classes,
                                        freeze_early=False, pretrained=False)
    elif mtype == "tgcn":
        model = PoseTGCN(num_nodes=55, num_classes=num_classes, feat_dim=2, dropout=0.3)
    else:
        raise ValueError(f"Unknown model_type: {mtype}")

    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Universal ISLR Checkpoint Evaluator")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["2d_mean", "2d_gru", "2d_lstm", "3d", "transformer", "tgcn"])
    parser.add_argument("--model_name", type=str, default="x3d_m",
                        help="Architecture name for 3d/transformer (e.g. x3d_m, swin3d_t)")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        help="Backbone name for 2D temporal models (e.g. resnet50, efficientnet_b0)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--csv_path", type=str, required=True, help="Manifest CSV file")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Data root (video folder, feature folder, or pose folder)")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_json", type=str, default=None, help="Optional output JSON report")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Loading checkpoint: {args.checkpoint}")

    df = pd.read_csv(args.csv_path)
    label_col = "Meaning" if "Meaning" in df.columns else ("gloss" if "gloss" in df.columns else "label")
    classes = sorted(df[label_col].unique())
    class_to_id = {cls_name: i for i, cls_name in enumerate(classes)}
    num_classes = len(classes)

    test_df = df[df["split"] == "test"].copy() if "split" in df.columns else df.copy()

    # Create dataset
    if args.model_type in ["2d_mean", "2d_gru", "2d_lstm"]:
        test_dataset = FeatureDataset(test_df, args.data_dir, class_to_id, target_frames=args.frames)
    elif args.model_type in ["3d", "transformer"]:
        crop_size = 112 if args.model_name == "c3d" else 224
        short_side = 128 if args.model_name == "c3d" else 256
        transform = VideoCompose([
            CenterCrop(crop_size),
            VideoNormalizeKinetics(),
        ])
        test_dataset = VideoDataset(test_df, args.data_dir, class_to_id, num_frames=args.frames,
                                    transform=transform, short_side=short_side)
    elif args.model_type == "tgcn":
        test_dataset = PoseDataset(test_df, args.data_dir, class_to_id,
                                   target_frames=args.frames, augment=False)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = load_model(args, num_classes, device)

    total_samples = 0
    top1, top5, top10 = 0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets, _ in tqdm(test_loader, desc="Testing", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            bs = targets.size(0)

            logits = model(inputs)
            maxk = min(10, logits.size(1))
            _, pred = logits.topk(maxk, 1, True, True)
            correct = pred.eq(targets.view(-1, 1).expand_as(pred))

            total_samples += bs
            top1 += correct[:, :1].sum().item()
            if maxk >= 5:
                top5 += correct[:, :5].sum().item()
            if maxk >= 10:
                top10 += correct[:, :10].sum().item()

            all_preds.extend(pred[:, 0].cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    acc1 = 100.0 * top1 / max(total_samples, 1)
    acc5 = 100.0 * top5 / max(total_samples, 1)
    acc10 = 100.0 * top10 / max(total_samples, 1)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total Test Samples: {total_samples}")
    print(f"Top-1 Accuracy:     {acc1:.2f}%")
    print(f"Top-5 Accuracy:     {acc5:.2f}%")
    print(f"Top-10 Accuracy:    {acc10:.2f}%")
    print("=" * 50)

    results = {
        "model_type": args.model_type,
        "model_name": args.model_name if args.model_type in ["3d", "transformer"] else args.backbone,
        "checkpoint": args.checkpoint,
        "num_samples": total_samples,
        "top1": acc1,
        "top5": acc5,
        "top10": acc10,
    }

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
