"""
Training and Evaluation Pipeline for 3D CNNs (X3D-M, R3D-18, I3D, C3D, R(2+1)D-18).
Standardized on Kinetics-400 normalization and T=64 uniform frame sampling.
Supports both Original 1,182 classes (Table 4 & 5) and Signer-Independent 447 classes (Table 11).
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.datasets.transforms import (
    CenterCrop,
    RandomCrop,
    RandomHorizontalFlip,
    VideoCompose,
    VideoNormalizeKinetics,
)
from src.datasets.video_dataset import VideoDataset
from src.models.cnn3d import build_3d_model


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_samples = 0.0, 0
    top1, top5, top10 = 0, 0, 0

    with torch.no_grad():
        for inputs, targets, _ in tqdm(loader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            bs = targets.size(0)

            with autocast(enabled=torch.cuda.is_available()):
                logits = model(inputs)
                if criterion is not None:
                    loss = criterion(logits, targets)
                    total_loss += loss.item() * bs

            maxk = min(10, logits.size(1))
            _, pred = logits.topk(maxk, 1, True, True)
            correct = pred.eq(targets.view(-1, 1).expand_as(pred))

            total_samples += bs
            top1 += correct[:, :1].sum().item()
            if maxk >= 5:
                top5 += correct[:, :5].sum().item()
            if maxk >= 10:
                top10 += correct[:, :10].sum().item()

    avg_loss = total_loss / max(total_samples, 1)
    acc1 = 100.0 * top1 / max(total_samples, 1)
    acc5 = 100.0 * top5 / max(total_samples, 1)
    acc10 = 100.0 * top10 / max(total_samples, 1)
    return avg_loss, acc1, acc5, acc10


def main():
    parser = argparse.ArgumentParser(description="Train / Evaluate 3D CNNs for ISLR")
    parser.add_argument("--csv_path", type=str, required=True, help="Manifest CSV file")
    parser.add_argument("--video_dir", type=str, required=True, help="Root directory containing MP4 videos")
    parser.add_argument("--model_name", type=str, default="x3d_m",
                        choices=["x3d_m", "r3d_18", "i3d", "c3d", "r2plus1d_18"])
    parser.add_argument("--save_dir", type=str, default="checkpoints/3d")
    parser.add_argument("--frames", type=int, default=64, help="Uniformly sampled frames (default: 64)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--freeze_early", action="store_true", default=True,
                        help="Freeze early backbone layers per Table 3")
    parser.add_argument("--no_freeze_early", dest="freeze_early", action="store_false")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for testing or resume")
    parser.add_argument("--csv_100", type=str, default=None, help="Optional subset CSV (e.g. 100collins.csv)")
    parser.add_argument("--csv_300", type=str, default=None, help="Optional subset CSV (e.g. 300collins.csv)")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model_name} | Seed: {args.seed}")

    # Load manifest and build class mapping
    df = pd.read_csv(args.csv_path)
    label_col = "Meaning" if "Meaning" in df.columns else ("gloss" if "gloss" in df.columns else "label")
    classes = sorted(df[label_col].unique())
    class_to_id = {cls_name: i for i, cls_name in enumerate(classes)}
    num_classes = len(classes)
    print(f"Loaded {len(df)} samples across {num_classes} classes.")

    # Spatial resolution: C3D uses 112x112; others use 224x224
    crop_size = 112 if args.model_name == "c3d" else 224
    short_side = 128 if args.model_name == "c3d" else 256

    train_transform = VideoCompose([
        RandomCrop(crop_size),
        RandomHorizontalFlip(0.5),
        VideoNormalizeKinetics(),
    ])
    test_transform = VideoCompose([
        CenterCrop(crop_size),
        VideoNormalizeKinetics(),
    ])

    test_df = df[df["split"] == "test"].copy() if "split" in df.columns else df.copy()
    test_dataset = VideoDataset(test_df, args.video_dir, class_to_id, num_frames=args.frames,
                                transform=test_transform, short_side=short_side)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Build model
    model = build_3d_model(
        model_name=args.model_name,
        num_classes=num_classes,
        freeze_early=args.freeze_early,
        pretrained=True,
    ).to(device)

    # Separate parameter groups for backbone and head
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(h in name for h in ["fc", "classifier", "head", "proj"]):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    start_epoch = 1
    best_val_acc = 0.0

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        cleaned_sd = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                cleaned_sd[k[6:]] = v
            elif k.startswith("module."):
                cleaned_sd[k[7:]] = v
            else:
                cleaned_sd[k] = v
        model.load_state_dict(cleaned_sd)
        if "best_acc" in ckpt:
            best_val_acc = ckpt["best_acc"]
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1

    if args.mode == "test":
        print("\nEvaluating on Test set...")
        loss, top1, top5, top10 = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {loss:.4f} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}% | Top-10: {top10:.2f}%")
        return

    # Prepare Train / Val loaders
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"].isin(["valid", "val"])].copy()

    train_dataset = VideoDataset(train_df, args.video_dir, class_to_id, num_frames=args.frames,
                                 transform=train_transform, short_side=short_side)
    val_dataset = VideoDataset(val_df, args.video_dir, class_to_id, num_frames=args.frames,
                               transform=test_transform, short_side=short_side)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print(f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples.")
    patience_counter = 0
    best_ckpt_path = os.path.join(args.save_dir, f"{args.model_name}_best.pth")

    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss, train_samples = 0.0, 0
        train_top1 = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for inputs, targets, _ in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            bs = targets.size(0)

            optimizer.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                logits = model(inputs)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * bs
            train_samples += bs
            _, pred = logits.topk(1, 1, True, True)
            train_top1 += pred.eq(targets.view(-1, 1)).sum().item()
            pbar.set_postfix(loss=f"{train_loss / train_samples:.4f}", acc=f"{100.0 * train_top1 / train_samples:.2f}%")

        scheduler.step()

        val_loss, val_top1, val_top5, val_top10 = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}: Train Loss={train_loss/train_samples:.4f}, Train Acc={100.0*train_top1/train_samples:.2f}% | "
              f"Val Loss={val_loss:.4f}, Val Top-1={val_top1:.2f}%, Top-5={val_top5:.2f}%, Top-10={val_top10:.2f}%")

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss / train_samples,
            "train_acc": 100.0 * train_top1 / train_samples,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "val_top10": val_top10,
        }
        history.append(epoch_record)

        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_val_acc,
                "args": vars(args),
            }, best_ckpt_path)
            print(f"🔥 New best Val Top-1: {best_val_acc:.2f}%. Saved to {best_ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {args.patience} epochs without improvement.")
                break

    with open(os.path.join(args.save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Final evaluation on test set using best checkpoint
    print(f"\nEvaluating Best Model ({best_ckpt_path}) on Test Set...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    t_loss, t_top1, t_top5, t_top10 = evaluate(model, test_loader, criterion, device)
    print(f"Final Test: Top-1: {t_top1:.2f}% | Top-5: {t_top5:.2f}% | Top-10: {t_top10:.2f}%")


if __name__ == "__main__":
    main()
