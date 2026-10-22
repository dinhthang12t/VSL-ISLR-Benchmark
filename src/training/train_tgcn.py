"""
Training and Evaluation Pipeline for Spatio-Temporal Graph Convolutional Network (TGCN).
Processes 55 skeletal keypoints (13 upper body + 42 hand joints).
Supports both Original 1,182 classes (Table 4) and Signer-Independent 447 classes (Table 11).
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
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.datasets.pose_dataset import PoseDataset
from src.models.tgcn import PoseTGCN


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
    parser = argparse.ArgumentParser(description="Train / Evaluate TGCN on 55-keypoint pose data")
    parser.add_argument("--csv_path", type=str, required=True, help="Manifest CSV file")
    parser.add_argument("--pose_dir", type=str, required=True, help="Directory containing .npz pose files")
    parser.add_argument("--save_dir", type=str, default="checkpoints/tgcn")
    parser.add_argument("--target_len", type=int, default=64)
    parser.add_argument("--num_nodes", type=int, default=55)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for testing or resume")
    parser.add_argument("--augment", action="store_true", default=True, help="Enable pose data augmentation")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: TGCN (nodes={args.num_nodes}) | Dropout: {args.dropout} | Seed: {args.seed}")

    df = pd.read_csv(args.csv_path)
    label_col = "Meaning" if "Meaning" in df.columns else ("gloss" if "gloss" in df.columns else "label")
    classes = sorted(df[label_col].unique())
    class_to_id = {cls_name: i for i, cls_name in enumerate(classes)}
    num_classes = len(classes)
    print(f"Loaded {len(df)} samples across {num_classes} classes.")

    test_df = df[df["split"] == "test"].copy() if "split" in df.columns else df.copy()
    test_dataset = PoseDataset(test_df, args.pose_dir, class_to_id,
                               target_frames=args.target_len, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = PoseTGCN(
        num_nodes=args.num_nodes,
        num_classes=num_classes,
        feat_dim=2,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    best_val_acc = 0.0

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        if "best_acc" in ckpt:
            best_val_acc = ckpt["best_acc"]
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1

    if args.mode == "test":
        print("\nEvaluating TGCN on Test set...")
        loss, top1, top5, top10 = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {loss:.4f} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}% | Top-10: {top10:.2f}%")
        return

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"].isin(["valid", "val"])].copy()

    train_dataset = PoseDataset(train_df, args.pose_dir, class_to_id,
                                target_frames=args.target_len, augment=args.augment)
    val_dataset = PoseDataset(val_df, args.pose_dir, class_to_id,
                              target_frames=args.target_len, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print(f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples.")
    patience_counter = 0
    best_ckpt_path = os.path.join(args.save_dir, "tgcn_best.pth")

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
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

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

    print(f"\nEvaluating Best TGCN Model ({best_ckpt_path}) on Test Set...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    t_loss, t_top1, t_top5, t_top10 = evaluate(model, test_loader, criterion, device)
    print(f"Final Test: Top-1: {t_top1:.2f}% | Top-5: {t_top5:.2f}% | Top-10: {t_top10:.2f}%")


if __name__ == "__main__":
    main()
