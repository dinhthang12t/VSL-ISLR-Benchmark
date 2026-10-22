"""
Training Script for 2D CNN + Temporal Models (Mean, GRU, LSTM).
Supports both 1,182-class original split and 447-class signer-independent split.
"""

import argparse
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

from src.datasets.feature_dataset import FeatureDataset
from src.models.cnn2d_temporal import MeanPoolingClassifier, SignLanguageGRU, SignLanguageLSTM


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


BACKBONE_DIMS = {
    "resnet18": 512,
    "resnet50": 2048,
    "efficientnet_b0": 1280,
    "convnext_tiny": 768,
}


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_samples = 0.0, 0
    top1, top5, top10 = 0, 0, 0

    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            bs = y.size(0)
            logits = model(x)
            if criterion:
                loss = criterion(logits, y)
                total_loss += loss.item() * bs

            maxk = min(10, logits.size(1))
            _, pred = logits.topk(maxk, 1, True, True)
            correct = pred.eq(y.view(-1, 1).expand_as(pred))

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
    parser = argparse.ArgumentParser(description="Train 2D CNN Temporal Models")
    parser.add_argument("--csv_path", type=str, required=True, help="Manifest CSV file")
    parser.add_argument("--feature_dir", type=str, required=True, help="Path to pre-extracted .npy features")
    parser.add_argument("--save_dir", type=str, default="checkpoints/2d_temporal")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=list(BACKBONE_DIMS.keys()))
    parser.add_argument("--model", type=str, default="gru", choices=["mean", "gru", "lstm"])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv_path)
    split_col = "split_signer_independent" if "split_signer_independent" in df.columns else "split"
    classes = sorted(df["Meaning"].unique().tolist())
    class_to_id = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    feat_dim = BACKBONE_DIMS[args.backbone]

    # Build model
    if args.model == "mean":
        model = MeanPoolingClassifier(input_size=feat_dim, num_classes=num_classes, dropout=args.dropout)
    elif args.model == "gru":
        model = SignLanguageGRU(input_size=feat_dim, hidden_size=args.hidden_dim, num_layers=args.num_layers, num_classes=num_classes, dropout=args.dropout)
    elif args.model == "lstm":
        model = SignLanguageLSTM(input_size=feat_dim, hidden_size=args.hidden_dim, num_layers=args.num_layers, num_classes=num_classes, dropout=args.dropout)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Test mode
    if args.mode == "test":
        if args.checkpoint and os.path.exists(args.checkpoint):
            ckpt = torch.load(args.checkpoint, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            model.load_state_dict(state_dict)
            print(f"Loaded checkpoint: {args.checkpoint}")
        test_df = df[df[split_col] == "test"].reset_index(drop=True)
        test_ds = FeatureDataset(test_df, args.feature_dir, class_to_id, input_dim=feat_dim)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        _, t1, t5, t10 = evaluate(model, test_loader, criterion, device)
        print(f"TEST RESULT [{args.backbone} + {args.model} | Seed {args.seed}]: Top-1={t1:.2f}% | Top-5={t5:.2f}% | Top-10={t10:.2f}%")
        return

    # Train mode
    train_df = df[df[split_col] == "train"].reset_index(drop=True)
    val_df = df[df[split_col].isin(["val", "valid"])].reset_index(drop=True)
    test_df = df[df[split_col] == "test"].reset_index(drop=True)

    train_ds = FeatureDataset(train_df, args.feature_dir, class_to_id, input_dim=feat_dim, augment=True)
    val_ds = FeatureDataset(val_df, args.feature_dir, class_to_id, input_dim=feat_dim)
    test_ds = FeatureDataset(test_df, args.feature_dir, class_to_id, input_dim=feat_dim)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_val_top1 = 0.0
    epochs_no_improve = 0

    print(f"Starting training: {args.backbone} + {args.model} (Seed={args.seed}, Classes={num_classes})")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, total = 0.0, 0, 0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            train_correct += (pred == y).sum().item()
            total += y.size(0)

        val_loss, val_top1, val_top5, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_top1)

        print(f"Epoch {epoch:03d}: Train Loss={train_loss/total:.4f}, Train Acc={100*train_correct/total:.2f}% | Val Loss={val_loss:.4f}, Val@1={val_top1:.2f}%, Val@5={val_top5:.2f}%")

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            epochs_no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_top1": val_top1}, save_path / "best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}!")
                break

    # Final evaluation on test split using best model
    best_ckpt = torch.load(save_path / "best_model.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    _, t1, t5, t10 = evaluate(model, test_loader, criterion, device)
    print(f"\n=======================================================")
    print(f"FINAL TEST RESULT [{args.backbone} + {args.model} | Seed {args.seed}]:")
    print(f"  Top-1:  {t1:.2f}%")
    print(f"  Top-5:  {t5:.2f}%")
    print(f"  Top-10: {t10:.2f}%")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
