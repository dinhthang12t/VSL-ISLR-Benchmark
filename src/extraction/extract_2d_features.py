"""
Extract 2D CNN Frame-Level Features (T=64 frames) for Mean / GRU / LSTM training.
Supports: resnet18, resnet50, efficientnet_b0, convnext_tiny.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.datasets.video_dataset import VideoDataset
from src.datasets.transforms import get_rgb_transforms


def get_backbone(backbone_name: str, device: str):
    name = backbone_name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        feat_dim = 512
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Identity()
        feat_dim = 2048
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier = nn.Identity()
        feat_dim = 1280
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Identity()
        feat_dim = 768
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad = False
    return m, feat_dim


def main():
    parser = argparse.ArgumentParser(description="Extract 2D CNN features")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to manifest CSV")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for .npy features")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet50", "efficientnet_b0", "convnext_tiny"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    classes = sorted(df["Meaning"].unique().tolist())
    class_to_id = {c: i for i, c in enumerate(classes)}

    dataset = VideoDataset(df, args.video_dir, class_to_id, transforms=get_rgb_transforms("test", 224), frames=64)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model, feat_dim = get_backbone(args.backbone, args.device)
    print(f"Extracting {args.backbone} (feat_dim={feat_dim}) for {len(dataset)} videos...")

    with torch.no_grad():
        for videos, _, vids in tqdm(loader, desc=f"Extracting {args.backbone}"):
            # videos shape: [B, C, T, H, W] -> permute to [B*T, C, H, W]
            B, C, T, H, W = videos.shape
            x = videos.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).to(args.device)
            feats = model(x)  # [B*T, feat_dim]
            feats = feats.view(B, T, feat_dim).cpu().numpy()

            for i, vid in enumerate(vids):
                save_file = out_path / f"{vid}.npy"
                np.save(save_file, feats[i].astype(np.float32))

    print(f"Extraction complete! Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
