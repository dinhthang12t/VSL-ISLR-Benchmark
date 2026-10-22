"""
Feature Dataset for 2D CNN frame-level features (ResNet, EfficientNet, ConvNeXt).
Loads pre-extracted .npy feature sequences [T=64, Feature_Dim].
"""

import os
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_dir: str,
        class_to_id: Dict[str, int],
        input_dim: int = 2048,
        target_frames: int = 64,
        augment: bool = False,
    ):
        self.df = df.copy().reset_index(drop=True)
        self.feature_dir = Path(feature_dir)
        self.class_to_id = class_to_id
        self.input_dim = input_dim
        self.target_frames = target_frames
        self.augment = augment

        self.df["label_id"] = self.df["Meaning"].map(self.class_to_id)
        self.df = self.df.dropna(subset=["label_id"]).reset_index(drop=True)
        self.df["label_id"] = self.df["label_id"].astype(int)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        vid_id = str(row["ID_video"]).replace(".mp4", "").replace(".webm", "")
        label = int(row["label_id"])

        npy_path = self.feature_dir / f"{vid_id}.npy"
        if npy_path.exists():
            try:
                data = np.load(npy_path).astype(np.float32)
            except Exception:
                data = np.zeros((self.target_frames, self.input_dim), dtype=np.float32)
        else:
            data = np.zeros((self.target_frames, self.input_dim), dtype=np.float32)

        if data.ndim == 1:
            data = data[None, :]

        # Pad or slice to target_frames
        t, d = data.shape
        if t < self.target_frames:
            pad = np.repeat(data[-1][None, :], self.target_frames - t, axis=0) if t > 0 else np.zeros((self.target_frames - t, d), dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)
        elif t > self.target_frames:
            idxs = np.linspace(0, t - 1, self.target_frames).astype(int)
            data = data[idxs]

        # Simple feature jitter augmentation
        if self.augment and np.random.rand() < 0.5:
            scale = np.random.uniform(0.95, 1.05)
            data = data * scale
            if np.random.rand() < 0.5:
                data += np.random.normal(0, 0.01, data.shape).astype(np.float32)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long), vid_id
