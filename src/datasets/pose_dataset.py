"""
Pose Dataset for TGCN model.
Loads 55 skeletal keypoints (13 upper-body + 42 hand joints) from .npz files.
Coordinates normalized in [-1, 1], padded/interpolated to T=64 frames.
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def fix_pose_length_linspace(seq: np.ndarray, target_len: int = 64) -> np.ndarray:
    """Interpolate to target_len via uniform linear sampling; edge-pad if shorter."""
    t = seq.shape[0]
    if t == target_len:
        return seq
    elif t > target_len:
        indices = np.linspace(0, t - 1, target_len).astype(int)
        return seq[indices]
    else:
        if t == 0:
            return np.zeros((target_len, 55, 2), dtype=np.float32)
        pad = np.repeat(seq[-1][None, :, :], target_len - t, axis=0)
        return np.concatenate([seq, pad], axis=0).astype(np.float32)


class PoseDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        pose_dir: str,
        class_to_id: Dict[str, int],
        target_frames: int = 64,
        augment: bool = False,
    ):
        self.df = df.copy().reset_index(drop=True)
        self.pose_dir = Path(pose_dir)
        self.class_to_id = class_to_id
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

        npz_path = self.pose_dir / f"{vid_id}.npz"
        if npz_path.exists():
            try:
                npz_data = np.load(npz_path)
                data = npz_data["x"] if "x" in npz_data else npz_data[list(npz_data.keys())[0]]
            except Exception:
                data = np.zeros((self.target_frames, 55, 2), dtype=np.float32)
        else:
            data = np.zeros((self.target_frames, 55, 2), dtype=np.float32)

        if data.ndim == 2:
            t, d = data.shape
            k = d // 2
            data = data.reshape(t, k, 2)

        data = fix_pose_length_linspace(data, target_len=self.target_frames)

        # Pose augmentation (jitter & scale)
        if self.augment:
            if np.random.rand() < 0.5:
                scale = np.random.uniform(0.9, 1.1)
                data = data * scale
            if np.random.rand() < 0.5:
                noise = np.random.normal(0, 0.005, data.shape).astype(np.float32)
                data = data + noise

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long), vid_id
