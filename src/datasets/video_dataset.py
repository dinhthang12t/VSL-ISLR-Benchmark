"""
Video Dataset Loader with Decord / OpenCV fallback.
Implements uniform linear sampling (linspace) over full duration with T=64 frames.
"""

from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

from .transforms import video_to_tensor_kinetics


def sample_linear_indices(total_frames: int, target_frames: int = 64) -> np.ndarray:
    """Uniform iterative sampling covering the full video duration."""
    if total_frames <= 0:
        return np.zeros((target_frames,), dtype=int)
    if total_frames >= target_frames:
        idxs = np.linspace(0, total_frames - 1, target_frames)
        return np.rint(idxs).astype(int)
    # Edge padding when total < target
    pad = np.full((target_frames - total_frames,), total_frames - 1, dtype=int)
    return np.concatenate([np.arange(total_frames, dtype=int), pad])


def read_frames_opencv(video_path: str, max_frames: int = 64, crop_size: int = 224) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        return np.zeros((max_frames, crop_size, crop_size, 3), dtype=np.uint8)

    indices = sample_linear_indices(total_frames, max_frames)
    frames = []
    current_idx = 0
    success = True

    while success and len(frames) < len(indices):
        success, frame = cap.read()
        if not success:
            break
        while len(frames) < len(indices) and indices[len(frames)] == current_idx:
            # Resize short edge to 256
            h, w = frame.shape[:2]
            scale = 256.0 / min(h, w)
            resized = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))))
            frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        current_idx += 1
    cap.release()

    if len(frames) == 0:
        return np.zeros((max_frames, crop_size, crop_size, 3), dtype=np.uint8)
    while len(frames) < max_frames:
        frames.append(frames[-1])
    return np.asarray(frames[:max_frames], dtype=np.uint8)


def read_frames_decord(video_path: str, max_frames: int = 64, crop_size: int = 224) -> np.ndarray:
    if not HAS_DECORD:
        return read_frames_opencv(video_path, max_frames, crop_size)
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        indices = np.clip(sample_linear_indices(total_frames, max_frames), 0, total_frames - 1)
        batch = vr.get_batch(indices).asnumpy()
        processed = []
        for img in batch:
            h, w = img.shape[:2]
            scale = 256.0 / min(h, w)
            resized = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))))
            processed.append(resized)
        return np.asarray(processed, dtype=np.uint8)
    except Exception:
        return read_frames_opencv(video_path, max_frames, crop_size)


class VideoDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        video_dir: str,
        class_to_id: Dict[str, int],
        transforms: Optional[List] = None,
        transform: Optional[object] = None,
        frames: int = 64,
        num_frames: Optional[int] = None,
        crop_size: int = 224,
        short_side: int = 256,
    ):
        self.df = df.copy().reset_index(drop=True)
        self.video_dir = Path(video_dir)
        self.class_to_id = class_to_id

        t = transform if transform is not None else transforms
        if t is None:
            self.transforms = []
        elif isinstance(t, list):
            self.transforms = t
        else:
            self.transforms = [t]

        self.frames = num_frames if num_frames is not None else frames
        self.crop_size = crop_size
        self.short_side = short_side

        label_col = "Meaning" if "Meaning" in self.df.columns else ("gloss" if "gloss" in self.df.columns else "label")
        self.df["label_id"] = self.df[label_col].map(self.class_to_id)
        self.df = self.df.dropna(subset=["label_id"]).reset_index(drop=True)
        self.df["label_id"] = self.df["label_id"].astype(int)

    def __len__(self) -> int:
        return len(self.df)

    def find_video_path(self, vid_id: str) -> Optional[Path]:
        for ext in [".mp4", ".webm", "_yolo.mp4"]:
            p = self.video_dir / f"{vid_id}{ext}"
            if p.exists():
                return p
        return None

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        id_col = "ID_video" if "ID_video" in row else ("video" if "video" in row else "id")
        vid_id = str(row[id_col]).replace(".mp4", "").replace(".webm", "")
        label = int(row["label_id"])

        video_path = self.find_video_path(vid_id)
        if video_path is None:
            imgs = np.zeros((self.frames, self.crop_size, self.crop_size, 3), dtype=np.uint8)
        else:
            imgs = read_frames_decord(str(video_path), max_frames=self.frames, crop_size=self.crop_size)

        for t in self.transforms:
            imgs = t(imgs)

        if not isinstance(imgs, torch.Tensor):
            video_tensor = video_to_tensor_kinetics(imgs)
        else:
            video_tensor = imgs
        return video_tensor, torch.tensor(label, dtype=torch.long), vid_id
