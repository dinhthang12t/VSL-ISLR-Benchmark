"""
Preprocessing and Data Augmentations for Video & Pose Models.
Standardized on Kinetics-400 (for 3D CNNs / Swin3D) and ImageNet (for 2D CNNs).
"""

import random
import cv2
import numpy as np
import torch

KINETICS_MEAN = torch.tensor([0.45, 0.45, 0.45], dtype=torch.float32)
KINETICS_STD = torch.tensor([0.225, 0.225, 0.225], dtype=torch.float32)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def video_to_tensor_kinetics(pic: np.ndarray) -> torch.Tensor:
    """Normalize uint8 numpy [T, H, W, C] to torch [C, T, H, W] using Kinetics-400 stats."""
    x = torch.from_numpy(pic).permute(3, 0, 1, 2).contiguous().float().div_(255.0)
    mean = KINETICS_MEAN.view(3, 1, 1, 1)
    std = KINETICS_STD.view(3, 1, 1, 1)
    x.sub_(mean).div_(std)
    return x


def video_to_tensor_imagenet(pic: np.ndarray) -> torch.Tensor:
    """Normalize uint8 numpy [T, H, W, C] to torch [C, T, H, W] using ImageNet stats."""
    x = torch.from_numpy(pic).permute(3, 0, 1, 2).contiguous().float().div_(255.0)
    mean = IMAGENET_MEAN.view(3, 1, 1, 1)
    std = IMAGENET_STD.view(3, 1, 1, 1)
    x.sub_(mean).div_(std)
    return x


class RandomCrop:
    def __init__(self, size: int = 224):
        self.size = size

    def __call__(self, imgs: np.ndarray) -> np.ndarray:
        t, h, w, c = imgs.shape
        th, tw = self.size, self.size
        if w < tw or h < th:
            imgs = np.array([cv2.resize(f, (tw, th)) for f in imgs])
            h, w = th, tw
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return imgs[:, i : i + th, j : j + tw, :]


class CenterCrop:
    def __init__(self, size: int = 224):
        self.size = size

    def __call__(self, imgs: np.ndarray) -> np.ndarray:
        t, h, w, c = imgs.shape
        th, tw = self.size, self.size
        if w < tw or h < th:
            imgs = np.array([cv2.resize(f, (tw, th)) for f in imgs])
            h, w = th, tw
        i = max(0, (h - th) // 2)
        j = max(0, (w - tw) // 2)
        return imgs[:, i : i + th, j : j + tw, :]


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, imgs: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            return imgs[:, :, ::-1, :]
        return imgs


def get_rgb_transforms(mode: str = "train", crop_size: int = 224):
    if mode == "train":
        return [RandomCrop(crop_size), RandomHorizontalFlip(0.5)]
    else:
        return [CenterCrop(crop_size)]
