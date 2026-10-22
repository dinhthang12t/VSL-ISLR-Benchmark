"""
3D CNN Backbones for Spatiotemporal Isolated Sign Language Recognition:
- X3D-M (PyTorchVideo)
- R3D-18 (torchvision)
- I3D (Inflated 3D Inception-V1)
- C3D (Original 3D CNN, Tran et al. 2015)
- R(2+1)D (torchvision / pytorchvideo)

Supports both Protocol 1 (1,182 classes with signer overlap) and
Protocol 2 (447 classes signer-independent benchmark).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models


# ==============================================================================
# CHANNEL ADAPTER
# ==============================================================================

class ChannelFirstVideoModel(nn.Module):
    """Adapts dataset clips [B, T, C, H, W] to PyTorch 3D CNN layout [B, C, T, H, W]."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5 and x.shape[1] != 3 and x.shape[2] == 3:
            # x is [B, T, C, H, W] -> permute to [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.model(x)


# ==============================================================================
# 1. INCEPTION-I3D ARCHITECTURE (Piergiaj / Carreira & Zisserman)
# ==============================================================================

class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        batch, channel, t, h, w = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_t // 2, pad_t - pad_t // 2)
        x = F.pad(x, pad)
        return super().forward(x)


class Unit3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        output_channels: int,
        kernel_shape=(1, 1, 1),
        stride=(1, 1, 1),
        padding=0,
        activation_fn=F.relu,
        use_batch_norm: bool = True,
        use_bias: bool = False,
        name: str = 'unit_3d',
    ):
        super().__init__()
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=output_channels,
            kernel_size=kernel_shape,
            stride=stride,
            padding=0,
            bias=use_bias,
        )
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channel, t, h, w = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_t // 2, pad_t - pad_t // 2)
        x = F.pad(x, pad)
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: list, name: str):
        super().__init__()
        self.b0 = Unit3D(in_channels, out_channels[0], kernel_shape=[1, 1, 1], name=name + '/Branch_0')
        self.b1a = Unit3D(in_channels, out_channels[1], kernel_shape=[1, 1, 1], name=name + '/Branch_1a')
        self.b1b = Unit3D(out_channels[1], out_channels[2], kernel_shape=[3, 3, 3], name=name + '/Branch_1b')
        self.b2a = Unit3D(in_channels, out_channels[3], kernel_shape=[1, 1, 1], name=name + '/Branch_2a')
        self.b2b = Unit3D(out_channels[3], out_channels[4], kernel_shape=[3, 3, 3], name=name + '/Branch_2b')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels, out_channels[5], kernel_shape=[1, 1, 1], name=name + '/Branch_3b')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """
    Inflated 3D Inception-V1 Architecture.
    Matches piergiaj/pytorch-i3d state_dict keys exactly.
    """
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7', 'MaxPool3d_2a_3x3', 'Conv3d_2b_1x1', 'Conv3d_2c_3x3', 'MaxPool3d_3a_3x3',
        'Mixed_3b', 'Mixed_3c', 'MaxPool3d_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_4f',
        'MaxPool3d_5a_2x2', 'Mixed_5b', 'Mixed_5c', 'Logits', 'Predictions',
    )

    def __init__(
        self,
        num_classes: int = 400,
        spatial_squeeze: bool = True,
        name: str = 'inception_i3d',
        in_channels: int = 3,
        dropout_keep_prob: float = 0.5,
    ):
        super().__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze

        self.end_points = {}
        self.end_points['Conv3d_1a_7x7'] = Unit3D(in_channels, 64, kernel_shape=[7, 7, 7], stride=(2, 2, 2))
        self.end_points['MaxPool3d_2a_3x3'] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2))
        self.end_points['Conv3d_2b_1x1'] = Unit3D(64, 64, kernel_shape=[1, 1, 1])
        self.end_points['Conv3d_2c_3x3'] = Unit3D(64, 192, kernel_shape=[3, 3, 3])
        self.end_points['MaxPool3d_3a_3x3'] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2))

        self.end_points['Mixed_3b'] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + 'Mixed_3b')
        self.end_points['Mixed_3c'] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + 'Mixed_3c')
        self.end_points['MaxPool3d_4a_3x3'] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2))

        self.end_points['Mixed_4b'] = InceptionModule(480, [192, 96, 208, 16, 48, 64], name + 'Mixed_4b')
        self.end_points['Mixed_4c'] = InceptionModule(512, [160, 112, 224, 24, 64, 64], name + 'Mixed_4c')
        self.end_points['Mixed_4d'] = InceptionModule(512, [128, 128, 256, 24, 64, 64], name + 'Mixed_4d')
        self.end_points['Mixed_4e'] = InceptionModule(512, [112, 144, 288, 32, 64, 64], name + 'Mixed_4e')
        self.end_points['Mixed_4f'] = InceptionModule(528, [256, 160, 320, 32, 128, 128], name + 'Mixed_4f')
        self.end_points['MaxPool3d_5a_2x2'] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2))

        self.end_points['Mixed_5b'] = InceptionModule(832, [256, 160, 320, 32, 128, 128], name + 'Mixed_5b')
        self.end_points['Mixed_5c'] = InceptionModule(832, [384, 192, 384, 48, 128, 128], name + 'Mixed_5c')

        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(1.0 - dropout_keep_prob if dropout_keep_prob < 1.0 else 0.5)
        self.logits = Unit3D(
            1024, self._num_classes, kernel_shape=[1, 1, 1], activation_fn=None, use_batch_norm=False, use_bias=True
        )

    def replace_logits(self, num_classes: int):
        self._num_classes = num_classes
        self.logits = Unit3D(
            1024, self._num_classes, kernel_shape=[1, 1, 1], activation_fn=None, use_batch_norm=False, use_bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            x = x.squeeze(3).squeeze(3)
        return torch.mean(x, dim=2)


# ==============================================================================
# 2. C3D ARCHITECTURE (Tran et al. 2015)
# ==============================================================================

class C3D(nn.Module):
    """
    C3D implementation following Tran et al. 2015.
    Supports both individual layer attributes (conv1..fc8) and sequential features/classifier.
    """

    def __init__(self, num_classes: int = 1182, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        h = self.relu(self.conv1(x))
        h = self.pool1(h)
        h = self.relu(self.conv2(h))
        h = self.pool2(h)
        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)
        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)
        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)
        h = h.view(h.size(0), -1)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)
        logits = self.fc8(h)
        return logits


# ==============================================================================
# 3. BUILDER FUNCTION
# ==============================================================================

def build_3d_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
    freeze_early: bool = False,
    dropout: float = 0.5,
    state_dict_keys: Optional[list] = None,
) -> nn.Module:
    """
    Builds a 3D CNN model matching the exact checkpoint configuration.
    Wraps models with ChannelFirstVideoModel to automatically accept [B, T, C, H, W]
    and convert to [B, C, T, H, W].
    """
    name = model_name.lower().replace("-", "_")

    if "x3d" in name:
        model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=pretrained)
        in_features = model.blocks[5].proj.in_features
        model.blocks[5].proj = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        return ChannelFirstVideoModel(model)

    elif "r3d" in name:
        weights = video_models.R3D_18_Weights.KINETICS400_V1 if pretrained else None
        model = video_models.r3d_18(weights=weights)
        in_features = model.fc.in_features

        # Check if checkpoint used Sequential(Dropout, Linear) or direct Linear
        use_sequential_head = True
        if state_dict_keys:
            # If bare 'fc.weight' exists without 'fc.1.weight', it's direct Linear
            bare_fc = any(k in ["fc.weight", "model.fc.weight"] for k in state_dict_keys)
            use_sequential_head = not bare_fc

        if use_sequential_head:
            model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
        else:
            model.fc = nn.Linear(in_features, num_classes)
        return ChannelFirstVideoModel(model)

    elif "i3d" in name:
        model = InceptionI3d(num_classes=num_classes, in_channels=3, dropout_keep_prob=1.0 - dropout)
        return ChannelFirstVideoModel(model)

    elif "c3d" in name:
        model = C3D(num_classes=num_classes, dropout=dropout)
        return ChannelFirstVideoModel(model)

    elif "r2plus1d" in name:
        if "r50" in name:
            model = torch.hub.load("facebookresearch/pytorchvideo", "r2plus1d_r50", pretrained=pretrained)
            in_features = model.blocks[5].proj.in_features
            model.blocks[5].proj = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
        else:
            weights = video_models.R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
            model = video_models.r2plus1d_18(weights=weights)
            model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(model.fc.in_features, num_classes))

    else:
        raise ValueError(f"Unsupported 3D CNN model: {model_name}. Choices: ['x3d_m', 'r3d_18', 'i3d', 'c3d', 'r2plus1d_18', 'r2plus1d_r50']")

    # Apply freeze early layers policy per Table 3
    if freeze_early:
        if "c3d" in name:
            layers_to_freeze = ["conv1", "conv2", "conv3a", "conv3b"]
        elif "r3d" in name or "r2plus1d" in name:
            layers_to_freeze = ["stem", "layer1", "layer2"]
        elif "i3d" in name:
            layers_to_freeze = [
                "Conv3d_1a_7x7", "MaxPool3d_2a_3x3", "Conv3d_2b_1x1", "Conv3d_2c_3x3",
                "MaxPool3d_3a_3x3", "Mixed_3b", "Mixed_3c", "MaxPool3d_4a_3x3",
                "Mixed_4b", "Mixed_4c", "Mixed_4d", "Mixed_4e", "Mixed_4f",
            ]
        elif "x3d" in name:
            layers_to_freeze = ["blocks.0", "blocks.1", "blocks.2", "blocks.3"]
        else:
            layers_to_freeze = []

        for p_name, param in model.named_parameters():
            if any(l in p_name for l in layers_to_freeze):
                param.requires_grad = False

    return ChannelFirstVideoModel(model)
