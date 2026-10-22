"""
Video Transformer Models:
- Swin3D-Tiny (torchvision)
- TimeSformer-Base (Divided Space-Time Attention, Hugging Face)
- ViViT-B/16x2 (Google / Hugging Face with 3D trilinear positional interpolation)

Supports both Protocol 1 (1,182 classes with signer overlap) and
Protocol 2 (447 classes signer-independent benchmark).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models


class ChannelFirstTransformerModel(nn.Module):
    """Adapts dataset clips [B, T, C, H, W] to torchvision Swin3D layout [B, C, T, H, W]."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5 and x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.model(x)


class HFVideoClassifierWrapper(nn.Module):
    """Adapts Hugging Face video transformers to return logits from [B, T, C, H, W] input."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hugging Face models expect pixel_values: [B, T, C, H, W]
        return self.model(pixel_values=x).logits


def interpolate_vivit_positions(model: nn.Module, input_frames: int = 64, img_size: int = 224):
    """Interpolates ViViT learned 3D positional embeddings to match target frame count."""
    embeddings = model.vivit.embeddings
    tubelet_t, tubelet_h, tubelet_w = model.config.tubelet_size
    old_t = model.config.num_frames // tubelet_t
    new_t = input_frames // tubelet_t
    grid_h = img_size // tubelet_h
    grid_w = img_size // tubelet_w

    old_pos = embeddings.position_embeddings.detach()
    cls_pos, patch_pos = old_pos[:, :1], old_pos[:, 1:]
    patch_pos = patch_pos.reshape(1, old_t, grid_h, grid_w, old_pos.shape[-1]).permute(0, 4, 1, 2, 3)
    patch_pos = F.interpolate(
        patch_pos, size=(new_t, grid_h, grid_w), mode="trilinear", align_corners=False
    )
    patch_pos = patch_pos.permute(0, 2, 3, 4, 1).reshape(1, new_t * grid_h * grid_w, old_pos.shape[-1])
    embeddings.position_embeddings = nn.Parameter(torch.cat((cls_pos, patch_pos), dim=1))
    embeddings.patch_embeddings.num_patches = new_t * grid_h * grid_w
    model.config.num_frames = input_frames


def build_transformer_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
    freeze_early: bool = False,
    dropout: float = 0.5,
    state_dict_keys: Optional[list] = None,
) -> nn.Module:
    """Builds video transformer models matching published checkpoint heads."""
    name = model_name.lower().replace("-", "_")

    if "swin" in name:
        weights = video_models.Swin3D_T_Weights.KINETICS400_V1 if pretrained else None
        model = video_models.swin3d_t(weights=weights)
        in_features = model.head.in_features

        # Check if checkpoint used Sequential(Dropout, Linear) or direct Linear
        has_sequential_head = False
        if state_dict_keys:
            has_sequential_head = any("head.1." in k for k in state_dict_keys)

        if has_sequential_head:
            model.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
        else:
            model.head = nn.Linear(in_features, num_classes)

        if freeze_early:
            # Frozen early stages (features[0:6]); fine-tune final stage (features[6]), normalization, and replaced head
            for param in model.parameters():
                param.requires_grad = False
            for param in model.features[6].parameters():
                param.requires_grad = True
            if hasattr(model, "norm"):
                for param in model.norm.parameters():
                    param.requires_grad = True
            for param in model.head.parameters():
                param.requires_grad = True

        return ChannelFirstTransformerModel(model)

    elif "timesformer" in name:
        from transformers import TimesformerForVideoClassification, TimesformerConfig
        if pretrained:
            model = TimesformerForVideoClassification.from_pretrained(
                "facebook/timesformer-base-finetuned-k400",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = TimesformerConfig(num_labels=num_classes)
            model = TimesformerForVideoClassification(config)

        in_features = getattr(model.classifier, "in_features", 768)
        model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))

        if freeze_early:
            # Frozen patch embedding and transformer blocks 0-7; fine-tune blocks 8-11, final normalization, and replaced classifier
            base = getattr(model, "timesformer", None)
            if base is not None:
                if hasattr(base, "embeddings"):
                    for p in base.embeddings.parameters():
                        p.requires_grad = False
                layers = getattr(base, "layers", None)
                if layers is None and hasattr(base, "encoder"):
                    layers = getattr(base.encoder, "layer", None)
                if layers is not None:
                    for layer in layers[:8]:
                        for p in layer.parameters():
                            p.requires_grad = False
                    for layer in layers[8:]:
                        for p in layer.parameters():
                            p.requires_grad = True
                if hasattr(base, "layernorm"):
                    for p in base.layernorm.parameters():
                        p.requires_grad = True
            for p in model.classifier.parameters():
                p.requires_grad = True

        return HFVideoClassifierWrapper(model)

    elif "vivit" in name:
        from transformers import VivitForVideoClassification, VivitConfig
        if pretrained:
            model = VivitForVideoClassification.from_pretrained(
                "google/vivit-b-16x2-kinetics400",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = VivitConfig(num_labels=num_classes)
            model = VivitForVideoClassification(config)

        interpolate_vivit_positions(model, input_frames=64, img_size=224)
        in_features = getattr(model.classifier, "in_features", 768)
        model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))

        if freeze_early:
            # Frozen tubelet embedding and encoder layers 0-7; fine-tune layers 8-11, final normalization, and replaced classifier
            base = getattr(model, "vivit", None)
            if base is not None:
                if hasattr(base, "embeddings"):
                    for p in base.embeddings.parameters():
                        p.requires_grad = False
                layers = getattr(base, "layers", None)
                if layers is None and hasattr(base, "encoder"):
                    layers = getattr(base.encoder, "layer", None)
                if layers is not None:
                    for layer in layers[:8]:
                        for p in layer.parameters():
                            p.requires_grad = False
                    for layer in layers[8:]:
                        for p in layer.parameters():
                            p.requires_grad = True
                if hasattr(base, "layernorm"):
                    for p in base.layernorm.parameters():
                        p.requires_grad = True
            for p in model.classifier.parameters():
                p.requires_grad = True

        return HFVideoClassifierWrapper(model)

    else:
        raise ValueError(f"Unsupported transformer model: {model_name}. Choices: ['swin3d_t', 'timesformer_base', 'vivit_base']")
