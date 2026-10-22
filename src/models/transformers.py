"""
Video Transformer Models:
- Swin3D-Tiny (torchvision)
- TimeSformer-Base (Divided Space-Time Attention)
- ViViT-B/16x2 (Google)

Implements early-stage layer freezing and classifier head replacement per Table 3.
"""

import torch
import torch.nn as nn
import torchvision.models.video as video_models


def build_transformer_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    pretrained_path: str = None,
) -> nn.Module:
    name = model_name.lower()

    if "swin" in name:
        # Torchvision Video Swin-Tiny
        weights = video_models.Swin3D_T_Weights.KINETICS400_V1 if (pretrained and not pretrained_path) else None
        model = video_models.swin3d_t(weights=weights)

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_sd, strict=False)

        # Freeze early stages (features[0:6]); fine-tune final stage (features[6]), norm, and head
        for param in model.parameters():
            param.requires_grad = False

        if hasattr(model, "features") and len(model.features) > 6:
            for param in model.features[6].parameters():
                param.requires_grad = True

        if hasattr(model, "norm"):
            for param in model.norm.parameters():
                param.requires_grad = True

        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        for param in model.head.parameters():
            param.requires_grad = True

        return model

    elif "timesformer" in name:
        from transformers import TimesformerForVideoClassification

        if pretrained:
            model = TimesformerForVideoClassification.from_pretrained(
                "facebook/timesformer-base-finetuned-k400",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            from transformers import TimesformerConfig
            config = TimesformerConfig(num_labels=num_classes)
            model = TimesformerForVideoClassification(config)

        # Freeze patch embedding and transformer blocks 0-7; fine-tune blocks 8-11 and classifier
        for name_param, param in model.named_parameters():
            if any(f"layer.{i}." in name_param for i in range(8)) or "embeddings" in name_param:
                param.requires_grad = False
            else:
                param.requires_grad = True

        return model

    elif "vivit" in name:
        from transformers import VivitForVideoClassification

        if pretrained:
            model = VivitForVideoClassification.from_pretrained(
                "google/vivit-b-16x2-kinetics400",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            from transformers import VivitConfig
            config = VivitConfig(num_labels=num_classes)
            model = VivitForVideoClassification(config)

        # Freeze tubelet embedding and encoder layers 0-7; fine-tune layers 8-11 and classifier
        for name_param, param in model.named_parameters():
            if any(f"layer.{i}." in name_param for i in range(8)) or "embeddings" in name_param:
                param.requires_grad = False
            else:
                param.requires_grad = True

        return model

    else:
        raise ValueError(f"Unsupported transformer model: {model_name}. Choices: ['swin3d', 'timesformer', 'vivit']")
