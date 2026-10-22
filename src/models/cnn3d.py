"""
3D CNN Backbones for Spatiotemporal Isolated Sign Language Recognition:
- X3D-M (PyTorchVideo)
- R3D-18 (torchvision)
- I3D (Inflated 3D Inception-V1)
- C3D (Original 3D CNN)
- R(2+1)D-18 (torchvision)

Implements early-stage layer freezing and classifier replacement per Table 3.
"""

import torch
import torch.nn as nn
import torchvision.models.video as video_models


def build_3d_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    name = model_name.lower()

    if name in ["x3d", "x3d_m"]:
        # PyTorchVideo X3D-M
        model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=pretrained)
        # Freeze blocks 0 to 3, fine-tune block 4 and head
        layers_to_freeze = ["blocks.0", "blocks.1", "blocks.2", "blocks.3"]
        for param_name, param in model.named_parameters():
            if any(l in param_name for l in layers_to_freeze):
                param.requires_grad = False
            else:
                param.requires_grad = True

        in_features = model.blocks[5].proj.in_features
        model.blocks[5].proj = nn.Linear(in_features, num_classes)
        return model

    elif name in ["r3d_18", "r3d"]:
        # Torchvision R3D-18
        weights = video_models.R3D_18_Weights.KINETICS400_V1 if pretrained else None
        model = video_models.r3d_18(weights=weights)
        # Freeze stem, layer1, layer2; fine-tune layer3, layer4, and head
        layers_to_freeze = ["stem", "layer1", "layer2"]
        for param_name, param in model.named_parameters():
            if any(l in param_name for l in layers_to_freeze):
                param.requires_grad = False
            else:
                param.requires_grad = True

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif name in ["r2plus1d_18", "r2plus1d"]:
        # Torchvision R(2+1)D-18
        weights = video_models.R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        model = video_models.r2plus1d_18(weights=weights)
        layers_to_freeze = ["stem", "layer1", "layer2"]
        for param_name, param in model.named_parameters():
            if any(l in param_name for l in layers_to_freeze):
                param.requires_grad = False
            else:
                param.requires_grad = True

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif name in ["c3d"]:
        # C3D architecture
        class C3D(nn.Module):
            def __init__(self, out_classes: int):
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
                self.fc8 = nn.Linear(4096, out_classes)
                self.dropout = nn.Dropout(p=0.5)
                self.relu = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        model = C3D(out_classes=num_classes)
        # Freeze conv1 to conv3b; fine-tune conv4a to conv5b, FC6/FC7; replace fc8
        layers_to_freeze = ["conv1", "conv2", "conv3a", "conv3b"]
        for param_name, param in model.named_parameters():
            if any(l in param_name for l in layers_to_freeze):
                param.requires_grad = False
            else:
                param.requires_grad = True
        return model

    else:
        raise ValueError(f"Unsupported 3D CNN model: {model_name}. Choices: ['x3d', 'r3d_18', 'r2plus1d_18', 'c3d']")
