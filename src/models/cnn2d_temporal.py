"""
2D CNN Backbone + Temporal Modeling (Mean Pooling, GRU, LSTM)
Supports both Protocol 1 (1,182 classes with signer overlap) and 
Protocol 2 (447 classes signer-independent benchmark).

Implements exact training architectures:
- Mean Pooling:
    * TwoDCNNMeanPool: LayerNorm(D) -> Dropout -> Linear (Protocol 2)
    * TorchvisionClassifier: direct backbone heads (Protocol 1)
- Recurrent GRU:
    * SignLanguageGRU: GRU -> Intermediate Linear Projection -> GAP -> Classifier (Protocol 1)
    * GRUClassifier: LayerNorm -> GRU -> GAP -> Dropout -> Linear (Protocol 2)
- Recurrent LSTM:
    * SignLanguageLSTM: LSTM -> Intermediate Linear Projection -> GAP -> Classifier (Protocol 1)
    * LSTMClassifier: LayerNorm -> LSTM -> GAP -> Dropout -> Linear (Protocol 2)
"""

from typing import Optional, Union
import torch
import torch.nn as nn


# ── Feature dimension mapping ────────────────────────────────────────────────
BACKBONE_DIMS = {
    "resnet18": 512,
    "resnet50": 2048,
    "efficientnet_b0": 1280,
    "convnext_tiny": 768,
}


# ==============================================================================
# 1. MEAN POOLING CLASSIFIERS
# ==============================================================================

class MeanPoolingClassifier(nn.Module):
    """
    Universal 2D CNN Frame Features + Temporal Mean Pooling Classifier.
    Supports both Protocol 1 (Torchvision head) and Protocol 2 (LayerNorm + Dropout + Linear).
    """

    def __init__(
        self,
        feat_dim: int = 2048,
        num_classes: int = 1182,
        dropout: float = 0.3,
        use_layernorm: bool = False,
        backbone_name: Optional[str] = None,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.use_layernorm = use_layernorm

        if use_layernorm:
            # Protocol 2 (HPC si_reg_v1): LayerNorm(D) -> Dropout -> Linear(D, num_classes)
            self.classifier = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Dropout(p=dropout),
                nn.Linear(feat_dim, num_classes),
            )
        elif backbone_name == "efficientnet_b0":
            # Protocol 1 EfficientNet-B0 torchvision head: Dropout -> Linear
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feat_dim, num_classes),
            )
        elif backbone_name == "convnext_tiny":
            # Protocol 1 ConvNeXt-Tiny torchvision head: LayerNorm -> Linear
            self.classifier = nn.Sequential(
                nn.LayerNorm(feat_dim, eps=1e-6),
                nn.Linear(feat_dim, num_classes),
            )
        else:
            # Protocol 1 ResNet-18 / ResNet-50: Linear(feat_dim, num_classes)
            self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [Batch, Time, Feature_Dim] or [Batch, Feature_Dim]
        if x.dim() == 3:
            feat = torch.mean(x, dim=1)
        else:
            feat = x
        return self.classifier(feat)


class TwoDCNNMeanPool(nn.Module):
    """Protocol 2 canonical 2D CNN + temporal mean pooling model."""

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        head_dropout: float = 0.5,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p=head_dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            feat = torch.mean(x, dim=1)
        else:
            feat = x
        return self.classifier(feat)


# ==============================================================================
# 2. GRU TEMPORAL CLASSIFIERS
# ==============================================================================

class SignLanguageGRU(nn.Module):
    """
    Protocol 1 GRU Architecture (with intermediate hidden projection):
        Input [B, T, D] -> GRU(num_layers=2, hidden=256)
                        -> Linear(256, 256) -> ReLU -> Dropout(0.3)
                        -> Temporal Global Average Pooling -> [B, 256]
                        -> Linear(256, num_classes)
    """

    def __init__(
        self,
        feat_dim: int = 512,
        input_size: Optional[int] = None,
        hidden_size: int = 256,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        num_classes: int = 1182,
        dropout: float = 0.3,
        head_dropout: Optional[float] = None,
    ):
        super().__init__()
        in_dim = feat_dim if feat_dim is not None else (input_size or 512)
        h_dim = hidden_dim if hidden_dim is not None else hidden_size
        h_drop = head_dropout if head_dropout is not None else dropout

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=h_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.hidden_layer = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(h_drop),
        )
        self.classifier = nn.Linear(h_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = self.hidden_layer(out)
        out = torch.mean(out, dim=1)  # GAP
        out = self.classifier(out)
        return out


class GRUClassifier(nn.Module):
    """
    Protocol 2 GRU Architecture (HPC standardized regularization):
        Input [B, T, D] -> LayerNorm(D)
                        -> GRU(num_layers=2, hidden=256, dropout=0.3)
                        -> Temporal Global Average Pooling -> [B, 256]
                        -> Dropout(head_dropout)
                        -> Linear(256, num_classes)
    """

    def __init__(
        self,
        input_dim: int = 512,
        feat_dim: Optional[int] = None,
        num_classes: int = 447,
        hidden_size: int = 256,
        num_layers: int = 2,
        head_dropout: float = 0.3,
        temporal_dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        in_dim = feat_dim if feat_dim is not None else input_dim
        n_dirs = 2 if bidirectional else 1

        self.layer_norm = nn.LayerNorm(in_dim)
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=temporal_dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(head_dropout)
        self.fc = nn.Linear(hidden_size * n_dirs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        out, _ = self.gru(x)
        video_feature = self.dropout(out.mean(dim=1))
        return self.fc(video_feature)


# ==============================================================================
# 3. LSTM TEMPORAL CLASSIFIERS
# ==============================================================================

class SignLanguageLSTM(nn.Module):
    """
    Protocol 1 LSTM Architecture (with intermediate hidden projection):
        Input [B, T, D] -> LSTM(num_layers=2, hidden=256)
                        -> Linear(256, 256) -> ReLU -> Dropout(0.3)
                        -> Temporal Global Average Pooling -> [B, 256]
                        -> Linear(256, num_classes)
    """

    def __init__(
        self,
        feat_dim: int = 2048,
        input_size: Optional[int] = None,
        hidden_size: int = 256,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        num_classes: int = 1182,
        dropout: float = 0.3,
        head_dropout: Optional[float] = None,
    ):
        super().__init__()
        in_dim = feat_dim if feat_dim is not None else (input_size or 2048)
        h_dim = hidden_dim if hidden_dim is not None else hidden_size
        h_drop = head_dropout if head_dropout is not None else dropout

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=h_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.hidden_layer = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(h_drop),
        )
        self.classifier = nn.Linear(h_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.hidden_layer(out)
        out = torch.mean(out, dim=1)
        out = self.classifier(out)
        return out


class LSTMClassifier(nn.Module):
    """
    Protocol 2 LSTM Architecture:
        Input [B, T, D] -> LayerNorm(D)
                        -> LSTM(num_layers=2, hidden=256)
                        -> Temporal Global Average Pooling
                        -> Dropout(head_dropout)
                        -> Linear(256, num_classes)
    """

    def __init__(
        self,
        input_dim: int = 2048,
        feat_dim: Optional[int] = None,
        num_classes: int = 447,
        hidden_size: int = 256,
        num_layers: int = 2,
        head_dropout: float = 0.3,
        temporal_dropout: float = 0.3,
    ):
        super().__init__()
        in_dim = feat_dim if feat_dim is not None else input_dim

        self.layer_norm = nn.LayerNorm(in_dim)
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=temporal_dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(head_dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        out, _ = self.lstm(x)
        video_feature = self.dropout(out.mean(dim=1))
        return self.fc(video_feature)


# ==============================================================================
# 4. MODEL BUILDER
# ==============================================================================

def build_2d_temporal_model(
    model_type: str,
    backbone: str,
    num_classes: int,
    dropout: float = 0.3,
    state_dict_keys: Optional[list] = None,
) -> nn.Module:
    """
    Factory builder that instantiates the exact model architecture matching
    the checkpoint's weight layout.
    """
    feat_dim = BACKBONE_DIMS.get(backbone.lower(), 2048)
    mtype = model_type.lower()

    if mtype in ["mean", "2d_mean", "mean_pooling"]:
        # Auto-detect whether checkpoint uses LayerNorm (Protocol 2) or torchvision head (Protocol 1)
        if state_dict_keys:
            has_layernorm = any("classifier.0" in k for k in state_dict_keys) and any("classifier.2" in k for k in state_dict_keys)
            if has_layernorm:
                return TwoDCNNMeanPool(feat_dim=feat_dim, num_classes=num_classes, head_dropout=dropout)
        return MeanPoolingClassifier(
            feat_dim=feat_dim,
            num_classes=num_classes,
            dropout=dropout,
            use_layernorm=(num_classes == 447),
            backbone_name=backbone.lower(),
        )

    elif mtype in ["gru", "2d_gru"]:
        # Check if checkpoint has layer_norm & fc (Protocol 2) or hidden_layer & classifier (Protocol 1)
        if state_dict_keys:
            if any("layer_norm" in k for k in state_dict_keys) and any("fc." in k for k in state_dict_keys):
                return GRUClassifier(input_dim=feat_dim, num_classes=num_classes, head_dropout=dropout)
            else:
                return SignLanguageGRU(feat_dim=feat_dim, num_classes=num_classes, dropout=dropout)
        if num_classes == 447:
            return GRUClassifier(input_dim=feat_dim, num_classes=num_classes, head_dropout=dropout)
        else:
            return SignLanguageGRU(feat_dim=feat_dim, num_classes=num_classes, dropout=dropout)

    elif mtype in ["lstm", "2d_lstm"]:
        if state_dict_keys:
            if any("layer_norm" in k for k in state_dict_keys) and any("fc." in k for k in state_dict_keys):
                return LSTMClassifier(input_dim=feat_dim, num_classes=num_classes, head_dropout=dropout)
            else:
                return SignLanguageLSTM(feat_dim=feat_dim, num_classes=num_classes, dropout=dropout)
        if num_classes == 447:
            return LSTMClassifier(input_dim=feat_dim, num_classes=num_classes, head_dropout=dropout)
        else:
            return SignLanguageLSTM(feat_dim=feat_dim, num_classes=num_classes, dropout=dropout)

    else:
        raise ValueError(f"Unsupported 2D temporal model type: {model_type}")
