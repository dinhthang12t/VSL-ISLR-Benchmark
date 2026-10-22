"""
2D CNN Backbone + Temporal Modeling (Mean Pooling, GRU, LSTM)
Supports both original 1,182-class and 447-class signer-independent protocols.
"""

import torch
import torch.nn as nn


class MeanPoolingClassifier(nn.Module):
    """
    2D CNN Frame Features + Mean Pooling Baseline.
    Averages frame representations over time and applies a linear classification layer.
    """

    def __init__(self, input_size: int = 2048, num_classes: int = 1182, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Time, Feature_Dim]
        out = torch.mean(x, dim=1)  # Global temporal pooling -> [Batch, Feature_Dim]
        out = self.dropout(out)
        out = self.classifier(out)  # [Batch, num_classes]
        return out


class SignLanguageGRU(nn.Module):
    """
    2D CNN Frame Features + Recurrent GRU Temporal Aggregation.
    Architecture:
        GRU(input_size, hidden_size=256, num_layers=2, dropout=0.3, bidirectional=False)
        Intermediate Projection: Linear(256, 256) -> ReLU -> Dropout(0.3)
        Temporal Global Average Pooling: torch.mean(out, dim=1)
        Linear Classifier: Linear(256, num_classes)
    """

    def __init__(
        self,
        input_size: int = 2048,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 1182,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Time, Feature_Dim]
        out, _ = self.gru(x)
        out = self.hidden_layer(out)
        out = torch.mean(out, dim=1)  # Global Average Pooling -> [Batch, Hidden]
        out = self.classifier(out)    # [Batch, num_classes]
        return out


class SignLanguageLSTM(nn.Module):
    """
    2D CNN Frame Features + Recurrent LSTM Temporal Aggregation.
    Architecture:
        LSTM(input_size, hidden_size=256, num_layers=2, dropout=0.3, bidirectional=False)
        Intermediate Projection: Linear(256, 256) -> ReLU -> Dropout(0.3)
        Temporal Global Average Pooling: torch.mean(out, dim=1)
        Linear Classifier: Linear(256, num_classes)
    """

    def __init__(
        self,
        input_size: int = 2048,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 1182,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Time, Feature_Dim]
        out, _ = self.lstm(x)
        out = self.hidden_layer(out)
        out = torch.mean(out, dim=1)  # Global Average Pooling -> [Batch, Hidden]
        out = self.classifier(out)    # [Batch, num_classes]
        return out
