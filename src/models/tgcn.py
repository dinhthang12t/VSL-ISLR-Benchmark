"""
Spatial-Temporal Graph Convolutional Network (TGCN) for Pose-Based ISLR.
Processes 55 skeletal keypoints (13 upper body + 42 hand joints).

Architecture:
    Input: [B, T=64, V=55, C=2]
    Block 1: ResidualBlock(2 -> 64, num_nodes=55)
    Block 2: ResidualBlock(64 -> 128, num_nodes=55)
    Block 3: ResidualBlock(128 -> 256, num_nodes=55)
    Global Spatiotemporal Pooling: mean over (Time, Nodes) -> [B, 256]
    Dropout(0.3)
    Linear(256 -> num_classes)
"""

import torch
import torch.nn as nn


class GraphConv(nn.Module):
    """Spatial Graph Convolution with learnable adjacency matrix."""
    def __init__(self, in_feat: int, out_feat: int, num_nodes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_feat, out_feat) * (1.0 / (in_feat ** 0.5)))
        self.adj = nn.Parameter(torch.eye(num_nodes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, V, C]
        x = torch.matmul(x, self.weight)
        x = torch.einsum('ij,btjf->btif', self.adj, x)
        return torch.tanh(x)


class ResidualBlock(nn.Module):
    """Residual Graph Convolutional block."""
    def __init__(self, in_feat: int, out_feat: int, num_nodes: int):
        super().__init__()
        self.gc1 = GraphConv(in_feat, out_feat, num_nodes)
        self.gc2 = GraphConv(out_feat, out_feat, num_nodes)
        self.match_dim = (in_feat != out_feat)
        if self.match_dim:
            self.proj = nn.Linear(in_feat, out_feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.proj(x) if self.match_dim else x
        out = self.gc1(x)
        out = self.gc2(out)
        return out + res


class PoseTGCN(nn.Module):
    """TGCN architecture for whole-body skeleton sequence classification."""
    def __init__(self, num_nodes: int = 55, num_classes: int = 1182, feat_dim: int = 2, dropout: float = 0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.block1 = ResidualBlock(feat_dim, 64, num_nodes)
        self.block2 = ResidualBlock(64, 128, num_nodes)
        self.block3 = ResidualBlock(128, 256, num_nodes)

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T=64, V=55, C=2]
        if x.dim() == 4 and x.shape[1] == self.feat_dim:
            # If [B, C, T, V] permute to [B, T, V, C]
            x = x.permute(0, 2, 3, 1).contiguous()

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        # Global spatiotemporal average pooling over time (T) and nodes (V)
        out = out.mean(dim=(1, 2))  # [B, 256]
        out = self.drop(out)
        logits = self.fc(out)
        return logits


# Alias for backward-compatibility and consistency
TGCN = PoseTGCN
