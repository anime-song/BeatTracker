from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Backbone


class ProjectionHead(nn.Module):
    """Backbone のフレーム特徴を contrastive 学習用埋め込みへ写像する。"""

    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class PLPContrastiveBackboneModel(nn.Module):
    """Backbone + projector だけを持つ事前学習用の薄いラッパ。"""

    def __init__(
        self,
        backbone: Backbone,
        projection_dim: int,
        projection_hidden_dim: int,
        projector_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = ProjectionHead(
            input_dim=backbone.output_dim,
            projection_dim=projection_dim,
            hidden_dim=projection_hidden_dim,
            dropout=projector_dropout,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # 時間方向の特徴列をそのまま projector に通して frame-wise embedding を得る。
        frame_features = self.backbone(waveform)
        return self.projector(frame_features)
