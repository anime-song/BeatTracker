from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class PLPContrastiveModelOutput:
    """事前学習で使う各 head の出力をまとめる。"""

    embeddings: torch.Tensor
    chord_boundary_logits: torch.Tensor


class PLPContrastiveBackboneModel(nn.Module):
    """Backbone + projector + chord boundary head を持つ事前学習用ラッパ。"""

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
        self.chord_boundary_head = nn.Sequential(
            nn.LayerNorm(backbone.output_dim),
            nn.Linear(backbone.output_dim, 1),
        )

    def forward(self, waveform: torch.Tensor) -> PLPContrastiveModelOutput:
        # 時間方向の特徴列を projector と補助 head へ流す。
        frame_features = self.backbone(waveform)
        return PLPContrastiveModelOutput(
            embeddings=self.projector(frame_features),
            chord_boundary_logits=self.chord_boundary_head(frame_features).squeeze(-1),
        )
