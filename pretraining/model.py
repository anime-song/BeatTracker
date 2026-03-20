from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Backbone
from models.transcription_model import BackboneContext
from pretraining.masked_segment_model import (
    MaskedSegmentPredictionHead,
    build_masked_backbone_context,
)


def masked_mean(
    features: torch.Tensor, valid_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """padding を除いたフレーム平均を作る。"""

    if valid_mask is None:
        return features.mean(dim=1)

    weights = valid_mask.to(features.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (features * weights).sum(dim=1) / denom


class FrameProjectionHead(nn.Module):
    """フレーム特徴を SSL 用埋め込みへ写像する小さな head。"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


@dataclass
class DrumContrastiveModelOutput:
    """Drum-Contrastive SSL が参照する出力だけをまとめる。"""

    rhythm_summary: torch.Tensor
    chord_boundary_logits: torch.Tensor
    segment_logits: Optional[torch.Tensor] = None
    masked_segment_mask: Optional[torch.Tensor] = None
    segment_valid_mask: Optional[torch.Tensor] = None


class DrumContrastiveModel(nn.Module):
    """
    共有 backbone から rhythm 表現と chord boundary 補助 head を作る。
    """

    def __init__(
        self,
        backbone: Backbone,
        rhythm_dim: int,
        projection_hidden_dim: int,
        head_dropout: float = 0.0,
        num_segment_prototypes: int = 0,
        segment_mask_ratio: float = 0.4,
        segment_min_masks_per_sample: int = 1,
        segment_predictor_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.rhythm_head = FrameProjectionHead(
            input_dim=backbone.output_dim,
            output_dim=rhythm_dim,
            hidden_dim=projection_hidden_dim,
            dropout=head_dropout,
        )
        # chord boundary は rhythm 表現と分け、共有 frame feature から独立に予測する。
        self.chord_boundary_head = nn.Sequential(
            nn.LayerNorm(backbone.output_dim),
            nn.Linear(backbone.output_dim, 1),
        )
        self.segment_head = (
            None
            if num_segment_prototypes <= 0
            else MaskedSegmentPredictionHead(
                input_dim=backbone.output_dim,
                num_prototypes=num_segment_prototypes,
                mask_ratio=segment_mask_ratio,
                min_masks_per_sample=segment_min_masks_per_sample,
                head_dropout=head_dropout,
                predictor_hidden_dim=(
                    projection_hidden_dim
                    if segment_predictor_hidden_dim is None
                    else int(segment_predictor_hidden_dim)
                ),
            )
        )

    def forward(
        self,
        waveform: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        backbone_context: Optional[BackboneContext] = None,
        segment_start_frames: Optional[torch.Tensor] = None,
        segment_inner_start_frames: Optional[torch.Tensor] = None,
        segment_inner_end_frames: Optional[torch.Tensor] = None,
        segment_valid_mask: Optional[torch.Tensor] = None,
    ) -> DrumContrastiveModelOutput:
        # mix を再 forward する場合も feature extractor の計算は再利用したいので、
        # 既にあれば backbone_context を受け取れるようにしている。
        context = (
            self.backbone.feature_extractor(waveform)
            if backbone_context is None
            else backbone_context
        )

        segment_supervision_enabled = (
            self.segment_head is not None
            and segment_start_frames is not None
            and segment_inner_start_frames is not None
            and segment_inner_end_frames is not None
            and segment_valid_mask is not None
        )

        masked_segment_mask: Optional[torch.Tensor] = None
        resolved_segment_valid_mask: Optional[torch.Tensor] = None
        active_context = context
        if segment_supervision_enabled:
            active_context, masked_segment_mask = build_masked_backbone_context(
                context=context,
                segment_valid_mask=segment_valid_mask,
                mask_start_frames=segment_inner_start_frames,
                mask_end_frames=segment_inner_end_frames,
                mask_ratio=self.segment_head.mask_ratio,
                min_masks_per_sample=self.segment_head.min_masks_per_sample,
            )
            resolved_segment_valid_mask = segment_valid_mask

        frame_features = self.backbone(waveform, context=active_context)
        rhythm_embeddings = self.rhythm_head(frame_features)
        rhythm_summary = F.normalize(masked_mean(rhythm_embeddings, valid_mask), dim=-1)
        chord_boundary_logits = self.chord_boundary_head(frame_features).squeeze(-1)
        segment_logits: Optional[torch.Tensor] = None
        if segment_supervision_enabled and resolved_segment_valid_mask is not None:
            segment_output = self.segment_head(
                frame_features=frame_features,
                segment_start_frames=segment_start_frames,
                segment_inner_start_frames=segment_inner_start_frames,
                segment_inner_end_frames=segment_inner_end_frames,
            )
            segment_logits = segment_output.segment_logits
        return DrumContrastiveModelOutput(
            rhythm_summary=rhythm_summary,
            chord_boundary_logits=chord_boundary_logits,
            segment_logits=segment_logits,
            masked_segment_mask=masked_segment_mask,
            segment_valid_mask=resolved_segment_valid_mask,
        )
