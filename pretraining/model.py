from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from models import Backbone
from models.transcription_model import BackboneContext
from pretraining.masked_segment_model import (
    MaskedSegmentPredictionHead,
    build_masked_backbone_context,
)


@dataclass
class PretrainingModelOutput:
    """SSL pretraining で参照する出力だけをまとめる。"""

    chord_boundary_logits: torch.Tensor
    segment_logits: Optional[torch.Tensor] = None
    masked_segment_mask: Optional[torch.Tensor] = None
    segment_valid_mask: Optional[torch.Tensor] = None


class PretrainingModel(nn.Module):
    """
    共有 backbone から chord boundary と masked segment の補助 head を作る。
    """

    def __init__(
        self,
        backbone: Backbone,
        head_dropout: float = 0.0,
        num_segment_prototypes: int = 0,
        segment_mask_ratio: float = 0.4,
        segment_min_masks_per_sample: int = 1,
        segment_predictor_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
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
                    backbone.output_dim
                    if segment_predictor_hidden_dim is None
                    else int(segment_predictor_hidden_dim)
                ),
            )
        )

    def forward(
        self,
        waveform: torch.Tensor,
        backbone_context: Optional[BackboneContext] = None,
        segment_start_frames: Optional[torch.Tensor] = None,
        segment_inner_start_frames: Optional[torch.Tensor] = None,
        segment_inner_end_frames: Optional[torch.Tensor] = None,
        segment_valid_mask: Optional[torch.Tensor] = None,
    ) -> PretrainingModelOutput:
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

        return PretrainingModelOutput(
            chord_boundary_logits=chord_boundary_logits,
            segment_logits=segment_logits,
            masked_segment_mask=masked_segment_mask,
            segment_valid_mask=resolved_segment_valid_mask,
        )
