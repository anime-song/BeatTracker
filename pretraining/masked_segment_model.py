from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Backbone, BackboneContext


def pool_segment_means(
    frame_features: torch.Tensor,
    start_frames: torch.Tensor,
    end_frames: torch.Tensor,
) -> torch.Tensor:
    """可変長 segment の平均特徴を、累積和で一括計算する。"""

    if frame_features.ndim != 3:
        raise ValueError("frame_features must have shape [batch, frames, dim]")
    if start_frames.shape != end_frames.shape:
        raise ValueError("start_frames and end_frames must have matching shape")

    batch_size, num_frames, feature_dim = frame_features.shape
    if start_frames.numel() == 0:
        return frame_features.new_zeros((batch_size, 0, feature_dim))

    cumulative = torch.cat(
        [
            frame_features.new_zeros((batch_size, 1, feature_dim)),
            frame_features.cumsum(dim=1),
        ],
        dim=1,
    )
    start_index = start_frames.clamp(min=0, max=num_frames)
    end_index = end_frames.clamp(min=0, max=num_frames)
    gather_shape = (-1, -1, feature_dim)
    start_gather_index = start_index.unsqueeze(-1).expand(gather_shape)
    end_gather_index = end_index.unsqueeze(-1).expand(gather_shape)
    pooled_sum = torch.gather(cumulative, 1, end_gather_index) - torch.gather(
        cumulative,
        1,
        start_gather_index,
    )
    lengths = (end_index - start_index).clamp_min(1).to(frame_features.dtype)
    return pooled_sum / lengths.unsqueeze(-1)


@dataclass
class MaskedSegmentModelOutput:
    segment_logits: torch.Tensor
    masked_segment_mask: torch.Tensor
    segment_valid_mask: torch.Tensor


def build_masked_backbone_context(
    context: BackboneContext,
    segment_valid_mask: torch.Tensor,
    mask_start_frames: torch.Tensor,
    mask_end_frames: torch.Tensor,
    mask_ratio: float,
    min_masks_per_sample: int,
) -> tuple[BackboneContext, torch.Tensor]:
    """
    segment 単位の mask 指定を、Backbone 入力上の時間マスクへまとめて変換する。

    1. visible segment から mask 対象を抽選する
    2. inner 区間だけ時間軸 mask を作る
    3. spec 上の該当フレームを 0 埋めした context を返す
    """

    if segment_valid_mask.shape != mask_start_frames.shape:
        raise ValueError("segment_valid_mask and mask_start_frames must match")
    if mask_start_frames.shape != mask_end_frames.shape:
        raise ValueError("mask_start_frames and mask_end_frames must match")
    if context.spec.shape[0] != segment_valid_mask.shape[0]:
        raise ValueError("segment masks batch size must match context.spec")
    if context.spec.ndim != 4:
        raise ValueError("context.spec must have shape [batch, channels, frames, bins]")

    if segment_valid_mask.numel() == 0:
        return context, segment_valid_mask.clone()

    masked_segment_mask = (
        torch.rand(segment_valid_mask.shape, device=segment_valid_mask.device)
        < float(mask_ratio)
    ) & segment_valid_mask

    min_masks_per_sample = max(0, int(min_masks_per_sample))
    if min_masks_per_sample > 0:
        for batch_index in range(segment_valid_mask.shape[0]):
            valid_indices = torch.nonzero(
                segment_valid_mask[batch_index], as_tuple=False
            ).flatten()
            if valid_indices.numel() == 0:
                continue
            current_mask_count = int(masked_segment_mask[batch_index].sum().item())
            target_mask_count = min(min_masks_per_sample, int(valid_indices.numel()))
            if current_mask_count >= target_mask_count:
                continue
            permutation = torch.randperm(
                valid_indices.numel(),
                device=valid_indices.device,
            )
            chosen = valid_indices[permutation[:target_mask_count]]
            masked_segment_mask[batch_index, chosen] = True

    num_frames = int(context.spec.shape[2])
    batch_size = masked_segment_mask.shape[0]
    frame_mask = torch.zeros(
        (batch_size, num_frames),
        dtype=torch.bool,
        device=masked_segment_mask.device,
    )
    for batch_index in range(batch_size):
        masked_indices = torch.nonzero(
            masked_segment_mask[batch_index], as_tuple=False
        ).flatten()
        for segment_index in masked_indices.tolist():
            start_frame = int(mask_start_frames[batch_index, segment_index].item())
            end_frame = int(mask_end_frames[batch_index, segment_index].item())
            start_frame = max(0, min(start_frame, num_frames))
            end_frame = max(start_frame, min(end_frame, num_frames))
            if end_frame > start_frame:
                frame_mask[batch_index, start_frame:end_frame] = True

    masked_spec = context.spec.clone()
    masked_spec = masked_spec.masked_fill(frame_mask[:, None, :, None], 0.0)
    return (
        BackboneContext(
            spec=masked_spec,
            crop_length=context.crop_length,
            original_time_steps=context.original_time_steps,
        ),
        masked_segment_mask,
    )


class MaskedSegmentPredictionHead(nn.Module):
    """mask 後の frame 表現を segment ごとに要約し、prototype 分布を予測する。"""

    def __init__(
        self,
        input_dim: int,
        num_prototypes: int,
        mask_ratio: float = 0.4,
        min_masks_per_sample: int = 1,
        head_dropout: float = 0.1,
        predictor_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if num_prototypes <= 0:
            raise ValueError("num_prototypes must be positive")
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be between 0 and 1")

        self.input_dim = int(input_dim)
        self.num_prototypes = int(num_prototypes)
        self.mask_ratio = float(mask_ratio)
        self.min_masks_per_sample = max(0, int(min_masks_per_sample))

        self.predictor = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, predictor_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(predictor_hidden_dim, num_prototypes),
        )

    def forward(
        self,
        frame_features: torch.Tensor,
        segment_start_frames: torch.Tensor,
        segment_inner_start_frames: torch.Tensor,
        segment_inner_end_frames: torch.Tensor,
    ) -> MaskedSegmentModelOutput:
        # 可視 segment が 0 本の crop は、そのまま空テンソルを返して上流で無視する。
        if segment_start_frames.shape[1] == 0:
            empty_logits = frame_features.new_zeros(
                (frame_features.shape[0], 0, self.num_prototypes)
            )
            empty_mask = frame_features.new_zeros(
                (frame_features.shape[0], 0), dtype=torch.bool
            )
            return MaskedSegmentModelOutput(
                segment_logits=empty_logits,
                masked_segment_mask=empty_mask,
                segment_valid_mask=empty_mask,
            )

        segment_features = pool_segment_means(
            frame_features=frame_features,
            start_frames=segment_inner_start_frames,
            end_frames=segment_inner_end_frames,
        )
        segment_logits = self.predictor(segment_features)
        return MaskedSegmentModelOutput(
            segment_logits=segment_logits,
            masked_segment_mask=frame_features.new_zeros(
                segment_start_frames.shape,
                dtype=torch.bool,
            ),
            segment_valid_mask=frame_features.new_zeros(
                segment_start_frames.shape,
                dtype=torch.bool,
            ),
        )


class MaskedSegmentModel(nn.Module):
    """masked segment prediction だけを単独で試すための薄い wrapper。"""

    def __init__(
        self,
        backbone: Backbone,
        num_prototypes: int,
        mask_ratio: float = 0.4,
        min_masks_per_sample: int = 1,
        head_dropout: float = 0.1,
        predictor_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.segment_head = MaskedSegmentPredictionHead(
            input_dim=backbone.output_dim,
            num_prototypes=num_prototypes,
            mask_ratio=mask_ratio,
            min_masks_per_sample=min_masks_per_sample,
            head_dropout=head_dropout,
            predictor_hidden_dim=predictor_hidden_dim,
        )

    def forward(
        self,
        waveform: torch.Tensor,
        segment_start_frames: torch.Tensor,
        segment_inner_start_frames: torch.Tensor,
        segment_inner_end_frames: torch.Tensor,
        segment_valid_mask: torch.Tensor,
        backbone_context: Optional[BackboneContext] = None,
    ) -> MaskedSegmentModelOutput:
        context = (
            self.backbone.feature_extractor(waveform)
            if backbone_context is None
            else backbone_context
        )
        masked_context, masked_segment_mask = build_masked_backbone_context(
            context=context,
            segment_valid_mask=segment_valid_mask,
            mask_start_frames=segment_inner_start_frames,
            mask_end_frames=segment_inner_end_frames,
            mask_ratio=self.segment_head.mask_ratio,
            min_masks_per_sample=self.segment_head.min_masks_per_sample,
        )
        frame_features = self.backbone(waveform, context=masked_context)
        head_output = self.segment_head(
            frame_features=frame_features,
            segment_start_frames=segment_start_frames,
            segment_inner_start_frames=segment_inner_start_frames,
            segment_inner_end_frames=segment_inner_end_frames,
        )
        return MaskedSegmentModelOutput(
            segment_logits=head_output.segment_logits,
            masked_segment_mask=masked_segment_mask,
            segment_valid_mask=segment_valid_mask,
        )


def compute_masked_segment_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masked_segment_mask: torch.Tensor,
    target_loss: str,
) -> torch.Tensor:
    if masked_segment_mask.numel() == 0 or not masked_segment_mask.any():
        return logits.new_zeros(())

    if target_loss == "bce":
        per_segment_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        ).mean(dim=-1)
    elif target_loss == "kl":
        normalized_targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        per_segment_loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            normalized_targets,
            reduction="none",
        ).sum(dim=-1)
    else:
        raise ValueError(f"Unsupported target_loss: {target_loss}")

    weights = masked_segment_mask.to(per_segment_loss.dtype)
    return (per_segment_loss * weights).sum() / weights.sum().clamp_min(1.0)
