from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class PLPContrastiveLoss(nn.Module):
    """
    PLP ピーク列に対する contrastive loss。

    基本方針は論文に寄せつつ、
    - positive: ピーク列上で 1, 2, 4 個先後ろの候補
    - negative: 近傍を避けたそれ以外のピーク
    という実装にしている。
    """

    def __init__(
        self,
        temperature: float = 0.1,
        positive_offsets: Sequence[int] = (1, 2, 4),
        negative_safety_radius: int = 4,
        num_negatives: int = 32,
        max_anchors_per_sample: int = 64,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        normalized_offsets = sorted({int(offset) for offset in positive_offsets if int(offset) > 0})
        if not normalized_offsets:
            raise ValueError("positive_offsets must contain at least one positive integer")

        self.temperature = float(temperature)
        self.positive_offsets = tuple(normalized_offsets)
        self.positive_offset_set = set(self.positive_offsets)
        self.negative_safety_radius = max(0, int(negative_safety_radius))
        self.num_negatives = int(num_negatives)
        self.max_anchors_per_sample = int(max_anchors_per_sample)

    def _positive_positions(self, anchor_position: int, num_peaks: int) -> list[int]:
        # anchor から見て「拍の倍周期っぽい距離」を positive 候補にする。
        positions: list[int] = []
        for offset in self.positive_offsets:
            forward = anchor_position + offset
            backward = anchor_position - offset
            if forward < num_peaks:
                positions.append(forward)
            if backward >= 0:
                positions.append(backward)
        return positions

    def forward(
        self,
        embeddings: torch.Tensor,
        peak_mask: torch.Tensor,
        peak_values: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, time, dim]")
        if peak_mask.shape != embeddings.shape[:2]:
            raise ValueError("peak_mask must have shape [batch, time]")
        if peak_values.shape != embeddings.shape[:2]:
            raise ValueError("peak_values must have shape [batch, time]")
        if valid_mask is not None and valid_mask.shape != embeddings.shape[:2]:
            raise ValueError("valid_mask must have shape [batch, time]")

        total_weighted_loss = embeddings.new_zeros(())
        total_weight = embeddings.new_zeros(())
        total_pairs = 0
        total_peaks = 0
        active_samples = 0

        for batch_index in range(embeddings.shape[0]):
            # まず PLP が立っているフレームだけを抜き出し、
            # contrastive の対象をピーク列へ圧縮する。
            active_peak_mask = peak_mask[batch_index] > 0.0
            if valid_mask is not None:
                active_peak_mask = active_peak_mask & (valid_mask[batch_index] > 0.0)

            peak_indices = active_peak_mask.nonzero(as_tuple=False).squeeze(-1)
            peak_count = int(peak_indices.numel())
            total_peaks += peak_count
            if peak_count < 2:
                continue

            sample_embeddings = embeddings[batch_index].index_select(0, peak_indices)
            sample_weights = peak_values[batch_index].index_select(0, peak_indices).clamp_min(1e-6)
            active_samples += 1

            if 0 < self.max_anchors_per_sample < peak_count:
                anchor_positions = torch.argsort(sample_weights, descending=True)[
                    : self.max_anchors_per_sample
                ]
            else:
                anchor_positions = torch.arange(peak_count, device=embeddings.device)

            for anchor_position_tensor in anchor_positions:
                anchor_position = int(anchor_position_tensor.item())
                positive_positions = self._positive_positions(anchor_position, peak_count)
                if not positive_positions:
                    continue

                positive_tensor = torch.tensor(
                    positive_positions,
                    device=embeddings.device,
                    dtype=torch.long,
                )
                positive_scores = sample_weights.index_select(0, positive_tensor)
                positive_position = int(positive_tensor[positive_scores.argmax()].item())

                negative_positions = [
                    position
                    for position in range(peak_count)
                    if position != anchor_position
                    and position != positive_position
                    and abs(position - anchor_position) > self.negative_safety_radius
                    and abs(position - anchor_position) not in self.positive_offset_set
                ]
                if not negative_positions:
                    negative_positions = [
                        position
                        for position in range(peak_count)
                        if position != anchor_position and position != positive_position
                    ]
                if not negative_positions:
                    continue

                negative_tensor = torch.tensor(
                    negative_positions,
                    device=embeddings.device,
                    dtype=torch.long,
                )
                if self.num_negatives > 0 and negative_tensor.numel() > self.num_negatives:
                    selection = torch.randperm(
                        negative_tensor.numel(), device=embeddings.device
                    )[: self.num_negatives]
                    negative_tensor = negative_tensor.index_select(0, selection)

                anchor_embedding = sample_embeddings[anchor_position]
                positive_embedding = sample_embeddings[positive_position]
                negative_embeddings = sample_embeddings.index_select(0, negative_tensor)

                positive_logit = torch.matmul(anchor_embedding, positive_embedding).view(1)
                negative_logits = negative_embeddings @ anchor_embedding
                logits = torch.cat([positive_logit, negative_logits], dim=0).unsqueeze(0)
                logits = logits / self.temperature

                # positive を class 0 とする標準的な InfoNCE 形式。
                pair_loss = F.cross_entropy(
                    logits,
                    torch.zeros(1, dtype=torch.long, device=embeddings.device),
                )
                anchor_weight = sample_weights[anchor_position]
                total_weighted_loss = total_weighted_loss + (pair_loss * anchor_weight)
                total_weight = total_weight + anchor_weight
                total_pairs += 1

        if total_pairs == 0 or float(total_weight) <= 0.0:
            loss = embeddings.sum() * 0.0
        else:
            loss = total_weighted_loss / total_weight

        batch_size = max(1, embeddings.shape[0])
        metrics = {
            "contrastive_loss": float(loss.detach()),
            "contrastive_pairs": float(total_pairs),
            "active_samples": float(active_samples),
            "avg_peaks_per_sample": float(total_peaks) / float(batch_size),
        }
        return loss, metrics
