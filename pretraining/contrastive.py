from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class PLPContrastiveLoss(nn.Module):
    """
    PLP ピーク列に対する contrastive loss。

    以前の「ピーク列添字ベース」ではなく、
    - positive: anchor 近傍で見積もった局所 tau の ±tau, ±2tau 付近
    - negative: frame 距離で安全半径の外にあるピーク
    を使う。

    これにより、テンポ揺れや odd meter でピーク添字が歪んでも、
    実時間上の rhythmic distance に近い形で対照学習できる。
    """

    def __init__(
        self,
        temperature: float = 0.1,
        positive_multiples: Sequence[int] = (1, 2),
        tau_neighborhood: int = 2,
        positive_tolerance_ratio: float = 0.25,
        positive_tolerance_frames: int = 2,
        negative_safety_radius_ratio: float = 0.5,
        negative_safety_radius_frames: int = 4,
        num_negatives: int = 32,
        max_anchors_per_sample: int = 64,
        anchor_sampling: Literal["topk", "random", "weighted_random"] = "weighted_random",
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        normalized_multiples = sorted(
            {int(multiple) for multiple in positive_multiples if int(multiple) > 0}
        )
        if not normalized_multiples:
            raise ValueError("positive_multiples must contain at least one positive integer")
        if tau_neighborhood < 1:
            raise ValueError("tau_neighborhood must be positive")
        if positive_tolerance_ratio < 0:
            raise ValueError("positive_tolerance_ratio must be non-negative")
        if positive_tolerance_frames < 0:
            raise ValueError("positive_tolerance_frames must be non-negative")
        if negative_safety_radius_ratio < 0:
            raise ValueError("negative_safety_radius_ratio must be non-negative")
        if negative_safety_radius_frames < 0:
            raise ValueError("negative_safety_radius_frames must be non-negative")
        if anchor_sampling not in {"topk", "random", "weighted_random"}:
            raise ValueError("anchor_sampling must be topk, random, or weighted_random")

        self.temperature = float(temperature)
        self.positive_multiples = tuple(normalized_multiples)
        self.tau_neighborhood = int(tau_neighborhood)
        self.positive_tolerance_ratio = float(positive_tolerance_ratio)
        self.positive_tolerance_frames = int(positive_tolerance_frames)
        self.negative_safety_radius_ratio = float(negative_safety_radius_ratio)
        self.negative_safety_radius_frames = int(negative_safety_radius_frames)
        self.num_negatives = int(num_negatives)
        self.max_anchors_per_sample = int(max_anchors_per_sample)
        self.anchor_sampling = anchor_sampling

    def _sample_anchor_positions(
        self,
        peak_count: int,
        sample_weights: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if self.max_anchors_per_sample <= 0 or self.max_anchors_per_sample >= peak_count:
            return torch.arange(peak_count, device=device)

        num_samples = min(self.max_anchors_per_sample, peak_count)
        if self.anchor_sampling == "topk":
            return torch.argsort(sample_weights, descending=True)[:num_samples]
        if self.anchor_sampling == "random":
            return torch.randperm(peak_count, device=device)[:num_samples]

        sampling_weights = sample_weights.clamp_min(1e-6)
        return torch.multinomial(
            sampling_weights,
            num_samples=num_samples,
            replacement=False,
        )

    def _estimate_local_tau(
        self,
        peak_indices: torch.Tensor,
        anchor_position: int,
    ) -> float | None:
        if peak_indices.numel() < 2:
            return None

        peak_intervals = peak_indices[1:] - peak_indices[:-1]
        start = max(0, anchor_position - self.tau_neighborhood)
        end = min(peak_intervals.numel(), anchor_position + self.tau_neighborhood + 1)
        local_intervals = peak_intervals[start:end].to(torch.float32)
        local_intervals = local_intervals[local_intervals > 0]
        if local_intervals.numel() == 0:
            return None
        return float(local_intervals.median().item())

    def _positive_positions(
        self,
        peak_indices: torch.Tensor,
        anchor_position: int,
        local_tau: float,
    ) -> torch.Tensor:
        anchor_frame = peak_indices[anchor_position].to(torch.float32)
        frame_distances = (peak_indices.to(torch.float32) - anchor_frame).abs()
        positive_mask = torch.zeros_like(frame_distances, dtype=torch.bool)

        for multiple in self.positive_multiples:
            target_distance = local_tau * float(multiple)
            tolerance = max(
                float(self.positive_tolerance_frames),
                target_distance * self.positive_tolerance_ratio,
            )
            positive_mask |= (frame_distances - target_distance).abs() <= tolerance

        positive_mask[anchor_position] = False
        return positive_mask.nonzero(as_tuple=False).squeeze(-1)

    def _negative_positions(
        self,
        peak_indices: torch.Tensor,
        anchor_position: int,
        positive_positions: torch.Tensor,
        local_tau: float,
    ) -> torch.Tensor:
        anchor_frame = peak_indices[anchor_position].to(torch.float32)
        frame_distances = (peak_indices.to(torch.float32) - anchor_frame).abs()
        safety_radius = max(
            float(self.negative_safety_radius_frames),
            local_tau * self.negative_safety_radius_ratio,
        )
        negative_mask = frame_distances > safety_radius
        negative_mask[anchor_position] = False
        if positive_positions.numel() > 0:
            negative_mask[positive_positions] = False

        negative_positions = negative_mask.nonzero(as_tuple=False).squeeze(-1)
        if negative_positions.numel() == 0:
            fallback_mask = torch.ones_like(frame_distances, dtype=torch.bool)
            fallback_mask[anchor_position] = False
            if positive_positions.numel() > 0:
                fallback_mask[positive_positions] = False
            negative_positions = fallback_mask.nonzero(as_tuple=False).squeeze(-1)

        return negative_positions

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
            sample_embeddings = F.normalize(sample_embeddings, dim=-1)
            sample_weights = peak_values[batch_index].index_select(0, peak_indices).clamp_min(1e-6)
            active_samples += 1

            anchor_positions = self._sample_anchor_positions(
                peak_count=peak_count,
                sample_weights=sample_weights,
                device=embeddings.device,
            )

            for anchor_position_tensor in anchor_positions:
                anchor_position = int(anchor_position_tensor.item())
                local_tau = self._estimate_local_tau(peak_indices=peak_indices, anchor_position=anchor_position)
                if local_tau is None or local_tau <= 0.0:
                    continue

                positive_tensor = self._positive_positions(
                    peak_indices=peak_indices,
                    anchor_position=anchor_position,
                    local_tau=local_tau,
                )
                if positive_tensor.numel() == 0:
                    continue

                negative_tensor = self._negative_positions(
                    peak_indices=peak_indices,
                    anchor_position=anchor_position,
                    positive_positions=positive_tensor,
                    local_tau=local_tau,
                )
                if negative_tensor.numel() == 0:
                    continue

                if self.num_negatives > 0 and negative_tensor.numel() > self.num_negatives:
                    selection = torch.randperm(
                        negative_tensor.numel(), device=embeddings.device
                    )[: self.num_negatives]
                    negative_tensor = negative_tensor.index_select(0, selection)

                anchor_embedding = sample_embeddings[anchor_position]
                positive_embeddings = sample_embeddings.index_select(0, positive_tensor)
                negative_embeddings = sample_embeddings.index_select(0, negative_tensor)

                positive_logits = (positive_embeddings @ anchor_embedding) / self.temperature
                negative_logits = (negative_embeddings @ anchor_embedding) / self.temperature
                all_logits = torch.cat([positive_logits, negative_logits], dim=0)

                # multi-positive InfoNCE。
                # anchor に対して正例群の和確率を上げる。
                pair_loss = torch.logsumexp(all_logits, dim=0) - torch.logsumexp(
                    positive_logits,
                    dim=0,
                )
                anchor_weight = sample_weights[anchor_position]
                total_weighted_loss = total_weighted_loss + (pair_loss * anchor_weight)
                total_weight = total_weight + anchor_weight
                total_pairs += int(positive_tensor.numel())

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
