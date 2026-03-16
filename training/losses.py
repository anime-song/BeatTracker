import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, List, Optional, Tuple


class BalancedSoftmaxLoss(nn.Module):
    def __init__(
        self,
        class_counts: Union[List[int], torch.Tensor],
        tau: float = 1.0,
        ignore_index: int = -100,
    ):
        """
        Args:
            class_counts (Union[List[int], torch.Tensor]):
                各クラスの出現回数のリストまたはテンソル。
                事前に Laplace 平滑化（全カウントに+1するなど）を推奨します。
            tau (float, optional): 補正のスケール係数. Defaults to 1.0.
        """
        super().__init__()

        class_counts = torch.as_tensor(class_counts, dtype=torch.float32)

        # log_prior を計算し、バッファとして登録
        # カウントが0のクラスは-infになるのを防ぐため、非常に小さい値にクリップ
        log_prior = torch.log(torch.clamp(class_counts, min=1e-9))

        self.register_buffer("log_prior", log_prior)
        self.tau = tau
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): モデルの出力ロジット (B, T, C)
            labels (torch.Tensor): 正解ラベル (B, T)

        Returns:
            torch.Tensor: 計算された損失値 (スカラー)
        """
        # 形状を合わせる
        if logits.dim() > 2:
            logits = logits.reshape(-1, logits.size(-1))  # (B*T, C)
            labels = labels.reshape(-1)  # (B*T,)

        # meter が未定義のフレームは ignore_index にして、そのまま落とす。
        valid = labels != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0

        logits = logits[valid]
        labels = labels[valid]

        # ロジット補正: z_k <- z_k + τ * log(n_k)
        adjusted_logits = logits + self.tau * self.log_prior
        loss = F.cross_entropy(adjusted_logits, labels)
        return loss


def masked_l1_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = (predictions - targets).abs()
    if mask is None:
        return diff.mean()

    weighted = diff * mask.to(diff.dtype)
    normalizer = mask.sum().clamp_min(1.0).to(diff.dtype)
    return weighted.sum() / normalizer


def masked_index_pair_l1_loss(
    sequence: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    sequence: (B, T)
    pair_indices: (B, P, 2)
    pair_mask: (B, P)
    """
    left = torch.gather(sequence, 1, pair_indices[..., 0])
    right = torch.gather(sequence, 1, pair_indices[..., 1])
    diff = (left - right).abs()
    if pair_mask is None:
        return diff.mean()

    weighted = diff * pair_mask.to(diff.dtype)
    normalizer = pair_mask.sum().clamp_min(1.0).to(diff.dtype)
    return weighted.sum() / normalizer


def build_meter_from_rhythm_examples(
    beat_logits: torch.Tensor,
    downbeat_logits: torch.Tensor,
    downbeat_targets: torch.Tensor,
    meter_targets: torch.Tensor,
    valid_mask: torch.Tensor,
    sample_length: int,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GT downbeat 区間ごとに、beat/downbeat の予測系列を固定長へ補間して
    meter classification 用のサンプルを作る。
    """
    beat_probabilities = torch.sigmoid(beat_logits)
    downbeat_probabilities = torch.sigmoid(downbeat_logits)

    interval_features: list[torch.Tensor] = []
    interval_targets: list[int] = []

    batch_size = beat_logits.shape[0]
    for batch_index in range(batch_size):
        sample_valid = valid_mask[batch_index] > 0.5
        event_indices = torch.nonzero(
            (downbeat_targets[batch_index] > 0.5) & sample_valid,
            as_tuple=False,
        ).flatten()
        if event_indices.numel() < 2:
            continue

        for start_frame, end_frame in zip(
            event_indices[:-1].tolist(),
            event_indices[1:].tolist(),
        ):
            if end_frame <= start_frame:
                continue

            interval_meter_targets = meter_targets[batch_index, start_frame:end_frame]
            interval_meter_targets = interval_meter_targets[
                interval_meter_targets != int(ignore_index)
            ]
            if interval_meter_targets.numel() == 0:
                continue

            target_class = int(interval_meter_targets[0].item())
            interval_sequence = torch.stack(
                [
                    beat_probabilities[batch_index, start_frame:end_frame],
                    downbeat_probabilities[batch_index, start_frame:end_frame],
                ],
                dim=0,
            ).unsqueeze(0)
            resized_sequence = F.interpolate(
                interval_sequence,
                size=sample_length,
                mode="linear",
                align_corners=False,
            ).squeeze(0)
            interval_features.append(resized_sequence)
            interval_targets.append(target_class)

    if not interval_features:
        empty_features = beat_logits.new_zeros((0, 2, sample_length))
        empty_targets = meter_targets.new_zeros((0,), dtype=torch.long)
        return empty_features, empty_targets

    return (
        torch.stack(interval_features, dim=0),
        torch.tensor(interval_targets, dtype=torch.long, device=beat_logits.device),
    )


# https://github.com/CPJKU/beat_this/blob/main/beat_this/model/loss.py
class ShiftTolerantBCELoss(torch.nn.Module):
    """
    BCE loss variant for sequence labeling that tolerates small shifts between
    predictions and targets. This is accomplished by max-pooling the
    predictions with a given tolerance and a stride of 1, so the gradient for a
    positive label affects the largest prediction in a window around it.
    Expects predictions to be given as logits, and accepts an optional mask
    with zeros indicating the entries to ignore. Note that the edges of the
    sequence will not receive a gradient, as it is assumed to be unknown
    whether there is a nearby positive annotation.

    Args:
        pos_weight (float): Weight for positive examples compared to negative
            examples (default: 1)
        tolerance (int): Tolerated shift in time steps in each direction
            (default: 1)
    """

    def __init__(self, pos_weight: float = 1, tolerance: int = 1):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight, dtype=torch.get_default_dtype()),
            persistent=False,
        )
        self.tolerance = tolerance

    def spread(self, x: torch.Tensor, factor: int = 1):
        if self.tolerance == 0:
            return x
        return F.max_pool1d(x, 1 + 2 * factor * self.tolerance, 1)

    def crop(self, x: torch.Tensor, factor: int = 1):
        return x[..., factor * self.tolerance : -factor * self.tolerance or None]

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        # spread preds and crop targets to match
        spreaded_preds = self.crop(self.spread(preds))
        cropped_targets = self.crop(targets, factor=2)
        # ignore around the positive targets
        look_at = cropped_targets + (1 - self.spread(targets, factor=2))
        if mask is not None:  # consider padding and no-downbeat mask
            look_at = look_at * self.crop(mask, factor=2)
        # compute loss
        return F.binary_cross_entropy_with_logits(
            spreaded_preds,
            cropped_targets,
            weight=look_at,
            pos_weight=self.pos_weight,
        )
