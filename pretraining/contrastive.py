from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def symmetric_info_nce(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """batch 内の他サンプルを negative とする軽量 InfoNCE。"""

    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    if anchor.shape[0] < 2:
        return 1.0 - F.cosine_similarity(anchor, positive, dim=-1).mean()

    logits = (anchor @ positive.transpose(0, 1)) / temperature
    labels = torch.arange(anchor.shape[0], device=anchor.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (loss_ab + loss_ba)


class DrumContrastiveObjective(nn.Module):
    """drum-only view に対する rhythm contrastive を担当する。"""

    def __init__(
        self,
        temperature: float = 0.1,
        rhythm_drum_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.temperature = float(temperature)
        self.rhythm_drum_weight = float(rhythm_drum_weight)

    def _info_nce_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
    ) -> torch.Tensor:
        return symmetric_info_nce(anchor, positive, temperature=self.temperature)
