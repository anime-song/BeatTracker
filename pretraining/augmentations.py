from __future__ import annotations

from typing import Sequence

import torch


def keep_only_stems(
    waveform: torch.Tensor,
    num_stems: int,
    keep_stem_indices: Sequence[int],
) -> torch.Tensor:
    if waveform.ndim != 3:
        raise ValueError("waveform must have shape [batch, channels, time]")
    if num_stems <= 0:
        raise ValueError("num_stems must be positive")
    if waveform.shape[1] % num_stems != 0:
        raise ValueError("channel count must be divisible by num_stems")

    channels_per_stem = waveform.shape[1] // num_stems
    kept = waveform.new_zeros(waveform.shape)
    for stem_index in keep_stem_indices:
        if not 0 <= int(stem_index) < num_stems:
            raise ValueError(f"stem_index out of range: {stem_index}")
        channel_start = int(stem_index) * channels_per_stem
        channel_end = channel_start + channels_per_stem
        kept[:, channel_start:channel_end] = waveform[:, channel_start:channel_end]
    return kept
