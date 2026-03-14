from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torchaudio.functional as AF


@torch.no_grad()
def time_stretch_waveform(
    waveform: torch.Tensor,
    rate: float,
    n_fft: int = 256,
    hop_length: int = 128,
    win_length: int = 256,
) -> torch.Tensor:
    """
    波形に phase vocoder の time stretch を適用する。

    rate > 1.0 で短く、rate < 1.0 で長くなる。
    入力は (C, T) または (B, C, T) を受け付ける。
    """
    is_unbatched = waveform.dim() == 2
    if is_unbatched:
        waveform = waveform.unsqueeze(0)

    batch_size, num_channels, _ = waveform.shape
    flat_waveform = waveform.reshape(batch_size * num_channels, waveform.shape[-1])

    window = torch.hann_window(
        win_length,
        device=waveform.device,
        dtype=waveform.dtype,
    )
    spec = torch.stft(
        flat_waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )

    phase_advance = torch.linspace(
        0,
        math.pi * hop_length,
        spec.size(-2),
        device=waveform.device,
        dtype=waveform.dtype,
    )[..., None]
    stretched_spec = AF.phase_vocoder(
        spec,
        rate=rate,
        phase_advance=phase_advance,
    )
    stretched_waveform = torch.istft(
        stretched_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )
    stretched_waveform = stretched_waveform.reshape(batch_size, num_channels, -1)

    if is_unbatched:
        stretched_waveform = stretched_waveform.squeeze(0)
    return stretched_waveform


@torch.no_grad()
def apply_batch_time_stretch(
    batch: dict,
    min_percent: float,
    max_percent: float,
    n_fft: int,
    hop_length: int,
    target_samples: int,
    target_num_frames: int,
    meter_ignore_index: int,
) -> dict:
    """
    train batch にだけ使う time stretch。

    - stretch の対象は waveform / beat / downbeat / meter / valid_mask
    - percent は duration の変化率として扱う
      例: +50 -> 1.5 倍に伸びる, -50 -> 0.5 倍に縮む
    - 出力テンソル長は target_samples / target_num_frames に戻す
    """
    waveform = batch.get("waveform")
    beat_targets = batch.get("beat_targets")
    downbeat_targets = batch.get("downbeat_targets")
    meter_targets = batch.get("meter_targets")
    valid_mask = batch.get("valid_mask")

    if waveform is None or beat_targets is None or downbeat_targets is None:
        return batch
    if meter_targets is None or valid_mask is None:
        return batch

    if min_percent > max_percent:
        raise ValueError("time stretch min_percent must be <= max_percent")
    if min_percent <= -100.0:
        raise ValueError("time stretch min_percent must be greater than -100")
    if max_percent <= -100.0:
        raise ValueError("time stretch max_percent must be greater than -100")
    if min_percent == 0.0 and max_percent == 0.0:
        return batch

    batch_size = waveform.shape[0]
    stretch_percent = torch.empty(batch_size, device=waveform.device).uniform_(
        min_percent,
        max_percent,
    )
    duration_scales = 1.0 + (stretch_percent / 100.0)
    phase_vocoder_rates = 1.0 / duration_scales

    stretched_waveforms: list[torch.Tensor] = []
    stretched_beat_targets: list[torch.Tensor] = []
    stretched_downbeat_targets: list[torch.Tensor] = []
    stretched_meter_targets: list[torch.Tensor] = []
    stretched_valid_masks: list[torch.Tensor] = []

    for sample_index in range(batch_size):
        duration_scale = float(duration_scales[sample_index].item())
        phase_vocoder_rate = float(phase_vocoder_rates[sample_index].item())

        stretched_waveform = time_stretch_waveform(
            waveform[sample_index],
            rate=phase_vocoder_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
        )
        if stretched_waveform.shape[-1] < target_samples:
            stretched_waveform = F.pad(
                stretched_waveform,
                (0, target_samples - stretched_waveform.shape[-1]),
            )
        else:
            stretched_waveform = stretched_waveform[..., :target_samples]
        stretched_waveforms.append(stretched_waveform)

        valid_frames = int(valid_mask[sample_index].sum().item())
        full_stretched_valid_frames = int(round(valid_frames * duration_scale))
        stretched_valid_frames = min(target_num_frames, full_stretched_valid_frames)

        sample_valid_mask = torch.zeros(
            target_num_frames,
            device=valid_mask.device,
            dtype=valid_mask.dtype,
        )
        sample_valid_mask[:stretched_valid_frames] = 1.0
        stretched_valid_masks.append(sample_valid_mask)

        stretched_beat = torch.zeros(
            target_num_frames,
            device=beat_targets.device,
            dtype=beat_targets.dtype,
        )
        beat_indices = torch.nonzero(
            beat_targets[sample_index, :valid_frames] > 0.5,
            as_tuple=False,
        ).flatten()
        if beat_indices.numel() > 0 and stretched_valid_frames > 0:
            stretched_indices = torch.round(
                beat_indices.to(torch.float32) * duration_scales[sample_index]
            ).long()
            stretched_indices = stretched_indices[
                stretched_indices < stretched_valid_frames
            ]
            if stretched_indices.numel() > 0:
                stretched_beat[stretched_indices.unique()] = 1.0
        stretched_beat_targets.append(stretched_beat)

        stretched_downbeat = torch.zeros(
            target_num_frames,
            device=downbeat_targets.device,
            dtype=downbeat_targets.dtype,
        )
        downbeat_indices = torch.nonzero(
            downbeat_targets[sample_index, :valid_frames] > 0.5,
            as_tuple=False,
        ).flatten()
        if downbeat_indices.numel() > 0 and stretched_valid_frames > 0:
            stretched_indices = torch.round(
                downbeat_indices.to(torch.float32) * duration_scales[sample_index]
            ).long()
            stretched_indices = stretched_indices[
                stretched_indices < stretched_valid_frames
            ]
            if stretched_indices.numel() > 0:
                stretched_downbeat[stretched_indices.unique()] = 1.0
        stretched_downbeat_targets.append(stretched_downbeat)

        stretched_meter = torch.full(
            (target_num_frames,),
            fill_value=meter_ignore_index,
            device=meter_targets.device,
            dtype=meter_targets.dtype,
        )
        if valid_frames > 0 and full_stretched_valid_frames > 0:
            source_meter = meter_targets[sample_index, :valid_frames]
            # meter は区間ラベルなので、まず全伸縮後の長さへ最近傍で伸ばし、
            # そのあと最終 crop 長へ切り詰める。
            source_meter = source_meter.to(torch.float32).view(1, 1, -1)
            resized_meter = F.interpolate(
                source_meter,
                size=full_stretched_valid_frames,
                mode="nearest",
            ).view(-1)[:stretched_valid_frames]
            stretched_meter[:stretched_valid_frames] = resized_meter.to(
                meter_targets.dtype
            )
        stretched_meter_targets.append(stretched_meter)

    batch["waveform"] = torch.stack(stretched_waveforms, dim=0)
    batch["beat_targets"] = torch.stack(stretched_beat_targets, dim=0)
    batch["downbeat_targets"] = torch.stack(stretched_downbeat_targets, dim=0)
    batch["meter_targets"] = torch.stack(stretched_meter_targets, dim=0)
    batch["valid_mask"] = torch.stack(stretched_valid_masks, dim=0)
    batch["time_stretch_percent"] = stretch_percent
    batch["time_stretch_duration_scale"] = duration_scales
    return batch
