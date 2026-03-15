import math

import torch
import torch.nn.functional as F
import torchaudio.functional as AF


@torch.no_grad()
def apply_ranked_stem_dropout(
    waveform: torch.Tensor,
    num_stems: int,
    max_dropout_stems: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    stem ごとのエネルギーを見て、弱い stem から順にランダム本数だけ落とす。
    最低 1 stem は残し、学習時の頑健性を上げるために使う。
    """
    if waveform.dim() != 3:
        raise ValueError("waveform must have shape (B, C, T)")
    if num_stems <= 0:
        raise ValueError("num_stems must be positive")
    if waveform.shape[1] % num_stems != 0:
        raise ValueError("audio channels must be divisible by num_stems")

    batch_size, num_channels, num_samples = waveform.shape
    channels_per_stem = num_channels // num_stems
    effective_max_dropout = min(int(max_dropout_stems), num_stems - 1)
    if effective_max_dropout <= 0:
        dropped_counts = torch.zeros(
            batch_size, dtype=torch.long, device=waveform.device
        )
        return waveform, dropped_counts

    stem_waveform = waveform.view(batch_size, num_stems, channels_per_stem, num_samples)

    # stereo をまとめた平均二乗エネルギーで stem を並べる。
    stem_energy = stem_waveform.square().mean(dim=(2, 3))
    energy_rank = torch.argsort(stem_energy, dim=1, descending=False)

    dropped_counts = torch.randint(
        low=1,
        high=effective_max_dropout + 1,
        size=(batch_size,),
        device=waveform.device,
    )
    drop_mask = torch.zeros(
        (batch_size, num_stems),
        dtype=torch.bool,
        device=waveform.device,
    )

    rank_index = torch.arange(num_stems, device=waveform.device).unsqueeze(0)
    should_drop_by_rank = rank_index < dropped_counts.unsqueeze(1)
    drop_mask.scatter_(dim=1, index=energy_rank, src=should_drop_by_rank)

    augmented = stem_waveform.masked_fill(
        drop_mask.unsqueeze(-1).unsqueeze(-1),
        0.0,
    )
    return augmented.view_as(waveform), dropped_counts


@torch.no_grad()
def time_stretch_waveform(
    waveform: torch.Tensor,
    rate: float,
    n_fft: int = 256,
    hop_length: int = 128,
    win_length: int = 256,
) -> torch.Tensor:
    """
    phase vocoder で波形を time stretch する。

    rate > 1.0 で短く、rate < 1.0 で長くなる。
    入力は (C, T) または (B, C, T) を受け付ける。
    """
    device = waveform.device
    dtype = waveform.dtype
    is_unbatched = waveform.dim() == 2
    if is_unbatched:
        waveform = waveform.unsqueeze(0)

    batch_size, num_channels, _ = waveform.shape
    flat_waveform = waveform.reshape(batch_size * num_channels, waveform.shape[-1])
    window = torch.hann_window(win_length, device=device, dtype=dtype)
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
        device=device,
        dtype=dtype,
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


def _stretch_binary_targets(
    targets: torch.Tensor,
    valid_frames: int,
    duration_scale: float,
    target_num_frames: int,
) -> torch.Tensor:
    """
    0/1 のイベント列を、フレーム index のスケーリングで伸縮する。
    """
    stretched_targets = torch.zeros(
        target_num_frames,
        dtype=targets.dtype,
        device=targets.device,
    )
    if valid_frames <= 0:
        return stretched_targets

    target_indices = torch.nonzero(
        targets[:valid_frames] > 0.5,
        as_tuple=False,
    ).flatten()
    if target_indices.numel() == 0:
        return stretched_targets

    stretched_indices = torch.round(
        target_indices.to(torch.float32) * duration_scale
    ).long()
    stretched_indices = stretched_indices[
        (stretched_indices >= 0) & (stretched_indices < target_num_frames)
    ]
    if stretched_indices.numel() > 0:
        stretched_targets[stretched_indices.unique()] = 1.0
    return stretched_targets


def _stretch_meter_targets(
    meter_targets: torch.Tensor,
    valid_frames: int,
    full_stretched_valid_frames: int,
    stretched_valid_frames: int,
    target_num_frames: int,
    meter_ignore_index: int,
) -> torch.Tensor:
    stretched_meter = torch.full(
        (target_num_frames,),
        fill_value=meter_ignore_index,
        device=meter_targets.device,
        dtype=meter_targets.dtype,
    )
    if valid_frames <= 0 or full_stretched_valid_frames <= 0 or stretched_valid_frames <= 0:
        return stretched_meter

    source_meter = meter_targets[:valid_frames].to(torch.float32).view(1, 1, -1)
    resized_meter = F.interpolate(
        source_meter,
        size=full_stretched_valid_frames,
        mode="nearest",
    ).view(-1)[:stretched_valid_frames]
    stretched_meter[:stretched_valid_frames] = resized_meter.to(meter_targets.dtype)
    return stretched_meter


@torch.no_grad()
def apply_sample_time_stretch(
    waveform: torch.Tensor,
    beat_targets: torch.Tensor,
    downbeat_targets: torch.Tensor,
    meter_targets: torch.Tensor,
    valid_frames: int,
    min_percent: float,
    max_percent: float,
    n_fft: int,
    hop_length: int,
    target_samples: int,
    target_num_frames: int,
    meter_ignore_index: int,
) -> dict[str, torch.Tensor | float]:
    """
    1 サンプルぶんの waveform / ラベルを同じ倍率で time stretch する。

    出力長は常に `target_samples` / `target_num_frames` に揃える。
    短くなる側は、dataset 側で事前に長めに読んだ waveform を使って埋める。
    """
    if min_percent > max_percent:
        raise ValueError("time stretch min_percent must be <= max_percent")
    if min_percent <= -100.0 or max_percent <= -100.0:
        raise ValueError("time stretch percent must be greater than -100")

    if min_percent == 0.0 and max_percent == 0.0:
        sample_valid_mask = torch.zeros(
            target_num_frames,
            dtype=torch.float32,
            device=waveform.device,
        )
        sample_valid_mask[: min(valid_frames, target_num_frames)] = 1.0
        return {
            "waveform": waveform[..., :target_samples],
            "beat_targets": beat_targets[:target_num_frames],
            "downbeat_targets": downbeat_targets[:target_num_frames],
            "meter_targets": meter_targets[:target_num_frames],
            "valid_mask": sample_valid_mask,
            "time_stretch_percent": 0.0,
            "time_stretch_duration_scale": 1.0,
        }

    stretch_percent = float(
        torch.empty(1, device=waveform.device).uniform_(min_percent, max_percent).item()
    )
    duration_scale = 1.0 + (stretch_percent / 100.0)
    phase_vocoder_rate = 1.0 / duration_scale

    stretched_waveform = time_stretch_waveform(
        waveform,
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

    full_stretched_valid_frames = int(round(valid_frames * duration_scale))
    stretched_valid_frames = min(target_num_frames, full_stretched_valid_frames)

    valid_mask = torch.zeros(
        target_num_frames,
        dtype=torch.float32,
        device=waveform.device,
    )
    valid_mask[:stretched_valid_frames] = 1.0

    stretched_beat_targets = _stretch_binary_targets(
        beat_targets,
        valid_frames=valid_frames,
        duration_scale=duration_scale,
        target_num_frames=target_num_frames,
    )
    stretched_downbeat_targets = _stretch_binary_targets(
        downbeat_targets,
        valid_frames=valid_frames,
        duration_scale=duration_scale,
        target_num_frames=target_num_frames,
    )
    stretched_meter_targets = _stretch_meter_targets(
        meter_targets,
        valid_frames=valid_frames,
        full_stretched_valid_frames=full_stretched_valid_frames,
        stretched_valid_frames=stretched_valid_frames,
        target_num_frames=target_num_frames,
        meter_ignore_index=meter_ignore_index,
    )

    return {
        "waveform": stretched_waveform,
        "beat_targets": stretched_beat_targets,
        "downbeat_targets": stretched_downbeat_targets,
        "meter_targets": stretched_meter_targets,
        "valid_mask": valid_mask,
        "time_stretch_percent": stretch_percent,
        "time_stretch_duration_scale": duration_scale,
    }


@torch.no_grad()
def apply_gpu_time_stretch_and_rebuild_targets(
    batch: dict,
    min_percent: float,
    max_percent: float,
    n_fft: int,
    hop_length: int,
    target_samples: int,
    target_num_frames: int,
    meter_ignore_index: int,
    aux_target_builder,
    repeat_pair_builder,
    drum_aux_loss_weight: float,
) -> dict:
    """
    GPU 上で batch 単位の time stretch を行い、
    伸縮後の waveform に合わせて補助ターゲットを作り直す。

    dataset 側では長めに読んだ元ラベルだけを返し、
    ここで最終的な学習窓へ揃える。
    """
    if min_percent == 0.0 and max_percent == 0.0:
        return batch

    updated = dict(batch)
    waveforms: list[torch.Tensor] = []
    beat_targets: list[torch.Tensor] = []
    downbeat_targets: list[torch.Tensor] = []
    meter_targets: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    broadband_flux_targets: list[torch.Tensor] = []
    onset_env_targets: list[torch.Tensor] = []
    high_frequency_flux_targets: list[torch.Tensor] = []
    repeat_pair_indices: list[torch.Tensor] = []
    repeat_pair_masks: list[torch.Tensor] = []
    stretch_percents: list[float] = []
    duration_scales: list[float] = []

    batch_size = batch["waveform"].shape[0]
    for sample_index in range(batch_size):
        valid_frames = int(batch["valid_mask"][sample_index].sum().item())
        stretched_sample = apply_sample_time_stretch(
            waveform=batch["waveform"][sample_index],
            beat_targets=batch["beat_targets"][sample_index],
            downbeat_targets=batch["downbeat_targets"][sample_index],
            meter_targets=batch["meter_targets"][sample_index],
            valid_frames=valid_frames,
            min_percent=min_percent,
            max_percent=max_percent,
            n_fft=n_fft,
            hop_length=hop_length,
            target_samples=target_samples,
            target_num_frames=target_num_frames,
            meter_ignore_index=meter_ignore_index,
        )

        stretched_waveform = stretched_sample["waveform"]
        stretched_beat_targets = stretched_sample["beat_targets"]
        stretched_downbeat_targets = stretched_sample["downbeat_targets"]
        stretched_meter_targets = stretched_sample["meter_targets"]
        stretched_valid_mask = stretched_sample["valid_mask"]
        stretched_valid_frames = int(stretched_valid_mask.sum().item())

        if drum_aux_loss_weight > 0.0:
            aux_targets = aux_target_builder.build(
                waveform=stretched_waveform,
                valid_frames=stretched_valid_frames,
            )
            broadband_flux_targets.append(aux_targets.broadband_flux_targets)
            onset_env_targets.append(aux_targets.onset_env_targets)
            high_frequency_flux_targets.append(
                aux_targets.high_frequency_flux_targets
            )
        else:
            zero_aux = torch.zeros(
                target_num_frames,
                dtype=torch.float32,
                device=stretched_waveform.device,
            )
            broadband_flux_targets.append(zero_aux)
            onset_env_targets.append(zero_aux.clone())
            high_frequency_flux_targets.append(zero_aux.clone())

        if repeat_pair_builder is None:
            repeat_pair_indices.append(
                torch.zeros(
                    (
                        batch["repeat_pair_indices"].shape[1],
                        2,
                    ),
                    dtype=torch.long,
                    device=stretched_waveform.device,
                )
            )
            repeat_pair_masks.append(
                torch.zeros(
                    batch["repeat_pair_mask"].shape[1],
                    dtype=torch.float32,
                    device=stretched_waveform.device,
                )
            )
        else:
            repeat_targets = repeat_pair_builder.build_from_frame_targets(
                waveform=stretched_waveform,
                beat_targets=stretched_beat_targets,
                downbeat_targets=stretched_downbeat_targets,
                meter_targets=stretched_meter_targets,
                valid_mask=stretched_valid_mask,
            )
            repeat_pair_indices.append(repeat_targets.pair_indices)
            repeat_pair_masks.append(repeat_targets.pair_mask)

        waveforms.append(stretched_waveform)
        beat_targets.append(stretched_beat_targets)
        downbeat_targets.append(stretched_downbeat_targets)
        meter_targets.append(stretched_meter_targets)
        valid_masks.append(stretched_valid_mask)
        stretch_percents.append(float(stretched_sample["time_stretch_percent"]))
        duration_scales.append(
            float(stretched_sample["time_stretch_duration_scale"])
        )

    updated["waveform"] = torch.stack(waveforms, dim=0)
    updated["beat_targets"] = torch.stack(beat_targets, dim=0)
    updated["downbeat_targets"] = torch.stack(downbeat_targets, dim=0)
    updated["meter_targets"] = torch.stack(meter_targets, dim=0)
    updated["valid_mask"] = torch.stack(valid_masks, dim=0)
    updated["broadband_flux_targets"] = torch.stack(broadband_flux_targets, dim=0)
    updated["onset_env_targets"] = torch.stack(onset_env_targets, dim=0)
    updated["high_frequency_flux_targets"] = torch.stack(
        high_frequency_flux_targets, dim=0
    )
    updated["repeat_pair_indices"] = torch.stack(repeat_pair_indices, dim=0)
    updated["repeat_pair_mask"] = torch.stack(repeat_pair_masks, dim=0)
    updated["time_stretch_percent"] = torch.tensor(
        stretch_percents,
        dtype=torch.float32,
        device=updated["waveform"].device,
    )
    updated["time_stretch_duration_scale"] = torch.tensor(
        duration_scales,
        dtype=torch.float32,
        device=updated["waveform"].device,
    )
    return updated
