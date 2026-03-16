import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import math


@torch.no_grad()
def apply_ranked_stem_dropout(
    waveform: torch.Tensor,
    num_stems: int,
    max_dropout_stems: int,
    prioritized_stem_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        drop_mask = torch.zeros(
            (batch_size, num_stems),
            dtype=torch.bool,
            device=waveform.device,
        )
        return waveform, dropped_counts, drop_mask

    stem_waveform = waveform.view(batch_size, num_stems, channels_per_stem, num_samples)

    # stereo をまとめた平均二乗エネルギーで stem を並べる。
    stem_energy = stem_waveform.square().mean(dim=(2, 3))
    energy_rank = torch.argsort(stem_energy, dim=1, descending=False)

    # 実験用に特定 stem を優先的に落としたいときは、
    # エネルギー順位の先頭へ寄せて「1本以上落とすならまずそこから」にする。
    if prioritized_stem_index is not None:
        if not (0 <= prioritized_stem_index < num_stems):
            raise ValueError("prioritized_stem_index is out of range")
        prioritized = energy_rank == prioritized_stem_index
        energy_rank = torch.cat(
            [
                energy_rank.masked_select(prioritized).view(batch_size, 1),
                energy_rank.masked_select(~prioritized).view(batch_size, num_stems - 1),
            ],
            dim=1,
        )

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
    return augmented.view_as(waveform), dropped_counts, drop_mask


@torch.no_grad()
def time_stretch_waveform(
    x: torch.Tensor, rate: float, n_fft=256, hop_length=128, win_length=256
) -> torch.Tensor:
    """
    x: (B, C, T) または (C, T) のCUDAテンソル想定
    rate: >1 で速く(短く), <1 で遅く(長く)
    """
    device, dtype = x.device, x.dtype
    # Batch次元の考慮: (C, T) -> (1, C, T)
    is_unbatched = x.dim() == 2
    if is_unbatched:
        x = x.unsqueeze(0)

    B, C, T = x.shape
    x_2d = x.reshape(B * C, T)

    window = torch.hann_window(win_length, device=device, dtype=dtype)

    # (B*C, F, Frames) complex
    spec = torch.stft(
        x_2d,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )

    # docsの例の通り phase_advance を作る
    freq = spec.size(-2)
    phase_advance = torch.linspace(
        0, math.pi * hop_length, freq, device=device, dtype=dtype
    )[..., None]

    spec_st = AF.phase_vocoder(spec, rate=rate, phase_advance=phase_advance)

    y_2d = torch.istft(
        spec_st,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )
    y = y_2d.reshape(B, C, -1)

    if is_unbatched:
        y = y.squeeze(0)
    return y


def apply_batch_time_stretch(batch: dict) -> dict:
    """
    バッチ内の各オーディオに対して個別の time_stretch_rate でタイムストレッチを適用し、
    目標の長さに揃えた新しいバッチを返します。
    """
    time_stretch_rate = batch.get("time_stretch_rate")
    target_samples = batch.get("target_samples")
    waveform = batch.get("audio")

    if time_stretch_rate is None or target_samples is None or waveform is None:
        return batch

    B = waveform.shape[0]
    processed_waves = []

    is_batched_rates = time_stretch_rate.dim() > 0
    is_batched_samples = target_samples.dim() > 0

    for b in range(B):
        rate = (
            float(time_stretch_rate[b])
            if is_batched_rates
            else float(time_stretch_rate)
        )
        in_samples = (
            int(target_samples[b]) if is_batched_samples else int(target_samples)
        )

        wave_b = waveform[b, :, :in_samples]

        if abs(rate - 1.0) > 1e-5:
            wave_b = time_stretch_waveform(wave_b, rate=rate)

        processed_waves.append(wave_b)

    # 全テンソルの長さを fixed_target_len に揃える
    fixed_target_len = (
        int(target_samples[0] / time_stretch_rate[0])
        if is_batched_rates
        else int(target_samples / time_stretch_rate)
    )

    padded_waves = []
    for w in processed_waves:
        if w.shape[-1] < fixed_target_len:
            w = F.pad(w, (0, fixed_target_len - w.shape[-1]))
        elif w.shape[-1] > fixed_target_len:
            w = w[:, :fixed_target_len]
        padded_waves.append(w)

    batch["audio"] = torch.stack(padded_waves, dim=0)
    return batch
