import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import random
import math


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


@torch.no_grad()
def apply_ranked_stem_dropout(
    waveform: torch.Tensor,
    num_stems: int,
    max_dropout_stems: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    stem ごとのエネルギーを見て、エネルギーが小さい stem から順に落とす。

    - drop 本数はサンプルごとに 1..max_dropout_stems の範囲でランダム
    - ただし必ず 1 stem は残す
    - 一番エネルギーが大きい stem は最後まで残る
    """
    if max_dropout_stems <= 0 or num_stems <= 1:
        batch_size = 1 if waveform.dim() == 2 else waveform.shape[0]
        dropped_counts = torch.zeros(batch_size, device=waveform.device, dtype=torch.long)
        return waveform, dropped_counts

    is_unbatched = waveform.dim() == 2
    if is_unbatched:
        waveform = waveform.unsqueeze(0)

    batch_size, num_channels, num_samples = waveform.shape
    if num_channels % num_stems != 0:
        raise ValueError(
            f"num_channels ({num_channels}) must be divisible by num_stems ({num_stems})"
        )

    channels_per_stem = num_channels // num_stems
    stem_waveform = waveform.view(batch_size, num_stems, channels_per_stem, num_samples)

    # stem ごとの平均二乗エネルギーを使って、落とす優先順位を決める。
    stem_energy = stem_waveform.square().mean(dim=(2, 3))
    energy_rank = torch.argsort(stem_energy, dim=1, descending=False)

    max_dropout_stems = min(int(max_dropout_stems), num_stems - 1)
    dropped_counts = torch.randint(
        low=1,
        high=max_dropout_stems + 1,
        size=(batch_size,),
        device=waveform.device,
    )

    # rank が小さい stem ほど先に落とす。
    rank_index = torch.arange(num_stems, device=waveform.device).unsqueeze(0)
    drop_by_rank = rank_index < dropped_counts.unsqueeze(1)
    drop_mask = torch.zeros(
        batch_size, num_stems, device=waveform.device, dtype=torch.bool
    )
    drop_mask.scatter_(1, energy_rank, drop_by_rank)

    kept_stem_mask = (~drop_mask).to(dtype=waveform.dtype).view(
        batch_size, num_stems, 1, 1
    )
    stem_waveform = stem_waveform * kept_stem_mask
    waveform = stem_waveform.view(batch_size, num_channels, num_samples)

    if is_unbatched:
        waveform = waveform.squeeze(0)
    return waveform, dropped_counts
