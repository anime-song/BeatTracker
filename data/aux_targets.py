from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


def _minmax_normalize_1d(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.numel() == 0:
        return x
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + eps)


def _smooth_1d_series(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size <= 1 or x.numel() == 0:
        return x

    weight = torch.full(
        (1, 1, kernel_size),
        fill_value=1.0 / kernel_size,
        dtype=x.dtype,
        device=x.device,
    )
    return F.conv1d(
        x.view(1, 1, -1),
        weight,
        padding=kernel_size // 2,
    ).view(-1)


@dataclass(frozen=True)
class StemAuxTargets:
    broadband_flux_targets: torch.Tensor
    onset_env_targets: torch.Tensor
    bass_low_flux_targets: torch.Tensor


class StemAuxTargetBuilder:
    """
    stem 波形から補助ターゲットを作るクラス。

    BeatStemDataset 側では crop と valid_frames だけを渡し、
    drum / bass の派生系列はここでまとめて作る。
    """

    def __init__(
        self,
        stem_names: Sequence[str],
        channels_per_stem: int,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        target_num_frames: int,
    ) -> None:
        self.stem_names = tuple(stem_names)
        self.channels_per_stem = int(channels_per_stem)
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.target_num_frames = int(target_num_frames)

        if "drums" not in self.stem_names:
            raise ValueError("auxiliary targets require a 'drums' stem")
        if "bass" not in self.stem_names:
            raise ValueError("auxiliary targets require a 'bass' stem")

        self.drums_stem_index = self.stem_names.index("drums")
        self.bass_stem_index = self.stem_names.index("bass")

    def build(self, waveform: torch.Tensor, valid_frames: int) -> StemAuxTargets:
        broadband_flux_targets, onset_env_targets = self._compute_drum_aux_targets(
            waveform=waveform,
            valid_frames=valid_frames,
        )
        bass_low_flux_targets = self._compute_bass_aux_targets(
            waveform=waveform,
            valid_frames=valid_frames,
        )
        return StemAuxTargets(
            broadband_flux_targets=broadband_flux_targets,
            onset_env_targets=onset_env_targets,
            bass_low_flux_targets=bass_low_flux_targets,
        )

    def _extract_mono_stem(
        self,
        waveform: torch.Tensor,
        stem_index: int,
    ) -> torch.Tensor:
        start_channel = stem_index * self.channels_per_stem
        end_channel = start_channel + self.channels_per_stem
        return waveform[start_channel:end_channel].mean(dim=0, keepdim=True)

    def _compute_log_magnitude(self, mono_waveform: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(
            self.n_fft,
            dtype=mono_waveform.dtype,
            device=mono_waveform.device,
        )
        spec = torch.stft(
            mono_waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
        )
        return torch.log1p(spec.abs())

    def _pad_to_target_length(
        self,
        x: torch.Tensor,
        valid_frames: int,
    ) -> torch.Tensor:
        targets = torch.zeros(
            self.target_num_frames,
            dtype=torch.float32,
            device=x.device,
        )
        target_length = min(self.target_num_frames, x.numel())
        targets[:target_length] = x[:target_length]
        if valid_frames < self.target_num_frames:
            targets[valid_frames:] = 0.0
        return targets

    def _compute_drum_aux_targets(
        self,
        waveform: torch.Tensor,
        valid_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # drums stem から broadband spectral flux と onset envelope を作る。
        empty = torch.zeros(
            self.target_num_frames,
            dtype=torch.float32,
            device=waveform.device,
        )
        if valid_frames <= 0:
            return empty, empty.clone()

        drums_waveform = self._extract_mono_stem(
            waveform=waveform,
            stem_index=self.drums_stem_index,
        )
        drum_mag = self._compute_log_magnitude(drums_waveform)

        diff = F.relu(drum_mag[:, :, 1:] - drum_mag[:, :, :-1])
        broadband_flux = F.pad(diff.sum(dim=1), (1, 0)).squeeze(0)
        onset_envelope = _smooth_1d_series(broadband_flux, kernel_size=5)

        broadband_flux = _minmax_normalize_1d(broadband_flux)
        onset_envelope = _minmax_normalize_1d(onset_envelope)
        return (
            self._pad_to_target_length(broadband_flux, valid_frames),
            self._pad_to_target_length(onset_envelope, valid_frames),
        )

    def _compute_bass_aux_targets(
        self,
        waveform: torch.Tensor,
        valid_frames: int,
    ) -> torch.Tensor:
        # bass stem から low-band flux を作る。
        empty = torch.zeros(
            self.target_num_frames,
            dtype=torch.float32,
            device=waveform.device,
        )
        if valid_frames <= 0:
            return empty

        bass_waveform = self._extract_mono_stem(
            waveform=waveform,
            stem_index=self.bass_stem_index,
        )
        bass_mag = self._compute_log_magnitude(bass_waveform)
        freqs = torch.linspace(
            0.0,
            self.sample_rate / 2.0,
            bass_mag.shape[1],
            dtype=bass_mag.dtype,
            device=bass_mag.device,
        )

        low_band_mag = bass_mag[:, freqs <= 300.0, :]
        low_band_diff = F.relu(low_band_mag[:, :, 1:] - low_band_mag[:, :, :-1])
        low_band_flux = F.pad(low_band_diff.sum(dim=1), (1, 0)).squeeze(0)
        low_band_flux = _smooth_1d_series(low_band_flux, kernel_size=5)

        low_band_flux = _minmax_normalize_1d(low_band_flux)
        return self._pad_to_target_length(low_band_flux, valid_frames)
