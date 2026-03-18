from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PLPTeacherOutput:
    """1 batch 分の PLP 擬似教師。"""

    pulse: torch.Tensor
    peak_mask: torch.Tensor
    peak_values: torch.Tensor


class PLPPseudoTeacher(nn.Module):
    """
    PLP を近似して擬似教師を作るモジュール。

    流れはかなり単純で、
    1. 多 ch stem 波形をモノラル化
    2. spectral flux から onset envelope を作成
    3. Fourier tempogram の卓越テンポ帯だけを残して pulse を復元
    4. pulse の局所ピークを擬似的な拍位置候補として返す
    という順で処理する。
    """

    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        n_fft: int,
        win_length: int = 384,
        tempo_min: float = 30.0,
        tempo_max: float = 300.0,
        peak_threshold: float = 0.3,
        min_peak_distance: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.tempo_min = float(tempo_min)
        self.tempo_max = float(tempo_max)
        self.peak_threshold = float(peak_threshold)
        self.min_peak_distance = None if min_peak_distance is None else int(min_peak_distance)

        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if self.win_length <= 0:
            raise ValueError("win_length must be positive")
        if self.tempo_max <= self.tempo_min:
            raise ValueError("tempo_max must be larger than tempo_min")

        self.register_buffer("analysis_window", torch.hann_window(self.n_fft), persistent=False)
        self.register_buffer("tempogram_window", torch.hann_window(self.win_length), persistent=False)

    @staticmethod
    def _normalize_to_unit_max(x: torch.Tensor) -> torch.Tensor:
        """無音区間で NaN を出さないように最大値だけで正規化する。"""

        scale = x.amax().clamp_min(1e-6)
        return x / scale

    def _detect_peaks(self, pulse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        復元した pulse から局所ピークを選ぶ。
        拍候補が過密になりすぎないよう、最小距離を入れている。
        """

        peak_mask = pulse.new_zeros(pulse.shape[0])
        peak_values = pulse.new_zeros(pulse.shape[0])
        if pulse.numel() == 0:
            return peak_mask, peak_values
        if pulse.numel() == 1:
            if float(pulse[0]) > 0.0:
                peak_mask[0] = 1.0
                peak_values[0] = pulse[0]
            return peak_mask, peak_values

        if self.min_peak_distance is None:
            frame_rate = float(self.sample_rate) / float(self.hop_length)
            shortest_beat_frames = max(1.0, frame_rate * 60.0 / self.tempo_max)
            min_distance = max(1, int(round(shortest_beat_frames * 0.5)))
        else:
            min_distance = max(1, self.min_peak_distance)

        threshold = float(pulse.amax()) * self.peak_threshold
        pooled = F.max_pool1d(
            pulse.view(1, 1, -1),
            kernel_size=(2 * min_distance) + 1,
            stride=1,
            padding=min_distance,
        ).view(-1)
        candidate_indices = ((pulse >= threshold) & (pulse == pooled)).nonzero(as_tuple=False).squeeze(-1)

        # しきい値が厳しすぎた場合でも、完全に候補ゼロにはしない。
        if candidate_indices.numel() == 0 and float(pulse.amax()) > 0.0:
            candidate_indices = pulse.argmax().view(1)

        if candidate_indices.numel() > 1:
            ordered_indices = torch.argsort(
                pulse.index_select(0, candidate_indices),
                descending=True,
            )
            selected: list[int] = []
            for ordered_index in ordered_indices.tolist():
                candidate_frame = int(candidate_indices[ordered_index].item())
                if any(abs(candidate_frame - existing) < min_distance for existing in selected):
                    continue
                selected.append(candidate_frame)
            candidate_indices = torch.tensor(
                sorted(selected),
                device=pulse.device,
                dtype=torch.long,
            )

        if candidate_indices.numel() > 0:
            peak_mask.index_fill_(0, candidate_indices, 1.0)
            peak_values[candidate_indices] = pulse[candidate_indices]

        return peak_mask, peak_values

    def _compute_single_example(
        self,
        waveform: torch.Tensor,
        frame_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        1 サンプルぶんの PLP 擬似教師をまとめて作る。
        細かい helper を増やさず、signal -> envelope -> pulse -> peaks を
        1 本の流れとして追えるようにしている。
        """

        pulse = waveform.new_zeros(frame_count, dtype=torch.float32)
        if frame_count <= 0:
            return pulse, pulse.clone(), pulse.clone()

        sample_count = min(
            waveform.shape[-1],
            self.n_fft + max(frame_count - 1, 0) * self.hop_length,
        )
        mono_waveform = waveform[:, :sample_count].mean(dim=0).to(torch.float32)
        if mono_waveform.numel() < self.n_fft:
            mono_waveform = F.pad(mono_waveform, (0, self.n_fft - mono_waveform.numel()))

        # まずは spectral flux ベースの onset envelope を作る。
        spec = torch.stft(
            mono_waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.analysis_window.to(device=waveform.device, dtype=torch.float32),
            center=False,
            return_complex=True,
        )
        magnitude = spec.abs()
        if magnitude.shape[-1] == 0:
            return pulse, pulse.clone(), pulse.clone()
        if magnitude.shape[-1] == 1:
            onset_envelope = magnitude.new_zeros(1, dtype=torch.float32)
        else:
            flux = (magnitude[:, 1:] - magnitude[:, :-1]).clamp_min(0.0).mean(dim=0)
            onset_envelope = torch.cat([flux.new_zeros(1), flux], dim=0)

        if onset_envelope.numel() >= 3:
            onset_envelope = F.avg_pool1d(
                onset_envelope.view(1, 1, -1),
                kernel_size=3,
                stride=1,
                padding=1,
            ).view(-1)
        if onset_envelope.shape[0] < frame_count:
            onset_envelope = F.pad(onset_envelope, (0, frame_count - onset_envelope.shape[0]))
        else:
            onset_envelope = onset_envelope[:frame_count]
        onset_envelope = self._normalize_to_unit_max(onset_envelope)

        # 次に tempo 候補のうち卓越する成分だけを残し、PLP pulse を復元する。
        max_supported_win_length = max(2, (2 * onset_envelope.numel()) - 1)
        effective_win_length = min(self.win_length, max_supported_win_length)
        tempogram_window = (
            self.tempogram_window.to(device=waveform.device, dtype=torch.float32)
            if effective_win_length == self.win_length
            else torch.hann_window(effective_win_length, device=waveform.device, dtype=torch.float32)
        )
        ftgram = torch.stft(
            onset_envelope,
            n_fft=effective_win_length,
            hop_length=1,
            win_length=effective_win_length,
            window=tempogram_window,
            center=True,
            return_complex=True,
        )
        tempo_frequencies = (
            torch.fft.rfftfreq(
                effective_win_length,
                d=float(self.hop_length) / float(self.sample_rate),
                device=waveform.device,
            )
            * 60.0
        )
        valid_tempo = (tempo_frequencies >= self.tempo_min) & (tempo_frequencies <= self.tempo_max)
        if not bool(valid_tempo.any()):
            return pulse, pulse.clone(), pulse.clone()

        ftgram = ftgram * valid_tempo.unsqueeze(-1)
        ftmag = torch.log1p(1e6 * ftgram.abs())
        peak_values = ftmag.max(dim=0, keepdim=True).values
        ftgram = torch.where(ftmag >= peak_values, ftgram, torch.zeros_like(ftgram))
        ftgram = ftgram / ftgram.abs().amax(dim=0, keepdim=True).clamp_min(1e-6)

        pulse = torch.istft(
            ftgram,
            n_fft=effective_win_length,
            hop_length=1,
            win_length=effective_win_length,
            window=tempogram_window,
            center=True,
            length=onset_envelope.shape[0],
        ).clamp_min(0.0)
        pulse = self._normalize_to_unit_max(pulse)

        peak_mask, peak_values = self._detect_peaks(pulse)
        return pulse, peak_mask, peak_values

    @torch.no_grad()
    def forward(
        self,
        waveform: torch.Tensor,
        valid_frames: torch.Tensor,
    ) -> PLPTeacherOutput:
        if waveform.ndim != 3:
            raise ValueError("waveform must have shape [batch, channels, time]")
        if valid_frames.ndim != 1:
            raise ValueError("valid_frames must have shape [batch]")

        batch_size, _, segment_samples = waveform.shape
        target_num_frames = 1 + ((segment_samples - self.n_fft) // self.hop_length)
        pulse = waveform.new_zeros((batch_size, target_num_frames), dtype=torch.float32)
        peak_mask = waveform.new_zeros((batch_size, target_num_frames), dtype=torch.float32)
        peak_values = waveform.new_zeros((batch_size, target_num_frames), dtype=torch.float32)

        for batch_index in range(batch_size):
            frame_count = int(valid_frames[batch_index].item())
            if frame_count <= 0:
                continue

            pulse_slice, peak_mask_slice, peak_values_slice = self._compute_single_example(
                waveform=waveform[batch_index],
                frame_count=frame_count,
            )
            pulse[batch_index, :frame_count] = pulse_slice
            peak_mask[batch_index, :frame_count] = peak_mask_slice
            peak_values[batch_index, :frame_count] = peak_values_slice

        return PLPTeacherOutput(
            pulse=pulse,
            peak_mask=peak_mask,
            peak_values=peak_values,
        )
