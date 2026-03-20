from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from models.cqt import RecursiveCQT
from pretraining.unlabeled_dataset import UnlabeledSongEntry

DEFAULT_HARMONIC_STEM_NAMES = ("bass", "guitar", "other", "piano")


@dataclass(frozen=True)
class SegmentTimeTable:
    segment_start_sec: torch.Tensor
    segment_end_sec: torch.Tensor
    inner_start_sec: torch.Tensor
    inner_end_sec: torch.Tensor

    @property
    def num_segments(self) -> int:
        return int(self.segment_start_sec.numel())


def resolve_harmonic_stem_names(
    stem_names: Sequence[str],
    harmonic_stem_names: Optional[Sequence[str]] = None,
) -> tuple[str, ...]:
    requested = (
        tuple(harmonic_stem_names)
        if harmonic_stem_names is not None
        else DEFAULT_HARMONIC_STEM_NAMES
    )
    missing = [stem_name for stem_name in requested if stem_name not in stem_names]
    if missing:
        raise ValueError(f"Harmonic stems are missing from the dataset: {missing}")
    return requested


def load_boundary_times(boundary_path: Path) -> Optional[torch.Tensor]:
    if not boundary_path.exists():
        return None

    payload = torch.load(boundary_path, map_location="cpu", weights_only=False)
    boundary_times_sec = payload.get("boundary_times_sec")
    if not torch.is_tensor(boundary_times_sec):
        return None
    return boundary_times_sec.to(torch.float32)


def build_segment_time_table(
    boundary_times_sec: torch.Tensor,
    duration_sec: float,
    boundary_guard_seconds: float,
    min_inner_seconds: float,
) -> SegmentTimeTable:
    """
    疑似 chord boundary を segment 列へ変換し、さらに各 segment の中心部を切り出す。

    inner 区間は boundary の誤差を避けるための安全側の領域で、
    学習 target と入力 mask の両方で使う。
    """

    duration_sec = float(duration_sec)
    if duration_sec <= 0.0:
        empty = torch.zeros(0, dtype=torch.float32)
        return SegmentTimeTable(empty, empty, empty, empty)

    clipped_boundaries = boundary_times_sec.to(torch.float32)
    clipped_boundaries = clipped_boundaries[
        (clipped_boundaries > 0.0) & (clipped_boundaries < duration_sec)
    ]
    if clipped_boundaries.numel() > 0:
        clipped_boundaries = torch.unique(clipped_boundaries, sorted=True)

    endpoints = torch.cat(
        [
            torch.zeros(1, dtype=torch.float32),
            clipped_boundaries,
            torch.tensor([duration_sec], dtype=torch.float32),
        ]
    )
    segment_start_sec = endpoints[:-1].contiguous()
    segment_end_sec = endpoints[1:].contiguous()

    guard_seconds = max(0.0, float(boundary_guard_seconds))
    min_inner_seconds = max(0.0, float(min_inner_seconds))
    # 端から `guard_seconds` だけ削りたいが、短い segment は潰したくないので
    # 最低でも `min_inner_seconds` は残すよう trim 量を制限する。
    durations = (segment_end_sec - segment_start_sec).clamp_min(0.0)
    max_trim = ((durations - min_inner_seconds).clamp_min(0.0)) * 0.5
    trim = torch.minimum(
        torch.full_like(max_trim, guard_seconds),
        max_trim,
    )
    inner_start_sec = segment_start_sec + trim
    inner_end_sec = segment_end_sec - trim

    return SegmentTimeTable(
        segment_start_sec=segment_start_sec,
        segment_end_sec=segment_end_sec,
        inner_start_sec=inner_start_sec,
        inner_end_sec=inner_end_sec,
    )


def time_range_to_frame_bounds(
    start_sec: float,
    end_sec: float,
    frame_rate: float,
    num_frames: int,
) -> tuple[int, int]:
    if num_frames <= 0:
        return 0, 0

    start_frame = int(math.floor(float(start_sec) * frame_rate))
    end_frame = int(math.ceil(float(end_sec) * frame_rate))
    start_frame = min(max(start_frame, 0), num_frames - 1)
    end_frame = min(max(end_frame, start_frame + 1), num_frames)
    return start_frame, end_frame


def _load_packed_harmonic_mix(
    song: UnlabeledSongEntry,
    stem_names: Sequence[str],
    harmonic_stem_names: Sequence[str],
) -> tuple[torch.Tensor, int]:
    packed_audio = song.packed_audio
    if packed_audio is None:
        raise ValueError(f"Packed audio metadata is missing for {song.song_id}")

    packed_array = np.load(packed_audio.array_path, mmap_mode="r")
    mix = np.zeros(packed_audio.num_frames, dtype=np.float32)

    for stem_name in harmonic_stem_names:
        stem_index = stem_names.index(stem_name)
        channel_start = stem_index * packed_audio.channels_per_stem
        channel_end = channel_start + packed_audio.channels_per_stem
        stem_audio = np.asarray(
            packed_array[channel_start:channel_end].mean(axis=0, dtype=np.float32),
            dtype=np.float32,
        )
        mix += stem_audio

    mix /= max(1, len(harmonic_stem_names))
    mmap_handle = getattr(packed_array, "_mmap", None)
    if mmap_handle is not None:
        mmap_handle.close()
    return torch.from_numpy(mix), int(packed_audio.sample_rate)


def _load_wav_harmonic_mix(
    song: UnlabeledSongEntry,
    harmonic_stem_names: Sequence[str],
) -> tuple[torch.Tensor, int]:
    mix: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None

    for stem_name in harmonic_stem_names:
        stem_audio, stem_sample_rate = sf.read(
            str(song.stem_paths[stem_name]),
            dtype="float32",
            always_2d=True,
        )
        mono = stem_audio.mean(axis=1, dtype=np.float32)
        if mix is None:
            mix = np.zeros_like(mono, dtype=np.float32)
            sample_rate = int(stem_sample_rate)
        if int(stem_sample_rate) != sample_rate:
            raise ValueError(
                f"Mismatched sample rates in {song.song_id}: {sample_rate} vs {stem_sample_rate}"
            )
        if mono.shape[0] < mix.shape[0]:
            mix = mix[: mono.shape[0]]
        elif mono.shape[0] > mix.shape[0]:
            mono = mono[: mix.shape[0]]
        mix += mono

    if mix is None or sample_rate is None:
        raise ValueError(f"No harmonic stems were loaded for {song.song_id}")

    mix /= max(1, len(harmonic_stem_names))
    return torch.from_numpy(mix), sample_rate


def load_harmonic_mono_waveform(
    song: UnlabeledSongEntry,
    stem_names: Sequence[str],
    harmonic_stem_names: Sequence[str],
    target_sample_rate: int,
) -> torch.Tensor:
    """和声を見る stem だけを mono mix して、必要なら sample rate を揃える。"""

    if song.packed_audio is not None:
        waveform, source_sample_rate = _load_packed_harmonic_mix(
            song=song,
            stem_names=stem_names,
            harmonic_stem_names=harmonic_stem_names,
        )
    else:
        waveform, source_sample_rate = _load_wav_harmonic_mix(
            song=song,
            harmonic_stem_names=harmonic_stem_names,
        )

    waveform = waveform.to(torch.float32)
    if int(source_sample_rate) != int(target_sample_rate):
        waveform = AF.resample(
            waveform.unsqueeze(0),
            orig_freq=int(source_sample_rate),
            new_freq=int(target_sample_rate),
        ).squeeze(0)
    return waveform.contiguous()


def fold_cqt_to_chroma(
    cqt_magnitude: torch.Tensor,
    bins_per_octave: int,
) -> torch.Tensor:
    """CQT を 12 次元 chroma へ畳み込む。"""

    if cqt_magnitude.ndim != 3:
        raise ValueError("cqt_magnitude must have shape [batch, bins, frames]")
    if bins_per_octave <= 0 or bins_per_octave % 12 != 0:
        raise ValueError("bins_per_octave must be a positive multiple of 12")

    batch_size, num_bins, num_frames = cqt_magnitude.shape
    usable_bins = (num_bins // bins_per_octave) * bins_per_octave
    if usable_bins <= 0:
        raise ValueError("n_bins must include at least one octave")

    bins_per_pitch_class = bins_per_octave // 12
    num_octaves = usable_bins // bins_per_octave
    reshaped = cqt_magnitude[:, :usable_bins].reshape(
        batch_size,
        num_octaves,
        12,
        bins_per_pitch_class,
        num_frames,
    )
    chroma = reshaped.mean(dim=(1, 3))
    chroma = chroma.transpose(1, 2).contiguous()
    chroma = chroma / chroma.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return chroma


def extract_song_chroma(
    waveform: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    n_bins: int,
    bins_per_octave: int,
    device: torch.device,
    cqt_module: Optional[RecursiveCQT] = None,
) -> torch.Tensor:
    if waveform.ndim != 1:
        raise ValueError("waveform must be mono [time]")

    local_cqt = cqt_module
    if local_cqt is None:
        local_cqt = RecursiveCQT(
            sr=sample_rate,
            hop_length=hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            filter_scale=0.4375,
        )
    local_cqt = local_cqt.to(device)

    with torch.inference_mode():
        cqt_magnitude = local_cqt(
            waveform.unsqueeze(0).to(device), return_complex=False
        )
        cqt_magnitude = torch.log1p(cqt_magnitude.clamp_min(0.0))
        chroma = fold_cqt_to_chroma(
            cqt_magnitude=cqt_magnitude,
            bins_per_octave=bins_per_octave,
        )
    return chroma.squeeze(0).cpu()


def summarize_segment_chroma(
    chroma: torch.Tensor,
    segments: SegmentTimeTable,
    sample_rate: int,
    hop_length: int,
) -> torch.Tensor:
    if chroma.ndim != 2 or chroma.shape[-1] != 12:
        raise ValueError("chroma must have shape [frames, 12]")

    num_frames = int(chroma.shape[0])
    frame_rate = float(sample_rate) / float(hop_length)
    features: list[torch.Tensor] = []

    for segment_index in range(segments.num_segments):
        inner_start_sec = float(segments.inner_start_sec[segment_index].item())
        inner_end_sec = float(segments.inner_end_sec[segment_index].item())
        segment_start_sec = float(segments.segment_start_sec[segment_index].item())
        segment_end_sec = float(segments.segment_end_sec[segment_index].item())

        start_frame, end_frame = time_range_to_frame_bounds(
            start_sec=inner_start_sec,
            end_sec=inner_end_sec,
            frame_rate=frame_rate,
            num_frames=num_frames,
        )
        segment_slice = chroma[start_frame:end_frame]
        if segment_slice.numel() == 0:
            start_frame, end_frame = time_range_to_frame_bounds(
                start_sec=segment_start_sec,
                end_sec=segment_end_sec,
                frame_rate=frame_rate,
                num_frames=num_frames,
            )
            segment_slice = chroma[start_frame:end_frame]
        if segment_slice.numel() == 0:
            segment_slice = chroma.new_zeros(1, chroma.shape[-1])

        summary = segment_slice.mean(dim=0)
        summary = summary / summary.sum().clamp_min(1e-8)
        features.append(summary)

    if not features:
        return chroma.new_zeros((0, chroma.shape[-1]))
    return torch.stack(features, dim=0)
