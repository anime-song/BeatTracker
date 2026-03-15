from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RepeatPairTargets:
    pair_indices: torch.Tensor
    pair_mask: torch.Tensor


@dataclass(frozen=True)
class RepeatPairRun:
    start_index_a: int
    start_index_b: int
    length_units: int
    mean_similarity: float


@dataclass(frozen=True)
class RepeatPairDebugInfo:
    sync_unit: str
    valid_frames: int
    sync_frame_indices: torch.Tensor
    sync_start_frames: torch.Tensor
    sync_labels: tuple[str, ...]
    sync_features: torch.Tensor
    similarity_matrix: torch.Tensor
    runs: tuple[RepeatPairRun, ...]
    targets: RepeatPairTargets


@dataclass(frozen=True)
class _DiagonalRun:
    start_a: int
    start_b: int
    length: int
    mean_similarity: float


class RepeatPairBuilder:
    """
    繰り返し区間の SSM を使って、対応する beat 開始フレームのペアを作る。

    ここでは `piano + guitar + other` の mono 和音波形から簡易 chroma を作り、
    ground-truth の beat または bar で同期した特徴列の自己類似行列を使って
    対角線を検出する。
    学習側では、この beat ペア上で downbeat 出力の一貫性 loss を掛ける。
    """

    def __init__(
        self,
        stem_names: Sequence[str],
        channels_per_stem: int,
        sample_rate: int,
        hop_length: int,
        target_num_frames: int,
        n_fft: int,
        sync_unit: str = "beat",
        repeat_stem_names: Sequence[str] = ("piano", "guitar", "other"),
        similarity_threshold: float = 0.85,
        min_diagonal_length_beats: int = 8,
        near_diagonal_margin_beats: int = 16,
        max_diagonal_length_beats: int = 16,
        max_pairs: int = 128,
        chroma_fmin_hz: float = 27.5,
        chroma_fmax_hz: float = 2000.0,
    ) -> None:
        self.stem_names = tuple(stem_names)
        self.channels_per_stem = int(channels_per_stem)
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.target_num_frames = int(target_num_frames)
        self.n_fft = int(n_fft)
        if sync_unit not in {"beat", "bar"}:
            raise ValueError("sync_unit must be either 'beat' or 'bar'")
        self.sync_unit = str(sync_unit)
        self.similarity_threshold = float(similarity_threshold)
        self.min_diagonal_length_beats = int(min_diagonal_length_beats)
        self.near_diagonal_margin_beats = int(near_diagonal_margin_beats)
        self.max_diagonal_length_beats = int(max_diagonal_length_beats)
        self.max_pairs = int(max_pairs)
        self.chroma_fmin_hz = float(chroma_fmin_hz)
        self.chroma_fmax_hz = float(chroma_fmax_hz)

        selected_stem_indices = [
            self.stem_names.index(stem_name)
            for stem_name in repeat_stem_names
            if stem_name in self.stem_names
        ]
        if not selected_stem_indices:
            raise ValueError(
                "repeat pair builder requires at least one of piano/guitar/other"
            )
        self.repeat_stem_indices = tuple(selected_stem_indices)

    def empty(self, device: torch.device) -> RepeatPairTargets:
        return RepeatPairTargets(
            pair_indices=torch.zeros(
                (self.max_pairs, 2), dtype=torch.long, device=device
            ),
            pair_mask=torch.zeros(self.max_pairs, dtype=torch.float32, device=device),
        )

    def build(
        self,
        waveform: torch.Tensor,
        beat_times: torch.Tensor,
        downbeat_times: torch.Tensor,
        meter_segments: Sequence[tuple[float, float, str]] | None,
        start_sec: float,
        valid_frames: int,
    ) -> RepeatPairTargets:
        targets, _ = self.analyze(
            waveform=waveform,
            beat_times=beat_times,
            downbeat_times=downbeat_times,
            meter_segments=meter_segments,
            start_sec=start_sec,
            valid_frames=valid_frames,
        )
        return targets

    def analyze(
        self,
        waveform: torch.Tensor,
        beat_times: torch.Tensor,
        downbeat_times: torch.Tensor,
        meter_segments: Sequence[tuple[float, float, str]] | None,
        start_sec: float,
        valid_frames: int,
    ) -> tuple[RepeatPairTargets, RepeatPairDebugInfo]:
        empty = self.empty(waveform.device)
        empty_debug = self._empty_debug_info(device=waveform.device, valid_frames=valid_frames)
        event_times = beat_times if self.sync_unit == "beat" else downbeat_times
        if valid_frames <= 1 or event_times.numel() < self.min_diagonal_length_beats + 1:
            return empty, empty_debug

        repeat_waveform = self._extract_repeat_waveform(waveform)
        chroma = self._compute_chroma(repeat_waveform)
        available_frames = min(valid_frames, int(chroma.shape[-1]))
        if available_frames <= 1:
            return empty, empty_debug

        sync_frame_indices, local_event_times = self._quantize_local_events(
            event_times=event_times,
            start_sec=start_sec,
            valid_frames=available_frames,
        )
        if sync_frame_indices.numel() < self.min_diagonal_length_beats + 1:
            return empty, RepeatPairDebugInfo(
                sync_unit=self.sync_unit,
                valid_frames=available_frames,
                sync_frame_indices=sync_frame_indices,
                sync_start_frames=torch.empty(0, dtype=torch.long, device=waveform.device),
                sync_labels=tuple(),
                sync_features=torch.empty((0, 12), dtype=waveform.dtype, device=waveform.device),
                similarity_matrix=torch.empty((0, 0), dtype=waveform.dtype, device=waveform.device),
                runs=tuple(),
                targets=empty,
            )

        sync_features, sync_start_frames = self._compute_sync_chroma(
            chroma=chroma[:, :available_frames],
            sync_frame_indices=sync_frame_indices,
        )
        sync_labels = self._compute_sync_labels(
            event_times=local_event_times,
            meter_segments=meter_segments,
        )
        if sync_features.shape[0] < self.min_diagonal_length_beats:
            return empty, RepeatPairDebugInfo(
                sync_unit=self.sync_unit,
                valid_frames=available_frames,
                sync_frame_indices=sync_frame_indices,
                sync_start_frames=sync_start_frames,
                sync_labels=sync_labels,
                sync_features=sync_features,
                similarity_matrix=torch.empty((0, 0), dtype=waveform.dtype, device=waveform.device),
                runs=tuple(),
                targets=empty,
            )

        similarity = self._compute_self_similarity(
            sync_features=sync_features,
            sync_labels=sync_labels,
        )
        runs = self._find_diagonal_runs(similarity)
        selected_runs = self._select_non_overlapping_runs(runs)
        targets = self._build_pair_targets(
            runs=selected_runs,
            sync_start_frames=sync_start_frames,
            device=waveform.device,
        )
        debug_info = RepeatPairDebugInfo(
            sync_unit=self.sync_unit,
            valid_frames=available_frames,
            sync_frame_indices=sync_frame_indices,
            sync_start_frames=sync_start_frames,
            sync_labels=sync_labels,
            sync_features=sync_features,
            similarity_matrix=similarity,
            runs=tuple(
                RepeatPairRun(
                    start_index_a=run.start_a,
                    start_index_b=run.start_b,
                    length_units=run.length,
                    mean_similarity=run.mean_similarity,
                )
                for run in selected_runs
            ),
            targets=targets,
        )
        return targets, debug_info

    def _empty_debug_info(
        self,
        device: torch.device,
        valid_frames: int,
    ) -> RepeatPairDebugInfo:
        empty_targets = self.empty(device)
        return RepeatPairDebugInfo(
            sync_unit=self.sync_unit,
            valid_frames=max(0, int(valid_frames)),
            sync_frame_indices=torch.empty(0, dtype=torch.long, device=device),
            sync_start_frames=torch.empty(0, dtype=torch.long, device=device),
            sync_labels=tuple(),
            sync_features=torch.empty((0, 12), dtype=torch.float32, device=device),
            similarity_matrix=torch.empty((0, 0), dtype=torch.float32, device=device),
            runs=tuple(),
            targets=empty_targets,
        )

    def _extract_repeat_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        mono_stems: list[torch.Tensor] = []
        for stem_index in self.repeat_stem_indices:
            start_channel = stem_index * self.channels_per_stem
            end_channel = start_channel + self.channels_per_stem
            mono_stems.append(waveform[start_channel:end_channel].mean(dim=0))
        return torch.stack(mono_stems, dim=0).mean(dim=0, keepdim=True)

    def _compute_chroma(self, mono_waveform: torch.Tensor) -> torch.Tensor:
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
        magnitude = torch.log1p(spec.abs())

        freqs = torch.linspace(
            0.0,
            self.sample_rate / 2.0,
            magnitude.shape[1],
            dtype=magnitude.dtype,
            device=magnitude.device,
        )
        valid = (freqs >= self.chroma_fmin_hz) & (freqs <= self.chroma_fmax_hz)
        if not bool(valid.any()):
            return torch.zeros((12, magnitude.shape[-1]), device=magnitude.device)

        valid_freqs = freqs[valid]
        midi = 69.0 + 12.0 * torch.log2(valid_freqs / 440.0)
        pitch_class = torch.remainder(torch.round(midi), 12).long()
        mapping = F.one_hot(pitch_class, num_classes=12).to(magnitude.dtype)
        chroma = torch.einsum("bft,fc->bct", magnitude[:, valid, :], mapping).squeeze(0)
        chroma = chroma / (chroma.sum(dim=0, keepdim=True) + 1e-8)
        return chroma

    def _quantize_local_events(
        self,
        event_times: torch.Tensor,
        start_sec: float,
        valid_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        end_sec = start_sec + (valid_frames * self.hop_length / self.sample_rate)
        within = (event_times >= start_sec) & (event_times < end_sec)
        if not torch.any(within):
            empty_frames = torch.empty(0, dtype=torch.long, device=event_times.device)
            empty_times = torch.empty(0, dtype=event_times.dtype, device=event_times.device)
            return empty_frames, empty_times

        local_event_times = event_times[within]
        relative_times = local_event_times - start_sec
        frame_indices = torch.round(
            relative_times * self.sample_rate / self.hop_length
        ).long()
        valid = (frame_indices >= 0) & (frame_indices < valid_frames)
        frame_indices = frame_indices[valid]
        local_event_times = local_event_times[valid]

        unique_frames: list[int] = []
        unique_times: list[float] = []
        previous_frame: int | None = None
        for frame_index, event_time in zip(
            frame_indices.tolist(),
            local_event_times.tolist(),
        ):
            if previous_frame == frame_index:
                continue
            unique_frames.append(frame_index)
            unique_times.append(event_time)
            previous_frame = frame_index

        return (
            torch.tensor(unique_frames, dtype=torch.long, device=event_times.device),
            torch.tensor(unique_times, dtype=event_times.dtype, device=event_times.device),
        )

    def _compute_sync_chroma(
        self,
        chroma: torch.Tensor,
        sync_frame_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sync_features: list[torch.Tensor] = []
        sync_start_frames: list[int] = []
        for start_frame, end_frame in zip(
            sync_frame_indices[:-1].tolist(),
            sync_frame_indices[1:].tolist(),
        ):
            if end_frame <= start_frame:
                continue
            sync_features.append(chroma[:, start_frame:end_frame].mean(dim=1))
            sync_start_frames.append(start_frame)

        if not sync_features:
            return (
                torch.empty((0, 12), dtype=chroma.dtype, device=chroma.device),
                torch.empty(0, dtype=torch.long, device=chroma.device),
            )

        return torch.stack(sync_features, dim=0), torch.tensor(
            sync_start_frames, dtype=torch.long, device=chroma.device
        )

    def _compute_sync_labels(
        self,
        event_times: torch.Tensor,
        meter_segments: Sequence[tuple[float, float, str]] | None,
    ) -> tuple[str, ...]:
        sync_count = max(0, int(event_times.numel()) - 1)
        if sync_count == 0:
            return tuple()
        if self.sync_unit != "bar" or not meter_segments:
            return tuple("" for _ in range(sync_count))

        labels: list[str] = []
        tolerance_sec = 1e-4
        for start_time in event_times[:-1].tolist():
            label = ""
            for segment_start, segment_end, segment_label in meter_segments:
                if (
                    (segment_start - tolerance_sec)
                    <= start_time
                    < (segment_end + tolerance_sec)
                ):
                    label = segment_label
                    break
            labels.append(label)
        return tuple(labels)

    def _compute_self_similarity(
        self,
        sync_features: torch.Tensor,
        sync_labels: Sequence[str],
    ) -> torch.Tensor:
        normalized = F.normalize(sync_features, p=2, dim=1, eps=1e-8)
        similarity = normalized @ normalized.transpose(0, 1)
        if self.sync_unit != "bar" or not sync_labels:
            return similarity

        label_mask = torch.tensor(
            [
                [left_label == right_label for right_label in sync_labels]
                for left_label in sync_labels
            ],
            dtype=torch.bool,
            device=similarity.device,
        )
        similarity = similarity.masked_fill(~label_mask, -1.0)
        return similarity

    def _find_diagonal_runs(self, similarity: torch.Tensor) -> list[_DiagonalRun]:
        num_beats = similarity.shape[0]
        runs: list[_DiagonalRun] = []
        for offset in range(self.near_diagonal_margin_beats, num_beats):
            diagonal = torch.diagonal(similarity, offset=offset)
            if diagonal.numel() < self.min_diagonal_length_beats:
                continue

            active = diagonal >= self.similarity_threshold
            run_start: int | None = None
            for index, is_active in enumerate(active.tolist()):
                if is_active and run_start is None:
                    run_start = index
                elif (not is_active) and run_start is not None:
                    self._maybe_append_run(
                        runs=runs,
                        diagonal=diagonal,
                        offset=offset,
                        run_start=run_start,
                        run_end=index,
                    )
                    run_start = None

            if run_start is not None:
                self._maybe_append_run(
                    runs=runs,
                    diagonal=diagonal,
                    offset=offset,
                    run_start=run_start,
                    run_end=int(diagonal.numel()),
                )
        return runs

    def _maybe_append_run(
        self,
        runs: list[_DiagonalRun],
        diagonal: torch.Tensor,
        offset: int,
        run_start: int,
        run_end: int,
    ) -> None:
        run_length = run_end - run_start
        if run_length < self.min_diagonal_length_beats:
            return

        mean_similarity = float(diagonal[run_start:run_end].mean().item())
        runs.append(
            _DiagonalRun(
                start_a=run_start,
                start_b=run_start + offset,
                length=run_length,
                mean_similarity=mean_similarity,
            )
        )

    def _select_non_overlapping_runs(
        self,
        runs: list[_DiagonalRun],
    ) -> list[_DiagonalRun]:
        if not runs:
            return []

        selected_runs: list[_DiagonalRun] = []
        sorted_runs = sorted(
            runs,
            key=lambda run: (run.mean_similarity, run.length),
            reverse=True,
        )
        for run in sorted_runs:
            trimmed_run = _DiagonalRun(
                start_a=run.start_a,
                start_b=run.start_b,
                length=min(run.length, self.max_diagonal_length_beats),
                mean_similarity=run.mean_similarity,
            )
            if trimmed_run.length < self.min_diagonal_length_beats:
                continue
            if any(
                self._runs_overlap_too_much(trimmed_run, existing_run)
                for existing_run in selected_runs
            ):
                continue
            selected_runs.append(trimmed_run)

        return selected_runs

    def _runs_overlap_too_much(
        self,
        candidate: _DiagonalRun,
        existing: _DiagonalRun,
    ) -> bool:
        overlap_a = self._interval_overlap(
            candidate.start_a,
            candidate.length,
            existing.start_a,
            existing.length,
        )
        overlap_b = self._interval_overlap(
            candidate.start_b,
            candidate.length,
            existing.start_b,
            existing.length,
        )
        min_length = max(1, min(candidate.length, existing.length))
        overlap_ratio_a = overlap_a / min_length
        overlap_ratio_b = overlap_b / min_length
        return overlap_ratio_a >= 0.5 and overlap_ratio_b >= 0.5

    @staticmethod
    def _interval_overlap(
        start_a: int,
        length_a: int,
        start_b: int,
        length_b: int,
    ) -> int:
        end_a = start_a + length_a
        end_b = start_b + length_b
        return max(0, min(end_a, end_b) - max(start_a, start_b))

    def _build_pair_targets(
        self,
        runs: list[_DiagonalRun],
        sync_start_frames: torch.Tensor,
        device: torch.device,
    ) -> RepeatPairTargets:
        pair_indices = torch.zeros((self.max_pairs, 2), dtype=torch.long, device=device)
        pair_mask = torch.zeros(self.max_pairs, dtype=torch.float32, device=device)

        pair_count = 0
        seen_pairs: set[tuple[int, int]] = set()
        sorted_runs = sorted(
            runs,
            key=lambda run: (run.mean_similarity, run.length),
            reverse=True,
        )
        for run in sorted_runs:
            for step in range(run.length):
                left = int(sync_start_frames[run.start_a + step].item())
                right = int(sync_start_frames[run.start_b + step].item())
                pair = (left, right)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                pair_indices[pair_count, 0] = left
                pair_indices[pair_count, 1] = right
                pair_mask[pair_count] = 1.0
                pair_count += 1
                if pair_count >= self.max_pairs:
                    return RepeatPairTargets(pair_indices=pair_indices, pair_mask=pair_mask)

        return RepeatPairTargets(pair_indices=pair_indices, pair_mask=pair_mask)
