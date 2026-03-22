from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass(frozen=True)
class ChordBoundaryTargets:
    target: torch.Tensor
    mask: torch.Tensor
    event_count: float


class ChordBoundaryTargetBuilder:
    """
    事前計算済み chord boundary cache を読み、crop ごとの frame target へ変換する。

    beat/pretraining の両 dataset から同じ実装を使えるように切り出しておく。
    """

    def __init__(
        self,
        cache_dir: Optional[str | Path],
        max_cached_entries: int = 256,
    ) -> None:
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.max_cached_entries = int(max_cached_entries)
        self._boundary_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        if self.max_cached_entries <= 0:
            raise ValueError("max_cached_entries must be positive")
        if self.cache_dir is not None and not self.cache_dir.exists():
            raise ValueError(f"cache_dir does not exist: {self.cache_dir}")

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_boundary_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._boundary_cache = OrderedDict()

    def clear_cache(self) -> None:
        self._boundary_cache.clear()

    def _get_boundary_path(self, song_id: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{song_id}.pt"

    def _load_boundary_times(self, song_id: str) -> Optional[torch.Tensor]:
        cache_key = str(song_id)
        cached = self._boundary_cache.pop(cache_key, None)
        if cached is not None:
            self._boundary_cache[cache_key] = cached
            return cached

        boundary_path = self._get_boundary_path(song_id)
        if boundary_path is None or not boundary_path.exists():
            return None

        payload = torch.load(boundary_path, map_location="cpu", weights_only=False)
        boundary_times = payload.get("boundary_times_sec")
        if not torch.is_tensor(boundary_times):
            return None

        boundary_times = boundary_times.to(torch.float32)
        self._boundary_cache[cache_key] = boundary_times
        while len(self._boundary_cache) > self.max_cached_entries:
            self._boundary_cache.popitem(last=False)
        return boundary_times

    def build(
        self,
        song_id: str,
        start_sec: float,
        segment_seconds: float,
        sample_rate: int,
        hop_length: int,
        target_num_frames: int,
        valid_mask: torch.Tensor,
    ) -> ChordBoundaryTargets:
        """
        秒単位の boundary 時刻列を、現在の crop に対応する binary frame target に変換する。
        cache が無い曲は mask を 0 にして、その sample を loss から外す。
        """

        target = torch.zeros(target_num_frames, dtype=torch.float32)
        boundary_times = self._load_boundary_times(song_id)
        if boundary_times is None:
            return ChordBoundaryTargets(
                target=target,
                mask=torch.zeros_like(valid_mask),
                event_count=0.0,
            )

        valid_frames = int(valid_mask.sum().item())
        if valid_frames <= 0:
            return ChordBoundaryTargets(
                target=target,
                mask=valid_mask.clone(),
                event_count=0.0,
            )

        relative_times = boundary_times - float(start_sec)
        visible = (relative_times >= 0.0) & (relative_times < float(segment_seconds))
        if not bool(visible.any()):
            return ChordBoundaryTargets(
                target=target,
                mask=valid_mask.clone(),
                event_count=0.0,
            )

        frame_indices = torch.round(
            relative_times[visible] * (float(sample_rate) / float(hop_length))
        ).to(torch.long)
        frame_indices = frame_indices[(frame_indices >= 0) & (frame_indices < valid_frames)]
        if frame_indices.numel() > 0:
            target.index_fill_(0, frame_indices.unique(sorted=True), 1.0)

        return ChordBoundaryTargets(
            target=target,
            mask=valid_mask.clone(),
            event_count=float(target.sum().item()),
        )
