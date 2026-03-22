from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset

from data.chord_boundary_targets import ChordBoundaryTargetBuilder
from data.beat_dataset import DEFAULT_STEM_NAMES, PackedAudioEntry


@dataclass(frozen=True)
class UnlabeledSongEntry:
    song_id: str
    sample_rate: int
    channels_per_stem: int
    duration_sec: float
    stem_paths: dict[str, Path]
    packed_audio: Optional[PackedAudioEntry] = None


class UnlabeledStemDataset(Dataset):
    """
    ビート注釈を持たない stem 音源からランダム crop を返す Dataset。

    audio_backend="wav" のとき:
    dataset_root/<song_id>/<song_id>_<stem>.wav

    audio_backend="packed" のとき:
    packed_audio_dir/<song_id>/<song_id>_stems_pitch_0st.npy
    packed_audio_dir/<song_id>/<song_id>_stems_pitch_0st.json
    """

    def __init__(
        self,
        dataset_root: str | Path = "dataset/unlabeled_dataset",
        segment_seconds: float = 30.0,
        sample_rate: Optional[int] = None,
        hop_length: int = 512,
        n_fft: int = 2048,
        stem_names: Sequence[str] = DEFAULT_STEM_NAMES,
        samples_per_epoch: Optional[int] = None,
        audio_backend: str = "wav",
        packed_audio_dir: Optional[str | Path] = None,
        use_file_handle_cache: bool = True,
        max_open_files: int = 64,
        manifest_path: Optional[str | Path] = None,
        rebuild_manifest: bool = False,
        chord_boundary_cache_dir: Optional[str | Path] = None,
        max_cached_boundary_entries: int = 256,
        prototype_cache_dir: Optional[str | Path] = None,
        min_visible_segments: int = 0,
        sample_retry_count: int = 1,
        max_cached_prototype_entries: int = 256,
    ) -> None:
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.segment_seconds = float(segment_seconds)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.stem_names = tuple(stem_names)
        self.samples_per_epoch = samples_per_epoch
        self.audio_backend = str(audio_backend)
        self.packed_audio_dir = (
            Path(packed_audio_dir)
            if packed_audio_dir is not None
            else (self.dataset_root / "songs_packed")
        )
        self.use_file_handle_cache = bool(use_file_handle_cache)
        self.max_open_files = int(max_open_files)
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else (self.dataset_root / ".unlabeled_stem_manifest.json")
        )
        self.rebuild_manifest = bool(rebuild_manifest)
        self.chord_boundary_cache_dir = (
            None if chord_boundary_cache_dir is None else Path(chord_boundary_cache_dir)
        )
        self.max_cached_boundary_entries = int(max_cached_boundary_entries)
        self.prototype_cache_dir = (
            None if prototype_cache_dir is None else Path(prototype_cache_dir)
        )
        self.min_visible_segments = max(0, int(min_visible_segments))
        self.sample_retry_count = max(1, int(sample_retry_count))
        self.max_cached_prototype_entries = int(max_cached_prototype_entries)
        self._audio_file_cache: OrderedDict[str, sf.SoundFile] = OrderedDict()
        self._packed_array_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._prototype_target_cache: OrderedDict[
            str, dict[str, torch.Tensor | str | float]
        ] = OrderedDict()
        self.prototype_song_target_dir: Optional[Path] = None
        self.num_prototypes = 0

        if self.segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if not self.stem_names:
            raise ValueError("stem_names must not be empty")
        if self.audio_backend not in {"wav", "packed"}:
            raise ValueError("audio_backend must be either 'wav' or 'packed'")
        if self.max_open_files <= 0:
            raise ValueError("max_open_files must be positive")
        if self.max_cached_prototype_entries <= 0:
            raise ValueError("max_cached_prototype_entries must be positive")
        if not self.dataset_root.exists():
            raise ValueError(f"dataset_root does not exist: {self.dataset_root}")
        if self.audio_backend == "packed" and not self.packed_audio_dir.exists():
            raise ValueError(
                f"packed_audio_dir does not exist: {self.packed_audio_dir}"
            )

        # 初回だけ stem 群を走査して manifest を作り、以後は JSON から復元する。
        songs = None if self.rebuild_manifest else self._load_manifest()
        if songs is None:
            songs = self._scan_song_entries()
            self._write_manifest(songs)
        if self.audio_backend == "packed":
            songs = self._attach_packed_entries(songs)

        if not songs:
            raise ValueError("No songs matched the expected unlabeled stem layout")

        source_sample_rates = {song.sample_rate for song in songs}
        if sample_rate is None:
            if len(source_sample_rates) != 1:
                raise ValueError(
                    "sample_rate must be set explicitly when source sample rates are mixed"
                )
            sample_rate = next(iter(source_sample_rates))
        self.sample_rate = int(sample_rate)

        channels_per_stem_set = {song.channels_per_stem for song in songs}
        if len(channels_per_stem_set) != 1:
            raise ValueError("All songs must have the same channel count per stem")
        self.channels_per_stem = next(iter(channels_per_stem_set))
        self.num_audio_channels = len(self.stem_names) * self.channels_per_stem

        self.segment_samples = int(round(self.segment_seconds * self.sample_rate))
        if self.segment_samples < self.n_fft:
            raise ValueError("segment_seconds is too short for the configured n_fft")
        self.target_num_frames = 1 + (
            (self.segment_samples - self.n_fft) // self.hop_length
        )
        self.songs = songs
        self._configure_prototype_cache()
        self.chord_boundary_target_builder = ChordBoundaryTargetBuilder(
            cache_dir=self.chord_boundary_cache_dir,
            max_cached_entries=self.max_cached_boundary_entries,
        )

    def _load_manifest(self) -> Optional[list[UnlabeledSongEntry]]:
        if not self.manifest_path.exists():
            return None

        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None
        if int(payload.get("manifest_version", -1)) != 1:
            return None
        if tuple(payload.get("stem_names", [])) != self.stem_names:
            return None

        songs: list[UnlabeledSongEntry] = []
        for raw_song in payload.get("songs", []):
            if not isinstance(raw_song, dict):
                return None

            raw_stem_paths = raw_song.get("stem_paths")
            if not isinstance(raw_stem_paths, dict):
                return None

            try:
                stem_paths = {
                    stem_name: self.dataset_root / str(raw_stem_paths[stem_name])
                    for stem_name in self.stem_names
                }
                songs.append(
                    UnlabeledSongEntry(
                        song_id=str(raw_song["song_id"]),
                        sample_rate=int(raw_song["sample_rate"]),
                        channels_per_stem=int(raw_song["channels_per_stem"]),
                        duration_sec=float(raw_song["duration_sec"]),
                        stem_paths=stem_paths,
                    )
                )
            except (KeyError, TypeError, ValueError):
                return None

        return songs

    def _scan_song_entries(self) -> list[UnlabeledSongEntry]:
        songs: list[UnlabeledSongEntry] = []
        # 既存 supervised dataset と同じ stem 命名規約だけを受け付ける。
        for song_dir in sorted(self.dataset_root.iterdir()):
            if not song_dir.is_dir():
                continue

            song_id = song_dir.name
            stem_paths = {
                stem_name: song_dir / f"{song_id}_{stem_name}.wav"
                for stem_name in self.stem_names
            }
            if not all(path.exists() for path in stem_paths.values()):
                continue

            stem_infos = [
                sf.info(str(stem_paths[stem_name])) for stem_name in self.stem_names
            ]
            reference_sample_rate = stem_infos[0].samplerate
            channels_per_stem = stem_infos[0].channels
            if any(info.samplerate != reference_sample_rate for info in stem_infos):
                raise ValueError(f"Mismatched sample rates in {song_dir}")
            if any(info.channels != channels_per_stem for info in stem_infos):
                raise ValueError(f"Mismatched channel counts in {song_dir}")

            duration_sec = min(info.frames for info in stem_infos) / float(
                reference_sample_rate
            )
            if duration_sec <= 0:
                continue

            songs.append(
                UnlabeledSongEntry(
                    song_id=song_id,
                    sample_rate=reference_sample_rate,
                    channels_per_stem=channels_per_stem,
                    duration_sec=duration_sec,
                    stem_paths=stem_paths,
                )
            )
        return songs

    def _load_packed_entry(self, song_id: str) -> Optional[PackedAudioEntry]:
        song_dir = self.packed_audio_dir / song_id
        metadata_path = song_dir / f"{song_id}_stems_pitch_0st.json"
        if not metadata_path.exists():
            return None

        array_path = metadata_path.with_suffix(".npy")
        if not array_path.exists():
            return None

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_stem_names = tuple(metadata.get("stem_names", []))
        if metadata_stem_names != self.stem_names:
            return None
        if int(metadata.get("semitone", 9999)) != 0:
            return None

        return PackedAudioEntry(
            array_path=array_path,
            metadata_path=metadata_path,
            sample_rate=int(metadata["sample_rate"]),
            channels_per_stem=int(metadata["channels_per_stem"]),
            num_channels=int(metadata["num_channels"]),
            num_frames=int(metadata["num_frames"]),
            storage_dtype=str(metadata.get("storage_dtype", "float32")),
        )

    def _attach_packed_entries(
        self,
        songs: Sequence[UnlabeledSongEntry],
    ) -> list[UnlabeledSongEntry]:
        resolved_songs: list[UnlabeledSongEntry] = []
        for song in songs:
            packed_audio = self._load_packed_entry(song.song_id)
            if packed_audio is None:
                continue

            expected_num_channels = (
                len(self.stem_names) * packed_audio.channels_per_stem
            )
            if packed_audio.num_channels != expected_num_channels:
                raise ValueError(
                    f"Unexpected num_channels in packed audio for {song.song_id}: {packed_audio.num_channels}"
                )

            resolved_songs.append(
                UnlabeledSongEntry(
                    song_id=song.song_id,
                    sample_rate=packed_audio.sample_rate,
                    channels_per_stem=packed_audio.channels_per_stem,
                    duration_sec=packed_audio.num_frames
                    / float(packed_audio.sample_rate),
                    stem_paths=song.stem_paths,
                    packed_audio=packed_audio,
                )
            )
        return resolved_songs

    def _write_manifest(self, songs: Sequence[UnlabeledSongEntry]) -> None:
        payload = {
            "manifest_version": 1,
            "dataset_root": str(self.dataset_root),
            "stem_names": list(self.stem_names),
            "songs": [
                {
                    "song_id": song.song_id,
                    "sample_rate": song.sample_rate,
                    "channels_per_stem": song.channels_per_stem,
                    "duration_sec": song.duration_sec,
                    "stem_paths": {
                        stem_name: str(
                            song.stem_paths[stem_name].relative_to(self.dataset_root)
                        )
                        for stem_name in self.stem_names
                    },
                }
                for song in songs
            ],
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def __len__(self) -> int:
        if self.samples_per_epoch is not None:
            return self.samples_per_epoch
        return len(self.songs)

    def _configure_prototype_cache(self) -> None:
        if self.prototype_cache_dir is None:
            self.prototype_song_target_dir = None
            self.num_prototypes = 0
            return

        metadata_path = self.prototype_cache_dir / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(
                f"Prototype cache metadata does not exist: {metadata_path}"
            )

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(payload.get("sample_rate", -1)) != self.sample_rate:
            raise ValueError(
                "Prototype cache sample_rate does not match the training dataset"
            )
        if int(payload.get("hop_length", -1)) != self.hop_length:
            raise ValueError(
                "Prototype cache hop_length does not match the training dataset"
            )

        song_target_dir = self.prototype_cache_dir / str(
            payload.get("song_target_dir", "song_targets")
        )
        if not song_target_dir.exists():
            raise ValueError(
                f"Prototype song target directory does not exist: {song_target_dir}"
            )

        self.prototype_song_target_dir = song_target_dir
        self.num_prototypes = int(payload.get("num_prototypes", 0))

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_audio_file_cache"] = OrderedDict()
        state["_packed_array_cache"] = OrderedDict()
        state["_prototype_target_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._audio_file_cache = OrderedDict()
        self._packed_array_cache = OrderedDict()
        self._prototype_target_cache = OrderedDict()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        while self._audio_file_cache:
            _, audio_file = self._audio_file_cache.popitem(last=False)
            audio_file.close()
        while self._packed_array_cache:
            _, packed_array = self._packed_array_cache.popitem(last=False)
            mmap_handle = getattr(packed_array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
        self.chord_boundary_target_builder.clear_cache()
        self._prototype_target_cache.clear()

    def _get_prototype_target_path(self, song_id: str) -> Optional[Path]:
        if self.prototype_song_target_dir is None:
            return None
        return self.prototype_song_target_dir / f"{song_id}.pt"

    def _load_prototype_targets(
        self, song_id: str
    ) -> Optional[dict[str, torch.Tensor | str | float]]:
        cache_key = str(song_id)
        cached = self._prototype_target_cache.pop(cache_key, None)
        if cached is not None:
            self._prototype_target_cache[cache_key] = cached
            return cached

        target_path = self._get_prototype_target_path(song_id)
        if target_path is None or not target_path.exists():
            return None

        payload = torch.load(target_path, map_location="cpu", weights_only=False)
        self._prototype_target_cache[cache_key] = payload
        while len(self._prototype_target_cache) > self.max_cached_prototype_entries:
            self._prototype_target_cache.popitem(last=False)
        return payload

    def _estimate_valid_frames(
        self,
        song: UnlabeledSongEntry,
        start_sec: float,
    ) -> int:
        remaining_sec = max(float(song.duration_sec) - float(start_sec), 0.0)
        valid_samples = min(
            self.segment_samples,
            int(round(remaining_sec * float(self.sample_rate))),
        )
        if valid_samples < self.n_fft:
            return 0
        return min(
            self.target_num_frames,
            1 + ((valid_samples - self.n_fft) // self.hop_length),
        )

    def _render_segment_prototype_targets(
        self,
        song: UnlabeledSongEntry,
        start_sec: float,
        valid_frames: int,
    ) -> dict[str, torch.Tensor | int]:
        """絶対時刻で保存された segment cache を、crop 内の相対 frame へ写す。"""

        empty_segments = {
            "segment_start_frames": torch.zeros(0, dtype=torch.long),
            "segment_end_frames": torch.zeros(0, dtype=torch.long),
            "segment_inner_start_frames": torch.zeros(0, dtype=torch.long),
            "segment_inner_end_frames": torch.zeros(0, dtype=torch.long),
            "segment_target_probs": torch.zeros(
                (0, self.num_prototypes),
                dtype=torch.float32,
            ),
            "segment_count": 0,
        }
        payload = self._load_prototype_targets(song.song_id)
        if payload is None or self.num_prototypes <= 0:
            return empty_segments

        frame_rate = float(self.sample_rate) / float(self.hop_length)
        segment_start_sec = payload["segment_start_sec"].to(torch.float32)
        segment_end_sec = payload["segment_end_sec"].to(torch.float32)
        inner_start_sec = payload["inner_start_sec"].to(torch.float32)
        inner_end_sec = payload["inner_end_sec"].to(torch.float32)
        prototype_distribution = payload["prototype_distribution"].to(torch.float32)

        relative_segment_start = segment_start_sec - float(start_sec)
        relative_segment_end = segment_end_sec - float(start_sec)
        relative_inner_start = inner_start_sec - float(start_sec)
        relative_inner_end = inner_end_sec - float(start_sec)

        # cache は秒単位なので、学習時は frame 単位へ丸め直して使う。
        segment_start_frames = torch.round(relative_segment_start * frame_rate).to(
            torch.long
        )
        segment_end_frames = torch.round(relative_segment_end * frame_rate).to(
            torch.long
        )
        inner_start_frames = torch.round(relative_inner_start * frame_rate).to(
            torch.long
        )
        inner_end_frames = torch.round(relative_inner_end * frame_rate).to(torch.long)

        # crop からはみ出る segment は捨て、完全に見えているものだけ supervision に使う。
        visible = (
            (segment_start_frames >= 0)
            & (segment_end_frames <= int(valid_frames))
            & (segment_end_frames > segment_start_frames)
            & (inner_end_frames > inner_start_frames)
        )
        if not visible.any():
            return empty_segments

        segment_start_frames = segment_start_frames[visible].clamp(
            min=0,
            max=max(0, int(valid_frames)),
        )
        segment_end_frames = segment_end_frames[visible].clamp(
            min=0,
            max=max(0, int(valid_frames)),
        )
        inner_start_frames = inner_start_frames[visible].clamp(
            min=0,
            max=max(0, int(valid_frames)),
        )
        inner_end_frames = inner_end_frames[visible].clamp(
            min=0,
            max=max(0, int(valid_frames)),
        )
        prototype_distribution = prototype_distribution[visible]

        return {
            "segment_start_frames": segment_start_frames,
            "segment_end_frames": segment_end_frames,
            "segment_inner_start_frames": inner_start_frames,
            "segment_inner_end_frames": inner_end_frames,
            "segment_target_probs": prototype_distribution,
            "segment_count": int(prototype_distribution.shape[0]),
        }

    def _choose_start_sec(self, song: UnlabeledSongEntry) -> float:
        max_start_sec = max(float(song.duration_sec) - self.segment_seconds, 0.0)
        if self.num_prototypes <= 0 or self.min_visible_segments <= 0:
            if max_start_sec <= 0.0:
                return 0.0
            return float(torch.rand(1).item() * max_start_sec)

        fallback_start_sec = 0.0
        best_start_sec = 0.0
        best_segment_count = -1
        for _ in range(self.sample_retry_count):
            if max_start_sec <= 0.0:
                start_sec = 0.0
            else:
                start_sec = float(torch.rand(1).item() * max_start_sec)
            valid_frames = self._estimate_valid_frames(song=song, start_sec=start_sec)
            segment_payload = self._render_segment_prototype_targets(
                song=song,
                start_sec=start_sec,
                valid_frames=valid_frames,
            )
            segment_count = int(segment_payload["segment_count"])
            if segment_count >= self.min_visible_segments:
                return start_sec
            if segment_count > best_segment_count:
                best_segment_count = segment_count
                best_start_sec = start_sec
            fallback_start_sec = start_sec
        return best_start_sec if best_segment_count >= 0 else fallback_start_sec

    def _get_cached_audio_file(self, wav_path: Path) -> sf.SoundFile:
        cache_key = str(wav_path)
        audio_file = self._audio_file_cache.pop(cache_key, None)
        if audio_file is None:
            audio_file = sf.SoundFile(cache_key, mode="r")
        self._audio_file_cache[cache_key] = audio_file

        while len(self._audio_file_cache) > self.max_open_files:
            _, oldest_audio_file = self._audio_file_cache.popitem(last=False)
            oldest_audio_file.close()
        return audio_file

    def _load_audio_crop(
        self,
        song: UnlabeledSongEntry,
        wav_path: Path,
        start_sec: float,
    ) -> tuple[torch.Tensor, int]:
        # 各 stem は必要区間だけ読み、足りない末尾はゼロ埋めする。
        source_sample_rate = song.sample_rate
        frame_offset = int(round(start_sec * source_sample_rate))
        num_frames = int(math.ceil(self.segment_seconds * source_sample_rate))

        if self.use_file_handle_cache:
            audio_file = self._get_cached_audio_file(wav_path)
            audio_file.seek(frame_offset)
            waveform = torch.from_numpy(
                audio_file.read(num_frames, dtype="float32", always_2d=True).T
            )
            loaded_sample_rate = source_sample_rate
        else:
            waveform, loaded_sample_rate = torchaudio.load(
                str(wav_path),
                frame_offset=frame_offset,
                num_frames=num_frames,
            )
            waveform = waveform.to(torch.float32)

        if waveform.shape[0] == 1 and self.channels_per_stem == 2:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] != self.channels_per_stem:
            raise ValueError(
                f"Expected {self.channels_per_stem} channels in {wav_path}, found {waveform.shape[0]}"
            )

        if loaded_sample_rate != self.sample_rate:
            waveform = AF.resample(
                waveform, orig_freq=loaded_sample_rate, new_freq=self.sample_rate
            )

        valid_samples = min(waveform.shape[-1], self.segment_samples)
        if waveform.shape[-1] < self.segment_samples:
            waveform = F.pad(waveform, (0, self.segment_samples - waveform.shape[-1]))
        else:
            waveform = waveform[..., : self.segment_samples]

        return waveform.contiguous(), valid_samples

    def _get_cached_packed_array(self, array_path: Path) -> np.ndarray:
        cache_key = str(array_path)
        packed_array = self._packed_array_cache.pop(cache_key, None)
        if packed_array is None:
            packed_array = np.load(array_path, mmap_mode="r")
        self._packed_array_cache[cache_key] = packed_array

        while len(self._packed_array_cache) > self.max_open_files:
            _, oldest_array = self._packed_array_cache.popitem(last=False)
            mmap_handle = getattr(oldest_array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
        return packed_array

    def _load_packed_audio_crop(
        self,
        song: UnlabeledSongEntry,
        start_sec: float,
    ) -> tuple[torch.Tensor, int]:
        packed_audio = song.packed_audio
        if packed_audio is None:
            raise ValueError(f"Packed audio metadata is missing for {song.song_id}")

        source_sample_rate = packed_audio.sample_rate
        frame_offset = int(round(start_sec * source_sample_rate))
        num_frames = int(math.ceil(self.segment_seconds * source_sample_rate))
        frame_end = min(frame_offset + num_frames, packed_audio.num_frames)

        if self.use_file_handle_cache:
            packed_array = self._get_cached_packed_array(packed_audio.array_path)
            crop = (
                np.zeros((packed_audio.num_channels, 0), dtype=np.float32)
                if frame_offset >= packed_audio.num_frames
                else np.array(packed_array[:, frame_offset:frame_end], copy=True)
            )
        else:
            packed_array = np.load(packed_audio.array_path, mmap_mode="r")
            crop = (
                np.zeros((packed_audio.num_channels, 0), dtype=np.float32)
                if frame_offset >= packed_audio.num_frames
                else np.array(packed_array[:, frame_offset:frame_end], copy=True)
            )
            mmap_handle = getattr(packed_array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()

        waveform = torch.from_numpy(crop).to(torch.float32)
        if waveform.shape[0] != self.num_audio_channels:
            raise ValueError(
                f"Expected {self.num_audio_channels} channels in {packed_audio.array_path}, found {waveform.shape[0]}"
            )

        if source_sample_rate != self.sample_rate:
            waveform = AF.resample(
                waveform, orig_freq=source_sample_rate, new_freq=self.sample_rate
            )

        valid_samples = min(waveform.shape[-1], self.segment_samples)
        if waveform.shape[-1] < self.segment_samples:
            waveform = F.pad(waveform, (0, self.segment_samples - waveform.shape[-1]))
        else:
            waveform = waveform[..., : self.segment_samples]

        return waveform.contiguous(), valid_samples

    def make_sample(
        self,
        song: UnlabeledSongEntry,
        start_sec: float,
    ) -> dict[str, torch.Tensor | str | float | int]:
        if self.audio_backend == "packed":
            waveform, valid_samples = self._load_packed_audio_crop(
                song=song,
                start_sec=start_sec,
            )
        else:
            # supervised 側と同じく、stem を channel 次元へ連結して返す。
            stem_waveforms: list[torch.Tensor] = []
            valid_samples = self.segment_samples
            for stem_name in self.stem_names:
                stem_waveform, stem_valid_samples = self._load_audio_crop(
                    song=song,
                    wav_path=song.stem_paths[stem_name],
                    start_sec=start_sec,
                )
                stem_waveforms.append(stem_waveform)
                valid_samples = min(valid_samples, stem_valid_samples)

            waveform = torch.cat(stem_waveforms, dim=0)

        valid_frames = 0
        if valid_samples >= self.n_fft:
            valid_frames = min(
                self.target_num_frames,
                1 + ((valid_samples - self.n_fft) // self.hop_length),
            )

        # loss から無効末尾フレームを外せるように valid_mask も返す。
        valid_mask = torch.zeros(self.target_num_frames, dtype=torch.float32)
        valid_mask[:valid_frames] = 1.0
        chord_boundary_targets = self.chord_boundary_target_builder.build(
            song_id=song.song_id,
            start_sec=start_sec,
            segment_seconds=self.segment_seconds,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            target_num_frames=self.target_num_frames,
            valid_mask=valid_mask,
        )
        segment_payload = self._render_segment_prototype_targets(
            song=song,
            start_sec=start_sec,
            valid_frames=valid_frames,
        )
        return {
            "waveform": waveform,
            "valid_mask": valid_mask,
            "valid_frames": valid_frames,
            "chord_boundary_target": chord_boundary_targets.target,
            "chord_boundary_mask": chord_boundary_targets.mask,
            "chord_boundary_event_count": chord_boundary_targets.event_count,
            "segment_start_frames": segment_payload["segment_start_frames"],
            "segment_end_frames": segment_payload["segment_end_frames"],
            "segment_inner_start_frames": segment_payload["segment_inner_start_frames"],
            "segment_inner_end_frames": segment_payload["segment_inner_end_frames"],
            "segment_target_probs": segment_payload["segment_target_probs"],
            "segment_count": int(segment_payload["segment_count"]),
            "num_prototypes": self.num_prototypes,
            "song_id": song.song_id,
            "start_sec": start_sec,
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float | int]:
        song = self.songs[index % len(self.songs)]
        start_sec = self._choose_start_sec(song)
        return self.make_sample(song=song, start_sec=start_sec)


def collate_unlabeled_stem_batch(
    batch: list[dict[str, torch.Tensor | str | float | int]],
) -> dict[str, torch.Tensor | list[str]]:
    batch_size = len(batch)
    num_prototypes = int(batch[0]["num_prototypes"])
    max_segments = max(int(item["segment_count"]) for item in batch)

    waveform = torch.stack([item["waveform"] for item in batch], dim=0)
    valid_mask = torch.stack([item["valid_mask"] for item in batch], dim=0)
    valid_frames = torch.tensor(
        [int(item["valid_frames"]) for item in batch],
        dtype=torch.long,
    )
    chord_boundary_target = torch.stack(
        [item["chord_boundary_target"] for item in batch],
        dim=0,
    )
    chord_boundary_mask = torch.stack(
        [item["chord_boundary_mask"] for item in batch],
        dim=0,
    )
    chord_boundary_event_count = torch.tensor(
        [float(item["chord_boundary_event_count"]) for item in batch],
        dtype=torch.float32,
    )
    start_sec = torch.tensor(
        [float(item["start_sec"]) for item in batch],
        dtype=torch.float32,
    )
    song_id = [str(item["song_id"]) for item in batch]

    segment_start_frames = torch.zeros((batch_size, max_segments), dtype=torch.long)
    segment_end_frames = torch.zeros((batch_size, max_segments), dtype=torch.long)
    segment_inner_start_frames = torch.zeros(
        (batch_size, max_segments), dtype=torch.long
    )
    segment_inner_end_frames = torch.zeros((batch_size, max_segments), dtype=torch.long)
    segment_target_probs = torch.zeros(
        (batch_size, max_segments, num_prototypes),
        dtype=torch.float32,
    )
    segment_valid_mask = torch.zeros((batch_size, max_segments), dtype=torch.bool)

    for batch_index, item in enumerate(batch):
        segment_count = int(item["segment_count"])
        if segment_count <= 0:
            continue
        segment_valid_mask[batch_index, :segment_count] = True
        segment_start_frames[batch_index, :segment_count] = item["segment_start_frames"]
        segment_end_frames[batch_index, :segment_count] = item["segment_end_frames"]
        segment_inner_start_frames[batch_index, :segment_count] = item[
            "segment_inner_start_frames"
        ]
        segment_inner_end_frames[batch_index, :segment_count] = item[
            "segment_inner_end_frames"
        ]
        segment_target_probs[batch_index, :segment_count] = item["segment_target_probs"]

    return {
        "waveform": waveform,
        "valid_mask": valid_mask,
        "valid_frames": valid_frames,
        "chord_boundary_target": chord_boundary_target,
        "chord_boundary_mask": chord_boundary_mask,
        "chord_boundary_event_count": chord_boundary_event_count,
        "song_id": song_id,
        "start_sec": start_sec,
        "segment_start_frames": segment_start_frames,
        "segment_end_frames": segment_end_frames,
        "segment_inner_start_frames": segment_inner_start_frames,
        "segment_inner_end_frames": segment_inner_end_frames,
        "segment_target_probs": segment_target_probs,
        "segment_valid_mask": segment_valid_mask,
        "num_prototypes": torch.full(
            (batch_size,),
            fill_value=num_prototypes,
            dtype=torch.long,
        ),
    }
