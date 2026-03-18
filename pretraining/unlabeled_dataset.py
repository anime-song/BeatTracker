from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset

from data.beat_dataset import DEFAULT_STEM_NAMES


@dataclass(frozen=True)
class UnlabeledSongEntry:
    song_id: str
    sample_rate: int
    channels_per_stem: int
    duration_sec: float
    stem_paths: dict[str, Path]


class UnlabeledStemDataset(Dataset):
    """
    ビート注釈を持たない stem 音源からランダム crop を返す Dataset。

    Expected layout:
    dataset_root/<song_id>/<song_id>_<stem>.wav
    """

    def __init__(
        self,
        dataset_root: str | Path = "dataset/unlabeled_dataset",
        segment_seconds: float = 30.0,
        sample_rate: Optional[int] = None,
        hop_length: int = 441,
        n_fft: int = 2048,
        stem_names: Sequence[str] = DEFAULT_STEM_NAMES,
        samples_per_epoch: Optional[int] = None,
        use_file_handle_cache: bool = True,
        max_open_files: int = 64,
        manifest_path: Optional[str | Path] = None,
        rebuild_manifest: bool = False,
    ) -> None:
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.segment_seconds = float(segment_seconds)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.stem_names = tuple(stem_names)
        self.samples_per_epoch = samples_per_epoch
        self.use_file_handle_cache = bool(use_file_handle_cache)
        self.max_open_files = int(max_open_files)
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else (self.dataset_root / ".unlabeled_stem_manifest.json")
        )
        self.rebuild_manifest = bool(rebuild_manifest)
        self._audio_file_cache: OrderedDict[str, sf.SoundFile] = OrderedDict()

        if self.segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if not self.stem_names:
            raise ValueError("stem_names must not be empty")
        if self.max_open_files <= 0:
            raise ValueError("max_open_files must be positive")
        if not self.dataset_root.exists():
            raise ValueError(f"dataset_root does not exist: {self.dataset_root}")

        # 初回だけ stem 群を走査して manifest を作り、以後は JSON から復元する。
        songs = None if self.rebuild_manifest else self._load_manifest()
        if songs is None:
            songs = self._scan_song_entries()
            self._write_manifest(songs)

        if not songs:
            raise ValueError("No songs matched the expected unlabeled stem layout")

        source_sample_rates = {song.sample_rate for song in songs}
        if sample_rate is None:
            if len(source_sample_rates) != 1:
                raise ValueError("sample_rate must be set explicitly when source sample rates are mixed")
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
        self.target_num_frames = 1 + ((self.segment_samples - self.n_fft) // self.hop_length)
        self.songs = songs

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

            stem_infos = [sf.info(str(stem_paths[stem_name])) for stem_name in self.stem_names]
            reference_sample_rate = stem_infos[0].samplerate
            channels_per_stem = stem_infos[0].channels
            if any(info.samplerate != reference_sample_rate for info in stem_infos):
                raise ValueError(f"Mismatched sample rates in {song_dir}")
            if any(info.channels != channels_per_stem for info in stem_infos):
                raise ValueError(f"Mismatched channel counts in {song_dir}")

            duration_sec = min(info.frames for info in stem_infos) / float(reference_sample_rate)
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
                        stem_name: str(song.stem_paths[stem_name].relative_to(self.dataset_root))
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

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_audio_file_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._audio_file_cache = OrderedDict()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        while self._audio_file_cache:
            _, audio_file = self._audio_file_cache.popitem(last=False)
            audio_file.close()

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

    def make_sample(
        self,
        song: UnlabeledSongEntry,
        start_sec: float,
    ) -> dict[str, torch.Tensor | str | float | int]:
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

        return {
            "waveform": waveform,
            "valid_mask": valid_mask,
            "valid_frames": valid_frames,
            "song_id": song.song_id,
            "start_sec": start_sec,
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float | int]:
        song = self.songs[index % len(self.songs)]
        max_start_sec = max(song.duration_sec - self.segment_seconds, 0.0)
        start_sec = 0.0 if max_start_sec <= 0 else float(torch.rand(1).item() * max_start_sec)
        return self.make_sample(song=song, start_sec=start_sec)
