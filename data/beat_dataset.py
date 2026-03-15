from __future__ import annotations

import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset

from .aux_targets import StemAuxTargetBuilder
from .repeat_pairs import RepeatPairBuilder


DEFAULT_STEM_NAMES = ("bass", "drums", "guitar", "other", "piano", "vocals")
PITCH_SUFFIX_PATTERN = re.compile(r"_pitch_(-?\d+)st$")
PACKED_PITCH_SUFFIX_PATTERN = re.compile(r"_pitch_(-?\d+)st$")


@dataclass(frozen=True)
class MeasureAnnotation:
    downbeat_sec: float
    time_sig_num: int
    time_sig_den: int
    tempo_bpm: float
    base_note: str


@dataclass(frozen=True)
class MeterAnnotation:
    start_sec: float
    end_sec: float
    meter_label: str


@dataclass
class SongEntry:
    song_id: str
    split: Optional[str]
    sample_rate: int
    channels_per_stem: int
    audio_duration_sec: float
    label_duration_sec: float
    effective_duration_sec: float
    stems_by_semitone: Dict[int, Dict[str, Path]]
    available_semitones: tuple[int, ...]
    beat_times: torch.Tensor
    downbeat_times: torch.Tensor
    meter_annotations: tuple[MeterAnnotation, ...]
    packed_variants: Dict[int, "PackedAudioEntry"] = field(default_factory=dict)


@dataclass(frozen=True)
class PackedAudioEntry:
    array_path: Path
    metadata_path: Path
    sample_rate: int
    channels_per_stem: int
    num_channels: int
    num_frames: int
    storage_dtype: str


def _read_split_map(split_path: Path) -> Dict[str, str]:
    split_map: Dict[str, str] = {}
    if not split_path.exists():
        return split_map

    # single.split は "<song_id>\t<split>" の単純な TSV。
    for line in split_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        song_id, split_name = line.split("\t", maxsplit=1)
        split_map[song_id] = split_name
    return split_map


def _parse_annotation_file(annotation_path: Path) -> list[MeasureAnnotation]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    measures: list[MeasureAnnotation] = []
    for raw in data.get("measures", []):
        measures.append(
            MeasureAnnotation(
                downbeat_sec=float(raw["downbeat_sec"]),
                time_sig_num=int(raw["time_sig_num"]),
                time_sig_den=int(raw["time_sig_den"]),
                tempo_bpm=float(raw.get("tempo_bpm", 0.0)),
                base_note=str(raw.get("base_note", "")),
            )
        )

    # 後段では downbeat の時系列順を前提に beat を補完する。
    measures.sort(key=lambda measure: measure.downbeat_sec)
    return measures


def _meter_label_sort_key(meter_label: str) -> tuple[int, int]:
    numerator, denominator = meter_label.split("/", maxsplit=1)
    return int(numerator), int(denominator)


def derive_beat_downbeat_and_meter_annotations(
    annotation_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, float, tuple[MeterAnnotation, ...]]:
    measures = _parse_annotation_file(annotation_path)
    if not measures:
        empty = torch.empty(0, dtype=torch.float32)
        return empty, empty, 0.0, tuple()

    beat_times: list[float] = []
    downbeat_times: list[float] = []
    meter_annotations: list[MeterAnnotation] = []
    previous_quarter_sec: Optional[float] = None
    song_end_sec = 0.0

    for index, measure in enumerate(measures):
        quarter_notes = measure.time_sig_num * (4.0 / measure.time_sig_den)
        if quarter_notes <= 0:
            raise ValueError(f"Invalid time signature for measure {measure}")

        next_measure = measures[index + 1] if index + 1 < len(measures) else None
        if next_measure is not None:
            # 基本は「次の小節頭 - 現小節頭」をその小節長とみなす。
            duration = next_measure.downbeat_sec - measure.downbeat_sec
            if duration <= 0:
                raise ValueError("Annotation downbeats must be strictly increasing")
        elif previous_quarter_sec is not None:
            # 最終小節だけは次の downbeat が無いので、
            # 直前までに観測できた四分音符長で補う。
            duration = previous_quarter_sec * quarter_notes
        elif measure.tempo_bpm > 0:
            # 先頭かつ単独小節の曲は tempo から長さを推定する。
            duration = (60.0 / measure.tempo_bpm) * quarter_notes
        else:
            raise ValueError(
                "Cannot infer the last measure duration without tempo or context"
            )

        previous_quarter_sec = duration / quarter_notes
        song_end_sec = measure.downbeat_sec + duration

        # meter は 4/4, 7/8 のような拍子文字列をそのまま 1 クラスとして持つ。
        meter_annotations.append(
            MeterAnnotation(
                start_sec=measure.downbeat_sec,
                end_sec=measure.downbeat_sec + duration,
                meter_label=f"{measure.time_sig_num}/{measure.time_sig_den}",
            )
        )

        # beat annotation が無いので、拍数は拍子の分子そのものとみなす。
        # 例: 7/8 -> 7 beats, 6/4 -> 6 beats
        beat_count = int(measure.time_sig_num)
        if beat_count <= 0:
            continue

        step_sec = duration / beat_count
        downbeat_times.append(measure.downbeat_sec)
        # downbeat も beat に含めたいので、小節頭から等間隔で beat を並べる。
        for beat_index in range(beat_count):
            beat_times.append(measure.downbeat_sec + (beat_index * step_sec))

    beat_tensor = torch.tensor(beat_times, dtype=torch.float32)
    downbeat_tensor = torch.tensor(downbeat_times, dtype=torch.float32)
    return (
        beat_tensor,
        downbeat_tensor,
        float(song_end_sec),
        tuple(meter_annotations),
    )


def derive_beat_and_downbeat_times(annotation_path: Path) -> tuple[torch.Tensor, torch.Tensor, float]:
    beat_tensor, downbeat_tensor, song_end_sec, _ = (
        derive_beat_downbeat_and_meter_annotations(annotation_path)
    )
    return beat_tensor, downbeat_tensor, song_end_sec


def _discover_stem_variants(song_dir: Path, song_id: str, stem_names: Sequence[str]) -> Dict[int, Dict[str, Path]]:
    variants: Dict[int, Dict[str, Path]] = {}
    for stem_name in stem_names:
        prefix = f"{song_id}_{stem_name}"
        for wav_path in song_dir.glob(f"{song_id}_{stem_name}*.wav"):
            stem_id = wav_path.stem
            if not stem_id.startswith(prefix):
                continue

            suffix = stem_id[len(prefix) :]
            if suffix == "":
                # suffix が無いものをオリジナル音源 (0 semitone) とみなす。
                semitone = 0
            else:
                match = PITCH_SUFFIX_PATTERN.fullmatch(suffix)
                if match is None:
                    continue
                semitone = int(match.group(1))

            variants.setdefault(semitone, {})[stem_name] = wav_path

    # 1 曲ぶんの学習サンプルは全 stem が揃っていることを前提にする。
    complete_variants: Dict[int, Dict[str, Path]] = {}
    for semitone, stem_map in variants.items():
        if all(stem_name in stem_map for stem_name in stem_names):
            complete_variants[semitone] = {stem_name: stem_map[stem_name] for stem_name in stem_names}
    return complete_variants


def _discover_packed_variants(
    song_dir: Path,
    song_id: str,
    stem_names: Sequence[str],
) -> Dict[int, PackedAudioEntry]:
    variants: Dict[int, PackedAudioEntry] = {}
    prefix = f"{song_id}_stems"

    for metadata_path in song_dir.glob(f"{song_id}_stems_pitch_*.json"):
        metadata_stem = metadata_path.stem
        if not metadata_stem.startswith(prefix):
            continue

        suffix = metadata_stem[len(prefix) :]
        match = PACKED_PITCH_SUFFIX_PATTERN.fullmatch(suffix)
        if match is None:
            continue

        array_path = metadata_path.with_suffix(".npy")
        if not array_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_stem_names = tuple(metadata.get("stem_names", []))
        # packed 側は channel 順がそのまま学習入力になるので、
        # stem 順が一致するファイルだけを採用する。
        if metadata_stem_names != tuple(stem_names):
            continue

        variants[int(match.group(1))] = PackedAudioEntry(
            array_path=array_path,
            metadata_path=metadata_path,
            sample_rate=int(metadata["sample_rate"]),
            channels_per_stem=int(metadata["channels_per_stem"]),
            num_channels=int(metadata["num_channels"]),
            num_frames=int(metadata["num_frames"]),
            storage_dtype=str(metadata.get("storage_dtype", "float32")),
        )
    return variants


class BeatStemDataset(Dataset):
    """
    Dataset for beat and downbeat transcription from separated stems.

    audio_backend="wav" のとき:
    - songs_separated/<song_id>/<song_id>_<stem>.wav
    - songs_separated/<song_id>/<song_id>_<stem>_pitch_<N>st.wav

    audio_backend="packed" のとき:
    - songs_packed/<song_id>/<song_id>_stems_pitch_<N>st.npy
    - songs_packed/<song_id>/<song_id>_stems_pitch_<N>st.json
    """

    def __init__(
        self,
        dataset_root: str | Path = "dataset/meter_dataset",
        split: Optional[str] = "train",
        segment_seconds: float = 8.0,
        sample_rate: Optional[int] = None,
        hop_length: int = 512,
        n_fft: int = 2048,
        stem_names: Sequence[str] = DEFAULT_STEM_NAMES,
        samples_per_epoch: Optional[int] = None,
        include_original: bool = True,
        random_pitch_shift: bool = True,
        allowed_pitch_shifts: Optional[Iterable[int]] = None,
        audio_backend: str = "wav",
        packed_audio_dir: Optional[str | Path] = None,
        meter_to_index: Optional[Dict[str, int]] = None,
        enable_repeat_pair_targets: bool = False,
        repeat_ssm_threshold: float = 0.85,
        repeat_ssm_min_length_beats: int = 8,
        repeat_ssm_near_diagonal_margin_beats: int = 16,
        repeat_ssm_max_length_beats: int = 16,
        repeat_ssm_max_pairs: int = 128,
        use_file_handle_cache: bool = True,
        max_open_files: int = 64,
    ) -> None:
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.songs_dir = self.dataset_root / "songs_separated"
        self.packed_audio_dir = (
            Path(packed_audio_dir) if packed_audio_dir is not None else (self.dataset_root / "songs_packed")
        )
        self.annotations_dir = self.dataset_root / "annotations" / "beats"
        self.split_path = self.dataset_root / "single.split"

        self.split = split
        self.segment_seconds = float(segment_seconds)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.stem_names = tuple(stem_names)
        self.samples_per_epoch = samples_per_epoch
        self.include_original = include_original
        self.random_pitch_shift = random_pitch_shift
        self.audio_backend = str(audio_backend)
        self.meter_ignore_index = -100
        self.enable_repeat_pair_targets = bool(enable_repeat_pair_targets)
        self.repeat_ssm_max_pairs = int(repeat_ssm_max_pairs)
        self.use_file_handle_cache = bool(use_file_handle_cache)
        self.max_open_files = int(max_open_files)
        self._audio_file_cache: OrderedDict[str, sf.SoundFile] = OrderedDict()
        self._packed_array_cache: OrderedDict[str, np.ndarray] = OrderedDict()

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

        allowed_pitch_set = None
        if allowed_pitch_shifts is not None:
            allowed_pitch_set = {int(shift) for shift in allowed_pitch_shifts}
            if include_original:
                allowed_pitch_set.add(0)

        split_map = _read_split_map(self.split_path)
        songs: list[SongEntry] = []
        audio_root = self.songs_dir if self.audio_backend == "wav" else self.packed_audio_dir
        if not audio_root.exists():
            raise ValueError(f"Audio root does not exist: {audio_root}")

        for song_dir in sorted(audio_root.iterdir()):
            if not song_dir.is_dir():
                continue

            song_id = song_dir.name
            song_split = split_map.get(song_id)
            if split is not None and song_split != split:
                continue

            annotation_path = self.annotations_dir / f"{song_id}.beat.beats.json"
            if not annotation_path.exists():
                continue

            packed_variants: Dict[int, PackedAudioEntry] = {}
            stems_by_semitone: Dict[int, Dict[str, Path]] = {}

            if self.audio_backend == "wav":
                # 曲ごとに「どの semitone なら全 stem が揃っているか」を先に集計する。
                stems_by_semitone = _discover_stem_variants(song_dir, song_id, self.stem_names)
                if not include_original:
                    stems_by_semitone.pop(0, None)
                if allowed_pitch_set is not None:
                    stems_by_semitone = {
                        semitone: stem_map
                        for semitone, stem_map in stems_by_semitone.items()
                        if semitone in allowed_pitch_set
                    }
                if not stems_by_semitone:
                    continue

                # メタデータ確認用の代表 variant。
                preferred_semitone = 0 if 0 in stems_by_semitone else sorted(stems_by_semitone)[0]
                reference_paths = stems_by_semitone[preferred_semitone]
                reference_infos = [sf.info(str(reference_paths[stem_name])) for stem_name in self.stem_names]

                reference_sample_rate = reference_infos[0].samplerate
                channels_per_stem = reference_infos[0].channels
                if any(info.samplerate != reference_sample_rate for info in reference_infos):
                    raise ValueError(f"Mismatched sample rates in {song_dir}")
                if any(info.channels != channels_per_stem for info in reference_infos):
                    raise ValueError(f"Mismatched channel counts in {song_dir}")

                # stem ごとに数サンプルの差があっても安全側に倒すため、最短長を採用する。
                audio_duration_sec = min(info.frames for info in reference_infos) / float(reference_sample_rate)
                available_semitones = tuple(sorted(stems_by_semitone))
            else:
                # packed backend では、曲ごとの npy/json を semitone 単位で解決する。
                packed_variants = _discover_packed_variants(song_dir, song_id, self.stem_names)
                if not include_original:
                    packed_variants.pop(0, None)
                if allowed_pitch_set is not None:
                    packed_variants = {
                        semitone: packed_entry
                        for semitone, packed_entry in packed_variants.items()
                        if semitone in allowed_pitch_set
                    }
                if not packed_variants:
                    continue

                preferred_semitone = 0 if 0 in packed_variants else sorted(packed_variants)[0]
                reference_variant = packed_variants[preferred_semitone]
                reference_sample_rate = reference_variant.sample_rate
                channels_per_stem = reference_variant.channels_per_stem
                expected_num_channels = len(self.stem_names) * channels_per_stem
                if reference_variant.num_channels != expected_num_channels:
                    raise ValueError(
                        f"Unexpected num_channels in packed audio for {song_id}: {reference_variant.num_channels}"
                    )
                if any(variant.sample_rate != reference_sample_rate for variant in packed_variants.values()):
                    raise ValueError(f"Mismatched packed sample rates in {song_dir}")
                if any(variant.channels_per_stem != channels_per_stem for variant in packed_variants.values()):
                    raise ValueError(f"Mismatched packed channel counts in {song_dir}")
                if any(variant.num_channels != expected_num_channels for variant in packed_variants.values()):
                    raise ValueError(f"Mismatched packed num_channels in {song_dir}")

                # packed variant ごとにフレーム長が少し違っても安全側へ倒す。
                audio_duration_sec = (
                    min(variant.num_frames for variant in packed_variants.values())
                    / float(reference_sample_rate)
                )
                available_semitones = tuple(sorted(packed_variants))

            (
                beat_times,
                downbeat_times,
                label_duration_sec,
                meter_annotations,
            ) = derive_beat_downbeat_and_meter_annotations(annotation_path)
            effective_duration_sec = min(audio_duration_sec, label_duration_sec) if label_duration_sec > 0 else audio_duration_sec
            if effective_duration_sec <= 0:
                continue

            songs.append(
                SongEntry(
                    song_id=song_id,
                    split=song_split,
                    sample_rate=reference_sample_rate,
                    channels_per_stem=channels_per_stem,
                    audio_duration_sec=audio_duration_sec,
                    label_duration_sec=label_duration_sec,
                    effective_duration_sec=effective_duration_sec,
                    stems_by_semitone=stems_by_semitone,
                    available_semitones=available_semitones,
                    beat_times=beat_times,
                    downbeat_times=downbeat_times,
                    meter_annotations=meter_annotations,
                    packed_variants=packed_variants,
                )
            )

        if not songs:
            raise ValueError("No songs matched the requested split and file layout")

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
        # モデル側の crop_length と一致するよう、center=True の STFT/CQT を前提にした長さで持つ。
        self.target_num_frames = 1 + ((self.segment_samples - self.n_fft) // self.hop_length)
        # 補助ターゲット生成は別クラスに切り出して、dataset 本体を薄く保つ。
        self.aux_target_builder = StemAuxTargetBuilder(
            stem_names=self.stem_names,
            channels_per_stem=self.channels_per_stem,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            target_num_frames=self.target_num_frames,
        )
        self.repeat_pair_builder = (
            RepeatPairBuilder(
                stem_names=self.stem_names,
                channels_per_stem=self.channels_per_stem,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
                target_num_frames=self.target_num_frames,
                n_fft=self.n_fft,
                similarity_threshold=repeat_ssm_threshold,
                min_diagonal_length_beats=repeat_ssm_min_length_beats,
                near_diagonal_margin_beats=repeat_ssm_near_diagonal_margin_beats,
                max_diagonal_length_beats=repeat_ssm_max_length_beats,
                max_pairs=self.repeat_ssm_max_pairs,
            )
            if self.enable_repeat_pair_targets
            else None
        )

        self.songs = songs
        detected_meter_labels = {
            meter_annotation.meter_label
            for song in self.songs
            for meter_annotation in song.meter_annotations
        }
        if not detected_meter_labels:
            raise ValueError("No meter labels found in the loaded annotations")

        # meter の class index は train/val で必ず一致させる。
        # val 側では train_dataset.meter_to_index を受け取って固定する。
        if meter_to_index is None:
            meter_labels = tuple(
                sorted(detected_meter_labels, key=_meter_label_sort_key)
            )
            resolved_meter_to_index = {
                meter_label: index for index, meter_label in enumerate(meter_labels)
            }
        else:
            resolved_meter_to_index = {
                str(meter_label): int(index)
                for meter_label, index in meter_to_index.items()
            }
            expected_indices = set(range(len(resolved_meter_to_index)))
            if set(resolved_meter_to_index.values()) != expected_indices:
                raise ValueError(
                    "meter_to_index must map classes to a contiguous index range"
                )
            missing_meter_labels = sorted(
                detected_meter_labels.difference(resolved_meter_to_index),
                key=_meter_label_sort_key,
            )
            if missing_meter_labels:
                raise ValueError(
                    "meter_to_index is missing classes required by the dataset: "
                    + ", ".join(missing_meter_labels)
                )
            meter_labels = tuple(
                meter_label
                for meter_label, _ in sorted(
                    resolved_meter_to_index.items(), key=lambda item: item[1]
                )
            )

        self.meter_labels = meter_labels
        self.meter_to_index = resolved_meter_to_index
        self.num_meter_classes = len(self.meter_labels)
        self.meter_class_counts = self._compute_meter_class_counts()

    def __len__(self) -> int:
        if self.samples_per_epoch is not None:
            return self.samples_per_epoch
        return len(self.songs)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # worker ごとに独立した handle cache を持たせたいので、
        # DataLoader に渡す際は open 済み handle を引き継がない。
        state["_audio_file_cache"] = OrderedDict()
        state["_packed_array_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._audio_file_cache = OrderedDict()
        self._packed_array_cache = OrderedDict()

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

    def _get_cached_packed_array(self, array_path: Path) -> np.ndarray:
        cache_key = str(array_path)
        packed_array = self._packed_array_cache.pop(cache_key, None)
        if packed_array is None:
            # npy は memmap で開いて、crop ごとに必要区間だけコピーする。
            packed_array = np.load(array_path, mmap_mode="r")
        self._packed_array_cache[cache_key] = packed_array

        while len(self._packed_array_cache) > self.max_open_files:
            _, oldest_array = self._packed_array_cache.popitem(last=False)
            mmap_handle = getattr(oldest_array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
        return packed_array

    def _load_audio_crop(
        self,
        song: SongEntry,
        wav_path: Path,
        start_sec: float,
    ) -> tuple[torch.Tensor, int]:
        source_sample_rate = song.sample_rate
        frame_offset = int(round(start_sec * source_sample_rate))
        num_frames = int(math.ceil(self.segment_seconds * source_sample_rate))

        if self.use_file_handle_cache:
            # 同じ worker が同じ stem を何度も読むので、open/close を避ける。
            audio_file = self._get_cached_audio_file(wav_path)
            audio_file.seek(frame_offset)
            waveform = torch.from_numpy(
                audio_file.read(num_frames, dtype="float32", always_2d=True).T
            )
            loaded_sample_rate = source_sample_rate
        else:
            # handle cache を使わない場合でも、sample_rate は song メタデータを再利用する。
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
            waveform = AF.resample(waveform, orig_freq=loaded_sample_rate, new_freq=self.sample_rate)

        # 終端付近で短く読まれた場合はゼロ埋めし、valid_samples で実データ範囲だけ管理する。
        valid_samples = min(waveform.shape[-1], self.segment_samples)
        if waveform.shape[-1] < self.segment_samples:
            waveform = F.pad(waveform, (0, self.segment_samples - waveform.shape[-1]))
        else:
            waveform = waveform[..., : self.segment_samples]

        return waveform.contiguous(), valid_samples

    def _load_packed_audio_crop(
        self,
        song: SongEntry,
        semitone: int,
        start_sec: float,
    ) -> tuple[torch.Tensor, int]:
        packed_entry = song.packed_variants[semitone]
        source_sample_rate = packed_entry.sample_rate
        frame_offset = int(round(start_sec * source_sample_rate))
        num_frames = int(math.ceil(self.segment_seconds * source_sample_rate))
        frame_end = min(frame_offset + num_frames, packed_entry.num_frames)

        if self.use_file_handle_cache:
            # 同じ packed 配列を何度も読むので、worker 内では memmap を再利用する。
            packed_array = self._get_cached_packed_array(packed_entry.array_path)
            crop = (
                np.zeros((packed_entry.num_channels, 0), dtype=np.float32)
                if frame_offset >= packed_entry.num_frames
                else np.array(packed_array[:, frame_offset:frame_end], copy=True)
            )
        else:
            packed_array = np.load(packed_entry.array_path, mmap_mode="r")
            crop = (
                np.zeros((packed_entry.num_channels, 0), dtype=np.float32)
                if frame_offset >= packed_entry.num_frames
                else np.array(packed_array[:, frame_offset:frame_end], copy=True)
            )
            mmap_handle = getattr(packed_array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()

        waveform = torch.from_numpy(crop).to(torch.float32)
        if waveform.shape[0] != self.num_audio_channels:
            raise ValueError(
                f"Expected {self.num_audio_channels} channels in {packed_entry.array_path}, found {waveform.shape[0]}"
            )

        if source_sample_rate != self.sample_rate:
            waveform = AF.resample(waveform, orig_freq=source_sample_rate, new_freq=self.sample_rate)

        valid_samples = min(waveform.shape[-1], self.segment_samples)
        if waveform.shape[-1] < self.segment_samples:
            waveform = F.pad(waveform, (0, self.segment_samples - waveform.shape[-1]))
        else:
            waveform = waveform[..., : self.segment_samples]

        return waveform.contiguous(), valid_samples

    def _events_to_frame_targets(
        self,
        event_times: torch.Tensor,
        start_sec: float,
        valid_frames: int,
    ) -> torch.Tensor:
        targets = torch.zeros(self.target_num_frames, dtype=torch.float32)
        if valid_frames <= 0 or event_times.numel() == 0:
            return targets

        end_sec = start_sec + self.segment_seconds
        # この crop に入るイベントだけを取り出す。
        within_crop = (event_times >= start_sec) & (event_times < end_sec)
        if not torch.any(within_crop):
            return targets

        relative_times = event_times[within_crop] - start_sec
        # 音声と同じ sample_rate / hop_length でフレーム番号へ量子化する。
        frame_indices = torch.round(relative_times * self.sample_rate / self.hop_length).long()
        frame_indices = frame_indices[(frame_indices >= 0) & (frame_indices < valid_frames)]
        if frame_indices.numel() > 0:
            targets[frame_indices.unique()] = 1.0
        return targets

    def _meter_annotations_to_frame_targets(
        self,
        song: SongEntry,
        start_sec: float,
        target_num_frames: int,
        valid_frames: int,
    ) -> torch.Tensor:
        targets = torch.full(
            (target_num_frames,),
            fill_value=self.meter_ignore_index,
            dtype=torch.long,
        )
        if valid_frames <= 0:
            return targets

        labeled_duration_sec = valid_frames * self.hop_length / self.sample_rate
        frame_scale = self.sample_rate / self.hop_length
        for meter_annotation in song.meter_annotations:
            relative_start_sec = meter_annotation.start_sec - start_sec
            relative_end_sec = meter_annotation.end_sec - start_sec
            if relative_end_sec <= 0.0 or relative_start_sec >= labeled_duration_sec:
                continue

            clipped_start_sec = max(relative_start_sec, 0.0)
            clipped_end_sec = min(relative_end_sec, labeled_duration_sec)
            # meter は downbeat 付近だけでなく、小節に属する全フレームへ貼る。
            frame_start = int(math.ceil((clipped_start_sec * frame_scale) - 1e-8))
            frame_end = int(math.ceil((clipped_end_sec * frame_scale) - 1e-8))
            frame_start = min(max(frame_start, 0), valid_frames)
            frame_end = min(max(frame_end, 0), valid_frames)
            if frame_end <= frame_start:
                continue

            meter_index = self.meter_to_index.get(meter_annotation.meter_label)
            if meter_index is None:
                raise ValueError(
                    f"Unknown meter label {meter_annotation.meter_label} for song {song.song_id}"
                )
            targets[frame_start:frame_end] = meter_index
        return targets

    def _compute_meter_class_counts(self) -> torch.Tensor:
        class_counts = torch.zeros(self.num_meter_classes, dtype=torch.long)
        for song in self.songs:
            # BalancedSoftmaxLoss 用に、train split 全体で
            # 「各 meter が何フレーム出たか」を数える。
            valid_samples = int(round(song.effective_duration_sec * self.sample_rate))
            if valid_samples < self.n_fft:
                continue

            valid_frames = 1 + ((valid_samples - self.n_fft) // self.hop_length)
            if valid_frames <= 0:
                continue

            meter_targets = self._meter_annotations_to_frame_targets(
                song=song,
                start_sec=0.0,
                target_num_frames=valid_frames,
                valid_frames=valid_frames,
            )
            labeled_targets = meter_targets[
                meter_targets != self.meter_ignore_index
            ]
            if labeled_targets.numel() == 0:
                continue

            class_counts += torch.bincount(
                labeled_targets, minlength=self.num_meter_classes
            )
        return class_counts

    def make_sample(
        self,
        song: SongEntry,
        start_sec: float,
        semitone: Optional[int] = None,
    ) -> dict[str, torch.Tensor | str | int | float]:
        if semitone is None:
            semitone = 0 if 0 in song.available_semitones else song.available_semitones[0]
        if self.audio_backend == "packed":
            if semitone not in song.packed_variants:
                raise ValueError(f"Semitone {semitone} is not available for song {song.song_id}")
            waveform, valid_samples = self._load_packed_audio_crop(song, semitone, start_sec)
        else:
            if semitone not in song.stems_by_semitone:
                raise ValueError(f"Semitone {semitone} is not available for song {song.song_id}")

            stem_waveforms: list[torch.Tensor] = []
            valid_samples = self.segment_samples
            for stem_name in self.stem_names:
                wav_path = song.stems_by_semitone[semitone][stem_name]
                stem_waveform, stem_valid_samples = self._load_audio_crop(song, wav_path, start_sec)
                stem_waveforms.append(stem_waveform)
                valid_samples = min(valid_samples, stem_valid_samples)

            # [stem, channel, time] ではなく、既存モデルに合わせて channel 次元へ連結する。
            waveform = torch.cat(stem_waveforms, dim=0)

        valid_frames = 0
        if valid_samples >= self.n_fft:
            valid_frames = min(self.target_num_frames, 1 + ((valid_samples - self.n_fft) // self.hop_length))

        # ゼロ埋め領域や、STFT 窓が成立しない末尾フレームを loss から外すためのマスク。
        valid_mask = torch.zeros(self.target_num_frames, dtype=torch.float32)
        valid_mask[:valid_frames] = 1.0

        beat_targets = self._events_to_frame_targets(song.beat_times, start_sec, valid_frames)
        downbeat_targets = self._events_to_frame_targets(song.downbeat_times, start_sec, valid_frames)
        meter_targets = self._meter_annotations_to_frame_targets(
            song=song,
            start_sec=start_sec,
            target_num_frames=self.target_num_frames,
            valid_frames=valid_frames,
        )
        aux_targets = self.aux_target_builder.build(
            waveform=waveform,
            valid_frames=valid_frames,
        )
        if self.repeat_pair_builder is None:
            repeat_pair_targets = {
                "pair_indices": torch.zeros(
                    (self.repeat_ssm_max_pairs, 2),
                    dtype=torch.long,
                    device=waveform.device,
                ),
                "pair_mask": torch.zeros(
                    self.repeat_ssm_max_pairs,
                    dtype=torch.float32,
                    device=waveform.device,
                ),
            }
        else:
            built_repeat_pair_targets = self.repeat_pair_builder.build(
                waveform=waveform,
                beat_times=song.beat_times,
                start_sec=start_sec,
                valid_frames=valid_frames,
            )
            repeat_pair_targets = {
                "pair_indices": built_repeat_pair_targets.pair_indices,
                "pair_mask": built_repeat_pair_targets.pair_mask,
            }

        return {
            "waveform": waveform,
            "beat_targets": beat_targets,
            "downbeat_targets": downbeat_targets,
            "meter_targets": meter_targets,
            "broadband_flux_targets": aux_targets.broadband_flux_targets,
            "onset_env_targets": aux_targets.onset_env_targets,
            "high_frequency_flux_targets": aux_targets.high_frequency_flux_targets,
            "repeat_pair_indices": repeat_pair_targets["pair_indices"],
            "repeat_pair_mask": repeat_pair_targets["pair_mask"],
            "valid_mask": valid_mask,
            "song_id": song.song_id,
            "semitone": semitone,
            "start_sec": start_sec,
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int | float]:
        song = self.songs[index % len(self.songs)]

        if not self.random_pitch_shift:
            semitone = 0 if 0 in song.available_semitones else song.available_semitones[0]
        elif len(song.available_semitones) == 1:
            semitone = song.available_semitones[0]
        else:
            # 1 サンプル内では全 stem 共通の semitone を使う。
            semitone_index = int(torch.randint(len(song.available_semitones), size=(1,)).item())
            semitone = song.available_semitones[semitone_index]

        max_start_sec = max(song.effective_duration_sec - self.segment_seconds, 0.0)
        start_sec = 0.0 if max_start_sec <= 0 else float(torch.rand(1).item() * max_start_sec)
        return self.make_sample(song, start_sec=start_sec, semitone=semitone)
