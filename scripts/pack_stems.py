from __future__ import annotations

import argparse
import os
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from data import BeatStemDataset
from data.beat_dataset import SongEntry


@dataclass(frozen=True)
class PackTask:
    song_id: str
    split: Optional[str]
    semitone: int
    stem_names: tuple[str, ...]
    stem_paths: tuple[str, ...]
    expected_sample_rate: int
    expected_channels_per_stem: int
    output_array_path: str
    output_metadata_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="songs_separated 以下の stem WAV を、曲ごとの packed 配列へ事前変換します。"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/meter_dataset"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="未指定なら <dataset-root>/songs_packed を使います。",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "val", "all"),
        default="all",
    )
    parser.add_argument(
        "--song-id",
        dest="song_ids",
        action="append",
        default=None,
        help="特定の曲だけ変換したいときに繰り返し指定します。",
    )
    parser.add_argument("--song-limit", type=int, default=None)
    parser.add_argument(
        "--allowed-pitch-shifts",
        type=int,
        nargs="*",
        default=None,
        help="指定した semitone だけ変換します。未指定なら利用可能なものを全て対象にします。",
    )
    parser.add_argument(
        "--exclude-original",
        action="store_true",
        help="オリジナル音源 (0 semitone) を packed 化しません。",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=1048576,
        help="1回に読むフレーム数。大きいほど速いがメモリ使用量も増えます。",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="並列に pack する variant 数です。ディスクが遅い場合は 2-4 程度が実用的です。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存の packed ファイルがあっても再生成します。",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="BeatStemDataset のメタデータ読込に使う値です。packed 出力自体は元の sample rate を保持します。",
    )
    parser.add_argument("--hop-length", type=int, default=441)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--segment-seconds", type=float, default=30.0)
    return parser.parse_args()


def resolve_split(split: str) -> Optional[str]:
    if split == "all":
        return None
    return split


def packed_file_stem(song_id: str, semitone: int) -> str:
    return f"{song_id}_stems_pitch_{semitone}st"


def output_paths(output_dir: Path, song_id: str, semitone: int) -> tuple[Path, Path]:
    song_output_dir = output_dir / song_id
    file_stem = packed_file_stem(song_id, semitone)
    return song_output_dir / f"{file_stem}.npy", song_output_dir / f"{file_stem}.json"


def select_songs(dataset: BeatStemDataset, args: argparse.Namespace) -> list[SongEntry]:
    songs = dataset.songs
    if args.song_ids:
        requested_ids = set(args.song_ids)
        songs = [song for song in songs if song.song_id in requested_ids]
    if args.song_limit is not None:
        songs = songs[: args.song_limit]
    return songs


def select_semitones(song: SongEntry, args: argparse.Namespace) -> list[int]:
    semitones = list(song.available_semitones)
    if args.exclude_original:
        semitones = [semitone for semitone in semitones if semitone != 0]
    if args.allowed_pitch_shifts is not None:
        allowed = {int(semitone) for semitone in args.allowed_pitch_shifts}
        semitones = [semitone for semitone in semitones if semitone in allowed]
    return sorted(semitones)


def build_tasks(
    songs: Iterable[SongEntry],
    stem_names: tuple[str, ...],
    args: argparse.Namespace,
) -> list[PackTask]:
    tasks: list[PackTask] = []
    output_dir = args.output_dir or (args.dataset_root / "songs_packed")
    for song in songs:
        for semitone in select_semitones(song, args):
            array_path, metadata_path = output_paths(output_dir, song.song_id, semitone)
            stem_paths = tuple(
                str(song.stems_by_semitone[semitone][stem_name]) for stem_name in stem_names
            )
            tasks.append(
                PackTask(
                    song_id=song.song_id,
                    split=song.split,
                    semitone=semitone,
                    stem_names=stem_names,
                    stem_paths=stem_paths,
                    expected_sample_rate=int(song.sample_rate),
                    expected_channels_per_stem=int(song.channels_per_stem),
                    output_array_path=str(array_path),
                    output_metadata_path=str(metadata_path),
                )
            )
    return tasks


def load_existing_metadata(metadata_path: Path) -> Optional[dict]:
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def should_skip_existing(
    task: PackTask,
    storage_dtype: np.dtype,
    force: bool,
) -> bool:
    array_path = Path(task.output_array_path)
    metadata_path = Path(task.output_metadata_path)
    if force or not array_path.exists() or not metadata_path.exists():
        return False

    metadata = load_existing_metadata(metadata_path)
    if metadata is None:
        return False

    return (
        metadata.get("song_id") == task.song_id
        and int(metadata.get("semitone", 9999)) == task.semitone
        and tuple(metadata.get("stem_names", [])) == task.stem_names
        and metadata.get("storage_dtype") == storage_dtype.name
        and int(metadata.get("sample_rate", -1)) == task.expected_sample_rate
        and int(metadata.get("channels_per_stem", -1)) == task.expected_channels_per_stem
    )


def pack_song_variant(
    task: PackTask,
    storage_dtype: np.dtype,
    chunk_frames: int,
    force: bool,
) -> tuple[str, Path]:
    array_path = Path(task.output_array_path)
    metadata_path = Path(task.output_metadata_path)
    if should_skip_existing(
        task=task,
        storage_dtype=storage_dtype,
        force=force,
    ):
        return "skipped", array_path

    array_path.parent.mkdir(parents=True, exist_ok=True)

    stem_paths = [Path(stem_path) for stem_path in task.stem_paths]
    stem_infos = [sf.info(str(stem_path)) for stem_path in stem_paths]

    sample_rate = int(stem_infos[0].samplerate)
    channels_per_stem = int(stem_infos[0].channels)
    num_frames = min(int(stem_info.frames) for stem_info in stem_infos)
    num_channels = len(task.stem_names) * channels_per_stem

    if any(int(stem_info.samplerate) != sample_rate for stem_info in stem_infos):
        raise ValueError(f"Mismatched sample rates in {task.song_id}")
    if any(int(stem_info.channels) != channels_per_stem for stem_info in stem_infos):
        raise ValueError(f"Mismatched channel counts in {task.song_id}")

    packed = np.lib.format.open_memmap(
        array_path,
        mode="w+",
        dtype=storage_dtype,
        shape=(num_channels, num_frames),
    )

    for stem_index, stem_path in enumerate(stem_paths):
        channel_start = stem_index * channels_per_stem
        written_frames = 0

        with sf.SoundFile(str(stem_path), mode="r") as audio_file:
            while written_frames < num_frames:
                frames_to_read = min(chunk_frames, num_frames - written_frames)
                block = audio_file.read(frames_to_read, dtype="float32", always_2d=True)
                if block.size == 0:
                    break

                # SoundFile は [time, channel] なので、学習で使いやすい [channel, time] に直す。
                block = block.T
                if block.shape[0] == 1 and channels_per_stem == 2:
                    block = np.repeat(block, 2, axis=0)
                elif block.shape[0] != channels_per_stem:
                    raise ValueError(
                        f"Expected {channels_per_stem} channels in {stem_path}, found {block.shape[0]}"
                    )

                if block.dtype != storage_dtype:
                    block = block.astype(storage_dtype, copy=False)

                block_frames = block.shape[1]
                packed[
                    channel_start : channel_start + channels_per_stem,
                    written_frames : written_frames + block_frames,
                ] = block
                written_frames += block_frames

        if written_frames < num_frames:
            packed[
                channel_start : channel_start + channels_per_stem,
                written_frames:num_frames,
            ] = 0

    del packed

    metadata = {
        "song_id": task.song_id,
        "split": task.split,
        "semitone": task.semitone,
        "sample_rate": sample_rate,
        "channels_per_stem": channels_per_stem,
        "num_channels": num_channels,
        "num_frames": num_frames,
        "storage_dtype": storage_dtype.name,
        "stem_names": list(task.stem_names),
        "source_kind": "songs_separated",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return "packed", array_path


def execute_pack_task(
    task: PackTask,
    storage_dtype_name: str,
    chunk_frames: int,
    force: bool,
) -> tuple[str, str]:
    status, array_path = pack_song_variant(
        task=task,
        storage_dtype=np.dtype(storage_dtype_name),
        chunk_frames=chunk_frames,
        force=force,
    )
    return status, str(array_path)


def summarize_directory_size(paths: Iterable[Path]) -> float:
    total_bytes = 0
    for path in paths:
        if path.exists():
            total_bytes += path.stat().st_size
    return total_bytes / (1024.0 * 1024.0)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.dataset_root / "songs_packed")
    storage_dtype = np.dtype(args.dtype)
    num_jobs = max(1, int(args.jobs))

    dataset = BeatStemDataset(
        dataset_root=args.dataset_root,
        split=resolve_split(args.split),
        segment_seconds=args.segment_seconds,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        random_pitch_shift=False,
        use_file_handle_cache=False,
    )

    songs = select_songs(dataset, args)
    if not songs:
        raise ValueError("No songs matched the requested filters")

    tasks = build_tasks(songs, dataset.stem_names, args)
    if not tasks:
        raise ValueError("No song/semitone combinations matched the requested filters")

    print(
        f"split={args.split}, songs={len(songs)}, variants={len(tasks)}, "
        f"dtype={storage_dtype.name}, jobs={num_jobs}, "
        f"chunk_frames={args.chunk_frames}, output_dir={output_dir}"
    )

    packed_paths: list[Path] = []
    packed_count = 0
    skipped_count = 0

    if num_jobs == 1:
        for task in tqdm(tasks, desc="Packing stems", unit="variant"):
            status, array_path = pack_song_variant(
                task=task,
                storage_dtype=storage_dtype,
                chunk_frames=args.chunk_frames,
                force=args.force,
            )
            packed_paths.append(array_path)
            if status == "packed":
                packed_count += 1
            else:
                skipped_count += 1
    else:
        with ProcessPoolExecutor(max_workers=num_jobs) as executor:
            futures = {
                executor.submit(
                    execute_pack_task,
                    task,
                    storage_dtype.name,
                    args.chunk_frames,
                    args.force,
                ): task
                for task in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Packing stems", unit="variant"):
                status, array_path_str = future.result()
                packed_paths.append(Path(array_path_str))
                if status == "packed":
                    packed_count += 1
                else:
                    skipped_count += 1

    packed_size_mib = summarize_directory_size(packed_paths)
    print(f"packed={packed_count}, skipped={skipped_count}, packed_size={packed_size_mib:.1f} MiB")


if __name__ == "__main__":
    main()
