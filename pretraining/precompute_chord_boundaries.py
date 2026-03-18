from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from pretraining.chord_boundary_teacher import (
    DEFAULT_CHORD_TRANSCRIPTION_CHECKPOINT,
    DEFAULT_CHORD_TRANSCRIPTION_CONFIG,
    ChordBoundaryTeacher,
)
from pretraining.unlabeled_dataset import UnlabeledStemDataset


def write_audacity_labels(
    output_path: Path,
    boundary_times_sec: torch.Tensor,
    label_text: str = "1",
) -> None:
    """
    Audacity のラベルトラックへそのまま import しやすい形式で保存する。
    点ラベルとして扱いたいので start=end として書く。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for time_sec in boundary_times_sec.tolist():
            fp.write(f"{time_sec:.6f}\t{time_sec:.6f}\t{label_text}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute chord boundary teacher labels for unlabeled stems."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/unlabeled_dataset"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="保存先。未指定なら <dataset-root>/.chord_boundary_cache",
    )
    parser.add_argument(
        "--song-id",
        type=str,
        default=None,
        help="特定の 1 曲だけ推論したいときの song_id。",
    )
    parser.add_argument(
        "--audacity-label-dir",
        type=Path,
        default=None,
        help="Audacity 用ラベルを書き出すディレクトリ。<song_id>.txt で保存する。",
    )
    parser.add_argument(
        "--audacity-label-text",
        type=str,
        default="1",
        help="Audacity ラベル 3 列目に入れる文字列。",
    )
    parser.add_argument(
        "--show-chunk-progress",
        action="store_true",
        help="各曲の chunk 推論進捗を表示する。1 曲テスト時の切り分け用。",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHORD_TRANSCRIPTION_CHECKPOINT,
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CHORD_TRANSCRIPTION_CONFIG,
    )
    parser.add_argument(
        "--device", type=str, default=None
    )
    parser.add_argument("--chunk-seconds", type=float, default=120.0)
    parser.add_argument("--overlap-seconds", type=float, default=8.0)
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument("--nms-window-radius", type=int, default=3)
    parser.add_argument("--min-boundary-distance-frames", type=int, default=4)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="無ラベル stem dataset の manifest JSON。",
    )
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="既存 manifest を無視して dataset 走査から再生成する。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存 cache を上書きする。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else (args.dataset_root / ".chord_boundary_cache")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[chord-boundary] loading teacher model...")
    teacher = ChordBoundaryTeacher(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        device=device,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        boundary_threshold=args.boundary_threshold,
        nms_window_radius=args.nms_window_radius,
        min_boundary_distance_frames=args.min_boundary_distance_frames,
    )

    print("[chord-boundary] loading dataset manifest...")
    dataset = UnlabeledStemDataset(
        dataset_root=args.dataset_root,
        segment_seconds=30.0,
        sample_rate=teacher.sample_rate,
        hop_length=teacher.hop_length,
        n_fft=teacher.n_fft,
        manifest_path=args.manifest_path,
        rebuild_manifest=args.rebuild_manifest,
    )

    metadata = {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(output_dir),
        "checkpoint_path": str(args.checkpoint_path),
        "config_path": str(args.config_path),
        "sample_rate": teacher.sample_rate,
        "hop_length": teacher.hop_length,
        "n_fft": teacher.n_fft,
        "boundary_threshold": args.boundary_threshold,
        "nms_window_radius": args.nms_window_radius,
        "min_boundary_distance_frames": args.min_boundary_distance_frames,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    songs = dataset.songs
    if args.song_id is not None:
        songs = [song for song in songs if song.song_id == args.song_id]
        if not songs:
            raise ValueError(f"song_id not found in dataset: {args.song_id}")

    print(f"[chord-boundary] start precompute songs={len(songs)} device={device}")
    progress = tqdm(songs, desc="chord-boundary", leave=False)
    for song in progress:
        output_path = output_dir / f"{song.song_id}.pt"
        if output_path.exists() and not args.overwrite:
            if args.audacity_label_dir is not None:
                payload = torch.load(output_path, map_location="cpu", weights_only=False)
                boundary_times_sec = payload.get("boundary_times_sec")
                if torch.is_tensor(boundary_times_sec):
                    write_audacity_labels(
                        output_path=args.audacity_label_dir / f"{song.song_id}.txt",
                        boundary_times_sec=boundary_times_sec.to(torch.float32),
                        label_text=args.audacity_label_text,
                    )
            continue

        if args.show_chunk_progress:
            print(f"[chord-boundary] song={song.song_id}")
        prediction = teacher.predict_from_stem_paths(
            song.stem_paths,
            song_id=song.song_id,
            show_chunk_progress=args.show_chunk_progress,
        )
        payload = {
            "song_id": song.song_id,
            "sample_rate": teacher.sample_rate,
            "hop_length": teacher.hop_length,
            "n_fft": teacher.n_fft,
            "duration_sec": prediction.duration_sec,
            "boundary_times_sec": prediction.boundary_times_sec,
            "boundary_scores": prediction.boundary_scores,
            "frame_times_sec": prediction.frame_times_sec,
            "boundary_probabilities": prediction.boundary_probabilities,
            "checkpoint_path": str(args.checkpoint_path),
            "config_path": str(args.config_path),
        }
        torch.save(payload, output_path)
        if args.audacity_label_dir is not None:
            write_audacity_labels(
                output_path=args.audacity_label_dir / f"{song.song_id}.txt",
                boundary_times_sec=prediction.boundary_times_sec,
                label_text=args.audacity_label_text,
            )
        progress.set_postfix(
            song=song.song_id,
            boundaries=int(prediction.boundary_times_sec.numel()),
        )


if __name__ == "__main__":
    main()
