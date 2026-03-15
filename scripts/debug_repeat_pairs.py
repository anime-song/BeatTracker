from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import torch

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from data.beat_dataset import DEFAULT_STEM_NAMES, BeatStemDataset, SongEntry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="繰り返し区間ペアの検出結果を可視化して確認する。"
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/meter_dataset"))
    parser.add_argument(
        "--audio-backend",
        type=str,
        choices=("wav", "packed"),
        default="packed",
    )
    parser.add_argument("--packed-audio-dir", type=Path, default=None)
    parser.add_argument("--song-id", type=str, required=True)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--segment-seconds", type=float, default=30.0)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=441)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--semitone", type=int, default=None)
    parser.add_argument(
        "--stem-names",
        nargs="+",
        default=list(DEFAULT_STEM_NAMES),
    )
    parser.add_argument(
        "--repeat-ssm-threshold",
        type=float,
        default=0.85,
        help="SSM 上で『似ている』とみなす最小類似度。上げるほど候補は減り、厳しくなる。",
    )
    parser.add_argument(
        "--repeat-ssm-sync-unit",
        type=str,
        choices=("beat", "bar"),
        default="beat",
        help="SSM を beat 単位で作るか、bar 単位で作るか。bar の方が構造反復寄りになりやすい。",
    )
    parser.add_argument(
        "--repeat-ssm-min-length-beats",
        type=int,
        default=8,
        help="採用する対角線 run の最小長。sync_unit=beat なら beat 数、bar なら bar 数として解釈する。",
    )
    parser.add_argument(
        "--repeat-ssm-near-diagonal-margin-beats",
        type=int,
        default=16,
        help="主対角線の近くを捨てるための最小オフセット。sync_unit=beat なら beat 数、bar なら bar 数。",
    )
    parser.add_argument(
        "--repeat-ssm-max-length-beats",
        type=int,
        default=16,
        help="1 本の対角線 run から使う最大長。長すぎる run を切って局所反復の暴走を抑える。",
    )
    parser.add_argument(
        "--repeat-ssm-max-pairs",
        type=int,
        default=128,
        help="最終的に保持する frame ペア数の上限。大きいほど loss には多く入るが、ノイズも増えやすい。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/repeat_debug"),
    )
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> BeatStemDataset:
    return BeatStemDataset(
        dataset_root=args.dataset_root,
        split=args.split,
        segment_seconds=args.segment_seconds,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        stem_names=args.stem_names,
        random_pitch_shift=False,
        audio_backend=args.audio_backend,
        packed_audio_dir=args.packed_audio_dir,
        enable_repeat_pair_targets=True,
        repeat_ssm_sync_unit=args.repeat_ssm_sync_unit,
        repeat_ssm_threshold=args.repeat_ssm_threshold,
        repeat_ssm_min_length_beats=args.repeat_ssm_min_length_beats,
        repeat_ssm_near_diagonal_margin_beats=args.repeat_ssm_near_diagonal_margin_beats,
        repeat_ssm_max_length_beats=args.repeat_ssm_max_length_beats,
        repeat_ssm_max_pairs=args.repeat_ssm_max_pairs,
    )


def find_song(dataset: BeatStemDataset, song_id: str) -> SongEntry:
    for song in dataset.songs:
        if song.song_id == song_id:
            return song
    raise ValueError(f"song_id が見つかりませんでした: {song_id}")


def write_pairs_csv(path: Path, pair_indices: torch.Tensor, pair_mask: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["pair_index", "left_frame", "right_frame"])
        count = int(pair_mask.sum().item())
        for index in range(count):
            writer.writerow(
                [
                    index,
                    int(pair_indices[index, 0].item()),
                    int(pair_indices[index, 1].item()),
                ]
            )


def write_summary_json(
    path: Path,
    args: argparse.Namespace,
    song: SongEntry,
    valid_frames: int,
    debug_info,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "song_id": song.song_id,
        "start_sec": args.start_sec,
        "segment_seconds": args.segment_seconds,
        "valid_frames": valid_frames,
        "sync_unit": debug_info.sync_unit,
        "event_count_in_window": int(debug_info.sync_frame_indices.numel()),
        "sync_count": int(debug_info.sync_start_frames.numel()),
        "sync_labels": list(debug_info.sync_labels),
        "repeat_pair_count": int(debug_info.targets.pair_mask.sum().item()),
        "repeat_ssm_sync_unit": args.repeat_ssm_sync_unit,
        "repeat_ssm_threshold": args.repeat_ssm_threshold,
        "repeat_ssm_min_length_beats": args.repeat_ssm_min_length_beats,
        "repeat_ssm_near_diagonal_margin_beats": args.repeat_ssm_near_diagonal_margin_beats,
        "repeat_ssm_max_length_beats": args.repeat_ssm_max_length_beats,
        "runs": [
            {
                "start_index_a": run.start_index_a,
                "start_index_b": run.start_index_b,
                "length_units": run.length_units,
                "mean_similarity": run.mean_similarity,
            }
            for run in debug_info.runs
        ],
        "selected_pairs": [
            {
                "left_frame": int(debug_info.targets.pair_indices[index, 0].item()),
                "right_frame": int(debug_info.targets.pair_indices[index, 1].item()),
            }
            for index in range(int(debug_info.targets.pair_mask.sum().item()))
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _svg_heatmap_cells(
    similarity: torch.Tensor,
    left: float,
    top: float,
    size: float,
) -> list[str]:
    cells: list[str] = []
    if similarity.numel() == 0:
        return cells

    rows, cols = similarity.shape
    cell_size = size / max(rows, cols)
    for row in range(rows):
        for col in range(cols):
            value = float(similarity[row, col].item())
            gray = int(round(255 * (1.0 - max(0.0, min(1.0, value)))))
            cells.append(
                f'<rect x="{left + col * cell_size:.2f}" y="{top + row * cell_size:.2f}" '
                f'width="{cell_size:.2f}" height="{cell_size:.2f}" '
                f'fill="rgb({gray},{gray},{gray})"/>'
            )
    return cells


def _svg_run_lines(
    runs: Iterable,
    beat_count: int,
    left: float,
    top: float,
    size: float,
) -> list[str]:
    lines: list[str] = []
    if beat_count <= 0:
        return lines
    cell_size = size / beat_count
    for run in runs:
        x1 = left + (run.start_index_b + 0.5) * cell_size
        y1 = top + (run.start_index_a + 0.5) * cell_size
        x2 = left + (run.start_index_b + run.length_units - 0.5) * cell_size
        y2 = top + (run.start_index_a + run.length_units - 0.5) * cell_size
        lines.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            'stroke="#d94841" stroke-width="2.0" stroke-linecap="round"/>'
        )
    return lines


def write_ssm_svg(path: Path, song_id: str, start_sec: float, debug_info) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 860
    height = 760
    left = 90
    top = 70
    matrix_size = 560
    similarity = debug_info.similarity_matrix.detach().cpu()
    beat_count = int(similarity.shape[0])

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        f'<text x="24" y="30" font-size="22" font-weight="700" fill="#15212b">Repeat Pair Debug</text>',
        f'<text x="24" y="55" font-size="13" fill="#425466">song={html.escape(song_id)} start_sec={start_sec:.2f} '
        f'{html.escape(debug_info.sync_unit)}_sync={int(debug_info.sync_start_frames.numel())} '
        f'selected_pairs={int(debug_info.targets.pair_mask.sum().item())}</text>',
        f'<rect x="{left}" y="{top}" width="{matrix_size}" height="{matrix_size}" fill="#ffffff" stroke="#425466" stroke-width="1.2"/>',
        *_svg_heatmap_cells(similarity, left, top, matrix_size),
        *_svg_run_lines(debug_info.runs, beat_count, left, top, matrix_size),
    ]

    if beat_count > 0:
        cell_size = matrix_size / beat_count
        tick_step = max(1, beat_count // 12)
        for beat_index in range(0, beat_count, tick_step):
            x = left + (beat_index + 0.5) * cell_size
            y = top + (beat_index + 0.5) * cell_size
            lines.append(
                f'<text x="{x:.2f}" y="{top + matrix_size + 18:.2f}" text-anchor="middle" font-size="11" fill="#425466">{beat_index}</text>'
            )
            lines.append(
                f'<text x="{left - 12:.2f}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#425466">{beat_index}</text>'
            )

    legend_y = top + matrix_size + 64
    lines.extend(
        [
            f'<rect x="{left}" y="{legend_y - 12}" width="18" height="18" fill="rgb(32,32,32)"/>',
            f'<text x="{left + 28}" y="{legend_y + 2}" font-size="12" fill="#425466">high similarity</text>',
            f'<line x1="{left + 180}" y1="{legend_y - 3}" x2="{left + 210}" y2="{legend_y - 3}" stroke="#d94841" stroke-width="2.0"/>',
            f'<text x="{left + 220}" y="{legend_y + 2}" font-size="12" fill="#425466">detected diagonal run</text>',
        ]
    )

    if debug_info.runs:
        run_text_y = legend_y + 36
        lines.append(
            f'<text x="{left}" y="{run_text_y}" font-size="13" font-weight="700" fill="#15212b">Top runs</text>'
        )
        for index, run in enumerate(debug_info.runs[:8], start=1):
            lines.append(
                f'<text x="{left}" y="{run_text_y + index * 18}" font-size="12" fill="#425466">'
                f'{index}. ({run.start_index_a}, {run.start_index_b}) len={run.length_units} sim={run.mean_similarity:.3f}</text>'
            )

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args)
    if dataset.repeat_pair_builder is None:
        raise RuntimeError("repeat pair builder が有効化されていません")

    song = find_song(dataset, args.song_id)
    sample = dataset.make_sample(song=song, start_sec=args.start_sec, semitone=args.semitone)
    valid_frames = int(sample["valid_mask"].sum().item())
    _, debug_info = dataset.repeat_pair_builder.analyze(
        waveform=sample["waveform"],
        beat_times=song.beat_times,
        downbeat_times=song.downbeat_times,
        meter_segments=tuple(
            (
                meter_annotation.start_sec,
                meter_annotation.end_sec,
                meter_annotation.meter_label,
            )
            for meter_annotation in song.meter_annotations
        ),
        start_sec=args.start_sec,
        valid_frames=valid_frames,
    )

    output_dir = (
        args.output_dir
        / args.repeat_ssm_sync_unit
        / song.song_id
        / f"start_{args.start_sec:.2f}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_json(output_dir / "repeat_debug.json", args, song, valid_frames, debug_info)
    write_pairs_csv(
        output_dir / "repeat_pairs.csv",
        debug_info.targets.pair_indices.cpu(),
        debug_info.targets.pair_mask.cpu(),
    )
    write_ssm_svg(output_dir / "repeat_ssm.svg", song.song_id, args.start_sec, debug_info)

    print(f"song_id={song.song_id}")
    print(f"sync_unit={debug_info.sync_unit}")
    print(f"valid_frames={valid_frames}")
    print(f"sync_count={int(debug_info.sync_start_frames.numel())}")
    print(f"repeat_pair_count={int(debug_info.targets.pair_mask.sum().item())}")
    print(f"json={output_dir / 'repeat_debug.json'}")
    print(f"csv={output_dir / 'repeat_pairs.csv'}")
    print(f"svg={output_dir / 'repeat_ssm.svg'}")


if __name__ == "__main__":
    main()
