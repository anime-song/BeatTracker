from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="time-only SpecAugment の sweep を順番に回す。"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="各実験の output_dir をまとめる親ディレクトリ。",
    )
    parser.add_argument(
        "--mask-spans",
        type=int,
        nargs="+",
        required=True,
        help="検証する time mask span の候補。",
    )
    parser.add_argument(
        "--mask-ratios",
        type=float,
        nargs="+",
        required=True,
        help="検証する time mask ratio の候補。",
    )
    parser.add_argument(
        "--specaugment-prob",
        type=float,
        default=1.0,
        help="SpecAugment を適用する確率。",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="train_beat_model.py を起動する Python 実行ファイル。",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("training/train_beat_model.py"),
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="失敗した実験があっても残りを続ける。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実行せず、生成されるコマンドだけ表示する。",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="`--` 以降は train_beat_model.py へそのまま渡す。",
    )
    return parser.parse_args()


def _normalize_train_args(train_args: list[str]) -> list[str]:
    if train_args and train_args[0] == "--":
        return train_args[1:]
    return train_args


def _format_ratio_tag(mask_ratio: float) -> str:
    return f"{mask_ratio:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _build_run_name(mask_span: int, mask_ratio: float) -> str:
    return f"time_mask_span{mask_span}_ratio{_format_ratio_tag(mask_ratio)}"


def _append_result(result_path: Path, payload: dict[str, object]) -> None:
    with result_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    train_script = (
        args.train_script
        if args.train_script.is_absolute()
        else (project_root / args.train_script)
    )
    if not train_script.exists():
        raise FileNotFoundError(f"train script does not exist: {train_script}")

    train_args = _normalize_train_args(list(args.train_args))
    if not train_args:
        raise ValueError("train_beat_model.py に渡す引数を `--` 以降に指定してください。")

    args.output_root.mkdir(parents=True, exist_ok=True)
    result_path = args.output_root / "sweep_results.jsonl"

    total_runs = len(args.mask_spans) * len(args.mask_ratios)
    run_index = 0
    for mask_span in args.mask_spans:
        if mask_span <= 0:
            raise ValueError("mask span must be positive")

        for mask_ratio in args.mask_ratios:
            if not 0.0 < mask_ratio <= 1.0:
                raise ValueError("mask ratio must be in (0.0, 1.0]")

            run_index += 1
            run_name = _build_run_name(mask_span=mask_span, mask_ratio=mask_ratio)
            output_dir = args.output_root / run_name
            command = [
                args.python,
                str(train_script),
                *train_args,
                "--specaugment-time-mask-span",
                str(mask_span),
                "--specaugment-time-mask-ratio",
                str(mask_ratio),
                "--specaugment-prob",
                str(args.specaugment_prob),
                "--specaugment-fixed-time-mask-size",
                "--output-dir",
                str(output_dir),
            ]

            print(f"[{run_index}/{total_runs}] {run_name}")
            print(" ".join(command))

            if args.dry_run:
                _append_result(
                    result_path,
                    {
                        "run_name": run_name,
                        "status": "dry_run",
                        "mask_span": mask_span,
                        "mask_ratio": mask_ratio,
                        "output_dir": str(output_dir),
                        "command": command,
                    },
                )
                continue

            started_at = time.time()
            completed = subprocess.run(command, cwd=project_root)
            elapsed_sec = time.time() - started_at
            status = "ok" if completed.returncode == 0 else "failed"
            _append_result(
                result_path,
                {
                    "run_name": run_name,
                    "status": status,
                    "returncode": completed.returncode,
                    "mask_span": mask_span,
                    "mask_ratio": mask_ratio,
                    "output_dir": str(output_dir),
                    "elapsed_sec": elapsed_sec,
                    "command": command,
                },
            )

            if completed.returncode != 0 and not args.keep_going:
                raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
