from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SweepConfig:
    name_suffix: str
    freq_mask_rate: float
    time_mask_rate: float


PRESETS: dict[str, list[SweepConfig]] = {
    "quick": [
        SweepConfig("f0_02_t0_00", 0.02, 0.00),
        SweepConfig("f0_00_t0_05", 0.00, 0.05),
        SweepConfig("f0_02_t0_05", 0.02, 0.05),
    ],
    "balanced": [
        SweepConfig("f0_02_t0_00", 0.02, 0.00),
        SweepConfig("f0_05_t0_00", 0.05, 0.00),
        SweepConfig("f0_00_t0_05", 0.00, 0.05),
        SweepConfig("f0_00_t0_10", 0.00, 0.10),
        SweepConfig("f0_02_t0_05", 0.02, 0.05),
        SweepConfig("f0_05_t0_10", 0.05, 0.10),
    ],
    "full": [
        SweepConfig("f0_02_t0_00", 0.02, 0.00),
        SweepConfig("f0_05_t0_00", 0.05, 0.00),
        SweepConfig("f0_08_t0_00", 0.08, 0.00),
        SweepConfig("f0_00_t0_02", 0.00, 0.02),
        SweepConfig("f0_00_t0_05", 0.00, 0.05),
        SweepConfig("f0_00_t0_10", 0.00, 0.10),
        SweepConfig("f0_02_t0_05", 0.02, 0.05),
        SweepConfig("f0_05_t0_10", 0.05, 0.10),
    ],
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run spec-augment experiments sequentially."
    )
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="balanced",
        help="Which built-in candidate set to run.",
    )
    parser.add_argument(
        "--base-run-name",
        default="exp_meter_classification_w0_0_5_specaug",
        help="Prefix used for output directory names.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--tensorboard-root",
        type=Path,
        default=Path("outputs/tb"),
        help="Root directory for TensorBoard outputs.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose history.jsonl already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not refresh experiment summary after each successful run.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue the sweep even if one experiment fails.",
    )
    return parser.parse_known_args()


def format_command(command: list[str]) -> str:
    return shlex.join(command)


def main() -> None:
    args, train_args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    train_script = project_root / "training" / "train_beat_model.py"
    summary_script = project_root / "scripts" / "summarize_experiments.py"

    # ベースラインに合わせた shared args を先に固定し、必要なら後ろの train_args で上書きする。
    shared_args = [
        "--audio-backend",
        "packed",
        "--meter-loss-weight",
        "0.05",
        "--seed",
        "42",
    ]

    failed_runs: list[str] = []
    for config in PRESETS[args.preset]:
        run_name = f"{args.base_run_name}_{config.name_suffix}"
        output_dir = args.outputs_root / run_name
        tensorboard_dir = args.tensorboard_root / run_name
        history_path = output_dir / "history.jsonl"

        if args.skip_existing and history_path.exists():
            print(f"skip={run_name} reason=history_exists")
            continue

        command = [
            sys.executable,
            str(train_script),
            "--output-dir",
            str(output_dir),
            "--tensorboard-dir",
            str(tensorboard_dir),
            "--spec-augment-freq-mask-rate",
            f"{config.freq_mask_rate:.3f}",
            "--spec-augment-time-mask-rate",
            f"{config.time_mask_rate:.3f}",
            *shared_args,
            *train_args,
        ]

        print(f"run={run_name}")
        print(format_command(command))
        if args.dry_run:
            continue

        try:
            subprocess.run(command, check=True, cwd=project_root)
        except subprocess.CalledProcessError:
            failed_runs.append(run_name)
            if not args.continue_on_error:
                raise
            print(f"failed={run_name}")
            continue

        if args.no_summary:
            continue

        subprocess.run(
            [sys.executable, str(summary_script)],
            check=True,
            cwd=project_root,
        )

    if failed_runs:
        print("failed_runs=" + ",".join(failed_runs))


if __name__ == "__main__":
    main()
