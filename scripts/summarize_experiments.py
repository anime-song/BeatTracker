from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class ExperimentSummary:
    run_name: str
    run_path: str
    status: str
    configured_epochs: Optional[int]
    last_epoch: Optional[int]
    last_downbeat_f1: Optional[float]
    best_epoch: Optional[int]
    best_downbeat_f1: Optional[float]
    best_beat_f1: Optional[float]
    best_val_loss: Optional[float]
    seed: Optional[int]
    lr: Optional[float]
    batch_size: Optional[int]
    train_samples_per_epoch: Optional[int]
    segment_seconds: Optional[float]
    meter_loss_weight: Optional[float]
    meter_conditioning: Optional[str]
    audio_backend: Optional[str]
    scheduler: Optional[str]
    ema_decay: Optional[float]
    num_layers: Optional[int]
    hidden_size: Optional[int]
    output_dim: Optional[int]
    resume: Optional[str]
    git_branch: Optional[str]
    git_commit: Optional[str]
    git_dirty: Optional[bool]

    @property
    def model_tag(self) -> str:
        parts: list[str] = []
        if self.num_layers is not None:
            parts.append(f"L{self.num_layers}")
        if self.hidden_size is not None:
            parts.append(f"H{self.hidden_size}")
        if self.output_dim is not None:
            parts.append(f"O{self.output_dim}")
        return "/".join(parts) if parts else "-"

    def to_csv_row(self) -> dict[str, str]:
        return {
            "run_name": self.run_name,
            "status": self.status,
            "best_epoch": _format_int(self.best_epoch),
            "best_downbeat_f1": _format_float(self.best_downbeat_f1),
            "best_beat_f1": _format_float(self.best_beat_f1),
            "best_val_loss": _format_float(self.best_val_loss),
            "last_epoch": _format_int(self.last_epoch),
            "last_downbeat_f1": _format_float(self.last_downbeat_f1),
            "configured_epochs": _format_int(self.configured_epochs),
            "seed": _format_int(self.seed),
            "lr": _format_float(self.lr, digits=6),
            "batch_size": _format_int(self.batch_size),
            "train_samples_per_epoch": _format_int(self.train_samples_per_epoch),
            "segment_seconds": _format_float(self.segment_seconds, digits=1),
            "meter_loss_weight": _format_float(self.meter_loss_weight, digits=3),
            "meter_conditioning": self.meter_conditioning or "",
            "audio_backend": self.audio_backend or "",
            "scheduler": self.scheduler or "",
            "ema_decay": _format_float(self.ema_decay, digits=4),
            "num_layers": _format_int(self.num_layers),
            "hidden_size": _format_int(self.hidden_size),
            "output_dim": _format_int(self.output_dim),
            "resume": self.resume or "",
            "git_branch": self.git_branch or "",
            "git_commit": self.git_commit or "",
            "git_dirty": "" if self.git_dirty is None else str(self.git_dirty).lower(),
            "run_path": self.run_path,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize experiment outputs into CSV and Markdown."
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Directory that contains experiment output subdirectories.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("reports/experiment_summary.csv"),
        help="Destination CSV summary path.",
    )
    parser.add_argument(
        "--markdown-path",
        type=Path,
        default=Path("reports/experiment_summary.md"),
        help="Destination Markdown summary path.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_history(history_path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not history_path.exists():
        return entries

    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entries.append(json.loads(line))
    return entries


def _as_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _as_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _format_int(value: Optional[int]) -> str:
    return "" if value is None else str(value)


def _format_float(value: Optional[float], digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _derive_status(configured_epochs: Optional[int], last_epoch: Optional[int]) -> str:
    if last_epoch is None:
        return "no_history"
    if configured_epochs is not None and last_epoch >= configured_epochs:
        return "complete"
    return "in_progress"


def _find_best_entry(history_entries: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    best_entry: Optional[dict[str, Any]] = None
    best_key: tuple[float, int] | None = None

    for entry in history_entries:
        val_metrics = entry.get("val")
        if not isinstance(val_metrics, dict):
            continue
        score = val_metrics.get("downbeat_f1")
        if score is None:
            continue

        entry_epoch = int(entry.get("epoch", 0))
        key = (float(score), entry_epoch)
        if best_key is None or key > best_key:
            best_key = key
            best_entry = entry
    return best_entry


def _build_summary(run_dir: Path) -> Optional[ExperimentSummary]:
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.jsonl"

    has_config = config_path.exists()
    has_history = history_path.exists()
    if not has_config and not has_history:
        return None

    config = _read_json(config_path) if has_config else {}
    history_entries = _read_history(history_path)

    last_entry = history_entries[-1] if history_entries else None
    best_entry = _find_best_entry(history_entries)

    last_metrics = last_entry.get("val", {}) if isinstance(last_entry, dict) else {}
    best_metrics = best_entry.get("val", {}) if isinstance(best_entry, dict) else {}

    configured_epochs = _as_int(config.get("epochs"))
    last_epoch = _as_int(last_entry.get("epoch")) if last_entry else None

    return ExperimentSummary(
        run_name=run_dir.name,
        run_path=str(run_dir),
        status=_derive_status(configured_epochs, last_epoch),
        configured_epochs=configured_epochs,
        last_epoch=last_epoch,
        last_downbeat_f1=_as_float(last_metrics.get("downbeat_f1")),
        best_epoch=_as_int(best_entry.get("epoch")) if best_entry else None,
        best_downbeat_f1=_as_float(best_metrics.get("downbeat_f1")),
        best_beat_f1=_as_float(best_metrics.get("beat_f1")),
        best_val_loss=_as_float(best_metrics.get("loss")),
        seed=_as_int(config.get("seed")),
        lr=_as_float(config.get("lr")),
        batch_size=_as_int(config.get("batch_size")),
        train_samples_per_epoch=_as_int(config.get("train_samples_per_epoch")),
        segment_seconds=_as_float(config.get("segment_seconds")),
        meter_loss_weight=_as_float(config.get("meter_loss_weight")),
        meter_conditioning=(
            str(config["meter_conditioning"])
            if config.get("meter_conditioning")
            else None
        ),
        audio_backend=str(config["audio_backend"]) if "audio_backend" in config else None,
        scheduler=str(config["scheduler"]) if "scheduler" in config else None,
        ema_decay=_as_float(config.get("ema_decay")),
        num_layers=_as_int(config.get("num_layers")),
        hidden_size=_as_int(config.get("hidden_size")),
        output_dim=_as_int(config.get("output_dim")),
        resume=str(config["resume"]) if config.get("resume") else None,
        git_branch=str(config["git_branch"]) if config.get("git_branch") else None,
        git_commit=str(config["git_commit"]) if config.get("git_commit") else None,
        git_dirty=_as_bool(config.get("git_dirty")),
    )


def collect_summaries(outputs_root: Path) -> list[ExperimentSummary]:
    summaries: list[ExperimentSummary] = []
    if not outputs_root.exists():
        return summaries

    for candidate in sorted(outputs_root.iterdir()):
        if not candidate.is_dir():
            continue
        summary = _build_summary(candidate)
        if summary is not None:
            summaries.append(summary)

    summaries.sort(
        key=lambda item: (
            item.best_downbeat_f1 is not None,
            item.best_downbeat_f1 or float("-inf"),
            item.last_epoch or -1,
            item.run_name,
        ),
        reverse=True,
    )
    return summaries


def write_csv(csv_path: Path, summaries: list[ExperimentSummary]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "status",
        "best_epoch",
        "best_downbeat_f1",
        "best_beat_f1",
        "best_val_loss",
        "last_epoch",
        "last_downbeat_f1",
        "configured_epochs",
        "seed",
        "lr",
        "batch_size",
        "train_samples_per_epoch",
        "segment_seconds",
        "meter_loss_weight",
        "meter_conditioning",
        "audio_backend",
        "scheduler",
        "ema_decay",
        "num_layers",
        "hidden_size",
        "output_dim",
        "resume",
        "git_branch",
        "git_commit",
        "git_dirty",
        "run_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.to_csv_row())


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def write_markdown(markdown_path: Path, summaries: list[ExperimentSummary]) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Experiment Summary",
        "",
        f"Generated: {generated_at}",
        "",
        f"Runs: {len(summaries)}",
        "",
    ]

    overview_rows = [
        [
            summary.run_name,
            summary.status,
            _format_int(summary.best_epoch) or "-",
            _format_float(summary.best_downbeat_f1) or "-",
            _format_int(summary.last_epoch) or "-",
            _format_float(summary.last_downbeat_f1) or "-",
            _format_int(summary.seed) or "-",
            _format_float(summary.lr, digits=6) or "-",
            _format_int(summary.batch_size) or "-",
            _format_float(summary.meter_loss_weight, digits=3) or "-",
            summary.meter_conditioning or "-",
            summary.model_tag,
            summary.git_branch or "-",
        ]
        for summary in summaries
    ]
    lines.append(
        _markdown_table(
            [
                "run",
                "status",
                "best_epoch",
                "best_downbeat_f1",
                "last_epoch",
                "last_downbeat_f1",
                "seed",
                "lr",
                "batch",
                "meter_w",
                "meter_cond",
                "model",
                "branch",
            ],
            overview_rows,
        )
    )
    lines.append("")

    for summary in summaries:
        lines.extend(
            [
                f"## {summary.run_name}",
                "",
                _markdown_table(
                    ["field", "value"],
                    [
                        ["path", summary.run_path],
                        ["status", summary.status],
                        ["best_epoch", _format_int(summary.best_epoch) or "-"],
                        [
                            "best_downbeat_f1",
                            _format_float(summary.best_downbeat_f1) or "-",
                        ],
                        ["best_beat_f1", _format_float(summary.best_beat_f1) or "-"],
                        ["best_val_loss", _format_float(summary.best_val_loss) or "-"],
                        ["last_epoch", _format_int(summary.last_epoch) or "-"],
                        [
                            "last_downbeat_f1",
                            _format_float(summary.last_downbeat_f1) or "-",
                        ],
                        [
                            "configured_epochs",
                            _format_int(summary.configured_epochs) or "-",
                        ],
                        ["seed", _format_int(summary.seed) or "-"],
                        ["lr", _format_float(summary.lr, digits=6) or "-"],
                        ["batch_size", _format_int(summary.batch_size) or "-"],
                        [
                            "train_samples_per_epoch",
                            _format_int(summary.train_samples_per_epoch) or "-",
                        ],
                        [
                            "segment_seconds",
                            _format_float(summary.segment_seconds, digits=1) or "-",
                        ],
                        [
                            "meter_loss_weight",
                            _format_float(summary.meter_loss_weight, digits=3) or "-",
                        ],
                        ["meter_conditioning", summary.meter_conditioning or "-"],
                        ["audio_backend", summary.audio_backend or "-"],
                        ["scheduler", summary.scheduler or "-"],
                        ["ema_decay", _format_float(summary.ema_decay) or "-"],
                        ["model", summary.model_tag],
                        ["resume", summary.resume or "-"],
                        ["git_branch", summary.git_branch or "-"],
                        ["git_commit", summary.git_commit or "-"],
                        [
                            "git_dirty",
                            "-" if summary.git_dirty is None else str(summary.git_dirty).lower(),
                        ],
                    ],
                ),
                "",
            ]
        )

    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summaries = collect_summaries(args.outputs_root)
    write_csv(args.csv_path, summaries)
    write_markdown(args.markdown_path, summaries)

    print(f"runs={len(summaries)}")
    print(f"csv={args.csv_path}")
    print(f"markdown={args.markdown_path}")


if __name__ == "__main__":
    main()
