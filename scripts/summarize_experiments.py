from __future__ import annotations

import argparse
import csv
import html
import json
import math
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
    drum_aux_loss_weight: Optional[float]
    drum_aux_use_high_frequency_flux: Optional[bool]
    repeat_consistency_loss_weight: Optional[float]
    repeat_ssm_threshold: Optional[float]
    repeat_ssm_min_length_beats: Optional[int]
    repeat_ssm_near_diagonal_margin_beats: Optional[int]
    repeat_ssm_max_length_beats: Optional[int]
    stem_dropout_max_count: Optional[int]
    init_scope: Optional[str]
    init_from: Optional[str]
    init_state_source: Optional[str]
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
    completed_timestamp: Optional[float]

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
            "drum_aux_loss_weight": _format_float(
                self.drum_aux_loss_weight, digits=3
            ),
            "drum_aux_use_high_frequency_flux": ""
            if self.drum_aux_use_high_frequency_flux is None
            else str(self.drum_aux_use_high_frequency_flux).lower(),
            "repeat_consistency_loss_weight": _format_float(
                self.repeat_consistency_loss_weight, digits=3
            ),
            "repeat_ssm_threshold": _format_float(
                self.repeat_ssm_threshold, digits=3
            ),
            "repeat_ssm_min_length_beats": _format_int(
                self.repeat_ssm_min_length_beats
            ),
            "repeat_ssm_near_diagonal_margin_beats": _format_int(
                self.repeat_ssm_near_diagonal_margin_beats
            ),
            "repeat_ssm_max_length_beats": _format_int(
                self.repeat_ssm_max_length_beats
            ),
            "stem_dropout_max_count": _format_int(self.stem_dropout_max_count),
            "init_scope": self.init_scope or "",
            "init_from": self.init_from or "",
            "init_state_source": self.init_state_source or "",
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
            "completed_at": _format_timestamp(self.completed_timestamp),
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


def _format_timestamp(value: Optional[float]) -> str:
    if value is None:
        return ""
    return datetime.fromtimestamp(value, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


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
    completed_timestamp = max(
        (path.stat().st_mtime for path in (config_path, history_path) if path.exists()),
        default=None,
    )

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
        drum_aux_loss_weight=_as_float(config.get("drum_aux_loss_weight")),
        drum_aux_use_high_frequency_flux=_as_bool(
            config.get("drum_aux_use_high_frequency_flux")
        ),
        repeat_consistency_loss_weight=_as_float(
            config.get("repeat_consistency_loss_weight")
        ),
        repeat_ssm_threshold=_as_float(config.get("repeat_ssm_threshold")),
        repeat_ssm_min_length_beats=_as_int(
            config.get("repeat_ssm_min_length_beats")
        ),
        repeat_ssm_near_diagonal_margin_beats=_as_int(
            config.get("repeat_ssm_near_diagonal_margin_beats")
        ),
        repeat_ssm_max_length_beats=_as_int(
            config.get("repeat_ssm_max_length_beats")
        ),
        stem_dropout_max_count=_as_int(config.get("stem_dropout_max_count")),
        init_scope=str(config["init_scope"]) if config.get("init_scope") else None,
        init_from=str(config["init_from"]) if config.get("init_from") else None,
        init_state_source=str(config["init_state_source"])
        if config.get("init_state_source")
        else None,
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
        completed_timestamp=completed_timestamp,
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
        "drum_aux_loss_weight",
        "drum_aux_use_high_frequency_flux",
        "repeat_consistency_loss_weight",
        "repeat_ssm_threshold",
        "repeat_ssm_min_length_beats",
        "repeat_ssm_near_diagonal_margin_beats",
        "repeat_ssm_max_length_beats",
        "stem_dropout_max_count",
        "init_scope",
        "init_from",
        "init_state_source",
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
        "completed_at",
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


def _completion_order(summaries: list[ExperimentSummary]) -> list[ExperimentSummary]:
    return sorted(
        summaries,
        key=lambda item: (
            item.completed_timestamp is None,
            item.completed_timestamp or float("inf"),
            item.run_name,
        ),
    )


def _write_progress_svg(svg_path: Path, summaries: list[ExperimentSummary]) -> None:
    ordered = [
        summary
        for summary in _completion_order(summaries)
        if summary.best_downbeat_f1 is not None
    ]
    if not ordered:
        return

    svg_path.parent.mkdir(parents=True, exist_ok=True)

    width = max(960, 140 + len(ordered) * 56)
    height = 520
    left = 72
    right = 24
    top = 28
    bottom = 170
    plot_width = width - left - right
    plot_height = height - top - bottom

    scores = [float(summary.best_downbeat_f1) for summary in ordered]
    y_min = max(0.0, math.floor((min(scores) - 0.02) * 20.0) / 20.0)
    y_max = min(1.0, math.ceil((max(scores) + 0.02) * 20.0) / 20.0)
    if y_max - y_min < 0.1:
        y_max = min(1.0, y_min + 0.1)

    def x_pos(index: int) -> float:
        if len(ordered) == 1:
            return left + plot_width / 2.0
        return left + (plot_width * index / (len(ordered) - 1))

    def y_pos(score: float) -> float:
        ratio = (score - y_min) / max(1e-8, (y_max - y_min))
        return top + plot_height - (ratio * plot_height)

    grid_lines: list[str] = []
    for step in range(6):
        value = y_min + (y_max - y_min) * step / 5.0
        y = y_pos(value)
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" '
            'stroke="#d7dce2" stroke-width="1"/>'
        )
        grid_lines.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            'font-size="12" fill="#425466">'
            f"{value:.3f}</text>"
        )

    bars: list[str] = []
    running_line_points: list[str] = []
    running_best = -1.0
    completion_rows: list[list[str]] = []

    for index, summary in enumerate(ordered, start=1):
        assert summary.best_downbeat_f1 is not None
        score = float(summary.best_downbeat_f1)
        running_best = max(running_best, score)
        x = x_pos(index - 1)
        y = y_pos(score)
        bar_top = y
        bar_height = top + plot_height - bar_top
        bars.append(
            "\n".join(
                [
                    f'<g>',
                    f'<title>{html.escape(summary.run_name)}&#10;best_downbeat_f1={score:.4f}'
                    f'&#10;completed_at~{html.escape(_format_timestamp(summary.completed_timestamp) or "-")}</title>',
                    f'<rect x="{x - 12:.1f}" y="{bar_top:.1f}" width="24" height="{bar_height:.1f}" '
                    'fill="#8fb8de" opacity="0.85"/>',
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#1f5f99"/>',
                    f'<text x="{x:.1f}" y="{top + plot_height + 18:.1f}" text-anchor="middle" '
                    'font-size="11" fill="#425466">'
                    f"{index}</text>",
                    f'<text x="{x - 4:.1f}" y="{top + plot_height + 32:.1f}" '
                    f'transform="rotate(60 {x - 4:.1f},{top + plot_height + 32:.1f})" '
                    'font-size="11" fill="#425466">'
                    f"{html.escape(summary.run_name)}</text>",
                    "</g>",
                ]
            )
        )
        running_line_points.append(f"{x:.1f},{y_pos(running_best):.1f}")
        completion_rows.append(
            [
                str(index),
                summary.run_name,
                _format_timestamp(summary.completed_timestamp) or "-",
                f"{score:.4f}",
                f"{running_best:.4f}",
            ]
        )

    path_d = " ".join(
        (
            [f"M {running_line_points[0]}"]
            + [f"L {point}" for point in running_line_points[1:]]
        )
    )
    legend_y = height - 26
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#f7f9fc"/>',
            '<text x="24" y="22" font-size="18" font-weight="700" fill="#15212b">Best Downbeat F1 Progress</text>',
            '<text x="24" y="42" font-size="12" fill="#425466">x-axis is estimated completion order from history/config file timestamps.</text>',
            *grid_lines,
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#425466" stroke-width="1.2"/>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#425466" stroke-width="1.2"/>',
            *bars,
            f'<path d="{path_d}" fill="none" stroke="#d94841" stroke-width="2.5"/>',
            f'<circle cx="{left}" cy="{legend_y - 4}" r="5" fill="#8fb8de"/>',
            f'<text x="{left + 12}" y="{legend_y}" font-size="12" fill="#425466">best_downbeat_f1 per run</text>',
            f'<line x1="{left + 170}" y1="{legend_y - 4}" x2="{left + 194}" y2="{legend_y - 4}" stroke="#d94841" stroke-width="2.5"/>',
            f'<text x="{left + 202}" y="{legend_y}" font-size="12" fill="#425466">running best</text>',
            "</svg>",
        ]
    )
    svg_path.write_text(svg, encoding="utf-8")

    table_path = svg_path.with_suffix(".csv")
    with table_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "completion_order",
                "run_name",
                "completed_at_estimated",
                "best_downbeat_f1",
                "running_best_downbeat_f1",
            ]
        )
        writer.writerows(completion_rows)


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
    progress_svg_name = "experiment_progress_downbeat.svg"
    lines.extend(
        [
            "## Progress",
            "",
            "Completion order is estimated from `history.jsonl` / `config.json` modification times.",
            "",
            f"![Downbeat Progress]({progress_svg_name})",
            "",
        ]
    )

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
            _format_float(summary.drum_aux_loss_weight, digits=3) or "-",
            "-"
            if summary.drum_aux_use_high_frequency_flux is None
            else str(summary.drum_aux_use_high_frequency_flux).lower(),
            _format_float(summary.repeat_consistency_loss_weight, digits=3) or "-",
            _format_int(summary.stem_dropout_max_count) or "-",
            (
                summary.init_scope
                if summary.init_state_source is None
                else f"{summary.init_scope}:{summary.init_state_source}"
            )
            or "-",
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
                "drum_aux_w",
                "drum_hf",
                "repeat_w",
                "stem_drop",
                "init",
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
                        [
                            "drum_aux_loss_weight",
                            _format_float(summary.drum_aux_loss_weight, digits=3)
                            or "-",
                        ],
                        [
                            "drum_aux_use_high_frequency_flux",
                            "-"
                            if summary.drum_aux_use_high_frequency_flux is None
                            else str(summary.drum_aux_use_high_frequency_flux).lower(),
                        ],
                        [
                            "repeat_consistency_loss_weight",
                            _format_float(
                                summary.repeat_consistency_loss_weight, digits=3
                            )
                            or "-",
                        ],
                        [
                            "repeat_ssm_threshold",
                            _format_float(summary.repeat_ssm_threshold, digits=3)
                            or "-",
                        ],
                        [
                            "repeat_ssm_min_length_beats",
                            _format_int(summary.repeat_ssm_min_length_beats) or "-",
                        ],
                        [
                            "repeat_ssm_near_diagonal_margin_beats",
                            _format_int(summary.repeat_ssm_near_diagonal_margin_beats)
                            or "-",
                        ],
                        [
                            "repeat_ssm_max_length_beats",
                            _format_int(summary.repeat_ssm_max_length_beats) or "-",
                        ],
                        [
                            "stem_dropout_max_count",
                            _format_int(summary.stem_dropout_max_count) or "-",
                        ],
                        ["init_scope", summary.init_scope or "-"],
                        ["init_from", summary.init_from or "-"],
                        ["init_state_source", summary.init_state_source or "-"],
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
    _write_progress_svg(args.markdown_path.with_name("experiment_progress_downbeat.svg"), summaries)

    print(f"runs={len(summaries)}")
    print(f"csv={args.csv_path}")
    print(f"markdown={args.markdown_path}")
    print(
        "progress_svg="
        f"{args.markdown_path.with_name('experiment_progress_downbeat.svg')}"
    )


if __name__ == "__main__":
    main()
