from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

try:
    import mir_eval
except ImportError:
    mir_eval = None

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from data import BeatStemDataset
from data.beat_dataset import SongEntry
from models import AudioFeatureExtractor, Backbone, BeatTranscriptionModel
from training.augmentations import apply_ranked_stem_dropout
from training.losses import BalancedSoftmaxLoss, ShiftTolerantBCELoss


class ValidationSegmentDataset(Dataset):
    """
    validation ではランダム crop ではなく、曲全体を固定窓で走査して評価する。
    """

    def __init__(self, base_dataset: BeatStemDataset, semitone: Optional[int] = None):
        self.base_dataset = base_dataset
        self.segment_index: list[tuple[SongEntry, float, Optional[int]]] = []

        for song in base_dataset.songs:
            num_segments = max(
                1, math.ceil(song.effective_duration_sec / base_dataset.segment_seconds)
            )
            for segment_idx in range(num_segments):
                start_sec = segment_idx * base_dataset.segment_seconds
                self.segment_index.append((song, start_sec, semitone))

    def __len__(self) -> int:
        return len(self.segment_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int | float]:
        song, start_sec, semitone = self.segment_index[index]
        return self.base_dataset.make_sample(
            song, start_sec=start_sec, semitone=semitone
        )


@dataclass
class MetricAverager:
    totals: dict[str, float] = field(default_factory=dict)
    count: int = 0

    def update(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value)
        self.count += 1

    def averages(self) -> dict[str, float]:
        if self.count == 0:
            return {key: 0.0 for key in self.totals}
        return {key: value / self.count for key, value in self.totals.items()}


class WarmupCosineScheduler:
    """
    optimizer step 単位で更新する簡易 warmup + cosine scheduler。
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0
        self._set_lr(self._lr_multiplier(step_index=0))

    def _lr_multiplier(self, step_index: int) -> float:
        if self.warmup_steps > 0 and step_index < self.warmup_steps:
            return max(1e-8, float(step_index + 1) / float(self.warmup_steps))

        if self.total_steps <= self.warmup_steps:
            return 1.0

        progress = (step_index - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _set_lr(self, multiplier: float) -> None:
        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            param_group["lr"] = base_lr * multiplier

    def step(self) -> None:
        self.step_count += 1
        self._set_lr(self._lr_multiplier(step_index=self.step_count))

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, float | int]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict: dict[str, float | int]) -> None:
        self.total_steps = int(state_dict["total_steps"])
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.min_lr_ratio = float(state_dict["min_lr_ratio"])
        self.step_count = int(state_dict["step_count"])
        self._set_lr(self._lr_multiplier(step_index=self.step_count))


class ModelEMA:
    """
    学習中モデルの指数移動平均を保持する。
    validation と checkpoint にはこちらを使えるようにする。
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(model).eval()
        self.ema_model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if not torch.is_floating_point(ema_value):
                ema_value.copy_(model_value)
                continue
            ema_value.lerp_(model_value, 1.0 - self.decay)

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "model_state_dict": self.ema_model.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.decay = float(state_dict["decay"])
        self.ema_model.load_state_dict(state_dict["model_state_dict"])


@dataclass(frozen=True)
class ResumeState:
    checkpoint_epoch: int
    start_epoch: int
    global_step: int
    best_downbeat_f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Beat / downbeat transcription model training"
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("dataset/meter_dataset")
    )
    parser.add_argument(
        "--audio-backend",
        type=str,
        choices=("wav", "packed"),
        default="packed",
    )
    parser.add_argument(
        "--packed-audio-dir",
        type=Path,
        default=None,
        help="audio_backend=packed のときに使う packed 音声ディレクトリ。未指定なら <dataset-root>/songs_packed。",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/beat_transcription")
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="checkpoint から optimizer / scheduler / EMA を含めて学習再開する。",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="学習初期値として使う checkpoint。resume と違って optimizer は復元しない。",
    )
    parser.add_argument(
        "--init-scope",
        type=str,
        choices=("backbone", "matching"),
        default="backbone",
        help="init-from でどの重みまで読むか。既定は backbone のみ。",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--disable-persistent-workers", action="store_true")
    parser.add_argument("--disable-file-handle-cache", action="store_true")
    parser.add_argument("--max-open-files", type=int, default=64)
    parser.add_argument("--train-samples-per-epoch", type=int, default=1024)
    parser.add_argument("--segment-seconds", type=float, default=30.0)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=441)
    parser.add_argument("--bins-per-octave", type=int, default=36)
    parser.add_argument("--n-bins", type=int, default=252)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--beat-pos-weight", type=float, default=5.0)
    parser.add_argument("--downbeat-pos-weight", type=float, default=20.0)
    parser.add_argument("--meter-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--meter-balanced-softmax-tau",
        type=float,
        default=0.5,
        help="meter 用 BalancedSoftmaxLoss の tau。今回の実験では 0.5 を既定にする。",
    )
    parser.add_argument(
        "--stem-dropout-max-count",
        type=int,
        default=4,
        help="train 時にエネルギーの小さい stem から最大何本まで落とすか。0 で無効。",
    )
    parser.add_argument("--loss-tolerance", type=int, default=1)
    parser.add_argument("--metric-tolerance-sec", type=float, default=0.07)
    parser.add_argument("--mir-eval-trim-beats-before-sec", type=float, default=5.0)
    parser.add_argument("--beat-threshold", type=float, default=0.5)
    parser.add_argument("--downbeat-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--disable-random-pitch-shift", action="store_true")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=("none", "warmup_cosine"),
        default="warmup_cosine",
    )
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--disable-ema", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_git_metadata(project_root: Path) -> dict[str, object]:
    def run_git_command(*args: str) -> Optional[str]:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None

        output = completed.stdout.strip()
        return output or None

    commit = run_git_command("rev-parse", "HEAD")
    if commit is None:
        return {}

    branch = run_git_command("branch", "--show-current")
    dirty_output = run_git_command("status", "--short")
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": bool(dirty_output),
    }


def _extract_model_state_dict(
    checkpoint: object,
) -> tuple[dict[str, torch.Tensor], str]:
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint does not contain a valid model state_dict")

    # 学習時に EMA を持っている checkpoint は、初期値としてはこちらを優先する。
    if "ema_state_dict" in checkpoint:
        ema_state = checkpoint["ema_state_dict"]
        if isinstance(ema_state, dict) and "model_state_dict" in ema_state:
            state_dict = ema_state["model_state_dict"]
        else:
            state_dict = ema_state

        if isinstance(state_dict, dict):
            return state_dict, "ema_state_dict"

    if "model_state_dict" in checkpoint and isinstance(
        checkpoint["model_state_dict"], dict
    ):
        return checkpoint["model_state_dict"], "model_state_dict"

    return checkpoint, "raw_checkpoint"


def initialize_model_from_checkpoint(
    model: BeatTranscriptionModel,
    checkpoint_path: Path,
    init_scope: str,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state, state_source = _extract_model_state_dict(checkpoint)
    target_state = model.state_dict()

    filtered_state: dict[str, torch.Tensor] = {}
    skipped_shape_keys: list[str] = []

    for key, value in source_state.items():
        if init_scope == "backbone" and not key.startswith("backbone."):
            continue
        if key not in target_state:
            continue
        if target_state[key].shape != value.shape:
            skipped_shape_keys.append(key)
            continue
        filtered_state[key] = value

    if not filtered_state:
        raise ValueError(
            f"No compatible parameters were found in {checkpoint_path} for init_scope={init_scope}"
        )

    load_result = model.load_state_dict(filtered_state, strict=False)
    return {
        "state_source": state_source,
        "loaded_keys": len(filtered_state),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "skipped_shape_keys": skipped_shape_keys,
    }


def build_dataloaders(
    args: argparse.Namespace,
) -> tuple[BeatStemDataset, DataLoader, DataLoader]:
    if args.num_workers > 0 and args.prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive when num_workers > 0")

    train_dataset = BeatStemDataset(
        dataset_root=args.dataset_root,
        split="train",
        segment_seconds=args.segment_seconds,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        samples_per_epoch=args.train_samples_per_epoch,
        random_pitch_shift=not args.disable_random_pitch_shift,
        audio_backend=args.audio_backend,
        packed_audio_dir=args.packed_audio_dir,
        use_file_handle_cache=not args.disable_file_handle_cache,
        max_open_files=args.max_open_files,
    )
    val_base_dataset = BeatStemDataset(
        dataset_root=args.dataset_root,
        split="val",
        segment_seconds=args.segment_seconds,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        random_pitch_shift=False,
        audio_backend=args.audio_backend,
        packed_audio_dir=args.packed_audio_dir,
        meter_to_index=train_dataset.meter_to_index,
        use_file_handle_cache=not args.disable_file_handle_cache,
        max_open_files=args.max_open_files,
    )
    val_dataset = ValidationSegmentDataset(val_base_dataset, semitone=None)

    loader_kwargs = {
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": False,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = not args.disable_persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        **loader_kwargs,
    )
    return train_dataset, train_loader, val_loader


def build_model(
    args: argparse.Namespace, train_dataset: BeatStemDataset
) -> BeatTranscriptionModel:
    feature_extractor = AudioFeatureExtractor(
        sampling_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        num_audio_channels=train_dataset.num_audio_channels,
        num_stems=len(train_dataset.stem_names),
        bins_per_octave=args.bins_per_octave,
        n_bins=args.n_bins,
        spec_augment_params=None,
    )
    backbone = Backbone(
        feature_extractor=feature_extractor,
        hidden_size=args.hidden_size,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_gradient_checkpoint=True,
    )
    return BeatTranscriptionModel(
        backbone=backbone,
        num_meter_classes=train_dataset.num_meter_classes,
        head_dropout=args.head_dropout,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    steps_per_epoch: int,
) -> Optional[WarmupCosineScheduler]:
    if args.scheduler == "none":
        return None

    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(round(steps_per_epoch * args.warmup_epochs))
    return WarmupCosineScheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )


def build_model_ema(
    model: BeatTranscriptionModel,
    args: argparse.Namespace,
) -> Optional[ModelEMA]:
    if args.disable_ema:
        return None
    if not (0.0 < args.ema_decay < 1.0):
        return None
    return ModelEMA(model=model, decay=args.ema_decay)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def write_scalar_metrics(
    writer: Optional[SummaryWriter],
    prefix: str,
    metrics: dict[str, float],
    step: int,
) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, step)


def append_history_entry(
    history_path: Path, epoch: int, train: dict, val: dict
) -> None:
    history_entry = {"epoch": epoch, "train": train, "val": val}
    with history_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(history_entry) + "\n")


def load_history_entries(history_path: Path) -> list[dict[str, object]]:
    if not history_path.exists():
        return []

    entries: list[dict[str, object]] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entries.append(json.loads(line))
    return entries


def rewrite_history_entries(
    history_path: Path, history_entries: list[dict[str, object]]
) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(json.dumps(entry) + "\n" for entry in history_entries)
    history_path.write_text(serialized, encoding="utf-8")


def trim_history_for_resume(
    history_path: Path, checkpoint_epoch: int
) -> list[dict[str, object]]:
    history_entries = load_history_entries(history_path)
    trimmed_entries = [
        entry
        for entry in history_entries
        if int(entry.get("epoch", 0)) <= checkpoint_epoch
    ]
    if len(trimmed_entries) != len(history_entries):
        rewrite_history_entries(history_path, trimmed_entries)
    return trimmed_entries


def infer_best_downbeat_f1(
    history_entries: list[dict[str, object]],
    checkpoint_epoch: int,
) -> Optional[float]:
    best_score: Optional[float] = None
    for entry in history_entries:
        entry_epoch = int(entry.get("epoch", 0))
        if entry_epoch > checkpoint_epoch:
            continue

        val_metrics = entry.get("val")
        if not isinstance(val_metrics, dict):
            continue
        if "downbeat_f1" not in val_metrics:
            continue

        score = float(val_metrics["downbeat_f1"])
        best_score = score if best_score is None else max(best_score, score)
    return best_score


def load_resume_state(
    checkpoint_path: Path,
    model: BeatTranscriptionModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[WarmupCosineScheduler],
    ema: Optional[ModelEMA],
    scaler: torch.amp.GradScaler,
    history_path: Path,
    steps_per_epoch: int,
) -> ResumeState:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    ema_state = checkpoint.get("ema_state_dict")
    if ema is not None:
        if ema_state is not None:
            ema.load_state_dict(ema_state)
        else:
            ema.ema_model.load_state_dict(model.state_dict())

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler.is_enabled() and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    checkpoint_epoch = int(checkpoint["epoch"])
    history_entries = trim_history_for_resume(history_path, checkpoint_epoch)

    best_downbeat_f1 = checkpoint.get("best_downbeat_f1")
    if best_downbeat_f1 is None:
        best_downbeat_f1 = infer_best_downbeat_f1(history_entries, checkpoint_epoch)

    if best_downbeat_f1 is None:
        checkpoint_history_path = checkpoint_path.parent / "history.jsonl"
        if checkpoint_history_path != history_path:
            best_downbeat_f1 = infer_best_downbeat_f1(
                load_history_entries(checkpoint_history_path), checkpoint_epoch
            )

    if best_downbeat_f1 is None:
        checkpoint_metrics = checkpoint.get("metrics")
        if isinstance(checkpoint_metrics, dict) and "downbeat_f1" in checkpoint_metrics:
            best_downbeat_f1 = float(checkpoint_metrics["downbeat_f1"])
        else:
            best_downbeat_f1 = -1.0

    global_step = checkpoint.get("global_step")
    if global_step is None:
        global_step = checkpoint_epoch * steps_per_epoch

    return ResumeState(
        checkpoint_epoch=checkpoint_epoch,
        start_epoch=checkpoint_epoch + 1,
        global_step=int(global_step),
        best_downbeat_f1=float(best_downbeat_f1),
    )


def compute_loss(
    output,
    batch: dict,
    beat_loss_fn: ShiftTolerantBCELoss,
    downbeat_loss_fn: ShiftTolerantBCELoss,
    meter_loss_fn: BalancedSoftmaxLoss,
    meter_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if output.meter_logits is None:
        raise ValueError("Model output must include meter_logits")

    # beat / downbeat は従来どおり BCE 系の loss を使う。
    beat_loss = beat_loss_fn(
        output.beat_logits, batch["beat_targets"], batch["valid_mask"]
    )
    downbeat_loss = downbeat_loss_fn(
        output.downbeat_logits, batch["downbeat_targets"], batch["valid_mask"]
    )

    # meter は小節全体のフレームへ貼った多クラス分類として扱う。
    # 末尾の無効フレームは ignore_index にして loss から外す。
    raw_meter_loss = meter_loss_fn(output.meter_logits, batch["meter_targets"])
    meter_loss = raw_meter_loss * meter_loss_weight
    valid_meter = batch["meter_targets"] != meter_loss_fn.ignore_index
    if bool(valid_meter.any()):
        meter_predictions = output.meter_logits.argmax(dim=-1)
        meter_accuracy = float(
            (meter_predictions[valid_meter] == batch["meter_targets"][valid_meter])
            .float()
            .mean()
            .detach()
        )
    else:
        meter_accuracy = 0.0

    # 今回は重みづけを増やさず、3 タスクの和をそのまま最適化する。
    total_loss = beat_loss + downbeat_loss + meter_loss
    return total_loss, {
        "loss": float(total_loss.detach()),
        "beat_loss": float(beat_loss.detach()),
        "downbeat_loss": float(downbeat_loss.detach()),
        "meter_loss": float(meter_loss.detach()),
        "raw_meter_loss": float(raw_meter_loss.detach()),
        "meter_accuracy": meter_accuracy,
    }


def pick_peak_indices(probabilities: torch.Tensor, threshold: float) -> list[int]:
    if probabilities.numel() == 0:
        return []

    active = probabilities >= threshold
    if not bool(active.any()):
        return []

    peak_indices: list[int] = []
    region_start: Optional[int] = None
    active_list = active.tolist()

    for idx, is_active in enumerate(active_list):
        if is_active and region_start is None:
            region_start = idx
        elif not is_active and region_start is not None:
            region = probabilities[region_start:idx]
            peak_indices.append(region_start + int(region.argmax().item()))
            region_start = None

    if region_start is not None:
        region = probabilities[region_start:]
        peak_indices.append(region_start + int(region.argmax().item()))

    return peak_indices


def frame_indices_to_times(
    frame_indices: list[int],
    start_sec: float,
    sample_rate: int,
    hop_length: int,
) -> list[float]:
    frame_duration_sec = hop_length / sample_rate
    return [
        start_sec + (frame_index * frame_duration_sec) for frame_index in frame_indices
    ]


def merge_close_events(event_times: list[float], min_interval_sec: float) -> np.ndarray:
    if not event_times:
        return np.empty(0, dtype=float)

    sorted_events = np.asarray(sorted(event_times), dtype=float)
    merged = [float(sorted_events[0])]
    for event_time in sorted_events[1:]:
        if event_time - merged[-1] >= min_interval_sec:
            merged.append(float(event_time))
    return np.asarray(merged, dtype=float)


def compute_mir_eval_scores(
    reference_events: np.ndarray,
    estimated_events: np.ndarray,
    tolerance_sec: float,
    trim_beats_before_sec: float,
) -> dict[str, float]:
    if trim_beats_before_sec > 0:
        reference_events = mir_eval.beat.trim_beats(
            reference_events, min_beat_time=trim_beats_before_sec
        )
        estimated_events = mir_eval.beat.trim_beats(
            estimated_events, min_beat_time=trim_beats_before_sec
        )

    if len(reference_events) == 0 and len(estimated_events) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if len(reference_events) == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if len(estimated_events) == 0:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

    matches = mir_eval.util.match_events(
        reference_events, estimated_events, tolerance_sec
    )
    true_positives = len(matches)
    precision = true_positives / len(estimated_events)
    recall = true_positives / len(reference_events)
    f1 = float(
        mir_eval.beat.f_measure(
            reference_events, estimated_events, f_measure_threshold=tolerance_sec
        )
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def train_one_epoch(
    model: BeatTranscriptionModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[WarmupCosineScheduler],
    ema: Optional[ModelEMA],
    scaler: torch.amp.GradScaler,
    beat_loss_fn: ShiftTolerantBCELoss,
    downbeat_loss_fn: ShiftTolerantBCELoss,
    meter_loss_fn: BalancedSoftmaxLoss,
    meter_loss_weight: float,
    num_stems: int,
    stem_dropout_max_count: int,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
    epoch: int,
    writer: Optional[SummaryWriter],
    global_step: int,
    log_interval: int,
) -> tuple[dict[str, float], int]:
    model.train()
    tracker = MetricAverager()
    progress = tqdm(
        train_loader,
        desc=f"Train {epoch}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch_index, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        # 低エネルギー stem から順に落とし、1 stem は必ず残す。
        dropped_stem_count = 0.0
        if stem_dropout_max_count > 0:
            batch["waveform"], dropped_counts = apply_ranked_stem_dropout(
                batch["waveform"],
                num_stems=num_stems,
                max_dropout_stems=stem_dropout_max_count,
            )
            dropped_stem_count = float(dropped_counts.float().mean().item())

        with (
            torch.autocast(device_type=device.type, dtype=torch.float16)
            if use_amp and device.type == "cuda"
            else nullcontext()
        ):
            output = model(batch["waveform"])
            loss, loss_info = compute_loss(
                output,
                batch,
                beat_loss_fn,
                downbeat_loss_fn,
                meter_loss_fn,
                meter_loss_weight,
            )
            loss_info["stem_dropout_count"] = dropped_stem_count

        if scaler.is_enabled():
            previous_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= previous_scale
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer_stepped = True

        if optimizer_stepped:
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)

        tracker.update(loss_info)
        global_step += 1

        if writer is not None:
            writer.add_scalar("train_step/loss", loss_info["loss"], global_step)
            writer.add_scalar(
                "train_step/beat_loss", loss_info["beat_loss"], global_step
            )
            writer.add_scalar(
                "train_step/downbeat_loss", loss_info["downbeat_loss"], global_step
            )
            writer.add_scalar(
                "train_step/meter_loss", loss_info["meter_loss"], global_step
            )
            writer.add_scalar(
                "train_step/meter_accuracy",
                loss_info["meter_accuracy"],
                global_step,
            )
            writer.add_scalar(
                "train_step/raw_meter_loss",
                loss_info["raw_meter_loss"],
                global_step,
            )
            writer.add_scalar(
                "train_step/stem_dropout_count",
                loss_info["stem_dropout_count"],
                global_step,
            )
            writer.add_scalar(
                "train_step/lr", optimizer.param_groups[0]["lr"], global_step
            )

        if batch_index == 1 or batch_index % max(1, log_interval) == 0:
            progress.set_postfix(
                loss=f"{loss_info['loss']:.4f}",
                beat=f"{loss_info['beat_loss']:.4f}",
                downbeat=f"{loss_info['downbeat_loss']:.4f}",
                meter=f"{loss_info['meter_loss']:.4f}",
            )

    return tracker.averages(), global_step


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    beat_loss_fn: ShiftTolerantBCELoss,
    downbeat_loss_fn: ShiftTolerantBCELoss,
    meter_loss_fn: BalancedSoftmaxLoss,
    meter_loss_weight: float,
    device: torch.device,
    beat_threshold: float,
    downbeat_threshold: float,
    tolerance_sec: float,
    trim_beats_before_sec: float,
    sample_rate: int,
    hop_length: int,
    epoch: int,
) -> dict[str, float]:
    if mir_eval is None:
        raise ImportError(
            "mir_eval is required for validation metrics. Install it with "
            "`uv pip install --python .venv/bin/python https://github.com/mir-evaluation/mir_eval/archive/main.zip`."
        )

    model.eval()
    tracker = MetricAverager()
    song_events: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "ref_beats": [],
            "est_beats": [],
            "ref_downbeats": [],
            "est_downbeats": [],
        }
    )

    progress = tqdm(
        val_loader,
        desc=f"Val {epoch}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in progress:
        batch = move_batch_to_device(batch, device)
        output = model(batch["waveform"])
        _, loss_info = compute_loss(
            output,
            batch,
            beat_loss_fn,
            downbeat_loss_fn,
            meter_loss_fn,
            meter_loss_weight,
        )

        tracker.update(loss_info)
        progress.set_postfix(loss=f"{loss_info['loss']:.4f}")

        beat_probabilities = torch.sigmoid(output.beat_logits.detach()).cpu()
        downbeat_probabilities = torch.sigmoid(output.downbeat_logits.detach()).cpu()
        beat_targets = batch["beat_targets"].detach().cpu()
        downbeat_targets = batch["downbeat_targets"].detach().cpu()
        valid_mask = batch["valid_mask"].detach().cpu()
        start_seconds = batch["start_sec"].detach().cpu().tolist()
        song_ids = batch["song_id"]

        for batch_idx, song_id in enumerate(song_ids):
            valid_frames = int(valid_mask[batch_idx].sum().item())
            start_sec = float(start_seconds[batch_idx])

            beat_pred_indices = pick_peak_indices(
                beat_probabilities[batch_idx, :valid_frames], beat_threshold
            )
            downbeat_pred_indices = pick_peak_indices(
                downbeat_probabilities[batch_idx, :valid_frames],
                downbeat_threshold,
            )
            beat_target_indices = (
                torch.nonzero(
                    beat_targets[batch_idx, :valid_frames] > 0.5, as_tuple=False
                )
                .flatten()
                .tolist()
            )
            downbeat_target_indices = (
                torch.nonzero(
                    downbeat_targets[batch_idx, :valid_frames] > 0.5, as_tuple=False
                )
                .flatten()
                .tolist()
            )

            song_events[song_id]["est_beats"].extend(
                frame_indices_to_times(
                    beat_pred_indices, start_sec, sample_rate, hop_length
                )
            )
            song_events[song_id]["est_downbeats"].extend(
                frame_indices_to_times(
                    downbeat_pred_indices, start_sec, sample_rate, hop_length
                )
            )
            song_events[song_id]["ref_beats"].extend(
                frame_indices_to_times(
                    beat_target_indices, start_sec, sample_rate, hop_length
                )
            )
            song_events[song_id]["ref_downbeats"].extend(
                frame_indices_to_times(
                    downbeat_target_indices, start_sec, sample_rate, hop_length
                )
            )

    running = tracker.averages()

    beat_precisions: list[float] = []
    beat_recalls: list[float] = []
    beat_f1s: list[float] = []
    downbeat_precisions: list[float] = []
    downbeat_recalls: list[float] = []
    downbeat_f1s: list[float] = []
    dedupe_interval_sec = (hop_length / sample_rate) * 0.5

    for events in song_events.values():
        beat_scores = compute_mir_eval_scores(
            reference_events=merge_close_events(
                events["ref_beats"], dedupe_interval_sec
            ),
            estimated_events=merge_close_events(
                events["est_beats"], dedupe_interval_sec
            ),
            tolerance_sec=tolerance_sec,
            trim_beats_before_sec=trim_beats_before_sec,
        )
        downbeat_scores = compute_mir_eval_scores(
            reference_events=merge_close_events(
                events["ref_downbeats"], dedupe_interval_sec
            ),
            estimated_events=merge_close_events(
                events["est_downbeats"], dedupe_interval_sec
            ),
            tolerance_sec=tolerance_sec,
            trim_beats_before_sec=trim_beats_before_sec,
        )

        beat_precisions.append(beat_scores["precision"])
        beat_recalls.append(beat_scores["recall"])
        beat_f1s.append(beat_scores["f1"])
        downbeat_precisions.append(downbeat_scores["precision"])
        downbeat_recalls.append(downbeat_scores["recall"])
        downbeat_f1s.append(downbeat_scores["f1"])

    running.update(
        {
            "beat_precision": float(np.mean(beat_precisions))
            if beat_precisions
            else 0.0,
            "beat_recall": float(np.mean(beat_recalls)) if beat_recalls else 0.0,
            "beat_f1": float(np.mean(beat_f1s)) if beat_f1s else 0.0,
            "downbeat_precision": float(np.mean(downbeat_precisions))
            if downbeat_precisions
            else 0.0,
            "downbeat_recall": float(np.mean(downbeat_recalls))
            if downbeat_recalls
            else 0.0,
            "downbeat_f1": float(np.mean(downbeat_f1s)) if downbeat_f1s else 0.0,
        }
    )
    return running


def save_checkpoint(
    checkpoint_path: Path,
    model: BeatTranscriptionModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[WarmupCosineScheduler],
    ema: Optional[ModelEMA],
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_downbeat_f1: float,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None
            if scheduler is None
            else scheduler.state_dict(),
            "ema_state_dict": None if ema is None else ema.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": epoch,
            "global_step": global_step,
            "best_downbeat_f1": best_downbeat_f1,
            "metrics": metrics,
            "args": vars(args),
        },
        checkpoint_path,
    )


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    ordered_keys = [
        "loss",
        "beat_loss",
        "downbeat_loss",
        "meter_loss",
        "raw_meter_loss",
        "meter_accuracy",
        "stem_dropout_count",
        "beat_precision",
        "beat_recall",
        "beat_f1",
        "downbeat_precision",
        "downbeat_recall",
        "downbeat_f1",
    ]
    values = [f"{key}={metrics[key]:.4f}" for key in ordered_keys if key in metrics]
    return f"{prefix}: " + ", ".join(values)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    project_root = Path(__file__).resolve().parent.parent

    if args.resume is not None and args.init_from is not None:
        raise ValueError("--resume and --init-from cannot be used together")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = args.tensorboard_dir or (args.output_dir / "tensorboard")
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, train_loader, val_loader = build_dataloaders(args)
    model = build_model(args, train_dataset).to(device)
    init_info: Optional[dict[str, object]] = None
    if args.init_from is not None:
        # 和音採譜モデルなど別 task の checkpoint は、まず backbone だけ読むのが安全。
        init_info = initialize_model_from_checkpoint(
            model=model,
            checkpoint_path=args.init_from,
            init_scope=args.init_scope,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(
        optimizer=optimizer, args=args, steps_per_epoch=len(train_loader)
    )
    ema = build_model_ema(model=model, args=args)
    scaler = torch.amp.GradScaler(
        device=device.type, enabled=args.use_amp and device.type == "cuda"
    )
    beat_loss_fn = ShiftTolerantBCELoss(
        pos_weight=args.beat_pos_weight, tolerance=args.loss_tolerance
    ).to(device)
    downbeat_loss_fn = ShiftTolerantBCELoss(
        pos_weight=args.downbeat_pos_weight, tolerance=args.loss_tolerance
    ).to(device)
    meter_loss_fn = BalancedSoftmaxLoss(
        class_counts=train_dataset.meter_class_counts,
        tau=args.meter_balanced_softmax_tau,
        ignore_index=train_dataset.meter_ignore_index,
    ).to(device)

    history_path = args.output_dir / "history.jsonl"
    best_downbeat_f1 = -1.0
    global_step = 0
    start_epoch = 1

    if args.resume is not None:
        resume_state = load_resume_state(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            scaler=scaler,
            history_path=history_path,
            steps_per_epoch=len(train_loader),
        )
        best_downbeat_f1 = resume_state.best_downbeat_f1
        global_step = resume_state.global_step
        start_epoch = resume_state.start_epoch
        if start_epoch > args.epochs:
            raise ValueError(
                f"Resume checkpoint epoch={resume_state.checkpoint_epoch} already reaches "
                f"--epochs={args.epochs}. Increase --epochs to continue training."
            )

    config_path = args.output_dir / "config.json"
    config_payload = dict(vars(args))
    config_payload.update(collect_git_metadata(project_root))
    config_payload.update(
        {
            "meter_labels": list(train_dataset.meter_labels),
            "meter_class_counts": train_dataset.meter_class_counts.tolist(),
            "init_state_source": None
            if init_info is None
            else init_info["state_source"],
            "init_loaded_keys": None if init_info is None else init_info["loaded_keys"],
            "init_skipped_shape_keys": []
            if init_info is None
            else init_info["skipped_shape_keys"],
        }
    )
    config_text = json.dumps(config_payload, indent=2, default=str)
    config_path.write_text(config_text, encoding="utf-8")
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_text("config", f"```json\n{config_text}\n```")

    print(f"device={device}")
    print(f"train_songs={len(train_dataset.songs)}, train_batches={len(train_loader)}")
    print(f"val_segments={len(val_loader.dataset)}, val_batches={len(val_loader)}")
    print(
        "meter="
        f"classes={train_dataset.num_meter_classes}, "
        f"labels={list(train_dataset.meter_labels)}"
    )
    print(f"metric_tolerance_sec={args.metric_tolerance_sec}")
    print(f"meter_loss_weight={args.meter_loss_weight}")
    print(f"meter_balanced_softmax_tau={args.meter_balanced_softmax_tau}")
    print(f"stem_dropout_max_count={args.stem_dropout_max_count}")
    print(f"tensorboard_dir={tensorboard_dir}")
    print(f"scheduler={args.scheduler}")
    print(f"ema={'disabled' if ema is None else f'decay={ema.decay}'}")
    if args.init_from is not None:
        print(
            "init="
            f"path={args.init_from}, scope={args.init_scope}, "
            f"source={init_info['state_source'] if init_info is not None else '-'}, "
            f"loaded_keys={init_info['loaded_keys'] if init_info is not None else 0}, "
            f"skipped_shape_keys={len(init_info['skipped_shape_keys']) if init_info is not None else 0}"
        )
    if args.resume is not None:
        print(
            "resume="
            f"path={args.resume}, start_epoch={start_epoch}, "
            f"global_step={global_step}, best_downbeat_f1={best_downbeat_f1:.4f}"
        )
    print(
        "audio="
        f"backend={args.audio_backend}, "
        f"packed_audio_dir={args.packed_audio_dir or (args.dataset_root / 'songs_packed')}"
    )
    print(
        "dataloader="
        f"num_workers={args.num_workers}, "
        f"persistent_workers={args.num_workers > 0 and not args.disable_persistent_workers}, "
        f"prefetch_factor={args.prefetch_factor if args.num_workers > 0 else 'n/a'}, "
        f"file_handle_cache={not args.disable_file_handle_cache}, "
        f"max_open_files={args.max_open_files}"
    )

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics, global_step = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                scaler=scaler,
                beat_loss_fn=beat_loss_fn,
                downbeat_loss_fn=downbeat_loss_fn,
                meter_loss_fn=meter_loss_fn,
                meter_loss_weight=args.meter_loss_weight,
                num_stems=len(train_dataset.stem_names),
                stem_dropout_max_count=args.stem_dropout_max_count,
                device=device,
                use_amp=args.use_amp,
                grad_clip=args.grad_clip,
                epoch=epoch,
                writer=writer,
                global_step=global_step,
                log_interval=args.log_interval,
            )
            eval_model = ema.ema_model if ema is not None else model
            val_metrics = validate(
                model=eval_model,
                val_loader=val_loader,
                beat_loss_fn=beat_loss_fn,
                downbeat_loss_fn=downbeat_loss_fn,
                meter_loss_fn=meter_loss_fn,
                meter_loss_weight=args.meter_loss_weight,
                device=device,
                beat_threshold=args.beat_threshold,
                downbeat_threshold=args.downbeat_threshold,
                tolerance_sec=args.metric_tolerance_sec,
                trim_beats_before_sec=args.mir_eval_trim_beats_before_sec,
                sample_rate=args.sample_rate,
                hop_length=args.hop_length,
                epoch=epoch,
            )

            print(f"Epoch {epoch}/{args.epochs}")
            print(format_metrics("  train", train_metrics))
            print(format_metrics("  val", val_metrics))

            write_scalar_metrics(writer, "train_epoch", train_metrics, epoch)
            write_scalar_metrics(writer, "val", val_metrics, epoch)
            if scheduler is not None:
                writer.add_scalar(
                    "train_epoch/lr", optimizer.param_groups[0]["lr"], epoch
                )
            writer.flush()

            append_history_entry(history_path, epoch, train_metrics, val_metrics)

            best_downbeat_f1 = max(best_downbeat_f1, val_metrics["downbeat_f1"])

            save_checkpoint(
                args.output_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                ema,
                scaler,
                epoch,
                global_step,
                best_downbeat_f1,
                val_metrics,
                args,
            )

            if val_metrics["downbeat_f1"] >= best_downbeat_f1:
                save_checkpoint(
                    args.output_dir / "best_downbeat_f1.pt",
                    model,
                    optimizer,
                    scheduler,
                    ema,
                    scaler,
                    epoch,
                    global_step,
                    best_downbeat_f1,
                    val_metrics,
                    args,
                )

            if args.save_every_epoch:
                save_checkpoint(
                    args.output_dir / f"epoch_{epoch:03d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    ema,
                    scaler,
                    epoch,
                    global_step,
                    best_downbeat_f1,
                    val_metrics,
                    args,
                )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
