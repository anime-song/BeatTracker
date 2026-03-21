from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from models import AudioFeatureExtractor, Backbone
from pretraining.masked_segment_model import compute_masked_segment_loss
from pretraining.model import PretrainingModel
from pretraining.unlabeled_dataset import (
    UnlabeledStemDataset,
    collate_unlabeled_stem_batch,
)
from training.losses import ShiftTolerantBCELoss


@dataclass
class MetricAverager:
    """step ごとの metric を epoch 平均へ畳み込む。"""

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
    """optimizer step 単位で更新する簡易 warmup + cosine scheduler。"""

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


@dataclass(frozen=True)
class ResumeState:
    checkpoint_epoch: int
    start_epoch: int
    global_step: int


@dataclass
class TrainingComponents:
    dataset: UnlabeledStemDataset
    train_loader: DataLoader
    model: PretrainingModel
    chord_boundary_loss_fn: ShiftTolerantBCELoss
    chord_boundary_loss_weight: float
    masked_segment_loss_weight: float
    optimizer: torch.optim.Optimizer
    scheduler: Optional[WarmupCosineScheduler]
    scaler: torch.amp.GradScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chord-boundary + masked-segment self-supervised pretraining for Backbone"
    )
    # dataset / cache 周り
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("dataset/unlabeled_dataset")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ssl_pretrain"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="無ラベル stem dataset の manifest JSON。",
    )
    parser.add_argument(
        "--audio-backend",
        type=str,
        choices=("wav", "packed"),
        default="wav",
        help="入力音声の読込元。packed を使うと songs_packed の npy/json を読みます。",
    )
    parser.add_argument(
        "--packed-audio-dir",
        type=Path,
        default=None,
        help="packed 音声ディレクトリ。未指定なら <dataset-root>/songs_packed を使います。",
    )
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="既存 manifest を無視して dataset 走査から再生成する。",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="このスクリプトが出力した checkpoint から学習再開する。",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=Path("model_epoch_200.pt"),
        help="事前学習前の backbone 初期値として読む checkpoint。",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--disable-persistent-workers", action="store_true")
    parser.add_argument(
        "--disable-file-handle-cache",
        action="store_true",
        help="WAV handle / packed memmap cache を無効化する。",
    )
    parser.add_argument("--max-open-files", type=int, default=64)
    parser.add_argument(
        "--chord-boundary-cache-dir",
        type=Path,
        default=None,
        help="事前推論済み chord boundary cache。未指定なら <dataset-root>/.chord_boundary_cache を自動検出する。",
    )
    parser.add_argument(
        "--prototype-cache-dir",
        type=Path,
        default=None,
        help="事前計算済み segment prototype cache。未指定なら <dataset-root>/.segment_prototype_cache を自動検出する。",
    )
    parser.add_argument("--train-samples-per-epoch", type=int, default=1024)
    parser.add_argument("--segment-seconds", type=float, default=30.0)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=441)
    parser.add_argument("--bins-per-octave", type=int, default=36)
    parser.add_argument("--n-bins", type=int, default=252)

    # model / optimizer
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=("none", "cosine"),
        default="cosine",
    )
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)

    # auxiliary SSL targets
    parser.add_argument(
        "--chord-boundary-loss-weight",
        type=float,
        default=1.0,
        help="事前推論済み chord boundary を使う補助 loss の重み。",
    )
    parser.add_argument(
        "--masked-segment-loss-weight",
        type=float,
        default=1.0,
        help="segment prototype を使う masked segment prediction loss の重み。",
    )
    parser.add_argument(
        "--masked-segment-target-loss",
        type=str,
        choices=("bce", "kl"),
        default="bce",
        help="prototype soft target に対する損失。",
    )
    parser.add_argument(
        "--segment-mask-ratio",
        type=float,
        default=0.4,
        help="visible segment のうち mask する比率。",
    )
    parser.add_argument(
        "--segment-min-masks-per-sample",
        type=int,
        default=1,
        help="各 sample で最低限 mask する segment 数。",
    )
    parser.add_argument(
        "--segment-predictor-hidden-dim",
        type=int,
        default=256,
        help="segment prototype head の hidden dim。",
    )
    parser.add_argument(
        "--min-visible-segments",
        type=int,
        default=2,
        help="prototype supervision 有効時、crop 内に最低限含めたい segment 数。",
    )
    parser.add_argument(
        "--sample-retry-count",
        type=int,
        default=8,
        help="visible segment 数を増やすため start offset を引き直す回数。",
    )
    parser.add_argument("--chord-boundary-pos-weight", type=float, default=5.0)
    parser.add_argument("--chord-boundary-tolerance", type=int, default=2)

    # runtime
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--enable-amp",
        action="store_true",
        help="CUDA 実行時に mixed precision を有効化する。既定では無効。",
    )
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=50)
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_scalar_metrics(
    writer: Optional[SummaryWriter],
    prefix: str,
    metrics: dict[str, float],
    step: int,
) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, step)


def sample_nonzero_integer(max_abs_value: int) -> int:
    if max_abs_value <= 0:
        return 0
    candidates = [
        value for value in range(-max_abs_value, max_abs_value + 1) if value != 0
    ]
    return random.choice(candidates)


def scaled_backward(
    scaler: torch.amp.GradScaler,
    loss: torch.Tensor,
    *,
    retain_graph: bool = False,
) -> None:
    """AMP の有無を隠蔽しつつ loss を逐次 backward する。"""

    if not loss.requires_grad:
        return
    scaler.scale(loss).backward(retain_graph=retain_graph)


def build_training_components(
    args: argparse.Namespace,
    device: torch.device,
) -> TrainingComponents:
    if args.num_workers > 0 and args.prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive when num_workers > 0")
    if args.chord_boundary_loss_weight <= 0.0 and args.masked_segment_loss_weight <= 0.0:
        raise ValueError("At least one SSL objective must be enabled")

    # cache は既定位置を自動検出し、masked segment が有効なときだけ prototype を必須にする。
    chord_boundary_cache_dir = args.chord_boundary_cache_dir
    if chord_boundary_cache_dir is None:
        default_boundary_cache_dir = args.dataset_root / ".chord_boundary_cache"
        if default_boundary_cache_dir.exists():
            chord_boundary_cache_dir = default_boundary_cache_dir
    prototype_cache_dir = args.prototype_cache_dir
    if prototype_cache_dir is None and args.masked_segment_loss_weight > 0.0:
        default_prototype_cache_dir = args.dataset_root / ".segment_prototype_cache"
        if default_prototype_cache_dir.exists():
            prototype_cache_dir = default_prototype_cache_dir
    if args.masked_segment_loss_weight > 0.0 and prototype_cache_dir is None:
        raise ValueError(
            "masked segment loss is enabled, but no prototype cache was found."
        )

    dataset = UnlabeledStemDataset(
        dataset_root=args.dataset_root,
        segment_seconds=args.segment_seconds,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        samples_per_epoch=args.train_samples_per_epoch,
        audio_backend=args.audio_backend,
        packed_audio_dir=args.packed_audio_dir,
        use_file_handle_cache=not args.disable_file_handle_cache,
        max_open_files=args.max_open_files,
        manifest_path=args.manifest_path,
        rebuild_manifest=args.rebuild_manifest,
        chord_boundary_cache_dir=chord_boundary_cache_dir,
        prototype_cache_dir=prototype_cache_dir,
        min_visible_segments=args.min_visible_segments,
        sample_retry_count=args.sample_retry_count,
    )
    if args.masked_segment_loss_weight > 0.0 and dataset.num_prototypes <= 0:
        raise ValueError(
            "masked segment loss is enabled, but the prototype cache did not expose any prototypes"
        )

    # dataset 側は crop 済み音声と、boundary / segment prototype の両方を返す。
    loader_kwargs = {
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": False,
        "collate_fn": collate_unlabeled_stem_batch,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = not args.disable_persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(dataset, batch_size=args.batch_size, **loader_kwargs)

    feature_extractor = AudioFeatureExtractor(
        sampling_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        num_audio_channels=dataset.num_audio_channels,
        num_stems=len(dataset.stem_names),
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
    model = PretrainingModel(
        backbone=backbone,
        head_dropout=args.head_dropout,
        num_segment_prototypes=(
            dataset.num_prototypes if args.masked_segment_loss_weight > 0.0 else 0
        ),
        segment_mask_ratio=args.segment_mask_ratio,
        segment_min_masks_per_sample=args.segment_min_masks_per_sample,
        segment_predictor_hidden_dim=args.segment_predictor_hidden_dim,
    ).to(device)

    chord_boundary_loss_fn = ShiftTolerantBCELoss(
        pos_weight=args.chord_boundary_pos_weight,
        tolerance=args.chord_boundary_tolerance,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler: Optional[WarmupCosineScheduler]
    if args.scheduler == "none":
        scheduler = None
    else:
        total_steps = max(1, len(train_loader) * args.epochs)
        warmup_steps = int(round(len(train_loader) * args.warmup_epochs))
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and args.enable_amp),
    )

    return TrainingComponents(
        dataset=dataset,
        train_loader=train_loader,
        model=model,
        chord_boundary_loss_fn=chord_boundary_loss_fn,
        chord_boundary_loss_weight=float(args.chord_boundary_loss_weight),
        masked_segment_loss_weight=float(args.masked_segment_loss_weight),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )


def initialize_backbone_from_checkpoint(
    model: PretrainingModel,
    checkpoint_path: Path,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format")

    if "backbone_state_dict" in checkpoint and isinstance(
        checkpoint["backbone_state_dict"], dict
    ):
        source_state = checkpoint["backbone_state_dict"]
        state_source = "backbone_state_dict"
    elif "model_state_dict" in checkpoint and isinstance(
        checkpoint["model_state_dict"], dict
    ):
        source_state = checkpoint["model_state_dict"]
        state_source = "model_state_dict"
    else:
        source_state = checkpoint
        state_source = "raw_checkpoint"

    target_state = model.backbone.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    skipped_shape_keys: list[str] = []

    for key, value in source_state.items():
        normalized_key = key
        if normalized_key.startswith("module.backbone."):
            normalized_key = normalized_key[len("module.backbone.") :]
        elif normalized_key.startswith("backbone."):
            normalized_key = normalized_key[len("backbone.") :]
        elif normalized_key.startswith("module."):
            normalized_key = normalized_key[len("module.") :]

        if normalized_key not in target_state:
            continue
        if target_state[normalized_key].shape != value.shape:
            skipped_shape_keys.append(key)
            continue
        filtered_state[normalized_key] = value

    if not filtered_state:
        raise ValueError(
            f"No compatible backbone parameters were found in {checkpoint_path}"
        )

    load_result = model.backbone.load_state_dict(filtered_state, strict=False)
    return {
        "state_source": state_source,
        "loaded_keys": len(filtered_state),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "skipped_shape_keys": skipped_shape_keys,
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    model: PretrainingModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[WarmupCosineScheduler],
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exported_backbone_state = {
        f"backbone.{key}": value.detach().cpu()
        for key, value in model.backbone.state_dict().items()
    }
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "task": "ssl_pretrain",
        "model_state_dict": exported_backbone_state,
        "backbone_state_dict": model.backbone.state_dict(),
        "pretrain_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "metrics": metrics,
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def load_resume_state(
    checkpoint_path: Path,
    model: PretrainingModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[WarmupCosineScheduler],
) -> ResumeState:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pretrain_state = checkpoint.get("pretrain_state_dict")
    if not isinstance(pretrain_state, dict):
        raise ValueError(
            "resume checkpoint must contain pretrain_state_dict produced by train_ssl.py"
        )

    model.load_state_dict(pretrain_state, strict=True)

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    checkpoint_epoch = int(checkpoint["epoch"])
    global_step = int(checkpoint.get("global_step", 0))
    return ResumeState(
        checkpoint_epoch=checkpoint_epoch,
        start_epoch=checkpoint_epoch + 1,
        global_step=global_step,
    )


def train_one_epoch(
    components: TrainingComponents,
    args: argparse.Namespace,
    device: torch.device,
    global_step: int,
    log_interval: int,
    writer: Optional[SummaryWriter],
) -> tuple[dict[str, float], int]:
    components.model.train()
    averager = MetricAverager()
    progress = tqdm(components.train_loader, desc="train", leave=False)

    for step_index, batch in enumerate(progress, start=1):
        components.optimizer.zero_grad(set_to_none=True)
        batch = {
            key: value.to(device, non_blocking=True)
            if torch.is_tensor(value)
            else value
            for key, value in batch.items()
        }
        mix_waveform = batch.pop("waveform")

        zero = mix_waveform.new_zeros(())
        chord_boundary_loss = zero
        masked_segment_loss = zero
        base_enabled = components.chord_boundary_loss_weight > 0.0
        masked_segment_enabled = components.masked_segment_loss_weight > 0.0

        mix_context = None
        if masked_segment_enabled:
            # masked segment は Backbone 入力前に spec を欠損させるので、
            # feature extractor だけ先に 1 回回して context を渡す。
            with torch.no_grad():
                mix_context = components.model.backbone.feature_extractor(mix_waveform)

        total_loss = zero
        with torch.amp.autocast(
            device_type=device.type,
            enabled=components.scaler.is_enabled(),
        ):
            mix_output = components.model(
                mix_waveform,
                backbone_context=mix_context,
                segment_start_frames=(
                    batch["segment_start_frames"] if masked_segment_enabled else None
                ),
                segment_inner_start_frames=(
                    batch["segment_inner_start_frames"]
                    if masked_segment_enabled
                    else None
                ),
                segment_inner_end_frames=(
                    batch["segment_inner_end_frames"] if masked_segment_enabled else None
                ),
                segment_valid_mask=(
                    batch["segment_valid_mask"] if masked_segment_enabled else None
                ),
            )

            if base_enabled:
                chord_boundary_loss = components.chord_boundary_loss_fn(
                    preds=mix_output.chord_boundary_logits,
                    targets=batch["chord_boundary_target"],
                    mask=batch["chord_boundary_mask"],
                )
                total_loss = total_loss + (
                    components.chord_boundary_loss_weight * chord_boundary_loss
                )

            if (
                masked_segment_enabled
                and mix_output.segment_logits is not None
                and mix_output.masked_segment_mask is not None
                and mix_output.segment_valid_mask is not None
            ):
                effective_segment_mask = (
                    mix_output.masked_segment_mask & mix_output.segment_valid_mask
                )
                masked_segment_loss = compute_masked_segment_loss(
                    logits=mix_output.segment_logits,
                    targets=batch["segment_target_probs"],
                    masked_segment_mask=effective_segment_mask,
                    target_loss=args.masked_segment_target_loss,
                )
                total_loss = total_loss + (
                    components.masked_segment_loss_weight * masked_segment_loss
                )
            else:
                effective_segment_mask = batch["segment_valid_mask"].new_zeros(
                    batch["segment_valid_mask"].shape
                )

        did_step = bool(total_loss.requires_grad)
        if did_step:
            scaled_backward(
                components.scaler,
                total_loss,
                retain_graph=False,
            )
        ssl_loss = components.masked_segment_loss_weight * masked_segment_loss
        loss = total_loss

        ssl_metrics = {
            "ssl_loss": float(ssl_loss.detach()),
            "masked_segment_loss": float(masked_segment_loss.detach()),
        }

        if did_step and args.grad_clip > 0:
            components.scaler.unscale_(components.optimizer)
            torch.nn.utils.clip_grad_norm_(
                components.model.parameters(), args.grad_clip
            )
        if did_step:
            components.scaler.step(components.optimizer)
            components.scaler.update()
            if components.scheduler is not None:
                components.scheduler.step()
            global_step += 1
        batch_metrics = {
            "loss": float(loss.detach()),
            "lr": float(components.optimizer.param_groups[0]["lr"]),
            "chord_boundary_loss": float(chord_boundary_loss.detach()),
            "chord_boundary_events": float(
                batch["chord_boundary_event_count"].to(torch.float32).mean().detach()
            ),
            "chord_boundary_supervised_samples": float(
                (batch["chord_boundary_mask"].sum(dim=1) > 0)
                .to(torch.float32)
                .mean()
                .detach()
            ),
            "visible_segments": float(
                batch["segment_valid_mask"].sum(dim=1).to(torch.float32).mean().detach()
            ),
            "masked_segments": float(
                effective_segment_mask.sum(dim=1).to(torch.float32).mean().detach()
            ),
            "masked_segment_supervised_samples": float(
                (batch["segment_valid_mask"].sum(dim=1) > 0)
                .to(torch.float32)
                .mean()
                .detach()
            ),
            **ssl_metrics,
        }
        averager.update(batch_metrics)
        if did_step:
            log_scalar_metrics(writer, "train_step", batch_metrics, global_step)

        if step_index % max(1, log_interval) == 0:
            progress.set_postfix(
                loss=f"{batch_metrics['loss']:.4f}",
                boundary=f"{batch_metrics['chord_boundary_loss']:.4f}",
                seg=f"{batch_metrics['masked_segment_loss']:.4f}",
            )

        del mix_waveform
        del mix_output
        if mix_context is not None:
            del mix_context
        del batch

    return averager.averages(), global_step


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device is None
        else torch.device(args.device)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )

    components = build_training_components(args, device)
    writer = SummaryWriter(log_dir=str(args.output_dir / "tensorboard"))
    history_path = args.output_dir / "history.jsonl"

    global_step = 0
    start_epoch = 1
    if args.resume is not None:
        resume_state = load_resume_state(
            checkpoint_path=args.resume,
            model=components.model,
            optimizer=components.optimizer,
            scheduler=components.scheduler,
        )
        start_epoch = resume_state.start_epoch
        global_step = resume_state.global_step
    elif args.init_from is not None:
        load_stats = initialize_backbone_from_checkpoint(
            components.model, args.init_from
        )
        (args.output_dir / "init_from.json").write_text(
            json.dumps(load_stats, indent=2, default=str),
            encoding="utf-8",
        )

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics, global_step = train_one_epoch(
                components=components,
                args=args,
                device=device,
                global_step=global_step,
                log_interval=args.log_interval,
                writer=writer,
            )

            with history_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps({"epoch": epoch, "train": train_metrics}) + "\n")
            log_scalar_metrics(writer, "train_epoch", train_metrics, epoch)

            metrics_text = ", ".join(
                f"{key}={value:.4f}" for key, value in sorted(train_metrics.items())
            )
            print(f"epoch {epoch}: {metrics_text}")

            save_checkpoint(
                path=args.output_dir / "checkpoint_last.pt",
                epoch=epoch,
                global_step=global_step,
                model=components.model,
                optimizer=components.optimizer,
                scheduler=components.scheduler,
                args=args,
                metrics=train_metrics,
            )

            if epoch % max(1, args.save_every) == 0:
                save_checkpoint(
                    path=args.output_dir / f"model_epoch_{epoch}.pt",
                    epoch=epoch,
                    global_step=global_step,
                    model=components.model,
                    optimizer=components.optimizer,
                    scheduler=components.scheduler,
                    args=args,
                    metrics=train_metrics,
                )
    finally:
        components.dataset.close()
        writer.close()


if __name__ == "__main__":
    main()
