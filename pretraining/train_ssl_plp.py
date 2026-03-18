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
from pretraining.contrastive import PLPContrastiveLoss
from pretraining.model import PLPContrastiveBackboneModel
from pretraining.plp_teacher import PLPPseudoTeacher
from pretraining.unlabeled_dataset import UnlabeledStemDataset


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
    """学習ループが必要とする部品をまとめて保持する。"""

    dataset: UnlabeledStemDataset
    train_loader: DataLoader
    model: PLPContrastiveBackboneModel
    teacher: PLPPseudoTeacher
    criterion: PLPContrastiveLoss
    optimizer: torch.optim.Optimizer
    scheduler: Optional[WarmupCosineScheduler]
    scaler: torch.amp.GradScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PLP-guided contrastive self-supervised pretraining for Backbone"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/unlabeled_dataset"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ssl_plp_pretrain"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="無ラベル stem dataset の manifest JSON。未指定なら <dataset-root>/.unlabeled_stem_manifest.json を使う。",
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
        help="Resume pretraining from a checkpoint produced by this script.",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=Path("model_epoch_200.pt"),
        help="Initial checkpoint to load backbone weights from before SSL pretraining.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
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
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-hidden-dim", type=int, default=256)
    parser.add_argument("--projector-dropout", type=float, default=0.1)
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
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--positive-offsets",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Positive peak-order offsets used by the contrastive objective.",
    )
    parser.add_argument("--negative-safety-radius", type=int, default=4)
    parser.add_argument("--num-negatives", type=int, default=32)
    parser.add_argument("--max-anchors-per-sample", type=int, default=64)
    parser.add_argument("--plp-win-length", type=int, default=384)
    parser.add_argument("--plp-tempo-min", type=float, default=30.0)
    parser.add_argument("--plp-tempo-max", type=float, default=300.0)
    parser.add_argument("--plp-peak-threshold", type=float, default=0.3)
    parser.add_argument(
        "--plp-min-peak-distance",
        type=int,
        default=None,
        help="Peak distance in frames. If omitted, a tempo-aware default is used.",
    )
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


def build_training_components(
    args: argparse.Namespace,
    device: torch.device,
) -> TrainingComponents:
    """
    データ・モデル・最適化器を一箇所で組み立てる。
    main から学習準備の流れが追いやすいよう、初期化処理をここへ集約する。
    """

    if args.num_workers > 0 and args.prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive when num_workers > 0")

    dataset = UnlabeledStemDataset(
        dataset_root=args.dataset_root,
        segment_seconds=args.segment_seconds,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        samples_per_epoch=args.train_samples_per_epoch,
        use_file_handle_cache=not args.disable_file_handle_cache,
        max_open_files=args.max_open_files,
        manifest_path=args.manifest_path,
        rebuild_manifest=args.rebuild_manifest,
    )

    loader_kwargs = {
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": False,
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
    model = PLPContrastiveBackboneModel(
        backbone=backbone,
        projection_dim=args.projection_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        projector_dropout=args.projector_dropout,
    ).to(device)

    teacher = PLPPseudoTeacher(
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        win_length=args.plp_win_length,
        tempo_min=args.plp_tempo_min,
        tempo_max=args.plp_tempo_max,
        peak_threshold=args.plp_peak_threshold,
        min_peak_distance=args.plp_min_peak_distance,
    ).to(device)

    criterion = PLPContrastiveLoss(
        temperature=args.temperature,
        positive_offsets=args.positive_offsets,
        negative_safety_radius=args.negative_safety_radius,
        num_negatives=args.num_negatives,
        max_anchors_per_sample=args.max_anchors_per_sample,
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
        teacher=teacher,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )


def initialize_backbone_from_checkpoint(
    model: PLPContrastiveBackboneModel,
    checkpoint_path: Path,
) -> dict[str, object]:
    """
    supervised 用 checkpoint でも SSL checkpoint でも読めるように、
    backbone 部分だけを抽出して初期化する。
    """

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
    model: PLPContrastiveBackboneModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[WarmupCosineScheduler],
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    """
    既存 supervised 学習からも読みやすいよう、
    `model_state_dict` には `backbone.*` 付きで保存する。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    exported_backbone_state = {
        f"backbone.{key}": value.detach().cpu()
        for key, value in model.backbone.state_dict().items()
    }
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "task": "ssl_plp_pretrain",
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
    model: PLPContrastiveBackboneModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[WarmupCosineScheduler],
) -> ResumeState:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pretrain_state = checkpoint.get("pretrain_state_dict")
    if not isinstance(pretrain_state, dict):
        raise ValueError(
            "resume checkpoint must contain pretrain_state_dict produced by train_ssl_plp.py"
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
    device: torch.device,
    global_step: int,
    log_interval: int,
    writer: Optional[SummaryWriter],
    grad_clip: float,
) -> tuple[dict[str, float], int]:
    """
    1 epoch の本体。
    各 batch で
    1. 無ラベル波形から PLP 教師を作る
    2. Backbone から埋め込みを作る
    3. contrastive loss で更新する
    という流れを順番に追えるようにしている。
    """

    components.model.train()
    averager = MetricAverager()
    progress = tqdm(components.train_loader, desc="train", leave=False)

    for step_index, batch in enumerate(progress, start=1):
        batch = {
            key: value.to(device, non_blocking=True)
            if torch.is_tensor(value)
            else value
            for key, value in batch.items()
        }
        valid_frames = batch["valid_frames"]
        valid_mask = batch["valid_mask"]

        # PLP は教師信号生成だけなので勾配は不要。
        with torch.no_grad():
            teacher_output = components.teacher(batch["waveform"], valid_frames)

        components.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=components.scaler.is_enabled(),
        ):
            embeddings = components.model(batch["waveform"])
            loss, loss_metrics = components.criterion(
                embeddings=embeddings,
                peak_mask=teacher_output.peak_mask,
                peak_values=teacher_output.peak_values,
                valid_mask=valid_mask,
            )

        components.scaler.scale(loss).backward()
        if grad_clip > 0:
            components.scaler.unscale_(components.optimizer)
            torch.nn.utils.clip_grad_norm_(components.model.parameters(), grad_clip)
        components.scaler.step(components.optimizer)
        components.scaler.update()
        if components.scheduler is not None:
            components.scheduler.step()

        global_step += 1
        teacher_metrics = {
            "plp_peak_count": float(
                teacher_output.peak_mask.sum(dim=1).mean().detach()
            ),
            "plp_pulse_mean": float(teacher_output.pulse.mean().detach()),
        }
        batch_metrics = {
            "loss": float(loss.detach()),
            "lr": float(components.optimizer.param_groups[0]["lr"]),
            **loss_metrics,
            **teacher_metrics,
        }
        averager.update(batch_metrics)
        log_scalar_metrics(writer, "train_step", batch_metrics, global_step)

        if step_index % max(1, log_interval) == 0:
            progress.set_postfix(
                loss=f"{batch_metrics['loss']:.4f}",
                pairs=f"{batch_metrics['contrastive_pairs']:.0f}",
                peaks=f"{batch_metrics['plp_peak_count']:.1f}",
            )

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
                device=device,
                global_step=global_step,
                log_interval=args.log_interval,
                writer=writer,
                grad_clip=args.grad_clip,
            )

            # 履歴は jsonl へ追記して、後で supervised 移行時の比較にも使えるようにする。
            with history_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps({"epoch": epoch, "train": train_metrics}) + "\n")
            log_scalar_metrics(writer, "train_epoch", train_metrics, epoch)

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
            if args.save_every > 0 and epoch % args.save_every == 0:
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

            metric_text = ", ".join(
                f"{key}={value:.4f}" for key, value in sorted(train_metrics.items())
            )
            print(f"epoch {epoch}: {metric_text}")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
