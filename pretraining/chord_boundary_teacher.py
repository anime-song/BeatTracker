from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
import yaml
from tqdm.auto import tqdm

from models import AudioFeatureExtractor, Backbone
from models.transformer import RMSNorm


CHORD_TRANSCRIPTION_ASSETS_DIR = (
    Path(__file__).resolve().parent / "assets" / "chord_transcription"
)
DEFAULT_CHORD_TRANSCRIPTION_CHECKPOINT = (
    CHORD_TRANSCRIPTION_ASSETS_DIR / "model_epoch_200.pt"
)
DEFAULT_CHORD_TRANSCRIPTION_CONFIG = CHORD_TRANSCRIPTION_ASSETS_DIR / "config.yaml"


@dataclass(frozen=True)
class ChordBoundaryPrediction:
    """1 曲ぶんの chord boundary 推論結果。"""

    boundary_times_sec: torch.Tensor
    boundary_scores: torch.Tensor
    frame_times_sec: torch.Tensor
    boundary_probabilities: torch.Tensor
    duration_sec: float


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def infer_backbone_num_layers_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int | None:
    """checkpoint の key から backbone の層数を推定する。"""

    layer_indices: list[int] = []
    for key in state_dict:
        match = re.match(r"backbone\.layers\.(\d+)\.", key)
        if match is not None:
            layer_indices.append(int(match.group(1)))
    if not layer_indices:
        return None
    return max(layer_indices) + 1


class ChordBoundaryModel(nn.Module):
    """コード境界推論に必要な最小構成だけを持つモデル。"""

    def __init__(
        self,
        backbone: Backbone,
        hidden_size: int,
        dropout_probability: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.norm = RMSNorm(hidden_size) if use_layer_norm else nn.Identity()
        self.dropout = (
            nn.Dropout(dropout_probability) if dropout_probability > 0.0 else nn.Identity()
        )
        self.boundary_head = nn.Linear(hidden_size, 1)

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        frame_features = self.backbone(waveform)
        frame_features = self.dropout(self.norm(frame_features))
        return {
            "initial_boundary_logits": self.boundary_head(frame_features),
        }


def build_chord_boundary_model_from_config(cfg: dict) -> ChordBoundaryModel:
    """Chord-Transcription checkpoint と互換な最小モデルだけを組み立てる。"""

    model_cfg = cfg["model"]
    backbone_cfg = model_cfg["backbone"]
    classifier_cfg = model_cfg["classifier"]
    loader_cfg = cfg["data_loader"]

    stem_order = tuple(str(name) for name in loader_cfg["stem_order"])
    mixdown_to_mono = bool(loader_cfg.get("mixdown_to_mono", False))
    channels_per_stem = 1 if mixdown_to_mono else 2
    num_audio_channels = len(stem_order) * channels_per_stem

    feature_extractor = AudioFeatureExtractor(
        sampling_rate=int(backbone_cfg["sampling_rate"]),
        n_fft=int(backbone_cfg["n_fft"]),
        hop_length=int(backbone_cfg["hop_length"]),
        num_audio_channels=num_audio_channels,
        num_stems=len(stem_order),
        spec_augment_params=backbone_cfg.get("spec_augment_params"),
    )
    backbone = Backbone(
        feature_extractor=feature_extractor,
        hidden_size=int(backbone_cfg["hidden_size"]),
        output_dim=int(classifier_cfg["hidden_size"]),
        num_layers=int(backbone_cfg.get("num_layers", 1)),
        dropout=float(backbone_cfg.get("dropout", 0.0)),
        use_gradient_checkpoint=True,
    )
    return ChordBoundaryModel(
        backbone=backbone,
        hidden_size=int(classifier_cfg["hidden_size"]),
        dropout_probability=float(classifier_cfg.get("dropout_probability", 0.0)),
        use_layer_norm=bool(classifier_cfg.get("use_layer_norm", True)),
    )


class ChordBoundaryTeacher:
    """
    Chord-Transcription の checkpoint から chord boundary を推論する薄いラッパ。

    公開用に依存を減らすため、このクラスでは
    - config / checkpoint は BeatTracker 側の assets を使う
    - モデル本体は boundary 推論に必要な最小構成だけをローカルで再構築する
    - 既に分離済みの stems をそのまま読む
    という構成にしている。
    """

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHORD_TRANSCRIPTION_CHECKPOINT,
        config_path: Path = DEFAULT_CHORD_TRANSCRIPTION_CONFIG,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        chunk_seconds: float = 120.0,
        overlap_seconds: float = 8.0,
        boundary_threshold: float = 0.5,
        nms_window_radius: int = 3,
        min_boundary_distance_frames: int = 4,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.device = torch.device(device)
        self.chunk_seconds = float(chunk_seconds)
        self.overlap_seconds = float(overlap_seconds)
        self.boundary_threshold = float(boundary_threshold)
        self.nms_window_radius = int(nms_window_radius)
        self.min_boundary_distance_frames = int(min_boundary_distance_frames)

        if self.chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")
        if self.overlap_seconds < 0:
            raise ValueError("overlap_seconds must be non-negative")
        if self.boundary_threshold <= 0 or self.boundary_threshold >= 1:
            raise ValueError("boundary_threshold must be in (0, 1)")
        if self.nms_window_radius < 0:
            raise ValueError("nms_window_radius must be non-negative")
        if self.min_boundary_distance_frames < 1:
            raise ValueError("min_boundary_distance_frames must be positive")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Chord checkpoint not found: {self.checkpoint_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Chord config not found: {self.config_path}")

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(
                "Chord checkpoint must contain ema_state_dict or model_state_dict"
            )
        self.config = _load_yaml(self.config_path)
        inferred_num_layers = infer_backbone_num_layers_from_state_dict(state_dict)
        if inferred_num_layers is not None:
            config_num_layers = int(self.config["model"]["backbone"].get("num_layers", 1))
            if config_num_layers != inferred_num_layers:
                self.config = copy.deepcopy(self.config)
                self.config["model"]["backbone"]["num_layers"] = inferred_num_layers

        self.model = build_chord_boundary_model_from_config(self.config).to(self.device)
        model_state = self.model.state_dict()
        filtered_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        load_result = self.model.load_state_dict(filtered_state, strict=False)
        if load_result.missing_keys:
            raise ValueError(
                "Chord boundary checkpoint is missing required keys: "
                + ", ".join(load_result.missing_keys)
            )
        self.model.eval()

        loader_cfg = self.config["data_loader"]
        self.sample_rate = int(loader_cfg["sample_rate"])
        self.hop_length = int(loader_cfg["hop_length"])
        self.n_fft = int(self.config["model"]["backbone"]["n_fft"])
        self.stem_order = tuple(str(name) for name in loader_cfg["stem_order"])
        self.mixdown_to_mono = bool(loader_cfg.get("mixdown_to_mono", False))
        self.channels_per_stem = 1 if self.mixdown_to_mono else 2
        self.expected_num_channels = len(self.stem_order) * self.channels_per_stem

    def _load_stem_waveform(self, stem_path: Path) -> torch.Tensor:
        waveform, source_sample_rate = torchaudio.load(str(stem_path))
        waveform = waveform.to(torch.float32)

        if source_sample_rate != self.sample_rate:
            waveform = AF.resample(
                waveform,
                orig_freq=source_sample_rate,
                new_freq=self.sample_rate,
            )

        if self.mixdown_to_mono:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.shape[0] == 1 and self.channels_per_stem == 2:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > self.channels_per_stem:
            waveform = waveform[: self.channels_per_stem]

        if waveform.shape[0] != self.channels_per_stem:
            raise ValueError(
                f"Unexpected channel count in {stem_path}: {waveform.shape[0]}"
            )
        return waveform

    def load_song_waveform(self, stem_paths: dict[str, Path]) -> torch.Tensor:
        """1 曲ぶんの stems を読み、モデル入力の多 ch waveform へ整形する。"""

        waveforms = [self._load_stem_waveform(stem_paths[stem_name]) for stem_name in self.stem_order]
        max_samples = max(waveform.shape[-1] for waveform in waveforms)
        padded = [
            F.pad(waveform, (0, max_samples - waveform.shape[-1])) for waveform in waveforms
        ]
        return torch.cat(padded, dim=0)

    def _infer_boundary_probabilities(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        長い曲でも推論できるよう、少し重なりを持たせた chunk ごとに処理する。
        keep するのは各 chunk の中央部だけにして、境界付近の不安定さを抑える。
        """

        total_samples = int(waveform.shape[-1])
        if total_samples <= 0:
            empty = waveform.new_zeros(0, dtype=torch.float32)
            return empty, empty

        chunk_samples = max(self.n_fft, int(round(self.chunk_seconds * self.sample_rate)))
        overlap_samples = int(round(self.overlap_seconds * self.sample_rate))
        step_samples = max(self.n_fft, chunk_samples - (2 * overlap_samples))

        chunks_probabilities: list[torch.Tensor] = []
        chunks_times: list[torch.Tensor] = []

        chunk_starts: list[int] = []
        chunk_start = 0
        while chunk_start < total_samples:
            chunk_starts.append(chunk_start)
            chunk_end = min(total_samples, chunk_start + chunk_samples)
            if chunk_end >= total_samples:
                break
            chunk_start += step_samples

        for chunk_start in chunk_starts:
            chunk_end = min(total_samples, chunk_start + chunk_samples)
            chunk_waveform = waveform[:, chunk_start:chunk_end]
            if chunk_waveform.shape[-1] < self.n_fft:
                chunk_waveform = F.pad(
                    chunk_waveform, (0, self.n_fft - chunk_waveform.shape[-1])
                )

            with torch.inference_mode():
                outputs = self.model(chunk_waveform.unsqueeze(0).to(self.device))
                boundary_logits = outputs["initial_boundary_logits"].squeeze(0).squeeze(-1)
                boundary_probabilities = torch.sigmoid(boundary_logits).to(torch.float32).cpu()

            frame_times = (
                torch.arange(
                    boundary_probabilities.shape[0],
                    dtype=torch.float32,
                )
                * (float(self.hop_length) / float(self.sample_rate))
            ) + (float(chunk_start) / float(self.sample_rate))

            keep_start_sec = float(chunk_start) / float(self.sample_rate)
            keep_end_sec = float(chunk_end) / float(self.sample_rate)
            if chunk_start > 0:
                keep_start_sec += self.overlap_seconds
            if chunk_end < total_samples:
                keep_end_sec -= self.overlap_seconds

            keep_mask = (frame_times >= keep_start_sec) & (frame_times < keep_end_sec)
            if chunk_end == total_samples:
                keep_mask = keep_mask | (frame_times == frame_times[-1])

            chunks_probabilities.append(boundary_probabilities[keep_mask])
            chunks_times.append(frame_times[keep_mask])

        if not chunks_probabilities:
            empty = waveform.new_zeros(0, dtype=torch.float32)
            return empty, empty

        return torch.cat(chunks_probabilities), torch.cat(chunks_times)

    def _detect_boundary_events(
        self,
        frame_times_sec: torch.Tensor,
        boundary_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if boundary_probabilities.numel() == 0:
            empty = boundary_probabilities.new_zeros(0)
            return empty, empty

        pooled = F.max_pool1d(
            boundary_probabilities.view(1, 1, -1),
            kernel_size=(2 * self.nms_window_radius) + 1,
            stride=1,
            padding=self.nms_window_radius,
        ).view(-1)
        candidate_indices = (
            (boundary_probabilities >= self.boundary_threshold)
            & (boundary_probabilities == pooled)
        ).nonzero(as_tuple=False).squeeze(-1)

        if candidate_indices.numel() == 0:
            return boundary_probabilities.new_zeros(0), boundary_probabilities.new_zeros(0)

        ordered = torch.argsort(
            boundary_probabilities.index_select(0, candidate_indices),
            descending=True,
        )
        selected: list[int] = []
        for order_index in ordered.tolist():
            candidate = int(candidate_indices[order_index].item())
            if any(abs(candidate - existing) < self.min_boundary_distance_frames for existing in selected):
                continue
            selected.append(candidate)

        selected_indices = torch.tensor(
            sorted(selected),
            dtype=torch.long,
        )
        return (
            frame_times_sec.index_select(0, selected_indices),
            boundary_probabilities.index_select(0, selected_indices),
        )

    def predict_from_stem_paths(
        self,
        stem_paths: dict[str, Path],
        song_id: str | None = None,
        show_chunk_progress: bool = False,
    ) -> ChordBoundaryPrediction:
        waveform = self.load_song_waveform(stem_paths)
        if show_chunk_progress:
            duration_sec = float(waveform.shape[-1]) / float(self.sample_rate)
            print(
                f"[chord-boundary] loaded stems"
                + (f" song={song_id}" if song_id is not None else "")
                + f" duration={duration_sec:.1f}s"
            )

        total_samples = int(waveform.shape[-1])
        chunk_samples = max(self.n_fft, int(round(self.chunk_seconds * self.sample_rate)))
        overlap_samples = int(round(self.overlap_seconds * self.sample_rate))
        step_samples = max(self.n_fft, chunk_samples - (2 * overlap_samples))
        chunk_starts: list[int] = []
        chunk_start = 0
        while chunk_start < total_samples:
            chunk_starts.append(chunk_start)
            chunk_end = min(total_samples, chunk_start + chunk_samples)
            if chunk_end >= total_samples:
                break
            chunk_start += step_samples

        chunks_probabilities: list[torch.Tensor] = []
        chunks_times: list[torch.Tensor] = []
        iterator = (
            tqdm(
                chunk_starts,
                desc=f"chunks:{song_id}" if song_id is not None else "chunks",
                leave=False,
            )
            if show_chunk_progress
            else chunk_starts
        )
        for chunk_start in iterator:
            chunk_end = min(total_samples, chunk_start + chunk_samples)
            chunk_waveform = waveform[:, chunk_start:chunk_end]
            if chunk_waveform.shape[-1] < self.n_fft:
                chunk_waveform = F.pad(
                    chunk_waveform, (0, self.n_fft - chunk_waveform.shape[-1])
                )

            with torch.inference_mode():
                outputs = self.model(chunk_waveform.unsqueeze(0).to(self.device))
                boundary_logits = outputs["initial_boundary_logits"].squeeze(0).squeeze(-1)
                boundary_probabilities = torch.sigmoid(boundary_logits).to(torch.float32).cpu()

            frame_times = (
                torch.arange(
                    boundary_probabilities.shape[0],
                    dtype=torch.float32,
                )
                * (float(self.hop_length) / float(self.sample_rate))
            ) + (float(chunk_start) / float(self.sample_rate))

            keep_start_sec = float(chunk_start) / float(self.sample_rate)
            keep_end_sec = float(chunk_end) / float(self.sample_rate)
            if chunk_start > 0:
                keep_start_sec += self.overlap_seconds
            if chunk_end < total_samples:
                keep_end_sec -= self.overlap_seconds

            keep_mask = (frame_times >= keep_start_sec) & (frame_times < keep_end_sec)
            if chunk_end == total_samples:
                keep_mask = keep_mask | (frame_times == frame_times[-1])

            chunks_probabilities.append(boundary_probabilities[keep_mask])
            chunks_times.append(frame_times[keep_mask])

        if not chunks_probabilities:
            boundary_probabilities = waveform.new_zeros(0, dtype=torch.float32)
            frame_times_sec = waveform.new_zeros(0, dtype=torch.float32)
        else:
            boundary_probabilities = torch.cat(chunks_probabilities)
            frame_times_sec = torch.cat(chunks_times)

        boundary_times_sec, boundary_scores = self._detect_boundary_events(
            frame_times_sec=frame_times_sec,
            boundary_probabilities=boundary_probabilities,
        )
        return ChordBoundaryPrediction(
            boundary_times_sec=boundary_times_sec,
            boundary_scores=boundary_scores,
            frame_times_sec=frame_times_sec,
            boundary_probabilities=boundary_probabilities,
            duration_sec=float(waveform.shape[-1]) / float(self.sample_rate),
        )
