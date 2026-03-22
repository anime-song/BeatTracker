from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from data.beat_dataset import DEFAULT_STEM_NAMES
from models import AudioFeatureExtractor, Backbone, BeatTranscriptionModel

SEPARATOR_SAMPLE_RATE = 44100


@dataclass(frozen=True)
class LoadedAudio:
    waveform: torch.Tensor
    sample_rate: int
    channels_per_stem: int
    source_type: str
    source_id: str
    monitor_waveform: Optional[torch.Tensor] = None


def load_audio_file(audio_path: Path, target_sample_rate: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(audio_path))
    waveform = waveform.to(torch.float32)
    if sample_rate != target_sample_rate:
        waveform = AF.resample(
            waveform, orig_freq=sample_rate, new_freq=target_sample_rate
        )
    return waveform.contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="学習済み beat/downbeat モデルで推論し、JSON とクリック付き WAV を出力する。"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="推論対象。songs_separated の曲ディレクトリ、songs_packed の曲ディレクトリ、packed metadata JSON、または単一のミックス音源を指定する。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="未指定なら checkpoint と同じディレクトリの config.json を使う。",
    )
    parser.add_argument(
        "--state-source",
        type=str,
        choices=("auto", "ema", "model", "raw"),
        default="auto",
        help="checkpoint からどの state_dict を使うか。既定は EMA 優先。",
    )
    parser.add_argument("--semitone", type=int, default=0)
    parser.add_argument(
        "--stem-names",
        nargs="+",
        default=list(DEFAULT_STEM_NAMES),
        help="stem WAV 入力時に使う stem 名の並び。",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=None,
        help="未指定なら学習時の segment_seconds を使う。",
    )
    parser.add_argument(
        "--segment-hop-seconds",
        type=float,
        default=None,
        help="未指定なら segment_seconds / 2 の overlap 推論を行う。",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--beat-threshold", type=float, default=None)
    parser.add_argument("--downbeat-threshold", type=float, default=None)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="未指定なら outputs/inference/<入力名> を使う。",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-audio", type=Path, default=None)
    parser.add_argument("--click-duration-ms", type=float, default=30.0)
    parser.add_argument("--beat-click-freq", type=float, default=1000.0)
    parser.add_argument("--downbeat-click-freq", type=float, default=1600.0)
    parser.add_argument("--click-amplitude", type=float, default=1.0)
    parser.add_argument("--audio-format", type=str, default="wav")
    parser.add_argument(
        "--separator-out-dir",
        type=Path,
        default=Path("outputs/inference/separated"),
        help="ミックス音源入力時に stem-splitter の出力を保存する先。",
    )
    parser.add_argument(
        "--separator-device",
        type=str,
        default=None,
        help="stem-splitter の device。未指定なら --device を流用する。",
    )
    parser.add_argument(
        "--force-reseparate",
        action="store_true",
        help="既存の分離結果があっても stem-splitter を再実行する。",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, object]:
    config_path = args.config or (args.checkpoint.parent / "config.json")
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json が見つかりません: {config_path}. --config を指定してください。"
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def extract_state_dict(
    checkpoint: object,
    state_source: str,
) -> tuple[dict[str, torch.Tensor], str]:
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint format is invalid")

    ema_state = checkpoint.get("ema_state_dict")
    if isinstance(ema_state, dict) and "model_state_dict" in ema_state:
        ema_state_dict = ema_state["model_state_dict"]
    else:
        ema_state_dict = ema_state

    model_state = checkpoint.get("model_state_dict")

    if state_source == "ema":
        if not isinstance(ema_state_dict, dict):
            raise ValueError("ema_state_dict is not available in this checkpoint")
        return ema_state_dict, "ema_state_dict"

    if state_source == "model":
        if not isinstance(model_state, dict):
            raise ValueError("model_state_dict is not available in this checkpoint")
        return model_state, "model_state_dict"

    if state_source == "raw":
        if not isinstance(checkpoint, dict):
            raise ValueError("raw checkpoint is not a valid state_dict")
        return checkpoint, "raw_checkpoint"

    if isinstance(ema_state_dict, dict):
        return ema_state_dict, "ema_state_dict"
    if isinstance(model_state, dict):
        return model_state, "model_state_dict"
    return checkpoint, "raw_checkpoint"


def infer_expected_num_audio_channels(state_dict: dict[str, torch.Tensor]) -> int:
    weight = state_dict.get("backbone.oct_frontend.shared.0.weight")
    if weight is None or weight.ndim != 4:
        raise ValueError("checkpoint から入力チャンネル数を推定できませんでした")
    return int(weight.shape[1])


def infer_num_meter_classes(
    state_dict: dict[str, torch.Tensor], config: dict[str, object]
) -> int:
    meter_labels = config.get("meter_labels")
    if isinstance(meter_labels, list) and meter_labels:
        return len(meter_labels)

    weight = state_dict.get("head.meter_head.weight")
    if weight is None or weight.ndim != 2:
        raise ValueError("checkpoint から meter class 数を推定できませんでした")
    return int(weight.shape[0])


def infer_use_drum_aux_head(
    state_dict: dict[str, torch.Tensor],
    config: dict[str, object],
) -> bool:
    # config に保存されていればそれを優先し、古い checkpoint は key の有無で推定する。
    if "use_drum_aux_head" in config:
        return bool(config["use_drum_aux_head"])

    return any(
        key.startswith("head.drum_aux_head.")
        for key in state_dict.keys()
    )


def infer_use_drum_high_frequency_flux(
    state_dict: dict[str, torch.Tensor],
    config: dict[str, object],
) -> bool:
    if "drum_aux_use_high_frequency_flux" in config:
        return bool(config["drum_aux_use_high_frequency_flux"])

    return any(
        key.startswith("head.drum_aux_head.high_frequency_flux.")
        for key in state_dict.keys()
    )


def infer_use_chord_boundary_head(
    state_dict: dict[str, torch.Tensor],
    config: dict[str, object],
) -> bool:
    if "use_chord_boundary_head" in config:
        return bool(config["use_chord_boundary_head"]) or any(
            key.startswith("head.chord_boundary_head.")
            for key in state_dict.keys()
        )

    return any(
        key.startswith("head.chord_boundary_head.")
        for key in state_dict.keys()
    )


def resolve_stem_file_paths(
    song_dir: Path,
    song_id: str,
    stem_names: list[str],
    semitone: int,
) -> dict[str, Path]:
    # ファイル名規約だけをここで解決し、実際の読み込み処理とは分ける。
    suffix_candidates = (
        [f"_pitch_{semitone}st"] if semitone != 0 else ["", "_pitch_0st"]
    )
    stem_paths: dict[str, Path] = {}

    for stem_name in stem_names:
        for suffix in suffix_candidates:
            candidate = song_dir / f"{song_id}_{stem_name}{suffix}.wav"
            if candidate.exists():
                stem_paths[stem_name] = candidate
                break

    return stem_paths


def build_loaded_audio_from_stem_paths(
    source_id: str,
    stem_paths: dict[str, Path],
    stem_names: list[str],
    target_sample_rate: int,
    source_type: str,
    monitor_waveform: Optional[torch.Tensor] = None,
) -> LoadedAudio:
    # stem の読み込み・resample・長さ合わせはここだけで行う。
    stem_waveforms: list[torch.Tensor] = []
    channels_per_stem: Optional[int] = None
    min_samples: Optional[int] = None

    for stem_name in stem_names:
        wav_path = stem_paths.get(stem_name)
        if wav_path is None:
            raise FileNotFoundError(
                f"{stem_name} stem が見つかりませんでした: {source_id}"
            )

        waveform = load_audio_file(wav_path, target_sample_rate)

        if channels_per_stem is None:
            channels_per_stem = int(waveform.shape[0])
        elif waveform.shape[0] != channels_per_stem:
            raise ValueError(f"stem ごとのチャンネル数が一致しません: {source_id}")

        min_samples = (
            waveform.shape[-1]
            if min_samples is None
            else min(min_samples, int(waveform.shape[-1]))
        )
        stem_waveforms.append(waveform)

    assert channels_per_stem is not None
    assert min_samples is not None
    trimmed = [waveform[..., :min_samples] for waveform in stem_waveforms]
    if monitor_waveform is not None:
        monitor_waveform = monitor_waveform[..., :min_samples].contiguous()

    return LoadedAudio(
        waveform=torch.cat(trimmed, dim=0).contiguous(),
        sample_rate=target_sample_rate,
        channels_per_stem=channels_per_stem,
        source_type=source_type,
        source_id=source_id,
        monitor_waveform=monitor_waveform,
    )


def load_stem_directory(
    song_dir: Path,
    stem_names: list[str],
    semitone: int,
    target_sample_rate: int,
) -> LoadedAudio:
    song_id = song_dir.name
    stem_paths = resolve_stem_file_paths(
        song_dir=song_dir,
        song_id=song_id,
        stem_names=stem_names,
        semitone=semitone,
    )
    return build_loaded_audio_from_stem_paths(
        source_id=song_id,
        stem_paths=stem_paths,
        stem_names=stem_names,
        target_sample_rate=target_sample_rate,
        source_type="stem_wav",
    )


def load_packed_audio(
    input_path: Path,
    semitone: int,
    target_sample_rate: int,
) -> LoadedAudio:
    if input_path.is_dir():
        song_id = input_path.name
        metadata_path = input_path / f"{song_id}_stems_pitch_{semitone}st.json"
    else:
        metadata_path = input_path
        song_id = metadata_path.stem.replace("_stems_pitch_0st", "")

    if not metadata_path.exists():
        raise FileNotFoundError(f"packed metadata が見つかりません: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    array_path = metadata_path.with_suffix(".npy")
    if not array_path.exists():
        raise FileNotFoundError(f"packed array が見つかりません: {array_path}")

    waveform = torch.from_numpy(np.load(array_path)).to(torch.float32)
    sample_rate = int(metadata["sample_rate"])
    if sample_rate != target_sample_rate:
        waveform = AF.resample(
            waveform, orig_freq=sample_rate, new_freq=target_sample_rate
        )
        sample_rate = target_sample_rate

    return LoadedAudio(
        waveform=waveform.contiguous(),
        sample_rate=sample_rate,
        channels_per_stem=int(metadata["channels_per_stem"]),
        source_type="packed",
        source_id=song_id,
    )


def separate_mixed_audio(
    input_path: Path,
    args: argparse.Namespace,
    target_sample_rate: int,
) -> LoadedAudio:
    try:
        from stem_splitter.inference import (
            SeparationConfig,
            _separate_one_file,
            load_mss_model,
            resolve_device,
        )
    except ImportError as exc:
        raise ImportError(
            "ミックス音源の簡易前処理には stem-splitter が必要です。"
        ) from exc

    source_id = input_path.stem
    output_root = args.separator_out_dir
    output_root.mkdir(parents=True, exist_ok=True)

    # ミックス音源だけはここで stem-splitter を通し、
    # その後は通常の stem 推論ルートへ合流させる。
    separate_config = SeparationConfig(
        target_sample_rate=SEPARATOR_SAMPLE_RATE,
        device_preference=args.separator_device or args.device,
        stem_names=tuple(str(name) for name in args.stem_names),
        skip_existing=not args.force_reseparate,
    )
    device = resolve_device(separate_config.device_preference)
    dtype = (
        torch.float16
        if (separate_config.use_half_precision and device.type == "cuda")
        else torch.float32
    )
    try:
        model = load_mss_model(separate_config, device=device)
    except RuntimeError as exc:
        raise RuntimeError(
            "stem-splitter のモデル読込に失敗しました。"
            f" model_name={separate_config.model_name} の重み互換が崩れている可能性があります。"
            " scripts/separate.py でも同じ環境エラーになるはずなので、"
            " stem-splitter 側の version / cached checkpoint を揃えるか、"
            " 先に songs_separated を用意してから推論してください。"
        ) from None

    try:
        separation_result = _separate_one_file(
            input_path,
            output_root,
            separate_config,
            model,
            device,
            dtype,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "stem-splitter の分離処理に失敗しました。"
            " 分離器のモデル互換または音声デコード周りを確認してください。"
        ) from None

    stem_paths = {
        str(stem_name): Path(stem_path)
        for stem_name, stem_path in separation_result.items()
    }
    if not stem_paths:
        song_dir = output_root / source_id
        stem_paths = resolve_stem_file_paths(
            song_dir=song_dir,
            song_id=source_id,
            stem_names=[str(name) for name in args.stem_names],
            semitone=0,
        )

    if not stem_paths:
        raise RuntimeError(f"stem-splitter の出力が見つかりませんでした: {input_path}")

    monitor_waveform = load_audio_file(input_path, target_sample_rate)
    return build_loaded_audio_from_stem_paths(
        source_id=source_id,
        stem_paths=stem_paths,
        stem_names=[str(name) for name in args.stem_names],
        target_sample_rate=target_sample_rate,
        source_type="mix_audio_separated",
        monitor_waveform=monitor_waveform,
    )


def load_input_audio(
    args: argparse.Namespace, config: dict[str, object]
) -> LoadedAudio:
    # 入力は 3 系統だけに絞る。
    # 1. packed JSON / packed ディレクトリ
    # 2. stem WAV ディレクトリ
    # 3. 単一ミックス音源 -> stem-splitter で前処理
    target_sample_rate = int(config["sample_rate"])
    input_path = args.input_path

    if input_path.is_file() and input_path.suffix.lower() == ".json":
        return load_packed_audio(
            input_path, semitone=args.semitone, target_sample_rate=target_sample_rate
        )

    if input_path.is_file():
        return separate_mixed_audio(
            input_path=input_path,
            args=args,
            target_sample_rate=target_sample_rate,
        )

    if input_path.is_dir():
        has_packed = any(input_path.glob("*_stems_pitch_*.json"))
        has_wav = any(input_path.glob("*.wav"))
        if has_packed:
            return load_packed_audio(
                input_path,
                semitone=args.semitone,
                target_sample_rate=target_sample_rate,
            )
        if has_wav:
            return load_stem_directory(
                input_path,
                stem_names=[str(name) for name in args.stem_names],
                semitone=args.semitone,
                target_sample_rate=target_sample_rate,
            )

    raise ValueError(
        "input-path は stem 曲ディレクトリか packed 曲ディレクトリ/JSON を指定してください。"
    )


def build_model_from_config(
    config: dict[str, object],
    num_audio_channels: int,
    num_stems: int,
    num_meter_classes: int,
    use_chord_boundary_head: bool,
    use_drum_aux_head: bool,
    use_drum_high_frequency_flux: bool,
) -> BeatTranscriptionModel:
    feature_extractor = AudioFeatureExtractor(
        sampling_rate=int(config["sample_rate"]),
        n_fft=int(config["n_fft"]),
        hop_length=int(config["hop_length"]),
        num_audio_channels=num_audio_channels,
        num_stems=num_stems,
        bins_per_octave=int(config["bins_per_octave"]),
        n_bins=int(config["n_bins"]),
        spec_augment_params=None,
    )
    backbone = Backbone(
        feature_extractor=feature_extractor,
        hidden_size=int(config["hidden_size"]),
        output_dim=int(config["output_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        use_gradient_checkpoint=False,
    )
    return BeatTranscriptionModel(
        backbone=backbone,
        num_meter_classes=num_meter_classes,
        use_chord_boundary_head=use_chord_boundary_head,
        use_drum_aux_head=use_drum_aux_head,
        use_drum_high_frequency_flux=use_drum_high_frequency_flux,
        head_dropout=float(config.get("head_dropout", 0.0)),
    )


def pick_peak_indices(probabilities: torch.Tensor, threshold: float) -> list[int]:
    if probabilities.numel() == 0:
        return []

    active = probabilities >= threshold
    if not bool(active.any()):
        return []

    peak_indices: list[int] = []
    region_start: Optional[int] = None
    for idx, is_active in enumerate(active.tolist()):
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


def merge_close_events(
    event_times: list[float], min_interval_sec: float
) -> list[float]:
    if not event_times:
        return []

    sorted_events = sorted(float(event_time) for event_time in event_times)
    merged = [sorted_events[0]]
    for event_time in sorted_events[1:]:
        if event_time - merged[-1] >= min_interval_sec:
            merged.append(event_time)
    return merged


def build_segment_starts(
    total_samples: int,
    segment_samples: int,
    hop_samples: int,
) -> list[int]:
    max_start = max(total_samples - segment_samples, 0)
    starts = list(range(0, max_start + 1, hop_samples))
    if not starts or starts[-1] != max_start:
        starts.append(max_start)
    return starts


def build_meter_segments_from_downbeats(
    meter_logits: torch.Tensor,
    downbeat_indices: list[int],
    frame_duration_sec: float,
    meter_labels: list[str],
) -> list[dict[str, object]]:
    """
    predicted downbeat で切った各区間に対して、meter を 1 つだけ割り当てる。

    基本は [downbeat_i, downbeat_{i+1}) を 1 小節として扱い、
    区間内の frame-level meter logits を平均して最終ラベルを決める。
    最後の downbeat 以降も残りがあれば 1 区間として出力する。
    """
    if meter_logits.numel() == 0 or not meter_labels:
        return []
    if not downbeat_indices:
        return []

    total_frames = int(meter_logits.shape[0])
    boundaries = sorted(int(frame_index) for frame_index in downbeat_indices)
    segments: list[tuple[int, int]] = []

    for start_frame, end_frame in zip(boundaries[:-1], boundaries[1:]):
        if end_frame > start_frame:
            segments.append((start_frame, end_frame))

    last_downbeat = boundaries[-1]
    if total_frames > last_downbeat:
        segments.append((last_downbeat, total_frames))

    meter_segments: list[dict[str, object]] = []
    for start_frame, end_frame in segments:
        segment_logits = meter_logits[start_frame:end_frame]
        if segment_logits.numel() == 0:
            continue

        segment_mean_logits = segment_logits.mean(dim=0)
        segment_probabilities = torch.softmax(segment_mean_logits, dim=-1)
        class_index = int(segment_probabilities.argmax().item())
        meter_segments.append(
            {
                "start_sec": start_frame * frame_duration_sec,
                "end_sec": end_frame * frame_duration_sec,
                "label": meter_labels[class_index],
                "confidence": float(segment_probabilities[class_index].item()),
            }
        )

    return meter_segments


def infer_track(
    model: BeatTranscriptionModel,
    waveform: torch.Tensor,
    device: torch.device,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    segment_seconds: float,
    segment_hop_seconds: float,
    batch_size: int,
    use_amp: bool,
    beat_threshold: float,
    downbeat_threshold: float,
    meter_labels: list[str],
) -> dict[str, object]:
    # 曲全体を固定長窓で走査し、重なったフレームは平均して 1 本の時系列へ戻す。
    segment_samples = int(round(segment_seconds * sample_rate))
    if segment_samples < n_fft:
        raise ValueError("segment_seconds が短すぎて STFT 窓を作れません")

    total_samples = int(waveform.shape[-1])
    total_frames = (
        0 if total_samples < n_fft else 1 + ((total_samples - n_fft) // hop_length)
    )
    if total_frames <= 0:
        return {
            "beat_times_sec": [],
            "downbeat_times_sec": [],
            "beat_probabilities": [],
            "downbeat_probabilities": [],
            "meter_segments": [],
        }

    target_num_frames = 1 + ((segment_samples - n_fft) // hop_length)
    hop_samples = max(1, int(round(segment_hop_seconds * sample_rate)))
    starts = build_segment_starts(total_samples, segment_samples, hop_samples)

    beat_prob_sum = torch.zeros(total_frames, dtype=torch.float32)
    downbeat_prob_sum = torch.zeros(total_frames, dtype=torch.float32)
    meter_logit_sum = torch.zeros(
        (total_frames, len(meter_labels)), dtype=torch.float32
    )
    frame_count = torch.zeros(total_frames, dtype=torch.float32)

    model.eval()
    if use_amp and device.type == "cuda":
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)
    elif use_amp and device.type == "cpu":
        autocast_context = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    else:
        autocast_context = nullcontext()

    with torch.no_grad():
        for batch_start in tqdm(
            range(0, len(starts), batch_size),
            desc="Infer",
            dynamic_ncols=True,
            leave=False,
        ):
            batch_starts = starts[batch_start : batch_start + batch_size]
            segments: list[torch.Tensor] = []
            valid_frames_list: list[int] = []
            frame_offsets: list[int] = []

            for start_sample in batch_starts:
                segment = waveform[:, start_sample : start_sample + segment_samples]
                valid_samples = int(segment.shape[-1])
                if valid_samples < segment_samples:
                    segment = F.pad(segment, (0, segment_samples - valid_samples))
                segments.append(segment)

                valid_frames = (
                    0
                    if valid_samples < n_fft
                    else min(
                        target_num_frames,
                        1 + ((valid_samples - n_fft) // hop_length),
                    )
                )
                valid_frames_list.append(valid_frames)
                frame_offsets.append(int(round(start_sample / hop_length)))

            batch_waveform = torch.stack(segments, dim=0).to(device)
            with autocast_context:
                output = model(batch_waveform)

            beat_probabilities = torch.sigmoid(output.beat_logits).detach().cpu()
            downbeat_probabilities = (
                torch.sigmoid(output.downbeat_logits).detach().cpu()
            )
            meter_logits = output.meter_logits.detach().cpu()

            for idx, valid_frames in enumerate(valid_frames_list):
                if valid_frames <= 0:
                    continue

                frame_offset = frame_offsets[idx]
                usable_frames = min(valid_frames, total_frames - frame_offset)
                if usable_frames <= 0:
                    continue

                beat_prob_sum[
                    frame_offset : frame_offset + usable_frames
                ] += beat_probabilities[idx, :usable_frames]
                downbeat_prob_sum[
                    frame_offset : frame_offset + usable_frames
                ] += downbeat_probabilities[idx, :usable_frames]
                meter_logit_sum[
                    frame_offset : frame_offset + usable_frames
                ] += meter_logits[idx, :usable_frames]
                frame_count[frame_offset : frame_offset + usable_frames] += 1.0

    frame_count = torch.clamp(frame_count, min=1.0)
    beat_prob = beat_prob_sum / frame_count
    downbeat_prob = downbeat_prob_sum / frame_count
    meter_logits = meter_logit_sum / frame_count.unsqueeze(-1)

    # 連続して threshold を超えた領域は 1 つのピークへ潰す。
    beat_indices = pick_peak_indices(beat_prob, beat_threshold)
    downbeat_indices = pick_peak_indices(downbeat_prob, downbeat_threshold)
    frame_duration_sec = hop_length / sample_rate
    dedupe_interval_sec = frame_duration_sec * 0.5
    beat_times_sec = merge_close_events(
        [frame_idx * frame_duration_sec for frame_idx in beat_indices],
        dedupe_interval_sec,
    )
    downbeat_times_sec = merge_close_events(
        [frame_idx * frame_duration_sec for frame_idx in downbeat_indices],
        dedupe_interval_sec,
    )

    meter_segments = build_meter_segments_from_downbeats(
        meter_logits=meter_logits,
        downbeat_indices=downbeat_indices,
        frame_duration_sec=frame_duration_sec,
        meter_labels=meter_labels,
    )
    if not meter_segments:
        # downbeat が十分に取れない曲だけは、従来の frame-wise 出力へ戻す。
        meter_prob = torch.softmax(meter_logits, dim=-1)
        meter_class = meter_prob.argmax(dim=-1)
        if len(meter_labels) > 0 and meter_class.numel() > 0:
            start_frame = 0
            current_class = int(meter_class[0].item())
            current_confidences = [float(meter_prob[0, current_class].item())]
            for frame_idx in range(1, int(meter_class.numel())):
                class_index = int(meter_class[frame_idx].item())
                if class_index == current_class:
                    current_confidences.append(
                        float(meter_prob[frame_idx, class_index].item())
                    )
                    continue

                meter_segments.append(
                    {
                        "start_sec": start_frame * frame_duration_sec,
                        "end_sec": frame_idx * frame_duration_sec,
                        "label": meter_labels[current_class],
                        "confidence": float(np.mean(current_confidences)),
                    }
                )
                start_frame = frame_idx
                current_class = class_index
                current_confidences = [float(meter_prob[frame_idx, class_index].item())]

            meter_segments.append(
                {
                    "start_sec": start_frame * frame_duration_sec,
                    "end_sec": meter_class.numel() * frame_duration_sec,
                    "label": meter_labels[current_class],
                    "confidence": float(np.mean(current_confidences)),
                }
            )

    return {
        "beat_times_sec": beat_times_sec,
        "downbeat_times_sec": downbeat_times_sec,
        "beat_probabilities": beat_prob.tolist(),
        "downbeat_probabilities": downbeat_prob.tolist(),
        "meter_segments": meter_segments,
    }


def make_click_track(
    event_times_sec: list[float],
    sample_rate: int,
    total_samples: int,
    frequency_hz: float,
    duration_ms: float,
    amplitude: float,
) -> torch.Tensor:
    click_samples = max(1, int(round((duration_ms / 1000.0) * sample_rate)))
    time_axis = torch.arange(click_samples, dtype=torch.float32) / float(sample_rate)
    window = torch.hann_window(click_samples, periodic=False)
    click = amplitude * torch.sin((2.0 * math.pi * frequency_hz) * time_axis) * window

    click_track = torch.zeros(total_samples, dtype=torch.float32)
    for event_time_sec in event_times_sec:
        start = int(round(event_time_sec * sample_rate))
        if start >= total_samples:
            continue
        end = min(total_samples, start + click_samples)
        click_track[start:end] += click[: end - start]
    return click_track


def filter_downbeats_from_beats(
    beat_times_sec: list[float], downbeat_times_sec: list[float], tolerance_sec: float
) -> list[float]:
    if not downbeat_times_sec:
        return beat_times_sec

    filtered: list[float] = []
    downbeat_idx = 0
    sorted_downbeats = sorted(downbeat_times_sec)
    for beat_time in sorted(beat_times_sec):
        while (
            downbeat_idx + 1 < len(sorted_downbeats)
            and sorted_downbeats[downbeat_idx + 1] <= beat_time
        ):
            downbeat_idx += 1
        current_distance = abs(sorted_downbeats[downbeat_idx] - beat_time)
        if current_distance > tolerance_sec:
            filtered.append(beat_time)
    return filtered


def mix_stems_for_monitoring(
    waveform: torch.Tensor,
    num_stems: int,
    channels_per_stem: int,
) -> torch.Tensor:
    reshaped = waveform.reshape(num_stems, channels_per_stem, waveform.shape[-1])
    return reshaped.mean(dim=0)


def write_outputs(
    args: argparse.Namespace,
    config: dict[str, object],
    loaded_audio: LoadedAudio,
    predictions: dict[str, object],
    selected_state_source: str,
) -> tuple[Path, Path]:
    # JSON は確認用、WAV は耳で挙動を見るための監聴用。
    default_prefix = Path("outputs") / "inference" / loaded_audio.source_id
    output_prefix = args.output_prefix or default_prefix
    output_json = args.output_json or output_prefix.with_suffix(".json")
    output_audio = args.output_audio or output_prefix.with_suffix(
        f".{args.audio_format}"
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_audio.parent.mkdir(parents=True, exist_ok=True)

    beat_times_sec = [float(value) for value in predictions["beat_times_sec"]]
    downbeat_times_sec = [float(value) for value in predictions["downbeat_times_sec"]]
    meter_segments = list(predictions["meter_segments"])

    payload = {
        "checkpoint": str(args.checkpoint),
        "state_source": selected_state_source,
        "input_path": str(args.input_path),
        "source_type": loaded_audio.source_type,
        "source_id": loaded_audio.source_id,
        "sample_rate": loaded_audio.sample_rate,
        "segment_seconds": float(
            args.segment_seconds
            if args.segment_seconds is not None
            else config["segment_seconds"]
        ),
        "segment_hop_seconds": float(
            args.segment_hop_seconds
            if args.segment_hop_seconds is not None
            else (float(config["segment_seconds"]) / 2.0)
        ),
        "hop_length": int(config["hop_length"]),
        "n_fft": int(config["n_fft"]),
        "beat_threshold": float(
            args.beat_threshold
            if args.beat_threshold is not None
            else config.get("beat_threshold", 0.5)
        ),
        "downbeat_threshold": float(
            args.downbeat_threshold
            if args.downbeat_threshold is not None
            else config.get("downbeat_threshold", 0.5)
        ),
        "meter_labels": list(predictions.get("meter_labels", [])),
        "beat_times_sec": beat_times_sec,
        "downbeat_times_sec": downbeat_times_sec,
        "meter_segments": meter_segments,
    }
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ミックス音源入力時は元のミックスにクリックを重ねる。
    if loaded_audio.monitor_waveform is not None:
        monitor_mix = loaded_audio.monitor_waveform
    else:
        monitor_mix = mix_stems_for_monitoring(
            loaded_audio.waveform,
            num_stems=len(args.stem_names),
            channels_per_stem=loaded_audio.channels_per_stem,
        )
    filtered_beat_times = filter_downbeats_from_beats(
        beat_times_sec,
        downbeat_times_sec,
        tolerance_sec=(args.click_duration_ms / 1000.0) * 0.5,
    )
    beat_click = make_click_track(
        filtered_beat_times,
        sample_rate=loaded_audio.sample_rate,
        total_samples=monitor_mix.shape[-1],
        frequency_hz=args.beat_click_freq,
        duration_ms=args.click_duration_ms,
        amplitude=args.click_amplitude,
    )
    downbeat_click = make_click_track(
        downbeat_times_sec,
        sample_rate=loaded_audio.sample_rate,
        total_samples=monitor_mix.shape[-1],
        frequency_hz=args.downbeat_click_freq,
        duration_ms=args.click_duration_ms,
        amplitude=args.click_amplitude * 1.1,
    )
    click_mix = (beat_click + downbeat_click).unsqueeze(0).expand_as(monitor_mix)
    combined = monitor_mix + click_mix
    peak = float(combined.abs().max().item())
    if peak > 0.99:
        combined = combined / (peak / 0.99)

    torchaudio.save(str(output_audio), combined.cpu(), loaded_audio.sample_rate)
    return output_json, output_audio


def main() -> None:
    args = parse_args()
    config = load_config(args)

    # 1. checkpoint / config を読み、入力音声を推論できる形へ揃える。
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict, selected_state_source = extract_state_dict(
        checkpoint, args.state_source
    )
    loaded_audio = load_input_audio(args, config)

    expected_num_audio_channels = infer_expected_num_audio_channels(state_dict)
    if loaded_audio.waveform.shape[0] != expected_num_audio_channels:
        raise ValueError(
            "入力チャンネル数が checkpoint と一致しません。"
            f" expected={expected_num_audio_channels}, actual={loaded_audio.waveform.shape[0]}. "
            "単一のミックス音源ではなく、学習時と同じ separated stems / packed stems を指定してください。"
        )

    num_meter_classes = infer_num_meter_classes(state_dict, config)
    use_chord_boundary_head = infer_use_chord_boundary_head(state_dict, config)
    use_drum_aux_head = infer_use_drum_aux_head(state_dict, config)
    use_drum_high_frequency_flux = infer_use_drum_high_frequency_flux(
        state_dict, config
    )
    model = build_model_from_config(
        config=config,
        num_audio_channels=int(loaded_audio.waveform.shape[0]),
        num_stems=len(args.stem_names),
        num_meter_classes=num_meter_classes,
        use_chord_boundary_head=use_chord_boundary_head,
        use_drum_aux_head=use_drum_aux_head,
        use_drum_high_frequency_flux=use_drum_high_frequency_flux,
    )
    model.load_state_dict(state_dict, strict=True)

    # 2. 推論設定を確定する。
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(device)

    segment_seconds = (
        float(args.segment_seconds)
        if args.segment_seconds is not None
        else float(config["segment_seconds"])
    )
    segment_hop_seconds = (
        float(args.segment_hop_seconds)
        if args.segment_hop_seconds is not None
        else (segment_seconds / 2.0)
    )
    if segment_hop_seconds <= 0:
        raise ValueError("segment-hop-seconds must be positive")

    beat_threshold = float(
        args.beat_threshold
        if args.beat_threshold is not None
        else config.get("beat_threshold", 0.5)
    )
    downbeat_threshold = float(
        args.downbeat_threshold
        if args.downbeat_threshold is not None
        else config.get("downbeat_threshold", 0.5)
    )
    meter_labels = [str(label) for label in config.get("meter_labels", [])]
    if not meter_labels:
        meter_labels = [f"class_{index}" for index in range(num_meter_classes)]

    # 3. 曲全体を推論し、イベント時刻を取り出す。
    predictions = infer_track(
        model=model,
        waveform=loaded_audio.waveform,
        device=device,
        sample_rate=loaded_audio.sample_rate,
        n_fft=int(config["n_fft"]),
        hop_length=int(config["hop_length"]),
        segment_seconds=segment_seconds,
        segment_hop_seconds=segment_hop_seconds,
        batch_size=args.batch_size,
        use_amp=args.use_amp,
        beat_threshold=beat_threshold,
        downbeat_threshold=downbeat_threshold,
        meter_labels=meter_labels,
    )
    predictions["meter_labels"] = meter_labels

    # 4. JSON とクリック付き音声を書き出す。
    output_json, output_audio = write_outputs(
        args=args,
        config=config,
        loaded_audio=loaded_audio,
        predictions=predictions,
        selected_state_source=selected_state_source,
    )

    print(f"state_source={selected_state_source}")
    print(f"beats={len(predictions['beat_times_sec'])}")
    print(f"downbeats={len(predictions['downbeat_times_sec'])}")
    print(f"json={output_json}")
    print(f"audio={output_audio}")


if __name__ == "__main__":
    main()
