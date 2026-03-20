from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from models.cqt import RecursiveCQT
from pretraining.segment_prototypes import (
    build_segment_time_table,
    extract_song_chroma,
    load_boundary_times,
    load_harmonic_mono_waveform,
    resolve_harmonic_stem_names,
    summarize_segment_chroma,
)
from pretraining.unlabeled_dataset import UnlabeledStemDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a GMM over chord-boundary segments and cache soft prototype targets."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/unlabeled_dataset"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <dataset-root>/.segment_prototype_cache",
    )
    parser.add_argument(
        "--chord-boundary-cache-dir",
        type=Path,
        default=None,
        help="Default: <dataset-root>/.chord_boundary_cache",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
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
    )
    parser.add_argument(
        "--song-id",
        dest="song_ids",
        action="append",
        default=None,
        help="Restrict processing to one or more songs.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=441,
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=252,
    )
    parser.add_argument(
        "--bins-per-octave",
        type=int,
        default=36,
    )
    parser.add_argument(
        "--num-prototypes",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--gmm-covariance-type",
        type=str,
        choices=("full", "tied", "diag", "spherical"),
        default="diag",
    )
    parser.add_argument(
        "--gmm-max-iter",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--gmm-reg-covar",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--boundary-guard-seconds",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--min-inner-seconds",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--harmonic-stems",
        nargs="+",
        default=None,
        help="Default: bass guitar other piano",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Default: cuda if available else cpu",
    )
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else (args.dataset_root / ".segment_prototype_cache")
    )
    boundary_cache_dir = (
        args.chord_boundary_cache_dir
        if args.chord_boundary_cache_dir is not None
        else (args.dataset_root / ".chord_boundary_cache")
    )
    if not boundary_cache_dir.exists():
        raise ValueError(f"Chord boundary cache does not exist: {boundary_cache_dir}")

    features_dir = output_dir / "segment_features"
    song_targets_dir = output_dir / "song_targets"
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    song_targets_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device is None
        else torch.device(args.device)
    )

    dataset = UnlabeledStemDataset(
        dataset_root=args.dataset_root,
        segment_seconds=30.0,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        samples_per_epoch=None,
        audio_backend=args.audio_backend,
        packed_audio_dir=args.packed_audio_dir,
        manifest_path=args.manifest_path,
        rebuild_manifest=args.rebuild_manifest,
    )
    harmonic_stem_names = resolve_harmonic_stem_names(
        stem_names=dataset.stem_names,
        harmonic_stem_names=args.harmonic_stems,
    )

    songs = dataset.songs
    if args.song_ids:
        allowed_song_ids = set(args.song_ids)
        songs = [song for song in songs if song.song_id in allowed_song_ids]
    if not songs:
        raise ValueError("No songs matched the requested filters")

    cqt_module = RecursiveCQT(
        sr=args.sample_rate,
        hop_length=args.hop_length,
        n_bins=args.n_bins,
        bins_per_octave=args.bins_per_octave,
        filter_scale=0.4375,
    ).to(device)

    collected_features: list[np.ndarray] = []
    processed_song_ids: list[str] = []
    skipped_missing_boundaries = 0

    progress = tqdm(songs, desc="extract-segments")
    for song in progress:
        boundary_path = boundary_cache_dir / f"{song.song_id}.pt"
        boundary_times_sec = load_boundary_times(boundary_path)
        if boundary_times_sec is None:
            skipped_missing_boundaries += 1
            continue

        # 1 曲ごとに
        # boundary -> segment 列 -> harmonic mix -> chroma -> segment 要約
        # の順で prototype 学習用の特徴を作る。
        segments = build_segment_time_table(
            boundary_times_sec=boundary_times_sec,
            duration_sec=song.duration_sec,
            boundary_guard_seconds=args.boundary_guard_seconds,
            min_inner_seconds=args.min_inner_seconds,
        )
        if segments.num_segments <= 0:
            continue

        waveform = load_harmonic_mono_waveform(
            song=song,
            stem_names=dataset.stem_names,
            harmonic_stem_names=harmonic_stem_names,
            target_sample_rate=args.sample_rate,
        )
        chroma = extract_song_chroma(
            waveform=waveform,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            n_bins=args.n_bins,
            bins_per_octave=args.bins_per_octave,
            device=device,
            cqt_module=cqt_module,
        )
        segment_features = summarize_segment_chroma(
            chroma=chroma,
            segments=segments,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
        )
        if segment_features.numel() == 0:
            continue

        feature_payload = {
            "song_id": song.song_id,
            "duration_sec": float(song.duration_sec),
            "segment_start_sec": segments.segment_start_sec,
            "segment_end_sec": segments.segment_end_sec,
            "inner_start_sec": segments.inner_start_sec,
            "inner_end_sec": segments.inner_end_sec,
            "segment_features": segment_features.to(torch.float32),
        }
        torch.save(feature_payload, features_dir / f"{song.song_id}.pt")
        collected_features.append(segment_features.cpu().numpy())
        processed_song_ids.append(song.song_id)
        progress.set_postfix(song=song.song_id, segments=segments.num_segments)

    if not collected_features:
        raise ValueError("No segment features were extracted")

    all_features = np.concatenate(collected_features, axis=0)
    if all_features.shape[0] < args.num_prototypes:
        raise ValueError(
            f"Need at least {args.num_prototypes} segments to fit the requested GMM, "
            f"but only found {all_features.shape[0]}"
        )

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_features)
    gmm = GaussianMixture(
        n_components=args.num_prototypes,
        covariance_type=args.gmm_covariance_type,
        max_iter=args.gmm_max_iter,
        reg_covar=args.gmm_reg_covar,
        random_state=args.seed,
    )
    gmm.fit(scaled_features)

    assignment_progress = tqdm(processed_song_ids, desc="write-targets")
    for song_id in assignment_progress:
        # 保存済み segment feature に対して、全 dataset で fit した共通 GMM から
        # soft prototype distribution を付与する。
        feature_payload = torch.load(
            features_dir / f"{song_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        segment_features = feature_payload["segment_features"].numpy()
        prototype_distribution = gmm.predict_proba(
            scaler.transform(segment_features)
        ).astype(np.float32)
        prototype_labels = prototype_distribution.argmax(axis=1).astype(np.int64)

        song_payload = {
            "song_id": song_id,
            "duration_sec": float(feature_payload["duration_sec"]),
            "segment_start_sec": feature_payload["segment_start_sec"].to(torch.float32),
            "segment_end_sec": feature_payload["segment_end_sec"].to(torch.float32),
            "inner_start_sec": feature_payload["inner_start_sec"].to(torch.float32),
            "inner_end_sec": feature_payload["inner_end_sec"].to(torch.float32),
            "segment_features": feature_payload["segment_features"].to(torch.float32),
            "prototype_distribution": torch.from_numpy(prototype_distribution),
            "prototype_labels": torch.from_numpy(prototype_labels),
        }
        torch.save(song_payload, song_targets_dir / f"{song_id}.pt")
        assignment_progress.set_postfix(
            song=song_id, segments=prototype_distribution.shape[0]
        )

    model_payload = {
        "num_prototypes": int(args.num_prototypes),
        "feature_dim": int(all_features.shape[1]),
        "harmonic_stem_names": list(harmonic_stem_names),
        "sample_rate": int(args.sample_rate),
        "hop_length": int(args.hop_length),
        "n_fft": int(args.n_fft),
        "n_bins": int(args.n_bins),
        "bins_per_octave": int(args.bins_per_octave),
        "boundary_guard_seconds": float(args.boundary_guard_seconds),
        "min_inner_seconds": float(args.min_inner_seconds),
        "gmm_covariance_type": args.gmm_covariance_type,
        "gmm_converged": bool(gmm.converged_),
        "gmm_n_iter": int(gmm.n_iter_),
        "scaler_mean": torch.from_numpy(scaler.mean_.astype(np.float32)),
        "scaler_scale": torch.from_numpy(scaler.scale_.astype(np.float32)),
        "gmm_weights": torch.from_numpy(gmm.weights_.astype(np.float32)),
        "gmm_means": torch.from_numpy(gmm.means_.astype(np.float32)),
        "gmm_covariances": torch.from_numpy(gmm.covariances_.astype(np.float32)),
        "gmm_precisions_cholesky": torch.from_numpy(
            gmm.precisions_cholesky_.astype(np.float32)
        ),
    }
    torch.save(model_payload, output_dir / "prototype_model.pt")

    metadata = {
        "cache_version": 1,
        "dataset_root": str(args.dataset_root),
        "audio_backend": args.audio_backend,
        "song_target_dir": "song_targets",
        "segment_feature_dir": "segment_features",
        "num_songs": len(processed_song_ids),
        "num_segments": int(all_features.shape[0]),
        "num_prototypes": int(args.num_prototypes),
        "harmonic_stem_names": list(harmonic_stem_names),
        "sample_rate": int(args.sample_rate),
        "hop_length": int(args.hop_length),
        "n_fft": int(args.n_fft),
        "n_bins": int(args.n_bins),
        "bins_per_octave": int(args.bins_per_octave),
        "boundary_guard_seconds": float(args.boundary_guard_seconds),
        "min_inner_seconds": float(args.min_inner_seconds),
        "gmm_covariance_type": args.gmm_covariance_type,
        "gmm_converged": bool(gmm.converged_),
        "gmm_n_iter": int(gmm.n_iter_),
        "skipped_missing_boundaries": int(skipped_missing_boundaries),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(
        f"saved {len(processed_song_ids)} songs / {all_features.shape[0]} segments "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    main()
