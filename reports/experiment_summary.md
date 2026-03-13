# Experiment Summary

Generated: 2026-03-13 01:30:42 UTC

Runs: 2

| run | status | best_epoch | best_downbeat_f1 | last_epoch | last_downbeat_f1 | seed | lr | batch | model | branch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_beat_plus_downbeat_logits | complete | 26 | 0.3371 | 30 | 0.3277 | 42 | 0.000300 | 8 | L6/H64/O256 | exp/beat-plus-downbeat-logits |
| beat_transcription | complete | 21 | 0.3141 | 30 | 0.2986 | 42 | 0.000300 | 8 | L6/H64/O256 | - |

## exp_beat_plus_downbeat_logits

| field | value |
| --- | --- |
| path | outputs/exp_beat_plus_downbeat_logits |
| status | complete |
| best_epoch | 26 |
| best_downbeat_f1 | 0.3371 |
| best_beat_f1 | 0.5242 |
| best_val_loss | 1.0373 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.3277 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/beat-plus-downbeat-logits |
| git_commit | 89b73fe9068db71734438b17169c4f782572d53a |
| git_dirty | true |

## beat_transcription

| field | value |
| --- | --- |
| path | outputs/beat_transcription |
| status | complete |
| best_epoch | 21 |
| best_downbeat_f1 | 0.3141 |
| best_beat_f1 | 0.4926 |
| best_val_loss | 0.9633 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.2986 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | - |
| git_commit | - |
| git_dirty | - |
