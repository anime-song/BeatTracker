# Experiment Summary

Generated: 2026-03-13 06:30:24 UTC

Runs: 6

| run | status | best_epoch | best_downbeat_f1 | last_epoch | last_downbeat_f1 | seed | lr | batch | meter_w | model | branch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_meter_classification_w0_0_5 | complete | 20 | 0.3903 | 30 | 0.3534 | 42 | 0.000300 | 8 | 0.050 | L6/H64/O256 | exp/meter-classification |
| exp_meter_classification_w0_1 | complete | 24 | 0.3508 | 30 | 0.3388 | 42 | 0.000300 | 8 | 0.100 | L6/H64/O256 | exp/meter-classification |
| exp_beat_plus_downbeat_logits | complete | 26 | 0.3371 | 30 | 0.3277 | 42 | 0.000300 | 8 | - | L6/H64/O256 | exp/beat-plus-downbeat-logits |
| exp_meter_classification_w0_3 | complete | 24 | 0.3249 | 30 | 0.3174 | 42 | 0.000300 | 8 | 0.300 | L6/H64/O256 | exp/meter-classification |
| beat_transcription | complete | 21 | 0.3141 | 30 | 0.2986 | 42 | 0.000300 | 8 | - | L6/H64/O256 | - |
| exp_meter_classification | complete | 25 | 0.2561 | 30 | 0.2559 | 42 | 0.000300 | 8 | - | L6/H64/O256 | exp/meter-classification |

## exp_meter_classification_w0_0_5

| field | value |
| --- | --- |
| path | outputs/exp_meter_classification_w0_0_5 |
| status | complete |
| best_epoch | 20 |
| best_downbeat_f1 | 0.3903 |
| best_beat_f1 | 0.5379 |
| best_val_loss | 1.0949 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.3534 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/meter-classification |
| git_commit | 8af5e0224aad55ca3e419c04bc182d2e86acb6b0 |
| git_dirty | true |

## exp_meter_classification_w0_1

| field | value |
| --- | --- |
| path | outputs/exp_meter_classification_w0_1 |
| status | complete |
| best_epoch | 24 |
| best_downbeat_f1 | 0.3508 |
| best_beat_f1 | 0.5380 |
| best_val_loss | 1.4538 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.3388 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.100 |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/meter-classification |
| git_commit | 07357ebe083aafc7234dad09257aa3b76e9e6ebe |
| git_dirty | true |

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
| meter_loss_weight | - |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/beat-plus-downbeat-logits |
| git_commit | 89b73fe9068db71734438b17169c4f782572d53a |
| git_dirty | true |

## exp_meter_classification_w0_3

| field | value |
| --- | --- |
| path | outputs/exp_meter_classification_w0_3 |
| status | complete |
| best_epoch | 24 |
| best_downbeat_f1 | 0.3249 |
| best_beat_f1 | 0.5550 |
| best_val_loss | 2.4279 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.3174 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.300 |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/meter-classification |
| git_commit | e74e83453e2b230a4b905351083d26bc3a45b53d |
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
| meter_loss_weight | - |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | - |
| git_commit | - |
| git_dirty | - |

## exp_meter_classification

| field | value |
| --- | --- |
| path | outputs/exp_meter_classification |
| status | complete |
| best_epoch | 25 |
| best_downbeat_f1 | 0.2561 |
| best_beat_f1 | 0.5061 |
| best_val_loss | 5.4904 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.2559 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | - |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/meter-classification |
| git_commit | bcef203d753b4517de3fc0dfb45e57021ac92e0e |
| git_dirty | true |
