# Experiment Summary

Generated: 2026-03-14 09:19:32 UTC

Runs: 22

| run | status | best_epoch | best_downbeat_f1 | last_epoch | last_downbeat_f1 | seed | lr | batch | meter_w | meter_tau | init | model | branch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_chord_preinit_stemdrop2 | complete | 18 | 0.4519 | 30 | 0.4275 | 42 | 0.000300 | 8 | 0.050 | - | backbone:ema_state_dict | L6/H64/O256 | exp/stem-dropout-energy-ranking |
| exp_chord_preinit_backbone | complete | 11 | 0.4471 | 30 | 0.3977 | 42 | 0.000300 | 8 | 0.050 | - | backbone:ema_state_dict | L6/H64/O256 | exp/chord-preinit-backbone |
| exp_chord_preinit_stemdrop2_phase | complete | 25 | 0.4409 | 30 | 0.4318 | 42 | 0.000300 | 8 | 0.050 | - | backbone:ema_state_dict | L6/H64/O256 | exp/beat-phase-classification |
| exp_chord_preinit_stemdrop2_metertau0_5 | complete | 16 | 0.4401 | 30 | 0.4131 | 42 | 0.000300 | 8 | 0.050 | 0.500 | backbone:ema_state_dict | L6/H64/O256 | exp/meter-balancedsoftmax-tau0_5 |
| exp_meter_classification_w0_0_5 | complete | 20 | 0.3903 | 30 | 0.3534 | 42 | 0.000300 | 8 | 0.050 | - | - | L6/H64/O256 | exp/meter-classification |
| exp_meter_classification_w0_0_5_specaug_f0_00_t0_05 | complete | 23 | 0.3743 | 30 | 0.3569 | 42 | 0.000300 | 8 | 0.050 | - | - | L6/H64/O256 | exp/spec-augment-mask-rate |
| exp_meter_classification_w0_0_5_specaug_f0_02_t0_05 | complete | 23 | 0.3705 | 30 | 0.3581 | 42 | 0.000300 | 8 | 0.050 | - | - | L6/H64/O256 | exp/spec-augment-mask-rate |
| exp_meter_classification_w0_0_5_specaug_f0_05_t0_10 | complete | 25 | 0.3604 | 30 | 0.3439 | 42 | 0.000300 | 8 | 0.050 | - | - | L6/H64/O256 | exp/spec-augment-mask-rate |
| exp_meter_classification_w0_0_5_beatpw10 | complete | 19 | 0.3599 | 30 | 0.3255 | 42 | 0.000300 | 8 | 0.050 | - | - | L6/H64/O256 | exp/beat-pos-weight |
| exp_meter_classification_w0_0_5_specaug_f0_00_t0_10 | complete | 24 | 0.3581 | 30 | 0.3552 | 42 | 0.000300 | 8 | 0.050 | - | - | L6/H64/O256 | exp/spec-augment-mask-rate |
| exp_meter_classification_w0_1 | complete | 24 | 0.3508 | 30 | 0.3388 | 42 | 0.000300 | 8 | 0.100 | - | - | L6/H64/O256 | exp/meter-classification |
| exp_meter_classification_w0_0_5_specaug_f0_02_t0_00 | complete | 29 | 0.3504 | 30 | 0.3468 | 42 | 0.000300 | 8 | 0.050 | - | - | L6/H64/O256 | exp/spec-augment-mask-rate |
| exp_meter_classification_w0_0_5_beatpw7_5 | complete | 16 | 0.3493 | 30 | 0.3248 | 42 | 0.000300 | 8 | 0.050 | - | - | L6/H64/O256 | exp/beat-pos-weight |
| exp_time_stretch_pm50 | complete | 22 | 0.3476 | 30 | 0.3372 | 42 | 0.000300 | 8 | 0.050 | - | - | L6/H64/O256 | exp/time-stretch-augmentation |
| exp_meter_numerator_w0_05 | complete | 28 | 0.3398 | 30 | 0.3379 | 42 | 0.000300 | 8 | 0.050 | - | - | L6/H64/O256 | exp/meter-numerator-classification |
| exp_meter_context_lowres_w0_05 | complete | 22 | 0.3382 | 30 | 0.3158 | 42 | 0.000300 | 8 | 0.050 | - | - | L6/H64/O256 | exp/meter-context-lowres |
| exp_beat_plus_downbeat_logits | complete | 26 | 0.3371 | 30 | 0.3277 | 42 | 0.000300 | 8 | - | - | - | L6/H64/O256 | exp/beat-plus-downbeat-logits |
| exp_meter_classification_w0_3 | complete | 24 | 0.3249 | 30 | 0.3174 | 42 | 0.000300 | 8 | 0.300 | - | - | L6/H64/O256 | exp/meter-classification |
| exp_meter_downbeat_conditioning_w0_05 | complete | 18 | 0.3248 | 30 | 0.3090 | 42 | 0.000300 | 8 | 0.050 | - | - | L6/H64/O256 | exp/meter-downbeat-conditioning |
| exp_meter_classification_w0_0_5_specaug_f0_05_t0_00 | complete | 21 | 0.3146 | 30 | 0.2990 | 42 | 0.000300 | 8 | 0.050 | - | - | L6/H64/O256 | exp/spec-augment-mask-rate |
| beat_transcription | complete | 21 | 0.3141 | 30 | 0.2986 | 42 | 0.000300 | 8 | - | - | - | L6/H64/O256 | - |
| exp_meter_classification | complete | 25 | 0.2561 | 30 | 0.2559 | 42 | 0.000300 | 8 | - | - | - | L6/H64/O256 | exp/meter-classification |

## exp_chord_preinit_stemdrop2

| field | value |
| --- | --- |
| path | outputs/exp_chord_preinit_stemdrop2 |
| status | complete |
| best_epoch | 18 |
| best_downbeat_f1 | 0.4519 |
| best_beat_f1 | 0.6497 |
| best_val_loss | 1.1892 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.4275 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| meter_balanced_softmax_tau | - |
| init_scope | backbone |
| init_from | model_epoch_200.pt |
| init_state_source | ema_state_dict |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/stem-dropout-energy-ranking |
| git_commit | f9c10e4c96d429b7ff085d05e21029188377c398 |
| git_dirty | true |

## exp_chord_preinit_backbone

| field | value |
| --- | --- |
| path | outputs/exp_chord_preinit_backbone |
| status | complete |
| best_epoch | 11 |
| best_downbeat_f1 | 0.4471 |
| best_beat_f1 | 0.5322 |
| best_val_loss | 1.0961 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.3977 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| meter_balanced_softmax_tau | - |
| init_scope | backbone |
| init_from | model_epoch_200.pt |
| init_state_source | ema_state_dict |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/chord-preinit-backbone |
| git_commit | 8af5e0224aad55ca3e419c04bc182d2e86acb6b0 |
| git_dirty | true |

## exp_chord_preinit_stemdrop2_phase

| field | value |
| --- | --- |
| path | outputs/exp_chord_preinit_stemdrop2_phase |
| status | complete |
| best_epoch | 25 |
| best_downbeat_f1 | 0.4409 |
| best_beat_f1 | 0.6570 |
| best_val_loss | 1.4329 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.4318 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| meter_balanced_softmax_tau | - |
| init_scope | backbone |
| init_from | /mnt/f/Github/BeatTracker/model_epoch_200.pt |
| init_state_source | ema_state_dict |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/beat-phase-classification |
| git_commit | f9c10e4c96d429b7ff085d05e21029188377c398 |
| git_dirty | true |

## exp_chord_preinit_stemdrop2_metertau0_5

| field | value |
| --- | --- |
| path | outputs/exp_chord_preinit_stemdrop2_metertau0_5 |
| status | complete |
| best_epoch | 16 |
| best_downbeat_f1 | 0.4401 |
| best_beat_f1 | 0.6440 |
| best_val_loss | 1.1168 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.4131 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| meter_balanced_softmax_tau | 0.500 |
| init_scope | backbone |
| init_from | model_epoch_200.pt |
| init_state_source | ema_state_dict |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/meter-balancedsoftmax-tau0_5 |
| git_commit | f9c10e4c96d429b7ff085d05e21029188377c398 |
| git_dirty | true |

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
| meter_balanced_softmax_tau | - |
| init_scope | - |
| init_from | - |
| init_state_source | - |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/meter-classification |
| git_commit | 8af5e0224aad55ca3e419c04bc182d2e86acb6b0 |
| git_dirty | true |

## exp_meter_classification_w0_0_5_specaug_f0_00_t0_05

| field | value |
| --- | --- |
| path | outputs/exp_meter_classification_w0_0_5_specaug_f0_00_t0_05 |
| status | complete |
| best_epoch | 23 |
| best_downbeat_f1 | 0.3743 |
| best_beat_f1 | 0.5330 |
| best_val_loss | 1.1145 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.3569 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| meter_balanced_softmax_tau | - |
| init_scope | - |
| init_from | - |
| init_state_source | - |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/spec-augment-mask-rate |
| git_commit | 1c6becc6e9ecd1f4043bc0dfb291c1ce1a861c55 |
| git_dirty | true |

## exp_meter_classification_w0_0_5_specaug_f0_02_t0_05

| field | value |
| --- | --- |
| path | outputs/exp_meter_classification_w0_0_5_specaug_f0_02_t0_05 |
| status | complete |
| best_epoch | 23 |
| best_downbeat_f1 | 0.3705 |
| best_beat_f1 | 0.5571 |
| best_val_loss | 1.1243 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.3581 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| meter_balanced_softmax_tau | - |
| init_scope | - |
| init_from | - |
| init_state_source | - |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/spec-augment-mask-rate |
| git_commit | 1c6becc6e9ecd1f4043bc0dfb291c1ce1a861c55 |
| git_dirty | true |

## exp_meter_classification_w0_0_5_specaug_f0_05_t0_10

| field | value |
| --- | --- |
| path | outputs/exp_meter_classification_w0_0_5_specaug_f0_05_t0_10 |
| status | complete |
| best_epoch | 25 |
| best_downbeat_f1 | 0.3604 |
| best_beat_f1 | 0.5355 |
| best_val_loss | 1.2867 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.3439 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| meter_balanced_softmax_tau | - |
| init_scope | - |
| init_from | - |
| init_state_source | - |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/spec-augment-mask-rate |
| git_commit | 1c6becc6e9ecd1f4043bc0dfb291c1ce1a861c55 |
| git_dirty | true |

## exp_meter_classification_w0_0_5_beatpw10

| field | value |
| --- | --- |
| path | outputs/exp_meter_classification_w0_0_5_beatpw10 |
| status | complete |
| best_epoch | 19 |
| best_downbeat_f1 | 0.3599 |
| best_beat_f1 | 0.6227 |
| best_val_loss | 1.2610 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.3255 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| meter_balanced_softmax_tau | - |
| init_scope | - |
| init_from | - |
| init_state_source | - |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/beat-pos-weight |
| git_commit | e545cd80edb38d95f645441fff508640064d0372 |
| git_dirty | true |

## exp_meter_classification_w0_0_5_specaug_f0_00_t0_10

| field | value |
| --- | --- |
| path | outputs/exp_meter_classification_w0_0_5_specaug_f0_00_t0_10 |
| status | complete |
| best_epoch | 24 |
| best_downbeat_f1 | 0.3581 |
| best_beat_f1 | 0.5582 |
| best_val_loss | 1.1460 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.3552 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| meter_balanced_softmax_tau | - |
| init_scope | - |
| init_from | - |
| init_state_source | - |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/spec-augment-mask-rate |
| git_commit | 1c6becc6e9ecd1f4043bc0dfb291c1ce1a861c55 |
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
| meter_balanced_softmax_tau | - |
| init_scope | - |
| init_from | - |
| init_state_source | - |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/meter-classification |
| git_commit | 07357ebe083aafc7234dad09257aa3b76e9e6ebe |
| git_dirty | true |

## exp_meter_classification_w0_0_5_specaug_f0_02_t0_00

| field | value |
| --- | --- |
| path | outputs/exp_meter_classification_w0_0_5_specaug_f0_02_t0_00 |
| status | complete |
| best_epoch | 29 |
| best_downbeat_f1 | 0.3504 |
| best_beat_f1 | 0.5507 |
| best_val_loss | 1.3130 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.3468 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| meter_balanced_softmax_tau | - |
| init_scope | - |
| init_from | - |
| init_state_source | - |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/spec-augment-mask-rate |
| git_commit | 1c6becc6e9ecd1f4043bc0dfb291c1ce1a861c55 |
| git_dirty | true |

## exp_meter_classification_w0_0_5_beatpw7_5

| field | value |
| --- | --- |
| path | outputs/exp_meter_classification_w0_0_5_beatpw7_5 |
| status | complete |
| best_epoch | 16 |
| best_downbeat_f1 | 0.3493 |
| best_beat_f1 | 0.6194 |
| best_val_loss | 1.1243 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.3248 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| meter_balanced_softmax_tau | - |
| init_scope | - |
| init_from | - |
| init_state_source | - |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/beat-pos-weight |
| git_commit | e545cd80edb38d95f645441fff508640064d0372 |
| git_dirty | true |

## exp_time_stretch_pm50

| field | value |
| --- | --- |
| path | outputs/exp_time_stretch_pm50 |
| status | complete |
| best_epoch | 22 |
| best_downbeat_f1 | 0.3476 |
| best_beat_f1 | 0.6170 |
| best_val_loss | 1.0687 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.3372 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| meter_balanced_softmax_tau | - |
| init_scope | - |
| init_from | - |
| init_state_source | - |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/time-stretch-augmentation |
| git_commit | 8af5e0224aad55ca3e419c04bc182d2e86acb6b0 |
| git_dirty | true |

## exp_meter_numerator_w0_05

| field | value |
| --- | --- |
| path | outputs/exp_meter_numerator_w0_05 |
| status | complete |
| best_epoch | 28 |
| best_downbeat_f1 | 0.3398 |
| best_beat_f1 | 0.5411 |
| best_val_loss | 1.2611 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.3379 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| meter_balanced_softmax_tau | - |
| init_scope | - |
| init_from | - |
| init_state_source | - |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/meter-numerator-classification |
| git_commit | e545cd80edb38d95f645441fff508640064d0372 |
| git_dirty | true |

## exp_meter_context_lowres_w0_05

| field | value |
| --- | --- |
| path | outputs/exp_meter_context_lowres_w0_05 |
| status | complete |
| best_epoch | 22 |
| best_downbeat_f1 | 0.3382 |
| best_beat_f1 | 0.5287 |
| best_val_loss | 1.1661 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.3158 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| meter_balanced_softmax_tau | - |
| init_scope | - |
| init_from | - |
| init_state_source | - |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/meter-context-lowres |
| git_commit | c0d4a5cb2ab7a1cbae03c46dd4b33363d2706921 |
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
| meter_balanced_softmax_tau | - |
| init_scope | - |
| init_from | - |
| init_state_source | - |
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
| meter_balanced_softmax_tau | - |
| init_scope | - |
| init_from | - |
| init_state_source | - |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/meter-classification |
| git_commit | e74e83453e2b230a4b905351083d26bc3a45b53d |
| git_dirty | true |

## exp_meter_downbeat_conditioning_w0_05

| field | value |
| --- | --- |
| path | outputs/exp_meter_downbeat_conditioning_w0_05 |
| status | complete |
| best_epoch | 18 |
| best_downbeat_f1 | 0.3248 |
| best_beat_f1 | 0.5252 |
| best_val_loss | 1.0562 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.3090 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| meter_balanced_softmax_tau | - |
| init_scope | - |
| init_from | - |
| init_state_source | - |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/meter-downbeat-conditioning |
| git_commit | 8af5e0224aad55ca3e419c04bc182d2e86acb6b0 |
| git_dirty | true |

## exp_meter_classification_w0_0_5_specaug_f0_05_t0_00

| field | value |
| --- | --- |
| path | outputs/exp_meter_classification_w0_0_5_specaug_f0_05_t0_00 |
| status | complete |
| best_epoch | 21 |
| best_downbeat_f1 | 0.3146 |
| best_beat_f1 | 0.4856 |
| best_val_loss | 1.1120 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.2990 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| meter_balanced_softmax_tau | - |
| init_scope | - |
| init_from | - |
| init_state_source | - |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/spec-augment-mask-rate |
| git_commit | 1c6becc6e9ecd1f4043bc0dfb291c1ce1a861c55 |
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
| meter_balanced_softmax_tau | - |
| init_scope | - |
| init_from | - |
| init_state_source | - |
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
| meter_balanced_softmax_tau | - |
| init_scope | - |
| init_from | - |
| init_state_source | - |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/meter-classification |
| git_commit | bcef203d753b4517de3fc0dfb45e57021ac92e0e |
| git_dirty | true |
