# Experiment Summary

Generated: 2026-03-16 07:40:56 UTC

Runs: 37

| run | status | best_epoch | best_downbeat_f1 | last_epoch | last_downbeat_f1 | seed | lr | batch | meter_w | drum_aux_w | drum_hf | stem_drop | init | model | branch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_chord_preinit_stemdrop2_drumaux_highfreq_repeatssm | complete | 23 | 0.4705 | 30 | 0.4444 | 42 | 0.000300 | 8 | 0.050 | 0.100 | true | 4 | backbone:ema_state_dict | L6/H64/O256 | exp/repeat-ssm-downbeat-consistency |
| exp_chord_preinit_stemdrop2_drumaux_highfreq | complete | 25 | 0.4642 | 30 | 0.4510 | 42 | 0.000300 | 8 | 0.050 | 0.100 | true | 4 | backbone:ema_state_dict | L6/H64/O256 | exp/drum-highfreq-flux-aux |
| exp_recover_43c5ebc_repeat_beat2 | complete | 18 | 0.4636 | 30 | 0.4391 | 42 | 0.000300 | 8 | 0.050 | 0.100 | true | 4 | backbone:ema_state_dict | L6/H64/O256 | exp/repeat-ssm-downbeat-consistency |
| exp_drum_highfreq_aux_mask_when_drums_dropped | complete | 20 | 0.4611 | 30 | 0.4420 | 42 | 0.000300 | 8 | 0.050 | 0.100 | true | 4 | backbone:ema_state_dict | L6/H64/O256 | exp/drum-highfreq-flux-aux |
| exp_chord_preinit_stemdrop2_drumaux | complete | 23 | 0.4598 | 30 | 0.4472 | 42 | 0.000300 | 8 | 0.050 | 0.100 | - | 4 | backbone:ema_state_dict | L6/H64/O256 | exp/drum-aux-flux-onset |
| exp_chord_preinit_stemdrop2_drumaux_highfreq_repeatssm_bar | complete | 21 | 0.4594 | 30 | 0.4370 | 42 | 0.000300 | 8 | 0.050 | 0.100 | true | 4 | backbone:ema_state_dict | L6/H64/O256 | exp/repeat-ssm-downbeat-consistency |
| exp_chord_preinit_stemdrop2_drumaux_basslowflux | complete | 17 | 0.4536 | 30 | 0.4387 | 42 | 0.000300 | 8 | 0.050 | 0.100 | - | 4 | backbone:ema_state_dict | L6/H64/O256 | exp/bass-aux-lowflux-harmonic-change |
| exp_chord_preinit_stemdrop2 | complete | 18 | 0.4519 | 30 | 0.4275 | 42 | 0.000300 | 8 | 0.050 | - | - | 4 | backbone:ema_state_dict | L6/H64/O256 | exp/stem-dropout-energy-ranking |
| exp_recover_43c5ebc_repeat_beat | complete | 22 | 0.4509 | 30 | 0.4351 | 42 | 0.000300 | 8 | 0.050 | 0.100 | true | 4 | backbone:ema_state_dict | L6/H64/O256 | exp/recover-43c5ebc |
| exp_chord_preinit_backbone | complete | 11 | 0.4471 | 30 | 0.3977 | 42 | 0.000300 | 8 | 0.050 | - | - | - | backbone:ema_state_dict | L6/H64/O256 | exp/chord-preinit-backbone |
| exp_meter_from_rhythm_head | complete | 13 | 0.4464 | 30 | 0.4135 | 42 | 0.000300 | 8 | 0.050 | 0.100 | true | 4 | backbone:ema_state_dict | L6/H64/O256 | exp/meter-from-rhythm-head |
| exp_chord_preinit_stemdrop2_drumaux_bassaux | complete | 20 | 0.4442 | 30 | 0.4281 | 42 | 0.000300 | 8 | 0.050 | 0.100 | - | 4 | backbone:ema_state_dict | L6/H64/O256 | exp/bass-aux-lowflux-harmonic-change |
| exp_chord_preinit_stemdrop2_drumaux_pianoaux | complete | 16 | 0.4440 | 30 | 0.4029 | 42 | 0.000300 | 8 | 0.050 | 0.100 | - | 4 | backbone:ema_state_dict | L6/H64/O256 | exp/piano-broadband-flux-aux |
| exp_chord_preinit_stemdrop2_phase | complete | 25 | 0.4409 | 30 | 0.4318 | 42 | 0.000300 | 8 | 0.050 | - | - | 4 | backbone:ema_state_dict | L6/H64/O256 | exp/beat-phase-classification |
| exp_chord_preinit_stemdrop2_drumaux_bassharmonic | complete | 12 | 0.4406 | 30 | 0.4150 | 42 | 0.000300 | 8 | 0.050 | 0.100 | - | 4 | backbone:ema_state_dict | L6/H64/O256 | exp/bass-aux-lowflux-harmonic-change |
| exp_chord_preinit_stemdrop2_metertau0_5 | complete | 16 | 0.4401 | 30 | 0.4131 | 42 | 0.000300 | 8 | 0.050 | - | - | 4 | backbone:ema_state_dict | L6/H64/O256 | exp/meter-balancedsoftmax-tau0_5 |
| exp_meter_from_rhythm_head_w0_0 | complete | 18 | 0.3979 | 30 | 0.3855 | 42 | 0.000300 | 8 | 0.050 | 0.100 | true | 4 | backbone:ema_state_dict | L6/H64/O256 | exp/meter-from-rhythm-head |
| exp_meter_classification_w0_0_5 | complete | 20 | 0.3903 | 30 | 0.3534 | 42 | 0.000300 | 8 | 0.050 | - | - | - | - | L6/H64/O256 | exp/meter-classification |
| exp_chord_preinit_stemdrop2_drumaux_highfreq_repeatssm_timestretch_pm10 | complete | 30 | 0.3784 | 30 | 0.3784 | 42 | 0.000300 | 8 | 0.050 | 0.100 | true | 4 | backbone:ema_state_dict | L6/H64/O256 | exp/tempo-augmentation-pm10 |
| exp_meter_classification_w0_0_5_specaug_f0_00_t0_05 | complete | 23 | 0.3743 | 30 | 0.3569 | 42 | 0.000300 | 8 | 0.050 | - | - | - | - | L6/H64/O256 | exp/spec-augment-mask-rate |
| exp_meter_classification_w0_0_5_specaug_f0_02_t0_05 | complete | 23 | 0.3705 | 30 | 0.3581 | 42 | 0.000300 | 8 | 0.050 | - | - | - | - | L6/H64/O256 | exp/spec-augment-mask-rate |
| exp_meter_classification_w0_0_5_specaug_f0_05_t0_10 | complete | 25 | 0.3604 | 30 | 0.3439 | 42 | 0.000300 | 8 | 0.050 | - | - | - | - | L6/H64/O256 | exp/spec-augment-mask-rate |
| exp_meter_classification_w0_0_5_beatpw10 | complete | 19 | 0.3599 | 30 | 0.3255 | 42 | 0.000300 | 8 | 0.050 | - | - | - | - | L6/H64/O256 | exp/beat-pos-weight |
| exp_meter_classification_w0_0_5_specaug_f0_00_t0_10 | complete | 24 | 0.3581 | 30 | 0.3552 | 42 | 0.000300 | 8 | 0.050 | - | - | - | - | L6/H64/O256 | exp/spec-augment-mask-rate |
| exp_meter_from_rhythm_head_w0_01 | in_progress | 13 | 0.3547 | 13 | 0.3547 | 42 | 0.000300 | 8 | 0.050 | 0.100 | true | 4 | backbone:ema_state_dict | L6/H64/O256 | exp/meter-from-rhythm-head |
| exp_meter_classification_w0_1 | complete | 24 | 0.3508 | 30 | 0.3388 | 42 | 0.000300 | 8 | 0.100 | - | - | - | - | L6/H64/O256 | exp/meter-classification |
| exp_meter_classification_w0_0_5_specaug_f0_02_t0_00 | complete | 29 | 0.3504 | 30 | 0.3468 | 42 | 0.000300 | 8 | 0.050 | - | - | - | - | L6/H64/O256 | exp/spec-augment-mask-rate |
| exp_meter_classification_w0_0_5_beatpw7_5 | complete | 16 | 0.3493 | 30 | 0.3248 | 42 | 0.000300 | 8 | 0.050 | - | - | - | - | L6/H64/O256 | exp/beat-pos-weight |
| exp_time_stretch_pm50 | complete | 22 | 0.3476 | 30 | 0.3372 | 42 | 0.000300 | 8 | 0.050 | - | - | - | - | L6/H64/O256 | exp/time-stretch-augmentation |
| exp_meter_numerator_w0_05 | complete | 28 | 0.3398 | 30 | 0.3379 | 42 | 0.000300 | 8 | 0.050 | - | - | - | - | L6/H64/O256 | exp/meter-numerator-classification |
| exp_meter_context_lowres_w0_05 | complete | 22 | 0.3382 | 30 | 0.3158 | 42 | 0.000300 | 8 | 0.050 | - | - | - | - | L6/H64/O256 | exp/meter-context-lowres |
| exp_beat_plus_downbeat_logits | complete | 26 | 0.3371 | 30 | 0.3277 | 42 | 0.000300 | 8 | - | - | - | - | - | L6/H64/O256 | exp/beat-plus-downbeat-logits |
| exp_meter_classification_w0_3 | complete | 24 | 0.3249 | 30 | 0.3174 | 42 | 0.000300 | 8 | 0.300 | - | - | - | - | L6/H64/O256 | exp/meter-classification |
| exp_meter_downbeat_conditioning_w0_05 | complete | 18 | 0.3248 | 30 | 0.3090 | 42 | 0.000300 | 8 | 0.050 | - | - | - | - | L6/H64/O256 | exp/meter-downbeat-conditioning |
| exp_meter_classification_w0_0_5_specaug_f0_05_t0_00 | complete | 21 | 0.3146 | 30 | 0.2990 | 42 | 0.000300 | 8 | 0.050 | - | - | - | - | L6/H64/O256 | exp/spec-augment-mask-rate |
| beat_transcription | complete | 21 | 0.3141 | 30 | 0.2986 | 42 | 0.000300 | 8 | - | - | - | - | - | L6/H64/O256 | - |
| exp_meter_classification | complete | 25 | 0.2561 | 30 | 0.2559 | 42 | 0.000300 | 8 | - | - | - | - | - | L6/H64/O256 | exp/meter-classification |

## exp_chord_preinit_stemdrop2_drumaux_highfreq_repeatssm

| field | value |
| --- | --- |
| path | outputs/exp_chord_preinit_stemdrop2_drumaux_highfreq_repeatssm |
| status | complete |
| best_epoch | 23 |
| best_downbeat_f1 | 0.4705 |
| best_beat_f1 | 0.6554 |
| best_val_loss | 1.2617 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.4444 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| drum_aux_loss_weight | 0.100 |
| drum_aux_use_high_frequency_flux | true |
| stem_dropout_max_count | 4 |
| init_scope | backbone |
| init_from | model_epoch_200.pt |
| init_state_source | ema_state_dict |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/repeat-ssm-downbeat-consistency |
| git_commit | 2dc24e5d708fbff50b002dddfb08a16eb5dc2ee4 |
| git_dirty | true |

## exp_chord_preinit_stemdrop2_drumaux_highfreq

| field | value |
| --- | --- |
| path | outputs/exp_chord_preinit_stemdrop2_drumaux_highfreq |
| status | complete |
| best_epoch | 25 |
| best_downbeat_f1 | 0.4642 |
| best_beat_f1 | 0.6441 |
| best_val_loss | 1.3692 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.4510 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| drum_aux_loss_weight | 0.100 |
| drum_aux_use_high_frequency_flux | true |
| stem_dropout_max_count | 4 |
| init_scope | backbone |
| init_from | model_epoch_200.pt |
| init_state_source | ema_state_dict |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/drum-highfreq-flux-aux |
| git_commit | e61c0b4e430a6f79535e8c907c0e5b0783051b4c |
| git_dirty | true |

## exp_recover_43c5ebc_repeat_beat2

| field | value |
| --- | --- |
| path | outputs/exp_recover_43c5ebc_repeat_beat2 |
| status | complete |
| best_epoch | 18 |
| best_downbeat_f1 | 0.4636 |
| best_beat_f1 | 0.6462 |
| best_val_loss | 1.2446 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.4391 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| drum_aux_loss_weight | 0.100 |
| drum_aux_use_high_frequency_flux | true |
| stem_dropout_max_count | 4 |
| init_scope | backbone |
| init_from | model_epoch_200.pt |
| init_state_source | ema_state_dict |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/repeat-ssm-downbeat-consistency |
| git_commit | eed926ab64bf5c489ebb504f5cea8767493a48f6 |
| git_dirty | true |

## exp_drum_highfreq_aux_mask_when_drums_dropped

| field | value |
| --- | --- |
| path | outputs/exp_drum_highfreq_aux_mask_when_drums_dropped |
| status | complete |
| best_epoch | 20 |
| best_downbeat_f1 | 0.4611 |
| best_beat_f1 | 0.6470 |
| best_val_loss | 1.2396 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.4420 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| drum_aux_loss_weight | 0.100 |
| drum_aux_use_high_frequency_flux | true |
| stem_dropout_max_count | 4 |
| init_scope | backbone |
| init_from | model_epoch_200.pt |
| init_state_source | ema_state_dict |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/drum-highfreq-flux-aux |
| git_commit | ae23cd4b48840e610c241cc723511153c0bfbb4f |
| git_dirty | true |

## exp_chord_preinit_stemdrop2_drumaux

| field | value |
| --- | --- |
| path | outputs/exp_chord_preinit_stemdrop2_drumaux |
| status | complete |
| best_epoch | 23 |
| best_downbeat_f1 | 0.4598 |
| best_beat_f1 | 0.6548 |
| best_val_loss | 1.2471 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.4472 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| drum_aux_loss_weight | 0.100 |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | 4 |
| init_scope | backbone |
| init_from | model_epoch_200.pt |
| init_state_source | ema_state_dict |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/drum-aux-flux-onset |
| git_commit | f9c10e4c96d429b7ff085d05e21029188377c398 |
| git_dirty | true |

## exp_chord_preinit_stemdrop2_drumaux_highfreq_repeatssm_bar

| field | value |
| --- | --- |
| path | outputs/exp_chord_preinit_stemdrop2_drumaux_highfreq_repeatssm_bar |
| status | complete |
| best_epoch | 21 |
| best_downbeat_f1 | 0.4594 |
| best_beat_f1 | 0.6514 |
| best_val_loss | 1.2455 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.4370 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| drum_aux_loss_weight | 0.100 |
| drum_aux_use_high_frequency_flux | true |
| stem_dropout_max_count | 4 |
| init_scope | backbone |
| init_from | model_epoch_200.pt |
| init_state_source | ema_state_dict |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/repeat-ssm-downbeat-consistency |
| git_commit | 43c5ebc0956f6303f2f07fab809fdc97af1efa14 |
| git_dirty | true |

## exp_chord_preinit_stemdrop2_drumaux_basslowflux

| field | value |
| --- | --- |
| path | outputs/exp_chord_preinit_stemdrop2_drumaux_basslowflux |
| status | complete |
| best_epoch | 17 |
| best_downbeat_f1 | 0.4536 |
| best_beat_f1 | 0.6364 |
| best_val_loss | 1.2103 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.4387 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| drum_aux_loss_weight | 0.100 |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | 4 |
| init_scope | backbone |
| init_from | model_epoch_200.pt |
| init_state_source | ema_state_dict |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/bass-aux-lowflux-harmonic-change |
| git_commit | adde35d3c2c9fce19fbff1d630b4ddd48e6b5b59 |
| git_dirty | true |

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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | 4 |
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

## exp_recover_43c5ebc_repeat_beat

| field | value |
| --- | --- |
| path | outputs/exp_recover_43c5ebc_repeat_beat |
| status | complete |
| best_epoch | 22 |
| best_downbeat_f1 | 0.4509 |
| best_beat_f1 | 0.6368 |
| best_val_loss | 1.3169 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.4351 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| drum_aux_loss_weight | 0.100 |
| drum_aux_use_high_frequency_flux | true |
| stem_dropout_max_count | 4 |
| init_scope | backbone |
| init_from | model_epoch_200.pt |
| init_state_source | ema_state_dict |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/recover-43c5ebc |
| git_commit | 43c5ebc0956f6303f2f07fab809fdc97af1efa14 |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | - |
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

## exp_meter_from_rhythm_head

| field | value |
| --- | --- |
| path | outputs/exp_meter_from_rhythm_head |
| status | complete |
| best_epoch | 13 |
| best_downbeat_f1 | 0.4464 |
| best_beat_f1 | 0.6033 |
| best_val_loss | 1.2967 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.4135 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| drum_aux_loss_weight | 0.100 |
| drum_aux_use_high_frequency_flux | true |
| stem_dropout_max_count | 4 |
| init_scope | backbone |
| init_from | model_epoch_200.pt |
| init_state_source | ema_state_dict |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/meter-from-rhythm-head |
| git_commit | 1ed6431a0e352535e30389ec4b94072498b6399b |
| git_dirty | true |

## exp_chord_preinit_stemdrop2_drumaux_bassaux

| field | value |
| --- | --- |
| path | outputs/exp_chord_preinit_stemdrop2_drumaux_bassaux |
| status | complete |
| best_epoch | 20 |
| best_downbeat_f1 | 0.4442 |
| best_beat_f1 | 0.6747 |
| best_val_loss | 1.2584 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.4281 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| drum_aux_loss_weight | 0.100 |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | 4 |
| init_scope | backbone |
| init_from | model_epoch_200.pt |
| init_state_source | ema_state_dict |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/bass-aux-lowflux-harmonic-change |
| git_commit | dee1996fc70872df2569fafc805fde22d31128c5 |
| git_dirty | true |

## exp_chord_preinit_stemdrop2_drumaux_pianoaux

| field | value |
| --- | --- |
| path | outputs/exp_chord_preinit_stemdrop2_drumaux_pianoaux |
| status | complete |
| best_epoch | 16 |
| best_downbeat_f1 | 0.4440 |
| best_beat_f1 | 0.6318 |
| best_val_loss | 1.1561 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.4029 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| drum_aux_loss_weight | 0.100 |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | 4 |
| init_scope | backbone |
| init_from | model_epoch_200.pt |
| init_state_source | ema_state_dict |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/piano-broadband-flux-aux |
| git_commit | dee1996fc70872df2569fafc805fde22d31128c5 |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | 4 |
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

## exp_chord_preinit_stemdrop2_drumaux_bassharmonic

| field | value |
| --- | --- |
| path | outputs/exp_chord_preinit_stemdrop2_drumaux_bassharmonic |
| status | complete |
| best_epoch | 12 |
| best_downbeat_f1 | 0.4406 |
| best_beat_f1 | 0.5886 |
| best_val_loss | 1.1710 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.4150 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| drum_aux_loss_weight | 0.100 |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | 4 |
| init_scope | backbone |
| init_from | model_epoch_200.pt |
| init_state_source | ema_state_dict |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/bass-aux-lowflux-harmonic-change |
| git_commit | 16b04fb9600983560eb50012866a4ae78312293e |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | 4 |
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

## exp_meter_from_rhythm_head_w0_0

| field | value |
| --- | --- |
| path | outputs/exp_meter_from_rhythm_head_w0_0 |
| status | complete |
| best_epoch | 18 |
| best_downbeat_f1 | 0.3979 |
| best_beat_f1 | 0.6316 |
| best_val_loss | 1.2141 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.3855 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| drum_aux_loss_weight | 0.100 |
| drum_aux_use_high_frequency_flux | true |
| stem_dropout_max_count | 4 |
| init_scope | backbone |
| init_from | model_epoch_200.pt |
| init_state_source | ema_state_dict |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/meter-from-rhythm-head |
| git_commit | 1ed6431a0e352535e30389ec4b94072498b6399b |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | - |
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

## exp_chord_preinit_stemdrop2_drumaux_highfreq_repeatssm_timestretch_pm10

| field | value |
| --- | --- |
| path | outputs/exp_chord_preinit_stemdrop2_drumaux_highfreq_repeatssm_timestretch_pm10 |
| status | complete |
| best_epoch | 30 |
| best_downbeat_f1 | 0.3784 |
| best_beat_f1 | 0.6596 |
| best_val_loss | 1.3199 |
| last_epoch | 30 |
| last_downbeat_f1 | 0.3784 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| drum_aux_loss_weight | 0.100 |
| drum_aux_use_high_frequency_flux | true |
| stem_dropout_max_count | 4 |
| init_scope | backbone |
| init_from | model_epoch_200.pt |
| init_state_source | ema_state_dict |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/tempo-augmentation-pm10 |
| git_commit | fa7462cfb77f811ce79e869947948c85cbf6d0ce |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | - |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | - |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | - |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | - |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | - |
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

## exp_meter_from_rhythm_head_w0_01

| field | value |
| --- | --- |
| path | outputs/exp_meter_from_rhythm_head_w0_01 |
| status | in_progress |
| best_epoch | 13 |
| best_downbeat_f1 | 0.3547 |
| best_beat_f1 | 0.5990 |
| best_val_loss | 1.2051 |
| last_epoch | 13 |
| last_downbeat_f1 | 0.3547 |
| configured_epochs | 30 |
| seed | 42 |
| lr | 0.000300 |
| batch_size | 8 |
| train_samples_per_epoch | 1024 |
| segment_seconds | 30.0 |
| meter_loss_weight | 0.050 |
| drum_aux_loss_weight | 0.100 |
| drum_aux_use_high_frequency_flux | true |
| stem_dropout_max_count | 4 |
| init_scope | backbone |
| init_from | model_epoch_200.pt |
| init_state_source | ema_state_dict |
| audio_backend | packed |
| scheduler | warmup_cosine |
| ema_decay | 0.9990 |
| model | L6/H64/O256 |
| resume | - |
| git_branch | exp/meter-from-rhythm-head |
| git_commit | 1ed6431a0e352535e30389ec4b94072498b6399b |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | - |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | - |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | - |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | - |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | - |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | - |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | - |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | - |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | - |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | - |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | - |
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
| drum_aux_loss_weight | - |
| drum_aux_use_high_frequency_flux | - |
| stem_dropout_max_count | - |
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
