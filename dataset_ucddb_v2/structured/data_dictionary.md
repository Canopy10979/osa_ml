# data_dictionary — dataset_ucddb_v2

`epoch_features.parquet` — 20,789 rows x 73 cols. One row = one 30 s epoch.

| column | dtype | null % | min | max |
|---|---|---|---|---|
| `subject` | str | 0.00 | — | 25 distinct |
| `epoch` | int64 | 0.00 | 0 | 924 |
| `spo2_mean` | float64 | 0.00 | -0.2922 | 99.96 |
| `spo2_std` | float64 | 0.00 | 0 | 49.5 |
| `spo2_min` | float64 | 0.00 | -0.379 | 99.87 |
| `spo2_max` | float64 | 0.00 | -0.2369 | 100.2 |
| `spo2_range` | float64 | 0.00 | 0 | 100.6 |
| `spo2_iqr` | float64 | 0.00 | 0 | 100.2 |
| `spo2_rel_amp` | float64 | 0.30 | 0 | 1 |
| `sound_mean` | float64 | 39.30 | -0.07155 | 1.596 |
| `sound_std` | float64 | 39.30 | 0 | 3.743 |
| `sound_min` | float64 | 39.30 | -4.024 | 0.0002442 |
| `sound_max` | float64 | 39.30 | -0.003663 | 3.978 |
| `sound_range` | float64 | 39.30 | 0 | 7.998 |
| `sound_iqr` | float64 | 39.30 | 0 | 7.882 |
| `sound_rel_amp` | float64 | 39.60 | 0 | 1 |
| `flow_mean` | float64 | 0.00 | -10.7 | 5.096 |
| `flow_std` | float64 | 0.00 | 0 | 19.11 |
| `flow_min` | float64 | 0.00 | -20.75 | 0.1575 |
| `flow_max` | float64 | 0.00 | -0.3504 | 20.59 |
| `flow_range` | float64 | 0.00 | 0 | 41.34 |
| `flow_iqr` | float64 | 0.00 | 0 | 40.29 |
| `flow_rel_amp` | float64 | 0.30 | 0 | 1 |
| `sum_mean` | float64 | 0.00 | -1.06 | 2.258 |
| `sum_std` | float64 | 0.00 | 0 | 3.674 |
| `sum_min` | float64 | 0.00 | -4.102 | 0.06667 |
| `sum_max` | float64 | 0.00 | 0.0002442 | 4.079 |
| `sum_range` | float64 | 0.00 | 0 | 8.178 |
| `sum_iqr` | float64 | 0.00 | 0 | 8.029 |
| `sum_rel_amp` | float64 | 0.30 | 0 | 1 |
| `ribcage_mean` | float64 | 0.00 | -0.9838 | 1.926 |
| `ribcage_std` | float64 | 0.00 | 0 | 3.613 |
| `ribcage_min` | float64 | 0.00 | -4.141 | 0.06276 |
| `ribcage_max` | float64 | 0.00 | 0.0002442 | 4.122 |
| `ribcage_range` | float64 | 0.00 | 0 | 8.264 |
| `ribcage_iqr` | float64 | 0.00 | 0 | 7.908 |
| `ribcage_rel_amp` | float64 | 0.30 | 0 | 1 |
| `abdo_mean` | float64 | 0.00 | -1.124 | 2.081 |
| `abdo_std` | float64 | 0.00 | 0 | 3.856 |
| `abdo_min` | float64 | 0.00 | -4.145 | 0.07057 |
| `abdo_max` | float64 | 0.00 | 0.0002442 | 4.107 |
| `abdo_range` | float64 | 0.00 | 0 | 8.248 |
| `abdo_iqr` | float64 | 0.00 | 0 | 8.139 |
| `abdo_rel_amp` | float64 | 0.30 | 0 | 1 |
| `pulse_mean` | float64 | 0.00 | -1.056 | 247.3 |
| `pulse_std` | float64 | 0.00 | 0 | 119.9 |
| `pulse_min` | float64 | 0.00 | -1.309 | 247 |
| `pulse_max` | float64 | 0.00 | -0.84 | 248 |
| `pulse_range` | float64 | 0.00 | 0 | 249.2 |
| `pulse_iqr` | float64 | 0.00 | 0 | 248.4 |
| `pulse_rel_amp` | float64 | 0.30 | 0 | 1 |
| `spo2_desat` | float64 | 0.00 | 0 | 99.96 |
| `spo2_below90` | float64 | 0.00 | 0 | 1 |
| `spo2_below92` | float64 | 0.00 | 0 | 1 |
| `effort_ratio` | float64 | 0.43 | 0.01197 | 156.9 |
| `effort_sum` | float64 | 0.00 | 0 | 7.297 |
| `stage` | int64 | 0.00 | 0 | 8 |
| `asleep` | int64 | 0.00 | 0 | 1 |
| `event` | int64 | 0.00 | 0 | 1 |
| `spo2_desat_ctx1` | float64 | 0.00 | 0 | 99.96 |
| `spo2_min_ctx1` | float64 | 0.00 | -0.3316 | 96.68 |
| `flow_rel_amp_ctx1` | float64 | 0.22 | 0 | 1 |
| `effort_sum_ctx1` | float64 | 0.00 | 0 | 7.143 |
| `pulse_mean_ctx1` | float64 | 0.00 | -1.044 | 238.1 |
| `pulse_std_ctx1` | float64 | 0.00 | 0 | 56.51 |
| `spo2_desat_ctx2` | float64 | 0.00 | 0 | 93.8 |
| `spo2_min_ctx2` | float64 | 0.00 | -0.3316 | 96.66 |
| `flow_rel_amp_ctx2` | float64 | 0.14 | 0 | 1 |
| `effort_sum_ctx2` | float64 | 0.00 | 0 | 6.9 |
| `pulse_mean_ctx2` | float64 | 0.00 | 0 | 175.1 |
| `pulse_std_ctx2` | float64 | 0.00 | 0 | 40.03 |
| `ahi` | int64 | 0.00 | 2 | 91 |
| `osa_15` | int64 | 0.00 | 0 | 1 |

## Notes

`event` = 1 if an expert-scored apnoea/hypopnoea (obstructive, central or mixed) overlaps the epoch. PB and POSSIBLE are excluded.
`asleep` = stage not in {0 wake, 8 artefact}. `ahi` is the recorded PSG AHI; `osa_15` = AHI >= 15. Both are subject-level and must never be used as features.
`*_rel_amp` = epoch amplitude over a 2-minute rolling maximum. `spo2_desat` = 2-minute baseline SpO2 minus epoch minimum.