# data_dictionary — minute_features

Rows: 13286 (one per minute of ECG). Records: 27. Positive class `apnea`=1 share: 0.489

| column | dtype | null % | min | max / cardinality |
|---|---|---|---|---|
| record | str | 0.0 | — | 27 distinct |
| minute | int64 | 0.0 | 0 | 576 |
| label | str | 0.0 | — | 2 distinct |
| n_beats | int64 | 0.0 | 0 | 112 |
| rr_mean | float64 | 0.1 | 0.4533 | 1.496 |
| rr_std | float64 | 0.1 | 0.006651 | 0.8115 |
| rr_min | float64 | 0.1 | 0.3 | 1.18 |
| rr_max | float64 | 0.1 | 0.61 | 2.5 |
| rr_range | float64 | 0.1 | 0.03 | 1.83 |
| rr_cv | float64 | 0.1 | 0.008893 | 0.6247 |
| rmssd | float64 | 0.1 | 0.007297 | 0.6965 |
| pnn50 | float64 | 0.1 | 0 | 0.9091 |
| hr_mean | float64 | 0.1 | 40.12 | 132.4 |
| rr_skew | float64 | 0.1 | -4.864 | 6.509 |
| rr_kurt | float64 | 0.1 | -2.462 | 43.18 |
| rr_iqr | float64 | 0.1 | 0.01 | 1.65 |
| p_cvhr | float64 | 0.0 | 0 | 0 |
| p_vlf | float64 | 0.0 | 0 | 0 |
| p_lf | float64 | 0.0 | 1.543e-05 | 0.03305 |
| p_hf | float64 | 0.0 | 6.454e-06 | 0.007111 |
| r_cvhr | float64 | 0.0 | 0 | 0 |
| r_lf | float64 | 0.0 | 0.07204 | 0.7428 |
| r_hf | float64 | 0.0 | 0.002235 | 0.7513 |
| lf_hf | float64 | 0.0 | 0.1319 | 79.96 |
| p_cvhr_log | float64 | 0.0 | -12 | -12 |
| peak_hz | float64 | 0.0 | 0.01562 | 0.2969 |
| edr_resp | float64 | 0.0 | 0.006182 | 0.8316 |
| edr_cvhr | float64 | 0.0 | 0 | 0 |
| edr_peak_hz | float64 | 0.0 | 0.01562 | 0.5469 |
| edr_std | float64 | 0.0 | 0.01155 | 4.852 |
| rr_std_ctx1 | float64 | 0.0 | 0.006783 | 0.5983 |
| rr_std_ctx2 | float64 | 0.0 | 0.009778 | 0.5725 |
| rr_std_ctx5 | float64 | 0.0 | 0.01021 | 0.4845 |
| rr_std_delta | float64 | 0.1 | -0.341 | 0.4511 |
| rr_cv_ctx1 | float64 | 0.0 | 0.008963 | 0.5564 |
| rr_cv_ctx2 | float64 | 0.0 | 0.01305 | 0.5043 |
| rr_cv_ctx5 | float64 | 0.0 | 0.01365 | 0.4747 |
| rr_cv_delta | float64 | 0.1 | -0.3546 | 0.4187 |
| rmssd_ctx1 | float64 | 0.0 | 0.007468 | 0.6965 |
| rmssd_ctx2 | float64 | 0.0 | 0.008264 | 0.4563 |
| rmssd_ctx5 | float64 | 0.0 | 0.00852 | 0.3882 |
| rmssd_delta | float64 | 0.1 | -0.2913 | 0.3502 |
| p_cvhr_log_ctx1 | float64 | 0.0 | -12 | -12 |
| p_cvhr_log_ctx2 | float64 | 0.0 | -12 | -12 |
| p_cvhr_log_ctx5 | float64 | 0.0 | -12 | -12 |
| p_cvhr_log_delta | float64 | 0.0 | 0 | 0 |
| r_cvhr_ctx1 | float64 | 0.0 | 0 | 0 |
| r_cvhr_ctx2 | float64 | 0.0 | 0 | 0 |
| r_cvhr_ctx5 | float64 | 0.0 | 0 | 0 |
| r_cvhr_delta | float64 | 0.0 | 0 | 0 |
| lf_hf_ctx1 | float64 | 0.0 | 0.1753 | 68.35 |
| lf_hf_ctx2 | float64 | 0.0 | 0.1918 | 58.29 |
| lf_hf_ctx5 | float64 | 0.0 | 0.2048 | 38.17 |
| lf_hf_delta | float64 | 0.0 | -23.74 | 54.19 |
| hr_mean_ctx1 | float64 | 0.0 | 44.93 | 132.4 |
| hr_mean_ctx2 | float64 | 0.0 | 46.88 | 132.4 |
| hr_mean_ctx5 | float64 | 0.0 | 47.48 | 91.22 |
| hr_mean_delta | float64 | 0.1 | -29.46 | 57.96 |
| edr_resp_ctx1 | float64 | 0.0 | 0.006728 | 0.8116 |
| edr_resp_ctx2 | float64 | 0.0 | 0.00873 | 0.8093 |
| edr_resp_ctx5 | float64 | 0.0 | 0.03742 | 0.8054 |
| edr_resp_delta | float64 | 0.0 | -0.3301 | 0.4869 |
| rr_mean_z | float64 | 0.1 | -6.141 | 7.842 |
| rr_std_z | float64 | 0.1 | -2.879 | 10.31 |
| rmssd_z | float64 | 0.1 | -3.396 | 16.81 |
| p_cvhr_log_z | float64 | 0.0 | 0 | 0 |
| hr_mean_z | float64 | 0.1 | -5.264 | 11.02 |
| apnea | int64 | 0.0 | 0 | 1 |
| class_prefix | str | 0.0 | — | 3 distinct |

## Notes
- `label` is the database's own per-minute annotation ('A'/'N'); `apnea` is its 0/1 encoding and is the modelling target.
- `p_cvhr` / `r_cvhr` capture 0.01–0.04 Hz cyclical variation in heart rate — the canonical ECG signature of obstructive apnoea.
- `edr_*` are ECG-derived respiration features from R-wave amplitude modulation.
- `*_ctx{k}` are centred rolling means over ±k minutes; `*_delta` is the deviation from the ±5-minute mean.
- `*_z` are within-record standardised values.
- Spectral features use a 5-minute window centred on the target minute, because CVHR cycles (25–100 s) cannot be resolved in 60 s.

## Leakage note
`class_prefix` and `record` encode the record identity and must NEVER be used as model features. `n_apnea_minutes` / `apnea_index` are derived from the labels and are subject-level outcome summaries, not predictors.