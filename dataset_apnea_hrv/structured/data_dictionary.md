# data_dictionary — dataset_apnea_hrv

`minute_features.parquet` — 30,445 rows x 57 cols. One row = one minute, featurised from the 5-minute frame centred on it.

| column | dtype | null % | min | max |
|---|---|---|---|---|
| `n_beats` | int64 | 0.00 | 188 | 503 |
| `rr_mean` | float64 | 0.00 | 596.5 | 1361 |
| `rr_median` | float64 | 0.00 | 601.4 | 1365 |
| `rr_std` | float64 | 0.00 | 7.056 | 215.1 |
| `rr_cv` | float64 | 0.00 | 0.008614 | 0.2166 |
| `rr_min` | float64 | 0.00 | 439.9 | 1229 |
| `rr_max` | float64 | 0.00 | 619.7 | 1891 |
| `rr_range` | float64 | 0.00 | 38.52 | 1100 |
| `rr_iqr` | float64 | 0.00 | 8.055 | 391.6 |
| `rr_mad` | float64 | 0.00 | 4.033 | 184.7 |
| `rr_skew` | float64 | 0.00 | -3.795 | 10.82 |
| `rr_kurt` | float64 | 0.00 | -1.627 | 155.5 |
| `hr_mean` | float64 | 0.00 | 44.09 | 100.6 |
| `rmssd` | float64 | 0.00 | 2.714 | 183 |
| `sdsd` | float64 | 0.00 | 2.715 | 183.5 |
| `pnn50` | float64 | 0.00 | 0 | 0.8342 |
| `pnn20` | float64 | 0.00 | 0 | 0.9397 |
| `sd1` | float64 | 0.00 | 1.92 | 129.8 |
| `sd2` | float64 | 0.00 | 9.483 | 297.8 |
| `sd_ratio` | float64 | 0.00 | 0.03827 | 1.3 |
| `total_power` | float64 | 0.00 | 21.47 | 4.102e+04 |
| `p_cvhr` | float64 | 0.00 | 2.467 | 1.946e+04 |
| `r_cvhr` | float64 | 0.00 | 0.009985 | 0.7489 |
| `p_lf` | float64 | 0.00 | 4.216 | 1.753e+04 |
| `r_lf` | float64 | 0.00 | 0.02618 | 0.9252 |
| `p_hf` | float64 | 0.00 | 1.421 | 5695 |
| `r_hf` | float64 | 0.00 | 0.002346 | 0.8653 |
| `lf_hf` | float64 | 0.00 | 0.07918 | 100.8 |
| `peak_hz` | float64 | 0.00 | 0 | 0.3917 |
| `cvhr_peak_hz` | float64 | 0.00 | 0.01667 | 0.03333 |
| `subject` | str | 0.00 | — | 77 distinct |
| `minute` | int64 | 0.00 | 2 | 471 |
| `apnea` | int64 | 0.00 | 0 | 1 |
| `rr_std_ctx1` | float64 | 0.00 | 7.884 | 209.3 |
| `rr_cv_ctx1` | float64 | 0.00 | 0.009095 | 0.2041 |
| `rmssd_ctx1` | float64 | 0.00 | 2.911 | 176.4 |
| `lf_hf_ctx1` | float64 | 0.00 | 0.09267 | 92.3 |
| `r_hf_ctx1` | float64 | 0.00 | 0.002945 | 0.8408 |
| `r_cvhr_ctx1` | float64 | 0.00 | 0.01501 | 0.7055 |
| `hr_mean_ctx1` | float64 | 0.00 | 44.2 | 100.1 |
| `rr_std_ctx2` | float64 | 0.00 | 8.346 | 203.7 |
| `rr_cv_ctx2` | float64 | 0.00 | 0.009282 | 0.1997 |
| `rmssd_ctx2` | float64 | 0.00 | 3.045 | 163.2 |
| `lf_hf_ctx2` | float64 | 0.00 | 0.09895 | 71.38 |
| `r_hf_ctx2` | float64 | 0.00 | 0.003786 | 0.8218 |
| `r_cvhr_ctx2` | float64 | 0.00 | 0.01653 | 0.6991 |
| `hr_mean_ctx2` | float64 | 0.00 | 44.29 | 100.1 |
| `rr_std_ctx5` | float64 | 0.00 | 8.686 | 178.8 |
| `rr_cv_ctx5` | float64 | 0.00 | 0.01151 | 0.1903 |
| `rmssd_ctx5` | float64 | 0.00 | 3.696 | 151.9 |
| `lf_hf_ctx5` | float64 | 0.00 | 0.1169 | 49.11 |
| `r_hf_ctx5` | float64 | 0.00 | 0.006665 | 0.8132 |
| `r_cvhr_ctx5` | float64 | 0.00 | 0.02174 | 0.6857 |
| `hr_mean_ctx5` | float64 | 0.00 | 44.8 | 99.38 |
| `osa` | int64 | 0.00 | 0 | 1 |
| `group` | str | 0.00 | — | 2 distinct |
| `split` | str | 0.00 | — | 2 distinct |

## Units

RR intervals in ms; `hr_mean` in bpm; `p_*` band powers in ms²/Hz; `r_*` relative (fraction of total power); `*_ctx{n}` = centred rolling mean over ±n minutes within subject.

`apnea` = expert per-minute label (1 = apnoea). `osa` = subject-level group (1 = APNEA cohort, AHI > 25). `split` = official L/T sets.