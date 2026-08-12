# Cross-dataset comparison

Five dataset folders have been run through the pipeline. This document compares what
each supports, tests whether the findings **transfer between cohorts**, and records two
defects that comparison exposed.

Produced by `common/cross_dataset_compare.py`. Figures trace to
`cross_dataset/results/transfer_metrics.csv` and `feature_agreement.csv`.

---

## 1. What each dataset can answer

| dataset | subjects | OSA cases | controls | verdict |
|---|---|---|---|---|
| **ucddb_v2** (UCD, full PSG) | 25 | 14 (AHI ≥ 15) | 11 | ✅ Best event detection; covers AHI 5–25 |
| **apnea_hrv** (HuGCDN2014) | 77 | 37 (AHI > 25) | **40** (AHI < 5) | ✅ Best screener |
| **apnea_ecg** (PhysioNet) | 27 | 23 | **2** | ✅ Minute detection; ❌ cannot rule out |
| **bioradiolocation** | 32 | **0** (max AHI 4.9) | 32 | ❌ No positive class exists |
| **cross_dataset** (legacy) | 1 | — | — | ⚠️ Target is asleep-vs-awake, mislabelled as OSA |

Only `apnea_hrv` has enough controls to estimate specificity. That single fact explains
almost every difference in outcome between the two working datasets: `apnea_ecg` reaches
subject-level balanced accuracy 0.582 and identifies 1 of 4 non-OSA subjects, while
`apnea_hrv` reaches 0.896 with specificity 0.900 against 40 genuine controls. Same
method, same feature philosophy — different cohort design.

## 2. Does it transfer? (`results/transfer_metrics.csv`)

`apnea_ecg` and `apnea_hrv` pose the same task at the same unit of observation — one
minute of ECG-derived HRV, expert-scored — and both featurise a 5-minute window centred
on the target minute. After removing degenerate columns, **34 features are shared**, so a
model trained on one cohort can be scored on the other.

This is the strongest available test of whether the results are physiology or artefact:
transfer performance cannot be inflated by anything internal to a single dataset.

| train → test | ROC-AUC | within-dataset reference | loss |
|---|---|---|---|
| **apnea_hrv → apnea_ecg** | **0.795** | 0.796 | **0.001** |
| **apnea_ecg → apnea_hrv** | 0.749 | 0.799 | 0.050 |

**A model trained on 77 Spanish hospital subjects detects apnoea minutes in PhysioNet
Apnea-ECG as well as a model trained on Apnea-ECG itself** — a loss of 0.001 ROC-AUC,
across different recording hardware, a different country, a different decade, and an
apnoea prevalence of 22% versus 49%. The reverse direction transfers with a modest loss
(PR-AUC 0.412 against a 0.221 base rate — a 1.86× lift, versus 2.53× within-dataset).

The asymmetry is expected: `apnea_hrv` has ~2.3× the minutes and 2.9× the subjects, so it
is the better training source.

**Feature agreement is high.** Across the shared features, univariate informativeness
ranks correlate at **Spearman 0.807** (p = 8.3e-09), and **76% of features point in the
same direction** in both cohorts (signed agreement r = 0.874). `r_hf` is the single
strongest shared feature in both (0.793 in ECG, 0.727 in HRV), with the `lf_hf` and
`rr_cv` families next in both.

**Conclusion: the physiology is real and cohort-independent.** This is the most defensible
claim the project can make.

## 3. Two defects this comparison exposed

Neither was visible from inside a single dataset. Both are recorded here because they
change how earlier results should be read.

### 3a. `apnea_ecg`'s CVHR features are identically zero — a bug, not a null result

Every CVHR column in `dataset_apnea_ecg` (`p_cvhr`, `r_cvhr`, `r_cvhr_ctx{1,2,5}`,
`edr_cvhr`) is **exactly 0.0 across all 13,286 minutes** (`p_cvhr_log` = −12, i.e.
log10(1e-12)).

The cause is a single line in that pipeline's `band_power()`:

```python
return float(np.trapezoid(pxx[m], f[m])) if m.sum() > 2 else 0.0
```

Its Welch PSD uses `nperseg=256` at a 4 Hz resample rate, giving a frequency resolution
of 4/256 = **0.015625 Hz**. The CVHR band spans 0.010–0.040 Hz and therefore contains
exactly **two** bins (0.015625 and 0.03125). `2 > 2` is false, so the guard returned 0.0
on every frame of every record.

**Why this matters:** the `apnea_ecg` report recorded, as a substantive scientific
finding, that it had expected CVHR band power to dominate and found that it "did not
appear in the top 10." That was not a physiological result — the feature was never
measured. On `apnea_hrv`, where the same band is computed at a resolution that can
resolve it (`nperseg` sized to the frame, 120 s segments), **CVHR is the single most
informative feature family**, peaking at univariate AUC 0.806.

The honest comparison is not "CVHR matters in one cohort but not the other." It is
"CVHR matters, and in one cohort it was silently discarded."

### 3b. The two pipelines share column names but not units

`dataset_apnea_ecg` stores RR intervals in **seconds** (median `rr_mean` = 0.916);
`dataset_apnea_hrv` stores them in **milliseconds** (median 903) — a factor of ~1000,
and ~10⁶ for the derived band powers.

A first attempt at transfer fitted the scaler on one cohort and applied it to the other,
which drove **every** transferred prediction to exactly 0.0 and produced a spurious
ROC-AUC of precisely 0.5000 — a number that reads like "no transferable signal" and is
in fact a saturation artefact. The corrected test standardises each cohort against its
own mean and s.d. (label-free, so nothing leaks), which removes both the unit mismatch
and any baseline offset. That is what the §2 numbers use.

The general lesson: **identical column names across independently written pipelines are
not evidence of identical definitions.** Comparing them requires checking units, not just
names.

## 3c. Sensors decide whether you detect events or recognise people

The within-subject shuffle measures how much of a model's skill survives when each
subject's event burden is preserved but epoch-level timing is destroyed. The spread
across three working datasets is the clearest structural finding in the project:

| dataset | signals | between-subject share | what it mostly does |
|---|---|---|---|
| **ucddb_v2** | SpO2 + flow + effort | **~7%** | **detects events** |
| apnea_ecg | ECG only | ~40% | mixed |
| apnea_hrv | ECG only | ~59% | recognises people |

Both ECG-only datasets derive 40–59% of their apparent skill from learning who the
patient is. The PSG dataset derives almost none. The explanation is direct: SpO2 and
airflow change *during* an event, whereas HRV signatures are a diffuse downstream
consequence that partly encodes the subject's overall severity.

This reframes the two ECG results. They are **screeners** — good at identifying which
people have OSA, weaker at saying which minute contains an apnoea — and their per-minute
scores should be read with that in mind. It also explains why `apnea_hrv` posts the
better subject-level number (ROC-AUC 0.937 vs 0.916) while `ucddb_v2` posts the better
per-observation number (0.928 vs 0.865).

## 4. `cross_dataset` — mislabelling identified, corrected, and re-run

### 4a. What was wrong

The legacy outputs reported up to **97% accuracy** for "With OSA" vs "Without OSA"
(`model_accuracy_by_osa_status.csv`). The target actually trained on was `Sleep_Label`,
which the generating code documents as **"0=awake, 1=asleep"**
(`regenerate_pipeline.py:121`, recoverable from git history). It was a sleep/wake
classifier presented as an OSA detector. Its own numbers gave SpO2 — the defining marker
of apnoea — an importance of **0.0019**, which should have been the tell.

### 4b. The source is UCDDB011

The export was unattributed. It is identifiable: recording start **22:47:44** and 8.9 h
duration match **UCDDB011** in `dataset_ucddb_v2` (PSG start 22:47:38, 7.5 h, recorded
**AHI 8**). The correct labels therefore exist — that subject's expert respiratory event
list — and the exported CSVs carry absolute wall-clock timestamps, so alignment is exact.

Validated: the realigned event index gives **7.8 events/h against a recorded AHI of 8**.

### 4c. The corrected run (`common/cross_dataset_relabel.py`)

Same features, real apnoea labels, five **contiguous time blocks** for CV (a random split
would leak neighbouring epochs from one night).

The two targets are barely related — they agree on only **24.3%** of epochs:

| target | positive rate | balanced accuracy | ROC-AUC |
|---|---|---|---|
| `Sleep_Label` (legacy, wrong) | 80.5% | 0.551 | 0.771 |
| respiratory event (correct) | 5.2% | 0.572 | **0.602** |

| model | recall | PR-AUC | ROC-AUC |
|---|---|---|---|
| Logistic Regression (L2) | 0.411 | **0.085** | **0.602** |
| Logistic Regression (L1) | 0.411 | 0.081 | 0.599 |
| XGBoost | **0.000** | 0.059 | 0.545 |
| Random Forest | **0.000** | 0.046 | 0.449 |
| baseline | 0.000 | 0.052 | 0.500 |

**This is a negative result and is reported as one.** On the correct target the best model
reaches PR-AUC 0.085 against a 0.052 baseline — barely above chance — and both tree
models predict no events at all. The 97% headline does not survive correction; it was
measuring sleep, and sleep is easy because 80% of the night is asleep.

Feature importances do move in the physiologically right direction: airflow reduction
(`flow_rel_amp_ctx2`, univariate AUC 0.731) becomes the top feature and SpO2's share of
importance rises from 0.0019 to **23.3%**. The signal is faintly there; there is simply
not enough of it here.

### 4d. Why it fails, and why that is consistent

`dataset_ucddb_v2` reaches ROC-AUC 0.928 on this same subject population. The difference
is not method, it is what this export retained:

- **One subject, one night** — 1,069 epochs with only **56 positive**.
- **Mild disease** — UCDDB011's AHI is 8, near the diagnostic floor.
- **Downsampled, partial channels** — 1 Hz HR and SpO2, 2 Hz flow, and **no ribcage or
  abdomen effort belts**, versus 8 Hz SpO2/flow plus effort in the full PSG.

So the corrected `cross_dataset` result does not contradict `ucddb_v2`; it shows what is
lost when the same night is reduced to three low-rate channels. These outputs should be
cited as a single-subject replication attempt, never as an independent OSA result.

## 5. Bottom line

1. **Two datasets support real claims, answering different questions.** `ucddb_v2`
   detects individual events (ROC-AUC 0.928, ~7% between-subject); `apnea_hrv` is the
   better screener (40 controls vs 37 patients, ROC-AUC 0.937, sensitivity 0.892,
   specificity 0.900) and works from heartbeat alone.
2. **The findings transfer between independent cohorts with almost no loss** (0.795 vs a
   0.796 within-dataset reference), which is strong evidence of real physiology.
3. **Cyclical variation in heart rate is the mechanism**, and the one dataset that
   appeared to contradict this had simply failed to measure it.
4. **Both working datasets are inflated by subject identity** — ~59% of skill in
   `apnea_hrv`, ~40% in `apnea_ecg`, is recognising the person rather than the event.
5. **The mild-to-moderate gap is now partly covered.** `apnea_hrv` is gapped by design
   (no subject between AHI 5 and 25), but 16 of `ucddb_v2`'s 25 subjects sit in exactly
   that band — though only with full PSG, and on 25 people.

## 6. Reproduce

```bash
python common/cross_dataset_compare.py
```

Requires `dataset_apnea_ecg/` and `dataset_apnea_hrv/` to have been built first. Note
that `dataset_apnea_ecg`'s pipeline scripts were deleted from `common/` in commit
`b80b397` and survive only in git history (`git show e61bfde:common/apnea_ecg_features.py`),
so that dataset is **not currently reproducible from the working tree**.
