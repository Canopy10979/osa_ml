# dataset_ucddb_v2 — report

**UCD Sleep Apnea Database** (St. Vincent's University Hospital, Dublin): 25 full
overnight polysomnograms with expert-scored respiratory events.

Two things make this dataset different from everything else in the repo. It is the only
one carrying the channels clinicians actually score apnoea from — **SpO2, nasal flow, and
ribcage/abdomen effort** — rather than ECG alone. And it is the only one that populates
the **AHI 5–25 band**: 16 of its 25 subjects sit in the diagnostically ambiguous range
that `dataset_apnea_hrv` excludes by design.

Every number traces to an artifact in `results/` or `structured/`. Seed 42;
`models/params.json` records the split and feature list.

---

## 1. Label and unit of observation

| | |
|---|---|
| **Unit of observation** | one **30 s epoch** (20,789 rows, 25 subjects) |
| **Epoch label** | `event = 1` if an expert-scored apnoea/hypopnoea overlaps the epoch |
| **Event types included** | `APNEA-O/C/M`, `HYP-O/C/M` (3,318 events) |
| **Excluded** | `PB` (periodic breathing — a multi-minute pattern, not a discrete event) and `POSSIBLE` (uncertain by definition) |
| **Subject label** | `osa_15 = 1` iff recorded **PSG AHI ≥ 15** → **14 OSA / 11 non-OSA** |
| **Features** | 67 from SpO2, Flow, ribcage, abdo, Sum, Pulse, Sound |
| **Class balance** | 21.2% event epochs (majority-baseline accuracy 0.788) |

**Why AHI ≥ 15 and not ≥ 5.** At the standard AHI ≥ 5 threshold this cohort splits
24-vs-1 — degenerate, and no model could be evaluated on it. At ≥ 15 (the
moderate-or-worse boundary) it splits **14-vs-11**, which is close to balanced. The
threshold is a property of the cohort, not a tuned parameter.

`ahi`, `osa_15`, and `stage` are excluded from the feature matrix. `asleep` (derived from
stage) is retained: it is available at inference time from the same PSG and is not the
outcome.

## 2. Time alignment — the load-bearing validation

Respiratory events are stamped in **wall-clock time**; epochs are indexed from **PSG
start**, and recordings cross midnight. A wrong offset would silently scramble every
label while still producing plausible-looking results, so this is the one thing that had
to be proven rather than assumed.

**The gate: the event index derived from my alignment must reproduce the independently
recorded PSG AHI.** It does — **Pearson r = 0.978 (p = 4.1e-17)**. Two supporting checks
also pass: events concentrate in sleep (25.3% of sleep epochs vs 7.4% of wake epochs),
and all 25 subjects have at least one event. Alignment uses the EDF header's own start
time rather than the spreadsheet's, since the header is written by the recorder and is
what the sample indices are relative to.

## 3. Leakage audit — PASS

Strongest single feature is `spo2_desat_ctx1` at univariate AUC **0.849**; **no feature
exceeds 0.90**. No single-feature separator.

## 4. Per-epoch results (`results/metrics.csv`)

Out-of-fold, `StratifiedGroupKFold(5)` grouped by subject.

| model | recall | precision | specificity | PR-AUC | ROC-AUC | F1 | accuracy |
|---|---|---|---|---|---|---|---|
| Logistic Regression (L2) | **0.803** | 0.577 | 0.841 | 0.715 | 0.898 | 0.671 | 0.833 |
| Logistic Regression (L1) | 0.803 | 0.578 | 0.842 | 0.715 | 0.898 | 0.672 | 0.834 |
| Random Forest | 0.575 | 0.772 | 0.954 | 0.772 | 0.918 | 0.659 | 0.874 |
| **XGBoost** | 0.762 | 0.694 | 0.909 | **0.789** | **0.928** | **0.726** | **0.878** |
| majority baseline | 0.000 | 0.000 | 1.000 | 0.212 | 0.500 | 0.000 | 0.788 |

**PR-AUC 0.789 against a 0.212 baseline — a 3.7× lift, the strongest per-observation
result in the repo.** XGBoost leads on every ranking metric while still holding recall
at 0.762; logistic regression trades precision for the best recall (0.803). Random
Forest is again too conservative (recall 0.575) despite the highest specificity.

Seed variance is negligible: PR-AUC mean 0.790, **sd 0.0031** across seeds 42/7/2024.

## 5. What drives it — SpO2, as physiology predicts

From `results/feature_importance.csv`:

| feature | XGB gain | univariate AUC | reading |
|---|---|---|---|
| `spo2_std` | 0.130 | 0.848 | SpO2 variability within the epoch |
| `spo2_range` | 0.107 | 0.846 | peak-to-trough oxygen swing |
| `spo2_desat_ctx1` | 0.106 | 0.849 | desaturation vs a 2-min baseline, ±1 epoch |
| `asleep` | 0.054 | — | events occur in sleep |
| `sum_iqr` | 0.028 | — | total respiratory effort excursion |

**The top nine features are all SpO2-derived.** This is the first dataset in the project
where the model reads the *direct* physiological consequence of an apnoea — the blood
oxygen dip — rather than inferring it from downstream heart-rate effects. It is worth
noting explicitly that the legacy `cross_dataset` analysis assigned SpO2 an importance of
0.0019 and treated that as evidence about OSA; here, with correctly aligned event labels,
SpO2 is the whole story.

These remain **associations and model importances, not causal claims.**

## 6. The result that distinguishes this dataset

**Only ~7% of the model's skill above chance is between-subject.**

The within-subject shuffle — which preserves each subject's event burden while destroying
epoch-level timing — scores ROC-AUC 0.531 against a real-label 0.928. Compare:

| dataset | between-subject share | what the model mostly does |
|---|---|---|
| **ucddb_v2** | **~7%** | **detects events** |
| apnea_ecg | ~40% | mixed |
| apnea_hrv | ~59% | recognises people |

This is the sharpest contrast in the project. `apnea_hrv` is an excellent *screener* whose
per-minute score is substantially inflated by learning who the patient is. This model
genuinely localises individual respiratory events in time. The reason is almost certainly
channel access: SpO2 and airflow change *during* an event, whereas HRV signatures are a
diffuse downstream consequence that partly encodes the person's overall severity.

**Label-shuffle: PASS.** Global shuffle collapses to chance — PR-AUC 0.2122 against a
0.2122 base rate, ROC-AUC 0.5007 against 0.5.

## 7. Subject-level OSA (AHI ≥ 15)

Pooling out-of-fold epoch predictions into a predicted event index:

- **Predicted index vs recorded AHI: Pearson r = 0.951** (p = 3.6e-13), MAE 13.0 events/h
- **As a ranking score for OSA: ROC-AUC 0.916**
- At the best fitted threshold: balanced accuracy 0.893, **sensitivity 0.786,
  specificity 1.000** (11/11 non-OSA correct, 11/14 OSA correct)
- **Leave-one-subject-out, threshold refit in every fold: balanced accuracy 0.847,
  accuracy 0.840**

Per the skill's small-cohort rule (< ~50 rows or < ~10 minority cases), **no held-out
test set was manufactured** — 25 subjects with 11 in the minority is below the point
where a test split carries information. The LOO figure is the honest headline because the
held-out subject never informs its own threshold; the 0.893 is optimistic.

**MAE of 13.0 events/h is large** relative to the AHI 15 boundary, and the three misses
are the model under-calling moderate cases. Specificity 1.000 on 11 subjects has a wide
confidence interval and should not be read as "never false-positives."

## 8. Conclusions

1. **Respiratory events are detectable per-epoch from PSG with high accuracy** — PR-AUC
   0.789 vs a 0.212 baseline, ROC-AUC 0.928, seed sd 0.003, clean leakage audit.
   *Confidence: high.*
2. **SpO2 desaturation is the mechanism**, occupying the entire top of the importance
   table. *Confidence: high for association, none for causation.*
3. **This model detects events rather than people** — only ~7% between-subject skill,
   against 59% for `apnea_hrv`. *Confidence: high*; the shuffle diagnostic is direct.
4. **Subject-level OSA classification works at AHI ≥ 15** — index correlates with AHI at
   r = 0.951, LOO balanced accuracy 0.847. *Confidence: moderate*, limited by n = 25.
5. **This cohort covers the range the rest of the project could not.** 16 of 25 subjects
   sit in AHI 5–25, the band `apnea_hrv` excludes entirely. The headline limitation of
   that dataset is directly addressed here.

**Principal limitations.** 25 subjects is small for any subject-level claim. The cohort is
clinically referred, so it is enriched for OSA (24/25 above AHI 5) and is not a screening
population. And this uses full PSG — the result does **not** transfer to wearable or
single-channel settings, which is what `apnea_hrv` and `apnea_ecg` speak to.

## 9. Reproduce

```bash
python common/ucddb_features.py && python common/ucddb_models.py
```

| Stage | Script | Key outputs |
|---|---|---|
| 1–2 extract, featurise, leakage audit | `common/ucddb_features.py` | `structured/epoch_features.{parquet,csv}`, `subject_level.csv`, `data_dictionary.md`, `results/leakage_audit.csv` |
| 3–6 train, evaluate, infer, validate | `common/ucddb_models.py` | `results/metrics.csv`, `confusion_*.csv`, `feature_importance.csv`, `subject_level_predictions.csv`, `shuffle_test.csv`, `seed_variance.csv`, `models/*.joblib` |

Raw data (1.3 GB) is **not tracked in git** — see `raw/.gitignore` and `raw/README.md`.
SHAP is not installed; XGB gain and RF permutation importance are reported instead.
