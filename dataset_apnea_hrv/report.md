# dataset_apnea_hrv — report

**HuGCDN2014** (Dr. Negrín University Hospital, Canary Islands): 77 single-lead ECG
recordings at 200 Hz, expert-scored for apnoea in every minute against simultaneous
polysomnography.

**This is the only dataset in this repo that properly supports the OSA-vs-non-OSA
question** — 40 controls (AHI < 5) against 37 patients (AHI > 25), a large, balanced,
cleanly separated cohort. Every other dataset here fails on that point: `apnea_ecg` has
2 controls, and `bioradiolocation` has no OSA cases at all.

Every number below is read from an artifact in `results/` or `structured/`. Seed 42;
`models/params.json` records the split and feature list.

---

## 1. Label and unit of observation

| | |
|---|---|
| **Unit of observation** | one **minute** (30,445 rows, 77 subjects) |
| **Minute label** | `apnea = 1` iff the expert scored apnoea in that minute (`salida_man_1m`) |
| **Subject label** | `osa = 1` iff the record is `APNxxx` (patient, AHI > 25); `CONxxx` = control (AHI < 5) |
| **Features** | 51 HRV features from the 5-minute frame centred on each minute |
| **Class balance** | 22.1% apnoea minutes (majority-baseline accuracy 0.7786) |

Each minute is featurised from the **5-minute frame centred on it**, shifted in 1-minute
increments — the framing the database itself ships. Subjects contribute ~400 minutes
each, so rows are **not independent**; all cross-validation is
`StratifiedGroupKFold(5)` **grouped by subject**.

**Group assignment comes from the record-name prefix**, the archive's own convention —
not inferred. It is cross-checked against the annotated apnoea index, which must
separate cleanly under the documented design, and does: **controls span 0.00–3.54
events/h, patients 16.83–54.10, with nothing in between.** That check is enforced in
`apnea_hrv_features.py` and would halt the run if it ever failed.

**Data quality:** 20,556 beats dropped as implausible (RR outside 300–2000 ms) or
ectopic (>20% jump); 342 of 30,787 minutes dropped for too few usable beats — **98.9%
retained**, logged per subject in `structured/rejects.csv`.

## 2. Leakage audit (`results/leakage_audit.csv`) — PASS

Subject identity, group, and split are excluded from the feature matrix by
construction. The strongest single feature is `r_cvhr_ctx5` at univariate AUC **0.806**;
**no feature exceeds 0.90**. No single-feature separator, so no evidence of outcome
leakage.

## 3. Minute-level results (`results/metrics.csv`)

Out-of-fold, grouped by subject. Leading with recall and PR-AUC.

| model | recall | precision | specificity | PR-AUC | ROC-AUC | F1 | accuracy |
|---|---|---|---|---|---|---|---|
| Logistic Regression (L2) | **0.755** | 0.500 | 0.785 | 0.681 | 0.857 | 0.601 | 0.778 |
| Logistic Regression (L1) | 0.752 | 0.499 | 0.785 | 0.681 | 0.857 | 0.600 | 0.778 |
| Random Forest | 0.451 | 0.736 | 0.954 | 0.676 | 0.860 | 0.559 | 0.843 |
| **XGBoost** | 0.646 | 0.629 | 0.892 | **0.683** | **0.865** | **0.637** | 0.837 |
| majority baseline | 0.000 | 0.000 | 1.000 | 0.221 | 0.500 | 0.000 | 0.779 |

**All models beat the baseline decisively** — PR-AUC 0.68 against 0.221, roughly 3×.
Note that raw *accuracy* is nearly useless here: the baseline already scores 0.779 by
calling everything normal, and Random Forest's headline 0.843 accuracy conceals a
recall of **0.451** — it misses more than half of all apnoeic minutes.

**Model choice depends on the goal.** XGBoost has the best ranking (PR-AUC, ROC-AUC),
but **logistic regression has by far the best recall (0.755 vs 0.451)** and is the right
choice for screening, where a missed apnoea is the costly error. That the linear model
is competitive at all says most of the signal is linearly accessible.

**The official holdout confirms it** (`results/holdout_metrics.csv`). Trained on the
database's learning subjects (L) and tested on the 39 test subjects (T), never seen in
training:

| model | recall | PR-AUC | ROC-AUC |
|---|---|---|---|
| Logistic Regression (L2) | 0.810 | 0.703 | 0.858 |
| XGBoost | 0.552 | 0.663 | 0.842 |
| Random Forest | 0.424 | 0.646 | 0.834 |

Logistic regression **improves** on unseen subjects (PR-AUC 0.703 vs 0.681 OOF). This is
a genuinely robust result and the sharpest contrast with `dataset_apnea_ecg`, where
holdout ROC-AUC collapsed from 0.825 to 0.675.

**Calibration** (`results/calibration.csv`) is good in the low range (predicted 0.036 →
observed 0.064) but compressed at the top: minutes given 0.925 are apnoeic 78.6% of the
time. Probabilities are usable as a ranking, but overstate confidence at the high end.

## 4. What drives the predictions — the CVHR finding

From `results/feature_importance.csv`:

| feature | XGB gain | LR odds ratio | univariate AUC | reading |
|---|---|---|---|---|
| `r_cvhr_ctx5` | 0.133 | 4.15 | 0.806 | relative power in the **0.01–0.04 Hz** band, averaged over ±5 min |
| `peak_hz` | 0.055 | 1.12 | — | dominant tachogram frequency shifts |
| `r_cvhr_ctx2` | 0.043 | 0.80 | 0.789 | same band, ±2 min |
| `r_hf_ctx2` | 0.036 | 1.58 | 0.744 | respiratory-band HRV |
| `hr_mean_ctx5` | 0.035 | 14.60 | — | heart-rate level over ±5 min |

**Cyclical variation in heart rate is the dominant signal.** The top four features by
univariate AUC are all `r_cvhr` variants (0.761–0.806). This is the physiological
signature of repetitive apnoea: each event ends in an arousal that spikes heart rate,
producing a slow ~0.01–0.04 Hz bradycardia/tachycardia oscillation.

**This explains a recorded "finding" in `dataset_apnea_ecg` that was actually a bug.**
That report noted an explicit failed prediction — CVHR band power was expected to
dominate but was "absent from the top 10" — and treated it as a physiological result. It
was not. **Every CVHR feature in that dataset is identically 0.0** across all 13,286
minutes. Its `band_power()` returns `0.0` unless more than two FFT bins fall inside the
band; at 4 Hz resampling with `nperseg=256` the resolution is 0.015625 Hz, so the
0.010–0.040 Hz band contains exactly two bins and the guard fired every time.

So the comparison is not "CVHR matters here but not there" — it is **"CVHR matters here,
and there it was never measured."** On this dataset, computed at a resolution that can
actually resolve the band, CVHR dominates, with HF power second (`r_hf_ctx5`, AUC 0.751).
See `cross_dataset/comparison.md`.

The **context features consistently outrank their instantaneous versions**
(`r_cvhr_ctx5` 0.806 > `r_cvhr` 0.761) — as expected, since a 0.01–0.04 Hz cycle has a
period of 25–100 s and simply cannot be resolved inside a single minute.

These are **associations and model importances, not causes.**

**Multicollinearity** (`results/vif_top15.csv`) is present among the overlapping context
windows of the same quantity — read the `r_cvhr_*` family as one signal, not as five
independent predictors.

## 5. Subject-level OSA classification — this one works

Pooling out-of-fold minute predictions into an estimated apnoea index
(`results/subject_level_predictions.csv`). Every subject was held out when its own
minutes were predicted.

- **Index estimation: Pearson r = 0.884** (p = 1.8e-26), Spearman 0.777,
  **MAE = 5.89 events/h**
- **As a ranking score, the predicted index separates the groups at ROC-AUC 0.937**
- At the best threshold (9.5 events/h): **balanced accuracy 0.896, sensitivity 0.892,
  specificity 0.900** — 36/40 controls and 33/37 patients correct
- **On the 39 official test subjects alone: ROC-AUC 0.921**

| group | n | mean predicted index | range |
|---|---|---|---|
| CONTROL | 40 | 4.97 | 0.00–21.37 |
| APNEA | 37 | 27.31 | 1.21–59.83 |

**This is the result the rest of the project could not produce.** `apnea_ecg` reached
balanced accuracy 0.582 and identified 1 of 4 non-OSA subjects; here specificity is
**0.900 on 40 real controls**. The difference is entirely cohort design, not modelling.

**Two honest caveats.** The 9.5 threshold was chosen by maximising balanced accuracy on
these same 77 subjects, so that figure is optimistic — the threshold-free ROC-AUC
(0.937 overall, **0.921 on unseen test subjects**) is the trustworthy number. And the
cohort is deliberately gapped: no subject sits between AHI 5 and 25, so this does **not**
demonstrate performance on the mild-to-moderate patients who are hardest to classify and
most common in practice.

## 6. Validating the analysis itself

**Label-shuffle (`results/shuffle_test.csv`) — PASS.** Under a global shuffle,
performance collapses to chance: PR-AUC 0.2193 against a 0.2214 base rate, ROC-AUC
0.4949 against 0.5. Nothing has leaked.

**Between- vs within-subject skill.** A within-subject shuffle — which preserves each
subject's apnoea burden while destroying minute-level timing — still scores ROC-AUC
0.7171. That is not leakage (a model that recognises "this person is a severe apnoeic"
needs no contamination to do it); it is a measure of *what kind* of skill the model has.
Here **~59% of the skill above chance is between-subject**, higher than the ~40% in
`apnea_ecg`. That is consistent with §5: this model is very good at characterising
people and correspondingly less impressive at pinpointing individual minutes.

**Seed variance (`results/seed_variance.csv`) — stable.** Seeds 42/7/2024 give PR-AUC
mean 0.6767, **sd 0.0057**; ROC-AUC mean 0.8593, sd 0.0049.

**Per-subject failure analysis (`results/per_subject_metrics.csv`).** Mean per-subject
accuracy 0.827 (sd 0.118, min 0.550). Accuracy correlates **negatively and strongly**
with a subject's apnoea rate (**Spearman −0.633, p = 6.8e-10**): the five worst subjects
are all patients with apnoea rates of 0.40–0.75. A single global 0.5 threshold is simply
mis-set for subjects whose apnoea burden is far from the 22% population rate. Their
*rankings* remain sound (e.g. `APN036`: accuracy 0.550 but ROC-AUC 0.804), so this is a
threshold problem, not a signal problem — per-subject threshold adaptation is the
obvious fix.

## 7. Conclusions

1. **Apnoea is detectable minute-by-minute from heartbeat timing alone.** XGBoost
   PR-AUC 0.683 vs a 0.221 baseline; logistic regression reaches recall 0.755, rising to
   0.810 on unseen test subjects. *Confidence: high* — clean leakage audit, stable across
   seeds, and it holds on the database's own held-out subjects.
2. **The mechanism is cyclical variation in heart rate (0.01–0.04 Hz).** Top four
   features by univariate AUC are all CVHR-band, peaking at 0.806. *Confidence: high for
   association, none for causation.*
3. **The CVHR "null result" recorded in `dataset_apnea_ecg` was a measurement bug**, not
   physiology — its CVHR features are identically zero because the band fell below the
   spectral resolution its guard clause required. *Confidence: high* — the features are
   provably constant and the failing condition is a single legible line.
4. **Findings transfer across cohorts.** A model trained here and tested on Apnea-ECG
   scores ROC-AUC 0.795 against a 0.796 within-dataset reference — essentially no
   generalisation loss, on different hardware, a different country, and a different
   apnoea prevalence. *Confidence: high* — this is the strongest evidence in the project
   that the signal is physiology rather than dataset artefact. See
   `cross_dataset/comparison.md`.
5. **Subject-level OSA screening is demonstrated here**: index ROC-AUC 0.937 overall and
   0.921 on unseen subjects, sensitivity 0.892 / specificity 0.900 against 40 genuine
   controls. *Confidence: moderate-to-high*, tempered by the fitted threshold and the
   deliberately gapped cohort.
6. **Around 59% of the model's skill is recognising the person, not the event.** Useful
   for screening; a weaker claim than per-minute scores alone suggest.
7. **Choose the model by the cost of error** — logistic regression for screening (recall
   0.755), XGBoost for ranking. Random Forest's high accuracy is a class-imbalance
   artefact and it should not be used here.

**Principal limitation:** the AHI 5–25 range is absent by design, and that is exactly
the diagnostically ambiguous band. Performance on mild and moderate OSA is untested and
should not be extrapolated from these numbers.

## 8. Reproduce

```bash
python common/apnea_hrv_features.py && python common/apnea_hrv_models.py && python common/apnea_hrv_validate.py
```

| Stage | Script | Key outputs |
|---|---|---|
| 1–2 extract, featurise, leakage audit | `common/apnea_hrv_features.py` | `structured/minute_features.{parquet,csv}`, `subject_level.csv`, `data_dictionary.md`, `rejects.csv`, `results/leakage_audit.csv` |
| 3–4 train, evaluate | `common/apnea_hrv_models.py` | `results/metrics.csv`, `holdout_metrics.csv`, `confusion_*.csv`, `feature_importance.csv`, `calibration.csv`, `models/*.joblib` |
| 5–6 infer, validate | `common/apnea_hrv_validate.py` | `results/subject_level_predictions.csv`, `shuffle_test.csv`, `seed_variance.csv`, `vif_top15.csv`, `per_subject_metrics.csv` |

Raw data (46 MB) is **not tracked in git** — see `raw/.gitignore` and `raw/README.md`
for provenance. SHAP is not installed; XGB gain and RF permutation importance are
reported instead.
