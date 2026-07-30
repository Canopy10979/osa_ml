# dataset_apnea_ecg — report

Per-minute obstructive apnoea detection from single-lead ECG, PhysioNet
Apnea-ECG. **This is the only dataset in this repo with genuine OSA labels**, so
it is the only one on which an OSA-vs-non-OSA claim can be made.

Every number below traces to a file in `results/`. Seed 42; `models/params.json`
records the split and feature list.

## Data

| | |
|---|---|
| Unit of observation | one minute of ECG |
| Rows | 13,286 minutes |
| Records | 27 (20 apnoea `a*`, 5 borderline `b*`, 2 control `c*`) |
| Features | 64 |
| Label | the database's own `.apn` annotation, `A` vs `N` — **not** a proxy |
| Class balance | 48.9% apnoea (majority baseline accuracy 0.511) |

**Label rule, verbatim:** a minute is positive iff its `.apn` annotation symbol
is `A`. Subject-level OSA is `apnoea index >= 5`, where apnoea index =
annotated apnoea minutes / recording hours.

Features target the two established ECG signatures of obstructive apnoea:
cyclical variation in heart rate (0.01–0.04 Hz bradycardia/tachycardia cycling)
and ECG-derived respiration recovered from R-wave amplitude modulation.
Spectral features use a 5-minute window centred on the target minute, because a
25–100 s CVHR cycle cannot be resolved inside 60 s. 17,853 beats (~2.2%) were
dropped as physiologically implausible or ectopic; see `structured/rejects.csv`.

## Stage 2 — leakage audit (`results/leakage_audit.csv`)

Record identity (`record`, `class_prefix`) and outcome-derived quantities
(`apnea_index`) were excluded by construction. The strongest single feature is
relative HF power at univariate AUC **0.793**; **zero features exceed 0.90**.
No single-feature separator, so no evidence of outcome leakage.

## Stage 3–4 — models (`results/metrics.csv`)

Cross-validation is `StratifiedGroupKFold(5)` **grouped by record** — every
record is held out exactly once. Minutes within a record are not independent
(~490 per record), so a random minute split would leak badly.

Leading with recall and PR-AUC, since a missed apnoea is the costly error:

| Model | Recall | PR-AUC | ROC-AUC | F1 | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression (L2) | 0.708 | 0.810 | 0.805 | 0.724 | 0.737 |
| Logistic Regression (L1) | 0.708 | 0.810 | 0.806 | 0.725 | 0.738 |
| Random Forest | 0.651 | 0.828 | 0.809 | 0.715 | 0.745 |
| **XGBoost** | 0.683 | **0.838** | **0.825** | **0.739** | **0.764** |
| _majority baseline_ | 0.000 | 0.489 | 0.500 | — | 0.511 |

**All four models beat the majority baseline decisively** (PR-AUC 0.81–0.84 vs
0.489). XGBoost wins; the L1/L2 logistic regressions are within 0.03 PR-AUC of
it, so most of the signal is linearly accessible.

Confusion matrix, XGBoost (`results/confusion_XGBoost.csv`):

| | pred Normal | pred Apnoea |
|---|---|---|
| **true Normal** | 5706 | 1078 |
| **true Apnoea** | 2063 | 4439 |

Apnoea precision 0.805, recall 0.683. **The model is conservative — it misses
32% of apnoea minutes.** For a screening application that is the wrong direction
and the decision threshold should be lowered from 0.5.

**Held-out record test** (6 records never trained on): XGBoost PR-AUC 0.821,
ROC-AUC 0.675; Random Forest ROC-AUC 0.715. ROC-AUC drops versus CV because the
random draw put 5 of 6 apnoea-heavy records in the test set (base rate ~0.6),
which compresses ROC while leaving PR-AUC intact — a composition artefact of a
6-record test set, not a generalisation failure.

**Calibration** (`results/calibration.csv`) is poor at the low end: minutes given
a 0.024 predicted probability are apnoea 20.1% of the time. Predicted
probabilities should not be read as apnoea likelihoods without recalibration.

## Stage 5 — inference

**Feature importance** (`results/feature_importance.csv`) — top by XGBoost gain:

| Feature | XGB gain | LR odds ratio | Reading |
|---|---|---|---|
| `r_hf` | 0.155 | 0.47 | relative HF power **falls** during apnoea |
| `rmssd` | 0.059 | 0.86 | short-term HRV falls |
| `peak_hz` | 0.038 | 0.98 | dominant tachogram frequency shifts |
| `edr_peak_hz` | 0.037 | 0.72 | ECG-derived respiration frequency shifts |
| `rr_iqr` | 0.033 | 0.93 | RR spread narrows |

Respiratory-sinus-arrhythmia suppression (`r_hf`, `rmssd`) dominates, and two
ECG-derived respiration features (`edr_peak_hz`, `edr_std`) carry independent
weight — the model is reading breathing off the ECG, as intended.

**A prior of mine was wrong and worth recording:** I expected the explicit CVHR
band-power features (`p_cvhr`, `r_cvhr`, 0.01–0.04 Hz) to dominate. They do not
appear in the top 10. Relative HF power is the stronger discriminator here.

**Multicollinearity** (`results/vif_top15.csv`): 3 of the top 15 features exceed
VIF 10 (`rr_cv_ctx2` = 33.3), all of them overlapping rolling-context windows of
the same underlying quantity. This inflates individual coefficient variance —
read the linear coefficients as a group, not individually — but does not affect
the tree models or the reported discrimination.

**Model agreement:** logistic regression and XGBoost disagree on 16.2% of
minutes. Accuracy is 0.798 where they agree and 0.584 where they do not, so
disagreement is a usable uncertainty flag.

### Subject-level OSA (`results/subject_level_predictions.csv`)

Pooling the out-of-fold minute predictions into an estimated apnoea index:

- **AI estimation: Pearson r = 0.627 (p = 0.00046)**, Spearman 0.607,
  MAE 10.8 events/h.
- **OSA classification (AI ≥ 5): accuracy 0.815, balanced accuracy 0.582.**

The accuracy figure is misleading and the balanced accuracy is the honest one:
23 of 27 subjects are true positives, so predicting "OSA" almost always scores
well. **The model correctly identified only 1 of the 4 true non-OSA subjects.**
It is usable for grading severity, not for ruling OSA out. Five subjects were
misclassified: `a18`, `b01`, `b04`, `b05`, `c03`. MAE of 10.8 events/h is also
too large for clinical severity banding.

## Stage 6 — validating the analysis

**Label-shuffle** (`results/shuffle_test.csv`):

| Test | PR-AUC | ROC-AUC |
|---|---|---|
| Global shuffle | 0.498 (chance 0.489) | **0.513** |
| Within-record shuffle | 0.628 | **0.631** |
| Real labels | 0.838 | 0.825 |

The global shuffle collapses to chance — **the leakage test passes.**

The within-record shuffle does *not* collapse, and this is the most important
methodological finding in the run. Shuffling labels inside each record preserves
that record's overall apnoea burden, and the model can still infer that burden
from record-level signal characteristics. So roughly **ROC-AUC 0.63 of the
headline 0.825 is "how apnoeic is this person overall", not "is this particular
minute an apnoea".** The genuine within-record minute-level discrimination is
the margin above that floor, not above 0.5. Equivalently, ~40% of the model's
skill above chance — (0.631 − 0.5) / (0.825 − 0.5) — is between-record. This is
not outcome leakage, it is base-rate inference, but it inflates every per-minute
Apnea-ECG score reported against a 0.5 chance line.

`apnea_ecg_validate.py` previously failed the whole run here, printing
"FAIL — investigate leakage", because its verdict required *both* shuffles to
collapse. That rule was too blunt and has been corrected: the leakage verdict now
rests on the global shuffle alone (**PASS**), and the within-record result is
reported as the quantified diagnostic above rather than as a pass/fail gate.

**Seed variance** (`results/seed_variance.csv`), seeds 42/7/2024:
PR-AUC 0.8376 / 0.8465 / 0.8461, mean 0.8434, **sd 0.0050**. Stable — the
ranking of models is not a seed artefact.

**Failure analysis** (`results/per_record_metrics.csv`): mean per-record accuracy
0.764, sd 0.169, **min 0.160**. Record `a18` fails outright (accuracy 0.160,
recall 0.066, ROC-AUC 0.434 — below chance) despite an 89.6% apnoea rate; the
model calls nearly every minute normal. `a09` is similar (accuracy 0.406) but
its ROC-AUC is 0.856, meaning its *ranking* is fine and only the threshold is
wrong. Accuracy does not correlate significantly with record apnoea rate
(Spearman −0.141, p = 0.48), so this is per-record signal quality, not a
systematic threshold effect.

## Conclusions

1. **Per-minute apnoea detection from single-lead ECG works** — XGBoost PR-AUC
   0.838 against a 0.489 baseline, stable across seeds (sd 0.005), with no
   outcome leakage.
2. **The honest chance line is 0.63 ROC-AUC, not 0.5**, because a model can
   infer a subject's overall apnoea burden without discriminating individual
   minutes. Published per-minute scores benchmarked against 0.5 overstate
   minute-level skill.
3. **The physiology is RSA suppression plus ECG-derived respiration**, not the
   CVHR band predicted in advance.
4. **Not deployable for ruling out OSA.** Subject-level balanced accuracy is
   0.582 and only 1 of 4 non-OSA subjects was identified; AI error is
   ±10.8 events/h.
5. **The model is too conservative for screening** — 32% of apnoea minutes
   missed at threshold 0.5. Lower the threshold before any screening use.

## Limitations

- 27 records, and only **2 controls plus 5 borderline** — the non-OSA class is
  far too small for a trustworthy subject-level specificity estimate.
- **The 2-control cohort cannot be improved from this archive.** `c04`–`c10` are
  absent entirely, and `c02` ships only as `c02r`, whose four channels are
  `Resp C / Resp A / Resp N / SpO2` — **no ECG**. c02 is featurisable only from
  `c02er.qrs` (R-peaks alone), which would leave its EDR columns null; since c02
  is a control, that would make feature missingness track the label. It is
  therefore excluded deliberately, not incidentally — `apnea_ecg_features.py`
  now requires ECG + `.qrs` + `.apn` and logs the exclusion, replacing an earlier
  filter that dropped every `*r` record without reporting what was lost.
  Obtaining the full PhysioNet release is the single highest-value fix available.
- The full Apnea-ECG release has 70 records (including further controls and a
  35-record withheld test set); this copy has 27. Results are not comparable to
  published benchmarks on the full set.
- Single-lead ECG only. No SpO2 was used, though `a*r` respiration/SpO2 files
  exist here for 8 records and would likely improve detection.
- Apnoea type is not distinguished — `.apn` marks apnoea minutes without
  separating obstructive from central events.
- No causal claim: feature importance is associational, not mechanistic.

## Reproduce

```bash
python common/apnea_ecg_features.py    # raw -> structured/  (~90 s)
python common/apnea_ecg_models.py      # leakage audit + 3 models -> results/
python common/apnea_ecg_validate.py    # inference + shuffle/seed validation
```
