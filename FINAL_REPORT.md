# FINAL_REPORT — OSA vs. non-OSA across five datasets

**Question:** can obstructive sleep apnoea be detected from heartbeat timing alone, and
can a patient be told apart from a healthy person?

**Answer:** yes to both. Two datasets now support real claims, and they answer different
questions: `ucddb_v2` (full sleep-lab sensors) pinpoints individual apnoeas, while
`apnea_hrv` (heartbeat only) is the better screener and transfers across hospitals.
Getting there took a corrected feature bug, a cross-cohort transfer test, and a
time-alignment gate.

Every number here traces to a saved artifact. Per-dataset detail is in each
`dataset_*/report.md`; the comparative analysis is in
[`cross_dataset/comparison.md`](cross_dataset/comparison.md); the visual summary is
[`results_dashboard.html`](results_dashboard.html).

---

## Inventory

| dataset | subjects | observations | status |
|---|---|---|---|
| [`dataset_ucddb_v2`](dataset_ucddb_v2/report.md) | 25 (11 non-OSA / 14 OSA) | 20,789 epochs | ✅ **Best event detection** — full PSG with SpO2 |
| [`dataset_apnea_hrv`](dataset_apnea_hrv/report.md) | 77 (40 control / 37 patient) | 30,445 minutes | ✅ **Best screener** — heartbeat only |
| [`dataset_apnea_ecg`](dataset_apnea_ecg/report.md) | 27 (2 control / 23 OSA) | 13,286 minutes | ✅ Minute detection; ❌ cannot rule OSA out |
| `dataset_bioradiolocation` | 32 | 1.6 GB radar, unanalysed | ❌ **Zero OSA cases** (max AHI 4.9) |
| [`cross_dataset`](cross_dataset/comparison.md) | 1 (= UCDDB011) | 1,069 epochs | ⚠️ **Mislabelling fixed; result is negative** |

## The headline results

Two datasets answer different questions, and the distinction matters more than either
headline number.

### `dataset_ucddb_v2` — detects events

25 full polysomnograms with **SpO2, airflow and effort belts**. Per-epoch ROC-AUC
**0.928**, PR-AUC 0.789 against a 0.212 baseline (3.7× lift), seed sd 0.003. The top nine
features are all SpO2-derived — the direct physiological consequence of an apnoea.

**Only ~7% of its skill is between-subject**, against 40% for `apnea_ecg` and 59% for
`apnea_hrv`. It genuinely localises events in time rather than recognising which patients
are sick. It also **fills the project's biggest gap**: 16 of its 25 subjects sit in the
AHI 5–25 band that `apnea_hrv` excludes by design.

Subject-level (AHI ≥ 15): predicted index vs recorded AHI r = 0.951, ROC-AUC 0.916,
leave-one-subject-out balanced accuracy 0.847. No test set was manufactured — 25 subjects
with 11 in the minority is below the threshold where a split carries information.

### `dataset_apnea_hrv` — best screener

The only cohort with enough healthy controls to answer the screening question.

| | |
|---|---|
| Patient vs. control separation | **ROC-AUC 0.937** |
| On unseen test subjects | **ROC-AUC 0.921** |
| Sensitivity / specificity | **0.892 / 0.900** (36/40 controls, 33/37 patients) |
| Per-minute detection | ROC-AUC 0.865, PR-AUC 0.683 vs 0.221 baseline |
| Seed variance | PR-AUC sd 0.006 |
| Label-shuffle test | ROC-AUC 0.495 — **passes**, no leakage |

`dataset_apnea_ecg` independently confirms that per-minute apnoea is detectable from
heartbeat alone (PR-AUC 0.838 vs 0.489 baseline) but, with **2 controls**, cannot
estimate specificity: it identified 1 of its 4 non-OSA subjects.

## The mechanism — depends on what you can measure

**With a pulse oximeter: SpO2 desaturation.** In `ucddb_v2` the entire top of the
importance table is SpO2-derived (`spo2_std`, `spo2_range`, `spo2_desat`, univariate AUC
up to 0.849). Breathing stops, oxygen falls, the gasp restores it.

**From heartbeat alone: cyclical variation in heart rate (CVHR, 0.010–0.040 Hz).** Every apnoea ends in an
arousal that spikes heart rate; repeated all night this produces a slow rhythmic
surge-and-settle roughly every 25–100 seconds. In `apnea_hrv` the four most informative
features are all CVHR variants, peaking at **univariate AUC 0.806**. Respiratory-band HRV
(`r_hf`) is second at 0.751 — it falls during apnoea, as physiology predicts.

These are **associations and model importances, not causal claims.**

## Findings transfer across cohorts

The strongest evidence that this is physiology rather than dataset artefact:

| train → test | ROC-AUC | within-dataset reference |
|---|---|---|
| apnea_hrv → apnea_ecg | **0.795** | 0.796 |
| apnea_ecg → apnea_hrv | 0.749 | 0.799 |

A model trained on 77 Spanish hospital subjects performs on PhysioNet Apnea-ECG
**within 0.001 ROC-AUC** of a model trained on Apnea-ECG itself — across different
hardware, country, decade, and an apnoea prevalence of 22% vs 49%. Feature
informativeness ranks correlate at Spearman 0.807, with 76% of features pointing the same
direction in both cohorts.

## Two defects found and corrected

1. **`apnea_ecg`'s CVHR features were identically zero.** Its `band_power()` returns 0.0
   unless more than two FFT bins fall in the band; at 4 Hz with `nperseg=256` the
   resolution is 0.015625 Hz, so the 0.010–0.040 Hz band held exactly two bins and the
   guard fired on every frame. That dataset's report had recorded "CVHR does not appear
   in the top 10" as a scientific finding — it was a measurement failure. Corrected in
   `cross_dataset/comparison.md` §3a and in the `apnea_hrv` report.

2. **`cross_dataset` was classifying sleep, not apnoea.** Its 97%-accuracy "OSA" result
   trained on `Sleep_Label` ("0=awake, 1=asleep"). The source is now identified as
   **UCDDB011** by start time and duration, its real respiratory-event labels attached,
   and the analysis re-run: the corrected target gives ROC-AUC **0.602** against a 0.052
   baseline, with both tree models predicting zero events. The old and new targets agree
   on only 24% of epochs. Reported as the negative result it is — see
   `cross_dataset/comparison.md` §4.

3. **The two pipelines share column names but not units** — RR in seconds vs
   milliseconds (~1000×). A naive transfer saturated every prediction to 0.0 and yielded
   a spurious ROC-AUC of exactly 0.5000, which reads like "no transferable signal." The
   corrected test standardises each cohort within itself.

A third correction, made earlier: `apnea_ecg`'s shuffle test **failed the whole run**
because its verdict required both the global *and* within-record shuffles to collapse.
Within-record shuffling preserves each subject's apnoea burden, so a grouped model scores
above chance there with nothing leaked. The verdict now rests on the global shuffle
(which passes), with the within-record result reported as a quantified diagnostic.

## Honest limitations

- **Both working models are inflated by subject identity.** ~59% of skill in `apnea_hrv`
  and ~40% in `apnea_ecg` comes from recognising *who the person is* rather than which
  minute is an apnoea. Good for screening; a weaker claim than per-minute scores suggest.
- **Mild-to-moderate OSA is untested.** `apnea_hrv` is gapped by design — no subject
  between AHI 5 and 25 — which is precisely the diagnostically ambiguous band.
- **The `apnea_hrv` operating threshold (9.5 events/h) was tuned on the same 77
  subjects**, so that figure is optimistic. The threshold-free ROC-AUC 0.921 on unseen
  subjects is the trustworthy number.
- **Probabilities are not calibrated** in any dataset and should not be read as risks.
- **`ucddb_v2` is 25 subjects** and clinically referred (24/25 above AHI 5), so it is not a
  screening population, and its perfect specificity (11/11) has a wide confidence interval.
- **`ucddb_v2` requires full PSG.** Its result does not transfer to wearable or
  single-channel settings — that is what the two heartbeat datasets speak to.
- **`dataset_apnea_ecg` is not currently reproducible.** Its pipeline scripts were deleted
  from `common/` in commit `b80b397` and survive only in git history.

## What would most improve this

1. Obtain PhysioNet Apnea-ECG's full control set (`c04`–`c10`) — `apnea_ecg`'s 2-control
   cohort is its binding limitation and cannot be fixed from the current archive
   (`c02` ships without an ECG channel).
2. Restore or rewrite the `apnea_ecg` pipeline scripts, recomputing CVHR at adequate
   spectral resolution — its results should be regenerated, not merely reinterpreted.

## Reproduce

```bash
python common/apnea_hrv_features.py && python common/apnea_hrv_models.py && python common/apnea_hrv_validate.py && python common/cross_dataset_compare.py
```

Seed 42 throughout; all cross-validation is `StratifiedGroupKFold(5)` grouped by subject.
Raw data is not tracked in git — see each `dataset_*/raw/README.md` for provenance.
