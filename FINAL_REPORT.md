# FINAL_REPORT

**Status: not yet written.** This file is required by the layout in
`osa-ml-skill.md`. It must not be populated with numbers that do not trace to a
saved artifact (operating principle 2), so it is left as a stub with an honest
inventory rather than a synthesised summary.

## What exists in this repo today

| Location | Contents |
|---|---|
| `dataset_apnea_ecg/` | PhysioNet Apnea-ECG, ~294 MB WFDB. **The only dataset here with real OSA labels.** ANALYSED — see [report](dataset_apnea_ecg/report.md): 13,286 minutes, XGBoost PR-AUC 0.838 vs 0.489 baseline. |
| `dataset_ucddb/raw/` | UCD Sleep Apnea Database subject details, 12 KB. Demographics only. |
| `dataset_bioradiolocation/raw/` | Sleep Bioradiolocation, 1.6 GB zip. **Contains zero OSA cases** (all AHI ≤ 4.9, threshold is ≥ 5). |
| `common/` | Shared pipeline code (formerly `scripts/`). |
| `cross_dataset/` | Prior run variants: `regenerated`, `batch_sampled`, `balanced_models`, plus flat results/models/figures. |
| `validation/` | `block_cross_validation` outputs. |

## Open items before this report can be written

1. ~~Per-dataset attribution.~~ `dataset_apnea_ecg/` is now fully populated
   (structured/, models/, results/, report.md). `ucddb` and `bioradiolocation`
   remain unanalysed; the pre-existing `cross_dataset/` run variants still
   cannot be attributed to one dataset.
2. ~~Label definition.~~ Stated verbatim in `dataset_apnea_ecg/report.md`:
   a minute is positive iff its `.apn` symbol is `A`; subject-level OSA is
   apnoea index >= 5.
3. **Provenance of `cross_dataset/structured/processed_dataset.csv`** is
   unresolved — its columns (`HR_Mean, SpO2_Mean, Flow_Mean`) are PSG-derived
   and therefore did *not* come from the ECG dataset.
4. **`cross_dataset/structured/osa_analysis.csv` is not a CSV** — it is a Python
   source file (begins `import os` / `import librosa`) with a `.csv` extension.
