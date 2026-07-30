---
name: osa-ml-pipeline
description: "End-to-end binary-classification data pipeline for OSA vs. non-OSA (or any two-class label) across multiple discrete datasets. Use when the user points at a zip/folder of RAW files and asks to extract them into structured data, train several ML models (logistic regression, random forest, XGBoost), and analyze/correlate/validate the results. Triggers include 'OSA vs non-OSA', 'analyze the files in this zip and run ML models', 'build an OSA-ML folder', 'classify these datasets and validate the results', or any request for a leakage-audited, reproducible multi-dataset classification study. Do NOT use for single-file spreadsheet cleanup (use xlsx), document generation (use docx), or unsupervised/forecasting-only tasks."
license: Proprietary
---

# OSA vs. Non-OSA ML Pipeline

Turn a folder/zip of raw files into a validated, reproducible binary-classification study:
**extract → structure → engineer → train 3 models → evaluate → infer → cross-validate the
analysis itself.** Built for OSA (obstructive sleep apnea) vs. non-OSA, but the label is a
parameter — it works for any two-class target across several discrete datasets.

## Operating principles (non-negotiable)

1. **Validate at every stage before advancing.** Print a checkpoint summary; halt on failure
   rather than proceeding with bad data.
2. **Every number in the final report must trace to a saved artifact.** No numbers invented
   in prose.
3. **Reproducible:** pinned seeds, logged library versions, no hidden state.
4. **Report negative or inconclusive results honestly.** Never tune until the story looks good.
5. **No causal claims from observational data.** Correlation and model importance are
   distinct concepts — say which one you mean.

## Two questions to resolve BEFORE Stage 1

Ask the user (use `ask_user`) unless the answer is unambiguous in the data:

- **How is the label defined?** Explicit column, AHI threshold (e.g. AHI >= 5 / 15 / 30),
  folder or filename convention, or ICD/diagnosis code? State the rule verbatim in the report.
- **What is the unit of observation?** Per-subject or per-epoch/per-night/per-window? If
  subjects contribute multiple rows you **must** use grouped splitting by subject ID.
  This is the single most common cause of inflated OSA-model scores.

## Workflow

### Stage 0 — Inventory & profiling
- Unzip to `OSA-ML/_inbox/`. Never mutate the original archive.
- Run `scripts/inventory.py <input_dir> <out_manifest.csv>`: name, ext, size, encoding,
  detected delimiter, row/col counts, sha256.
- Decide how many **discrete datasets** exist and what separates them (source, schema,
  cohort, time range). One folder per dataset downstream.
- Scaffold with `scripts/init_project.py <root> --datasets a b c` — copies `common/` and
  creates the tree below.

### Stage 1 — Extraction → structured files
- Parse each raw file into a tidy table: one row = one observation, `snake_case` columns,
  explicit dtypes. Do **not** assume schema — inspect first.
- Write **Parquet** (primary) + **CSV** (human inspection) to `dataset_<n>/structured/`.
- Emit `data_dictionary.md`: column, dtype, units, null %, min/max or cardinality.
- **Checkpoint:** rows in == rows out (± documented drops); no silent type coercion; every
  malformed record logged to `rejects.csv` with a reason.

### Stage 2 — Cleaning & feature engineering
- Missingness: state a strategy per column. **Never impute the target.**
- Outliers: flag and document; delete only with written justification.
- **Leakage audit is mandatory** — run `common/validation.audit_leakage()`. Drop any feature
  that encodes the outcome after the fact (diagnosis codes, CPAP/treatment flags, AHI itself
  when AHI defines the label, post-hoc severity scores, near-perfect single-feature
  separators).
- Report class balance per dataset.

### Stage 3 — Modeling (three models, identical splits)
Per dataset train: **Logistic Regression** (scaled; compare L1 vs L2 — the interpretable
baseline), **Random Forest**, **XGBoost / Gradient Boosting** (falls back to sklearn
`HistGradientBoosting` if `xgboost` is unavailable).
- Stratified train/test split **plus** stratified 5-fold CV on train; seed recorded.
- **Grouped splitting by subject ID** when rows are not independent.
- Imbalance: class weights and/or SMOTE **on training folds only**; report the delta.
- Modest documented hyperparameter grid. **Never tune on the test set.**

### Stage 4 — Evaluation
Per model per dataset: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix,
calibration curve. **Lead with recall and PR-AUC** — a missed OSA case is the costly error.
- Importance: LR coefficients **plus odds ratios**, RF impurity + permutation importance,
  XGB gain + SHAP (skip SHAP gracefully if not installed).
- Always compare to a majority-class baseline. A model that does not beat it is a failure,
  not a result — say so.

### Stage 5 — Correlation, inference, conclusions
- Correlation matrix + VIF for multicollinearity.
- Cross-dataset comparison: which features are consistently predictive vs. dataset-specific?
  Where do the three models agree/disagree, and why?
- Conclusions with explicit confidence levels, sample-size caveats, and limitations.

### Stage 6 — Validate the analysis itself
- Re-run with >= 2 alternate seeds; report metric variance.
- **Label-shuffle test:** performance must collapse to chance. If it does not, you have
  leakage — go back to Stage 2.
- Cross-dataset generalization: train on one, test on another where schemas align.
- Write `validation/leakage_audit.md` and `validation/seed_variance.md`.

## Output structure

```
OSA-ML/
├── common/                 # shared importable code — never duplicated per dataset
│   ├── io_utils.py         # loaders, writers, manifest, hashing
│   ├── preprocessing.py    # cleaning, features, splits (grouped-aware)
│   ├── models.py           # LR / RF / XGB definitions + training
│   ├── evaluation.py       # metrics, plots, importances, SHAP
│   └── validation.py       # leakage audit, shuffle test, seed variance
├── dataset_<name>/
│   ├── raw/
│   ├── structured/         # parquet + csv + data_dictionary.md + rejects.csv
│   ├── models/             # serialized models + params.json
│   ├── results/            # metrics.json, plots/, feature_importance.csv
│   └── report.md
├── cross_dataset/          # comparative analysis
├── validation/             # seed variance, shuffle test, leakage audit
├── FINAL_REPORT.md
└── README.md               # how to reproduce: env, versions, seeds, commands
```

## Prereqs

Python with `pandas`, `numpy`, `scikit-learn`, `pyarrow`, `matplotlib`.
Optional and used when present: `xgboost`, `shap`, `imbalanced-learn`, `statsmodels`.
Install only what is missing. `scripts/check_env.py` reports what is available and pins
versions into `README.md`.

## Notes
- Datasets differ; the `common/` helpers are a starting point, not a straitjacket. Extend
  them in `common/` rather than copy-pasting variants into dataset folders.
- If a dataset is too small for a meaningful test split (< ~50 rows or < ~10 minority cases),
  say so and report cross-validated metrics only — do not manufacture a test set.
