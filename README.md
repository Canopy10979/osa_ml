# OSA ML

Layout follows the `Output structure` section of [osa-ml-skill.md](osa-ml-skill.md).

```
OSA ML/
├── common/                      # shared pipeline code (was scripts/)
├── dataset_apnea_ecg/           # PhysioNet Apnea-ECG  — has real OSA labels
├── dataset_ucddb/               # UCD Sleep Apnea DB   — demographics only
├── dataset_bioradiolocation/    # Sleep Bioradiolocation — ZERO OSA cases
│   └── {raw,structured,models,results}/ + report.md
├── cross_dataset/               # prior run variants + comparative outputs
├── validation/                  # block cross-validation
├── FINAL_REPORT.md
└── README.md
```

Datasets are split **by raw source**. The pre-existing `results/` and `models/`
were pipeline *run variants* (`regenerated`, `batch_sampled`, `balanced_models`,
`block_cross_validation`) rather than per-dataset outputs, so they live in
`cross_dataset/` and `validation/`; the `dataset_*/results/` folders are empty
until the pipeline is re-run scoped to one dataset.

## Raw data is not tracked

Each `dataset_*/raw/` has a `.gitignore` that ignores everything but itself and
a provenance `README.md`. Note that `dataset_apnea_ecg/raw/` was committed
before this rule existed, so those files remain in the index and in history —
`.gitignore` only affects untracked files.

## Label availability (read before modelling)

| Dataset | OSA label? |
|---|---|
| apnea_ecg | **Yes** — record prefix `a*`=apnea, `b*`=borderline, `c*`=control |
| ucddb | Demographics only |
| bioradiolocation | **No** — all 32 subjects have AHI ≤ 4.9, below the AHI ≥ 5 threshold |

---

<details>
<summary>Previous README</summary>

# Sleep Apnea Detection Using Machine Learning

## Overview

This project develops a machine learning pipeline to detect sleep apnea using physiological data.

The workflow includes:

- Data preprocessing
- Feature engineering
- Model training
- Hyperparameter tuning
- Model evaluation
- Visualization
- Prediction using trained models

---

## Dataset

Features:

- HR_Mean (Average Heart Rate)
- SpO2_Mean (Average Blood Oxygen Saturation)
- Flow_Mean (Average Airflow)

Target:

- Sleep_Label
  - 0 = Non-Apnea
  - 1 = Apnea

---

## Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest

The Random Forest classifier achieved the best overall performance.

---

## Evaluation

Metrics used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- Precision–Recall Curve
- Confusion Matrix
- Cross Validation

---

## Project Structure

```
SleepApneaML/
│
├── data/
├── models/
├── figures/
├── results/
├── scripts/
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run scripts from the `scripts` folder:

```bash
python train_models.py
python optimized_model.py
python predict.py
```

---

## Results

The optimized Random Forest model demonstrated excellent predictive performance on the processed dataset.

Generated outputs include:

- Model evaluation metrics
- ROC Curve
- Precision–Recall Curve
- Feature Importance
- Correlation Heatmap
- Evaluation Dashboard

---

## Future Work

Potential improvements include:

- Deep Learning (LSTM/CNN)
- Additional physiological signals
- Real-time monitoring
- Larger datasets
- External validation

</details>
