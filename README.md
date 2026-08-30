# OSA ML

The repository is organized by dataset, with shared pipeline code and separate
cross-dataset, validation, and mentor-review outputs.

```
OSA ML/
|-- common/                              # Shared features, models, and validation code
|-- dataset_apnea_ecg/                   # PhysioNet Apnea-ECG pipeline
|   |-- raw/
|   |-- structured/
|   |-- models/
|   |-- results/
|   `-- report.md
|-- dataset_apnea_hrv/                   # Apnea-HRV pipeline
|   |-- raw/
|   |-- structured/
|   |-- models/
|   |-- results/
|   `-- report.md
|-- dataset_bioradiolocation/            # Sleep Bioradiolocation pipeline
|   |-- raw/
|   |-- structured/
|   |-- models/
|   |-- results/
|   `-- report.md
|-- dataset_ucddb_v2/                    # UCDDB feature and model pipeline
|   |-- raw/
|   |-- structured/
|   |-- models/
|   |-- results/
|   `-- report.md
|-- cross_dataset/                       # Harmonized data and transfer comparisons
|   |-- raw/
|   |-- structured/
|   |-- models/
|   |-- results/
|   |-- figures/
|   `-- comparison.md
|-- validation/
|   `-- block_cross_validation/          # Block CV summaries and fold results
|-- mentor_action_items/                 # Review findings, tables, and figures
|-- patient_level_aggregation_findings.md
|-- generate_aggregation_report.py
|-- aggregation_code.txt
|-- aggregation_model_code.txt
|-- FINAL_REPORT.md
|-- results_dashboard.html
|-- requirements.txt
|-- requirements_full.txt
|-- osa-ml-skill.md
`-- README.md
```

Each `dataset_*` directory owns its raw-data provenance, structured features,
model configuration, evaluation results, and report. Shared transformations and
model helpers live in `common/`. Comparative experiments live in
`cross_dataset/`, while validation runs are isolated under `validation/`.

## Raw data is not tracked

Each `dataset_*/raw/` has a `.gitignore` that ignores everything but itself and
a provenance `README.md`. Note that `dataset_apnea_ecg/raw/` was committed
before this rule existed, so those files remain in the index and in history —
`.gitignore` only affects untracked files.

## Label availability (read before modelling)

| Dataset | OSA label? |
|---|---|
| apnea_ecg | **Yes** — record prefix `a*`=apnea, `b*`=borderline, `c*`=control |
| apnea_hrv | **Yes** — subject/record labels supplied by the source dataset |
| bioradiolocation | **No** — all 32 subjects have AHI ≤ 4.9, below the AHI ≥ 5 threshold |
| ucddb_v2 | **Yes** — subject-level OSA outcomes and epoch-level features |

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
