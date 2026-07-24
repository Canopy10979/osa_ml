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
