from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path("data/processed_dataset.csv")
RESULTS_DIR = Path("results/block_cross_validation")

FEATURES = ["HR_Mean", "SpO2_Mean", "Flow_Mean"]
TARGET = "Sleep_Label"


def build_models():
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]),

        "decision_tree": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                DecisionTreeClassifier(
                    class_weight="balanced",
                    max_depth=6,
                    min_samples_leaf=4,
                    random_state=42,
                ),
            ),
        ]),

        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    if "Window" in df.columns:
        df = df.sort_values("Window").reset_index(drop=True)

    df = df.dropna(subset=FEATURES + [TARGET]).copy()

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    splitter = TimeSeriesSplit(
        n_splits=5,
        gap=1,
    )

    all_results = []

    for model_name, model in build_models().items():
        print("\n" + "=" * 60)
        print(model_name.replace("_", " ").title())
        print("=" * 60)

        fold_number = 1

        for train_indexes, validation_indexes in splitter.split(X):
            X_train = X.iloc[train_indexes]
            X_validation = X.iloc[validation_indexes]

            y_train = y.iloc[train_indexes]
            y_validation = y.iloc[validation_indexes]

            if y_train.nunique() < 2:
                print(
                    f"Skipping fold {fold_number}: "
                    "training block contains only one class."
                )
                fold_number += 1
                continue

            model.fit(X_train, y_train)
            predictions = model.predict(X_validation)

            result = {
                "Model": model_name,
                "Fold": fold_number,
                "Training_Rows": len(train_indexes),
                "Validation_Rows": len(validation_indexes),
                "Accuracy": accuracy_score(
                    y_validation,
                    predictions,
                ),
                "Balanced_Accuracy": balanced_accuracy_score(
                    y_validation,
                    predictions,
                ),
                "Class_0_Precision": precision_score(
                    y_validation,
                    predictions,
                    pos_label=0,
                    zero_division=0,
                ),
                "Class_0_Recall": recall_score(
                    y_validation,
                    predictions,
                    pos_label=0,
                    zero_division=0,
                ),
                "Class_1_Precision": precision_score(
                    y_validation,
                    predictions,
                    pos_label=1,
                    zero_division=0,
                ),
                "Class_1_Recall": recall_score(
                    y_validation,
                    predictions,
                    pos_label=1,
                    zero_division=0,
                ),
            }

            all_results.append(result)

            print(
                f"Fold {fold_number}: "
                f"balanced accuracy = "
                f"{result['Balanced_Accuracy']:.4f}"
            )

            fold_number += 1

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        RESULTS_DIR / "block_cv_results.csv",
        index=False,
    )

    if not results_df.empty:
        summary = (
            results_df
            .groupby("Model")[
                [
                    "Accuracy",
                    "Balanced_Accuracy",
                    "Class_0_Precision",
                    "Class_0_Recall",
                    "Class_1_Precision",
                    "Class_1_Recall",
                ]
            ]
            .mean()
        )

        summary.to_csv(
            RESULTS_DIR / "block_cv_summary.csv"
        )

        print("\nAverage results:")
        print(summary)

    print("\nBlock cross-validation complete.")


if __name__ == "__main__":
    main()
