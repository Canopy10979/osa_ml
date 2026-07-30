from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path("data/processed_dataset.csv")
RESULTS_DIR = Path("results/balanced_models")
MODELS_DIR = Path("models")

TARGET = "Sleep_Label"
FEATURES = ["HR_Mean", "SpO2_Mean", "Flow_Mean"]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Sort chronologically when a Window column exists.
    if "Window" in df.columns:
        df = df.sort_values("Window").reset_index(drop=True)

    # Remove rows missing necessary training values.
    df = df.dropna(subset=required_columns).copy()

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    # Use the first 80% for training and the final 20% for testing.
    split_index = int(len(df) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("Full dataset shape:", df.shape)
    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting class distribution:")
    print(y_test.value_counts().sort_index())

    models = {
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

    overall_results = []

    for model_name, model in models.items():
        print("\n" + "=" * 60)
        print(model_name.replace("_", " ").title())
        print("=" * 60)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        balanced_accuracy = balanced_accuracy_score(
            y_test,
            predictions,
        )

        print("\nAccuracy:", round(accuracy, 4))
        print(
            "Balanced accuracy:",
            round(balanced_accuracy, 4),
        )

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0,
            )
        )

        report = classification_report(
            y_test,
            predictions,
            zero_division=0,
            output_dict=True,
        )

        report_df = pd.DataFrame(report).transpose()

        report_df.to_csv(
            RESULTS_DIR /
            f"{model_name}_classification_report.csv"
        )

        matrix = confusion_matrix(y_test, predictions)

        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual_0", "Actual_1"],
            columns=["Predicted_0", "Predicted_1"],
        )

        matrix_df.to_csv(
            RESULTS_DIR /
            f"{model_name}_confusion_matrix.csv"
        )

        joblib.dump(
            model,
            MODELS_DIR / f"{model_name}.joblib",
        )

        overall_results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Balanced_Accuracy": balanced_accuracy,
        })

    pd.DataFrame(overall_results).to_csv(
        RESULTS_DIR / "overall_metrics.csv",
        index=False,
    )

    print("\nTraining completed successfully.")
    print("Results folder:", RESULTS_DIR.resolve())
    print("Models folder:", MODELS_DIR.resolve())


if __name__ == "__main__":
    main()
