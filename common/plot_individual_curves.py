from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
)


DATA_PATH = Path("data/processed_dataset.csv")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results/balanced_models")

FEATURES = ["HR_Mean", "SpO2_Mean", "Flow_Mean"]
TARGET = "Sleep_Label"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    if "Window" in df.columns:
        df = df.sort_values("Window").reset_index(drop=True)

    df = df.dropna(subset=FEATURES + [TARGET]).copy()

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    split_index = int(len(df) * 0.80)

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    model_names = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
    ]

    auc_rows = []

    for model_name in model_names:
        model_path = MODELS_DIR / f"{model_name}.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing trained model: {model_path}"
            )

        model = joblib.load(model_path)

        probabilities = model.predict_proba(X_test)
        classes = model.named_steps["model"].classes_

        for class_value in classes:
            class_index = np.where(
                classes == class_value
            )[0][0]

            class_scores = probabilities[:, class_index]

            # Treat this one class as positive.
            y_binary = (
                y_test.to_numpy() == class_value
            ).astype(int)

            if len(np.unique(y_binary)) < 2:
                print(
                    f"Skipping {model_name}, class "
                    f"{class_value}: test set does not "
                    "contain both outcomes."
                )
                continue

            precision_values, recall_values, _ = (
                precision_recall_curve(
                    y_binary,
                    class_scores,
                )
            )

            false_positive_rate, true_positive_rate, _ = (
                roc_curve(
                    y_binary,
                    class_scores,
                )
            )

            pr_auc = auc(
                recall_values,
                precision_values,
            )

            roc_auc = auc(
                false_positive_rate,
                true_positive_rate,
            )

            auc_rows.append({
                "Model": model_name,
                "Class": int(class_value),
                "Precision_Recall_AUC": pr_auc,
                "ROC_AUC": roc_auc,
            })

            plt.figure(figsize=(8, 6))
            plt.plot(
                recall_values,
                precision_values,
                label=f"AUC = {pr_auc:.3f}",
            )
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(
                f"{model_name.replace('_', ' ').title()}\n"
                f"Precision-Recall Curve for Class {class_value}"
            )
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()

            plt.savefig(
                RESULTS_DIR /
                f"{model_name}_class_{class_value}_precision_recall.png",
                dpi=200,
            )

            plt.close()

            plt.figure(figsize=(8, 6))
            plt.plot(
                false_positive_rate,
                true_positive_rate,
                label=f"AUC = {roc_auc:.3f}",
            )
            plt.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                label="Random classifier",
            )
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(
                f"{model_name.replace('_', ' ').title()}\n"
                f"ROC Curve for Class {class_value}"
            )
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()

            plt.savefig(
                RESULTS_DIR /
                f"{model_name}_class_{class_value}_roc.png",
                dpi=200,
            )

            plt.close()

    pd.DataFrame(auc_rows).to_csv(
        RESULTS_DIR / "individual_curve_auc.csv",
        index=False,
    )

    print("Individual class curves created successfully.")


if __name__ == "__main__":
    main()
