"""Train the three baseline models and plot accuracy separately for each OSA class."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed_dataset.csv"
FIGURES_FOLDER = PROJECT_ROOT / "figures"
RESULTS_FOLDER = PROJECT_ROOT / "results"

FEATURES = ["HR_Mean", "SpO2_Mean", "Flow_Mean"]
TARGET = "Sleep_Label"


def build_models():
    """Use the same model settings as compare_models.py."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                (
                    "scaler",
                    ColumnTransformer(
                        [("numeric", StandardScaler(), FEATURES)],
                        remainder="drop",
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),
    }


def calculate_class_accuracy():
    df = pd.read_csv(DATA_FILE)
    df[FEATURES + [TARGET]] = df[FEATURES + [TARGET]].apply(
        pd.to_numeric, errors="coerce"
    )
    df = df.dropna(subset=FEATURES + [TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=0.20,
        random_state=42,
        stratify=df[TARGET],
    )

    rows = []
    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()
        rows.append(
            {
                "Model": model_name,
                "Without OSA Accuracy": tn / (tn + fp),
                "With OSA Accuracy": tp / (tp + fn),
                "True 0 Samples": int(tn + fp),
                "True 1 Samples": int(tp + fn),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }
        )

    return pd.DataFrame(rows)


def add_bar_labels(ax, bars):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.6,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


def plot_class_accuracy(results):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Segoe UI", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.labelweight": "semibold",
        }
    )

    names = results["Model"].tolist()
    without_osa = results["Without OSA Accuracy"].mul(100).tolist()
    with_osa = results["With OSA Accuracy"].mul(100).tolist()
    colors = ["#287271", "#D1495B"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    panels = [
        (without_osa, "Without OSA (actual class 0)", colors[0]),
        (with_osa, "With OSA (actual class 1)", colors[1]),
    ]

    for ax, (values, title, color) in zip(axes, panels):
        bars = ax.bar(names, values, width=0.64, color=color)
        ax.set_title(title, fontsize=15, pad=14)
        ax.set_ylim(0, 108)
        ax.set_ylabel("Correct predictions within class (%)")
        ax.set_xlabel("Model")
        ax.grid(axis="y", color="#D9DEE3", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=12)
        add_bar_labels(ax, bars)

    n_zero = results.loc[0, "True 0 Samples"]
    n_one = results.loc[0, "True 1 Samples"]
    fig.suptitle(
        "Model Accuracy by OSA Status",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        f"Held-out test set: {n_zero} without OSA samples and {n_one} with OSA samples",
        ha="center",
        color="#4B5563",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.93), w_pad=3)
    fig.savefig(
        FIGURES_FOLDER / "model_accuracy_by_osa_status.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    # Keep the original comparison filename updated for existing references.
    fig.savefig(
        FIGURES_FOLDER / "LogRegression, Tree, Forest.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


if __name__ == "__main__":
    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    class_results = calculate_class_accuracy()
    class_results.to_csv(
        RESULTS_FOLDER / "model_accuracy_by_osa_status.csv", index=False
    )
    plot_class_accuracy(class_results)
    print(class_results.to_string(index=False))
    print("\nSaved class-specific accuracy chart and results.")
