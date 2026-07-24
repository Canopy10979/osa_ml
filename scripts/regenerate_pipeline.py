"""Rebuild the OSA ML analysis from raw signals with reproducible randomized systematic sampling.

Sampling rule requested by the project owner:
1. Build every valid 30-second record from the raw HR, SpO2, Flow_DR, and sleep-stage files.
2. Randomly shuffle records with a fixed seed.
3. Keep every fifth shuffled record (positions 0, 5, 10, ...).
4. Split the sampled records using a stratified train/test split.
5. Fit normalization on training rows only to prevent data leakage.
6. Regenerate datasets, figures, trained models, and evaluation tables.

Important label note: 50_sleep_stage.csv contains sleep stages, not scored apnea events.
The retained target therefore represents awake (0) versus asleep (1), matching the
legacy project mapping. It must not be described as a clinically validated apnea label.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DATA = DATA_DIR / "regenerated"
OUT_RESULTS = ROOT / "results" / "regenerated"
OUT_FIGURES = ROOT / "figures" / "regenerated"
OUT_MODELS = ROOT / "models" / "regenerated"

RANDOM_STATE = 42
WINDOW_SECONDS = 30
SYSTEMATIC_STEP = 5
TEST_SIZE = 0.20
FEATURES = ["HR_Mean", "SpO2_Mean", "Flow_Mean"]
TARGET = "Sleep_Label"


def time_to_seconds(value: object) -> float:
    """Convert hh:mm:ss.ms to seconds; relative timestamps need no midnight correction."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) != 3:
        return np.nan
    try:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return np.nan


def load_signal(filename: str, value_column: str, output_column: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / filename)
    time_column = "relative position (hh:mm:ss.ms)"
    df["Seconds"] = df[time_column].map(time_to_seconds)
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
    df["Window"] = np.floor(df["Seconds"] / WINDOW_SECONDS).astype("Int64")
    out = (
        df.dropna(subset=["Window", value_column])
        .groupby("Window", as_index=False)[value_column]
        .mean()
        .rename(columns={value_column: output_column})
    )
    out["Window"] = out["Window"].astype(int)
    return out


def build_structured_dataset() -> pd.DataFrame:
    hr = load_signal("50_HR.csv", 'Heart Rate ("bpm")', "HR_Mean")
    spo2 = load_signal("50_SpO2.csv", 'OSat ("%")', "SpO2_Mean")
    flow = load_signal("50_Flow_DR.csv", "Flow_DR", "Flow_Mean")

    features = hr.merge(spo2, on="Window", how="inner").merge(flow, on="Window", how="inner")

    sleep = pd.read_csv(DATA_DIR / "50_sleep_stage.csv")
    stage_col = 'Default Staging Set ("stage")'
    sleep["Stage"] = sleep[stage_col].astype(str).str.strip().str.upper()
    sleep[TARGET] = sleep["Stage"].map({"W": 0, "N1": 1, "N2": 1, "N3": 1, "N4": 1, "R": 1, "REM": 1})
    # Each sleep-stage row is one 30-second epoch, aligned to Window 0.
    sleep["Window"] = np.arange(len(sleep), dtype=int)
    labels = sleep[["Window", "Stage", TARGET]].dropna(subset=[TARGET])
    labels[TARGET] = labels[TARGET].astype(int)

    dataset = features.merge(labels, on="Window", how="inner")
    dataset = dataset.dropna(subset=FEATURES + [TARGET]).sort_values("Window").reset_index(drop=True)
    dataset.insert(0, "Record_ID", np.arange(len(dataset), dtype=int))
    return dataset


def save_distribution_figure(df: pd.DataFrame, column: str, filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[column].dropna(), bins=30, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / filename, dpi=160)
    plt.close(fig)


def generate_figures(sampled: pd.DataFrame, metrics_df: pd.DataFrame, model_payloads: dict) -> None:
    # Class distribution
    counts = sampled[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([str(i) for i in counts.index], counts.values)
    ax.set_title("Randomized Every-Fifth Sample: Class Distribution")
    ax.set_xlabel("Sleep_Label (0=awake, 1=asleep)")
    ax.set_ylabel("Records")
    for i, value in enumerate(counts.values):
        ax.text(i, value, str(value), ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(OUT_FIGURES / "class_distribution.png", dpi=160); plt.close(fig)

    save_distribution_figure(sampled, "HR_Mean", "hr_distribution.png", "Heart Rate Distribution")
    save_distribution_figure(sampled, "SpO2_Mean", "spo2_distribution.png", "SpO2 Distribution")
    save_distribution_figure(sampled, "Flow_Mean", "flow_distribution.png", "Airflow Distribution")

    # Correlation heatmap without seaborn.
    corr = sampled[FEATURES + [TARGET]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    ax.set_title("Correlation Heatmap — Regenerated Sample")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout(); fig.savefig(OUT_FIGURES / "correlation_heatmap.png", dpi=160); plt.close(fig)

    # Model comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics_df))
    width = 0.36
    ax.bar(x - width/2, metrics_df["Accuracy"], width, label="Accuracy")
    ax.bar(x + width/2, metrics_df["Balanced_Accuracy"], width, label="Balanced accuracy")
    ax.set_xticks(x, metrics_df["Model"], rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance — Randomized Every-Fifth Sample")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT_FIGURES / "model_comparison.png", dpi=160); plt.close(fig)

    # ROC and PR curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, payload in model_payloads.items():
        fpr, tpr, _ = roc_curve(payload["y_test"], payload["probabilities"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={payload['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves"); ax.legend(); fig.tight_layout()
    fig.savefig(OUT_FIGURES / "roc_curves.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, payload in model_payloads.items():
        precision, recall, _ = precision_recall_curve(payload["y_test"], payload["probabilities"])
        ax.plot(recall, precision, label=f"{name} (AP={payload['average_precision']:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves"); ax.legend(); fig.tight_layout()
    fig.savefig(OUT_FIGURES / "precision_recall_curves.png", dpi=160); plt.close(fig)

    # Confusion matrices
    for name, payload in model_payloads.items():
        cm = payload["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
        ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_title(f"{name} Confusion Matrix")
        fig.colorbar(image, ax=ax); fig.tight_layout()
        fig.savefig(OUT_FIGURES / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=160)
        plt.close(fig)

    # Random Forest feature importance
    rf = model_payloads["Random Forest"]["model"]
    importances = rf.feature_importances_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(FEATURES)[order], importances[order])
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    fig.tight_layout(); fig.savefig(OUT_FIGURES / "feature_importance.png", dpi=160); plt.close(fig)


def main() -> None:
    for directory in (OUT_DATA, OUT_RESULTS, OUT_FIGURES, OUT_MODELS):
        directory.mkdir(parents=True, exist_ok=True)

    full = build_structured_dataset()
    full.to_csv(OUT_DATA / "structured_full.csv", index=False)

    # Seeded randomization followed by systematic every-fifth selection.
    randomized = full.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    randomized.insert(0, "Randomized_Position", np.arange(len(randomized), dtype=int))
    sampled = randomized.iloc[::SYSTEMATIC_STEP].copy().reset_index(drop=True)
    sampled.insert(0, "Sampled_Row", np.arange(len(sampled), dtype=int))
    sampled.to_csv(OUT_DATA / "structured_randomized_every_5th.csv", index=False)
    sampled[["Sampled_Row", "Randomized_Position", "Record_ID", "Window", "Stage", TARGET]].to_csv(
        OUT_DATA / "sampled_record_manifest.csv", index=False
    )

    X = sampled[FEATURES].copy()
    y = sampled[TARGET].astype(int)
    train_idx, test_idx = train_test_split(
        np.arange(len(sampled)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    train_idx = np.sort(train_idx); test_idx = np.sort(test_idx)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler().fit(X_train)
    joblib.dump(scaler, OUT_MODELS / "standard_scaler.joblib")

    normalized = sampled.copy()
    normalized[FEATURES] = scaler.transform(X)
    normalized["Split"] = "train"
    normalized.loc[test_idx, "Split"] = "test"
    normalized.to_csv(OUT_DATA / "normalized_sampled_dataset.csv", index=False)
    normalized.iloc[train_idx].to_csv(OUT_DATA / "normalized_train.csv", index=False)
    normalized.iloc[test_idx].to_csv(OUT_DATA / "normalized_test.csv", index=False)

    pd.DataFrame({
        "Feature": FEATURES,
        "Training_Mean": scaler.mean_,
        "Training_Scale": scaler.scale_,
    }).to_csv(OUT_RESULTS / "normalization_parameters.csv", index=False)

    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=3000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_depth=6, min_samples_leaf=4, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=400, class_weight="balanced", min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
    }

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    metrics = []
    predictions_table = sampled.iloc[test_idx][["Record_ID", "Window", "Stage", TARGET]].reset_index(drop=True)
    payloads = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        prob = model.predict_proba(X_test_scaled)[:, 1]
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        report = pd.DataFrame(classification_report(y_test, pred, labels=[0, 1], output_dict=True, zero_division=0)).T
        slug = name.lower().replace(" ", "_")
        report.to_csv(OUT_RESULTS / f"{slug}_classification_report.csv")
        pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Predicted_0", "Predicted_1"]).to_csv(
            OUT_RESULTS / f"{slug}_confusion_matrix.csv"
        )
        joblib.dump(model, OUT_MODELS / f"{slug}.joblib")
        roc_auc = roc_auc_score(y_test, prob)
        avg_precision = average_precision_score(y_test, prob)
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Balanced_Accuracy": balanced_accuracy_score(y_test, pred),
            "ROC_AUC": roc_auc,
            "Average_Precision": avg_precision,
            "Test_Rows": len(y_test),
        })
        predictions_table[f"{slug}_prediction"] = pred
        predictions_table[f"{slug}_probability_1"] = prob
        payloads[name] = {"model": model, "y_test": y_test, "probabilities": prob,
                          "roc_auc": roc_auc, "average_precision": avg_precision,
                          "confusion_matrix": cm}

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT_RESULTS / "model_evaluation_results.csv", index=False)
    predictions_table.to_csv(OUT_RESULTS / "test_predictions.csv", index=False)

    class_summary = pd.DataFrame([
        {"Dataset": "Full structured", "Rows": len(full), "Class_0": int((full[TARGET] == 0).sum()), "Class_1": int((full[TARGET] == 1).sum())},
        {"Dataset": "Randomized every-fifth", "Rows": len(sampled), "Class_0": int((sampled[TARGET] == 0).sum()), "Class_1": int((sampled[TARGET] == 1).sum())},
        {"Dataset": "Training", "Rows": len(train_idx), "Class_0": int((y_train == 0).sum()), "Class_1": int((y_train == 1).sum())},
        {"Dataset": "Testing", "Rows": len(test_idx), "Class_0": int((y_test == 0).sum()), "Class_1": int((y_test == 1).sum())},
    ])
    class_summary["Positive_Fraction"] = class_summary["Class_1"] / class_summary["Rows"]
    class_summary.to_csv(OUT_RESULTS / "sampling_and_class_summary.csv", index=False)

    config = {
        "random_state": RANDOM_STATE,
        "window_seconds": WINDOW_SECONDS,
        "systematic_step": SYSTEMATIC_STEP,
        "test_size": TEST_SIZE,
        "features": FEATURES,
        "target": TARGET,
        "target_definition": "0=awake, 1=asleep; derived from sleep stages, not apnea-event annotations",
    }
    (OUT_RESULTS / "pipeline_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    generate_figures(sampled, metrics_df, payloads)

    print("Regeneration complete.")
    print(class_summary.to_string(index=False))
    print("\nModel metrics:\n", metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
