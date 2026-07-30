from __future__ import annotations

# Fixed batch-sampling pipeline for rebuilding the sleep/wake dataset and training evaluation models.
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
SLEEP_FRACTION = 0.70   # prioritized, but still includes awake records
SAMPLE_FRACTION = 0.20  # total fraction of rows to retain after batch-aware sampling
TEST_SIZE = 0.20

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT_DATA = DATA / "batch_sampled"
OUT_FIG = ROOT / "figures" / "batch_sampled"
OUT_MODELS = ROOT / "models" / "batch_sampled"
OUT_RESULTS = ROOT / "results" / "batch_sampled"
for folder in (OUT_DATA, OUT_FIG, OUT_MODELS, OUT_RESULTS):
    folder.mkdir(parents=True, exist_ok=True)


def time_to_seconds(value: str) -> float:
    h, m, s = str(value).split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def build_processed_dataset() -> pd.DataFrame:
    hr = pd.read_csv(DATA / "50_HR.csv")
    spo2 = pd.read_csv(DATA / "50_SpO2.csv")
    flow = pd.read_csv(DATA / "50_Flow_DR.csv")
    sleep = pd.read_csv(DATA / "50_sleep_stage.csv")

    def aggregate(df: pd.DataFrame, value_col: str, output_col: str) -> pd.DataFrame:
        df = df.copy()
        df["Seconds"] = df["relative position (hh:mm:ss.ms)"].map(time_to_seconds)
        df["Window"] = (df["Seconds"] // 30).astype(int)
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        return (
            df.dropna(subset=[value_col])
              .groupby("Window", as_index=False)[value_col]
              .mean()
              .rename(columns={value_col: output_col})
        )

    hr_f = aggregate(hr, 'Heart Rate ("bpm")', "HR_Mean")
    spo2_f = aggregate(spo2, 'OSat ("%")', "SpO2_Mean")
    flow_f = aggregate(flow, "Flow_DR", "Flow_Mean")

    features = hr_f.merge(spo2_f, on="Window", how="inner").merge(flow_f, on="Window", how="inner")

    stage_col = 'Default Staging Set ("stage")'
    stage_map = {"W": 0, "N1": 1, "N2": 1, "N3": 1, "REM": 1, "R": 1}
    labels = sleep[[stage_col]].copy()
    labels["Sleep_Label"] = labels[stage_col].astype(str).str.strip().map(stage_map)
    labels["Window"] = np.arange(len(labels))

    dataset = features.merge(labels[["Window", "Sleep_Label"]], on="Window", how="inner")
    dataset = dataset.dropna(subset=["HR_Mean", "SpO2_Mean", "Flow_Mean", "Sleep_Label"])
    dataset["Sleep_Label"] = dataset["Sleep_Label"].astype(int)
    dataset = dataset.sort_values("Window").reset_index(drop=True)
    dataset.to_csv(OUT_DATA / "processed_full.csv", index=False)
    return dataset


def add_batch_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Window").reset_index(drop=True).copy()
    label_changed = df["Sleep_Label"].ne(df["Sleep_Label"].shift())
    window_gap = df["Window"].diff().fillna(1).ne(1)
    df["Batch_ID"] = (label_changed | window_gap).cumsum()
    return df


def sample_from_batches(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    target_total = max(2, int(round(len(df) * SAMPLE_FRACTION)))
    target_sleep = int(round(target_total * SLEEP_FRACTION))
    target_awake = target_total - target_sleep

    sampled_parts = []
    for label, target in [(1, target_sleep), (0, target_awake)]:
        class_df = df[df["Sleep_Label"] == label]
        if class_df.empty:
            continue

        batches = class_df["Batch_ID"].drop_duplicates().to_numpy().copy()
        rng.shuffle(batches)
        chosen_indices: list[int] = []

        # Cycle through randomized batches, drawing random rows from each batch.
        while len(chosen_indices) < min(target, len(class_df)):
            made_progress = False
            for batch_id in batches:
                batch_indices = class_df.index[class_df["Batch_ID"] == batch_id].to_numpy()
                remaining = np.setdiff1d(batch_indices, np.asarray(chosen_indices, dtype=int), assume_unique=False)
                if remaining.size == 0:
                    continue
                chosen_indices.append(int(rng.choice(remaining)))
                made_progress = True
                if len(chosen_indices) >= min(target, len(class_df)):
                    break
            if not made_progress:
                break

        sampled_parts.append(df.loc[chosen_indices])

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    sampled.insert(0, "Sample_Order", np.arange(1, len(sampled) + 1))
    sampled.to_csv(OUT_DATA / "batch_randomized_sample.csv", index=False)
    return sampled


def distribution_table(name: str, df: pd.DataFrame) -> pd.DataFrame:
    counts = df["Sleep_Label"].value_counts().reindex([0, 1], fill_value=0)
    out = pd.DataFrame({
        "Dataset": name,
        "Label": [0, 1],
        "Meaning": ["Awake", "Sleep"],
        "Count": [int(counts.loc[0]), int(counts.loc[1])],
    })
    out["Percent"] = out["Count"] / max(len(df), 1) * 100
    return out


def train_and_evaluate(sampled: pd.DataFrame) -> None:
    feature_cols = ["HR_Mean", "SpO2_Mean", "Flow_Mean"]
    X = sampled[feature_cols]
    y = sampled["Sleep_Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    train_export = X_train_scaled.copy()
    train_export["Sleep_Label"] = y_train
    test_export = X_test_scaled.copy()
    test_export["Sleep_Label"] = y_test
    train_export.to_csv(OUT_DATA / "normalized_train.csv", index=False)
    test_export.to_csv(OUT_DATA / "normalized_test.csv", index=False)
    joblib.dump(scaler, OUT_MODELS / "standard_scaler.joblib")

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE, max_depth=6),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
    }

    summary_rows = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        prob = model.predict_proba(X_test_scaled)[:, 1]

        report = classification_report(y_test, pred, labels=[0, 1], target_names=["Awake", "Sleep"], output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(OUT_RESULTS / f"{name}_classification_report.csv")
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        pd.DataFrame(cm, index=["Actual_Awake", "Actual_Sleep"], columns=["Pred_Awake", "Pred_Sleep"]).to_csv(OUT_RESULTS / f"{name}_confusion_matrix.csv")

        summary_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Balanced_Accuracy": balanced_accuracy_score(y_test, pred),
            "ROC_AUC": roc_auc_score(y_test, prob),
            "Awake_Precision": report["Awake"]["precision"],
            "Awake_Recall": report["Awake"]["recall"],
            "Awake_F1": report["Awake"]["f1-score"],
            "Sleep_Precision": report["Sleep"]["precision"],
            "Sleep_Recall": report["Sleep"]["recall"],
            "Sleep_F1": report["Sleep"]["f1-score"],
        })
        joblib.dump(model, OUT_MODELS / f"{name}.joblib")

        ConfusionMatrixDisplay(cm, display_labels=["Awake", "Sleep"]).plot(values_format="d")
        plt.title(f"{name.replace('_', ' ').title()} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(OUT_FIG / f"{name}_confusion_matrix.png", dpi=200)
        plt.close()

        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, prob):.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{name.replace('_', ' ').title()} ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_FIG / f"{name}_roc_curve.png", dpi=200)
        plt.close()

        precision, recall, _ = precision_recall_curve(y_test, prob)
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{name.replace('_', ' ').title()} Precision–Recall Curve")
        plt.tight_layout()
        plt.savefig(OUT_FIG / f"{name}_precision_recall_curve.png", dpi=200)
        plt.close()

    pd.DataFrame(summary_rows).to_csv(OUT_RESULTS / "model_summary.csv", index=False)


def main() -> None:
    full = build_processed_dataset()
    batched = add_batch_ids(full)
    batched.to_csv(OUT_DATA / "processed_with_batches.csv", index=False)
    sampled = sample_from_batches(batched)

    distributions = pd.concat([
        distribution_table("Full", full),
        distribution_table("Batch-randomized sample", sampled),
    ], ignore_index=True)
    distributions.to_csv(OUT_RESULTS / "distribution_validation.csv", index=False)

    batch_summary = (
        batched.groupby(["Batch_ID", "Sleep_Label"], as_index=False)
               .agg(Start_Window=("Window", "min"), End_Window=("Window", "max"), Row_Count=("Window", "size"))
    )
    batch_summary.to_csv(OUT_RESULTS / "batch_summary.csv", index=False)

    plt.figure()
    plot_df = distributions.pivot(index="Dataset", columns="Meaning", values="Percent")
    plot_df.plot(kind="bar")
    plt.ylabel("Percent of records")
    plt.title("Awake vs Sleep Distribution")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "distribution_comparison.png", dpi=200)
    plt.close("all")

    train_and_evaluate(sampled)

    metadata = {
        "random_state": RANDOM_STATE,
        "sample_fraction": SAMPLE_FRACTION,
        "target_sleep_fraction": SLEEP_FRACTION,
        "test_size": TEST_SIZE,
        "full_rows": int(len(full)),
        "sampled_rows": int(len(sampled)),
        "note": "Labels are awake (0) vs sleep (1), not verified OSA/apnea-event labels.",
    }
    (OUT_RESULTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nPipeline completed.")
    print(distributions.to_string(index=False))
    print(f"\nOutputs: {OUT_DATA}, {OUT_FIG}, {OUT_MODELS}, {OUT_RESULTS}")
    print("\nIMPORTANT: Sleep_Label is awake vs sleep, not apnea vs non-OSA.")


if __name__ == "__main__":
    main()
