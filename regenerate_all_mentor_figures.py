from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
OUT = ROOT / "mentor_action_items"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA SOURCES
# ============================================================

SOURCES = {
    "Apnea ECG": ROOT / "dataset_apnea_ecg" / "results",
    "Apnea HRV": ROOT / "dataset_apnea_hrv" / "results",
    "UCDDB v2": ROOT / "dataset_ucddb_v2" / "results",
}


# ============================================================
# HELPERS
# ============================================================

def read_csv_if_exists(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def find_col(df, candidates):
    if df is None:
        return None

    lookup = {
        str(c).lower().replace(" ", "_"): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.lower().replace(" ", "_")

        if key in lookup:
            return lookup[key]

    return None


def numeric(df, col):
    return pd.to_numeric(
        df[col],
        errors="coerce"
    )


def normalize_metric(series):
    """
    Convert a metric into 0-1 form.

    If values appear to be percentages (e.g. 91.4),
    divide by 100.
    """

    s = pd.to_numeric(
        series,
        errors="coerce"
    )

    if s.dropna().empty:
        return s

    if s.max() > 1.5:
        s = s / 100.0

    return s


def load_dataset_metrics(dataset_name, result_dir):
    """
    Tries several existing result files and returns
    standardized model/metric records.
    """

    candidates = [
        result_dir / "holdout_metrics.csv",
        result_dir / "metrics.csv",
    ]

    df = None

    for candidate in candidates:
        df = read_csv_if_exists(candidate)

        if df is not None and len(df):
            break

    if df is None:
        return pd.DataFrame()

    model_col = find_col(
        df,
        [
            "model",
            "Model",
            "classifier",
            "algorithm"
        ]
    )

    if model_col is None:
        df = df.copy()
        df["Model"] = [
            f"Model {i+1}"
            for i in range(len(df))
        ]
        model_col = "Model"

    mappings = {
        "Accuracy": [
            "accuracy",
            "acc"
        ],

        "Balanced Accuracy": [
            "balanced_accuracy",
            "balanced_acc"
        ],

        "ROC-AUC": [
            "roc_auc",
            "auc",
            "roc-auc"
        ],

        "Sensitivity": [
            "sensitivity",
            "recall",
            "tpr"
        ],

        "Specificity": [
            "specificity",
            "tnr"
        ],

        "F1": [
            "f1",
            "f1_score",
            "f1-score"
        ],

        "Precision": [
            "precision"
        ],
    }

    rows = []

    for _, row in df.iterrows():
        record = {
            "Dataset": dataset_name,
            "Model": str(row[model_col])
        }

        for clean_name, aliases in mappings.items():
            col = find_col(
                df,
                aliases
            )

            if col is not None:
                value = pd.to_numeric(
                    pd.Series([row[col]]),
                    errors="coerce"
                ).iloc[0]

                if pd.notna(value):
                    if value > 1.5:
                        value = value / 100.0

                    record[clean_name] = value

        rows.append(record)

    return pd.DataFrame(rows)


# ============================================================
# BUILD MASTER METRIC TABLE
# ============================================================

metric_frames = []

for dataset_name, result_dir in SOURCES.items():
    frame = load_dataset_metrics(
        dataset_name,
        result_dir
    )

    if not frame.empty:
        metric_frames.append(frame)


if metric_frames:
    master = pd.concat(
        metric_frames,
        ignore_index=True
    )
else:
    master = pd.DataFrame()


# ============================================================
# ALSO LOAD ACTION 3 RESULTS
# ============================================================

action3_file = (
    OUT /
    "03_model_comparison.csv"
)

if action3_file.exists():
    a3 = pd.read_csv(action3_file)

    renamed = {}

    possible = {
        "Accuracy": "Accuracy",
        "Balanced_Accuracy": "Balanced Accuracy",
        "ROC_AUC": "ROC-AUC",
        "Sensitivity_Recall": "Sensitivity",
        "Specificity": "Specificity",
        "F1": "F1",
        "Precision": "Precision",
    }

    for old, new in possible.items():
        if old in a3.columns:
            renamed[old] = new

    a3 = a3.rename(
        columns=renamed
    )

    if "Experiment" in a3.columns:
        a3 = a3.rename(
            columns={
                "Experiment": "Model"
            }
        )

    if "Model" in a3.columns:
        a3["Dataset"] = "Action 3 ECG"

        metric_frames.append(
            a3[
                [
                    c
                    for c in [
                        "Dataset",
                        "Model",
                        "Accuracy",
                        "Balanced Accuracy",
                        "ROC-AUC",
                        "Sensitivity",
                        "Specificity",
                        "F1",
                        "Precision",
                    ]
                    if c in a3.columns
                ]
            ]
        )


# ============================================================
# LOAD ACTION 5 RESULTS
# ============================================================

action5_file = (
    OUT /
    "05_neural_network_model_comparison.csv"
)

if action5_file.exists():
    a5 = pd.read_csv(
        action5_file
    )

    a5 = a5.rename(
        columns={
            "Balanced_Accuracy":
                "Balanced Accuracy",

            "ROC_AUC":
                "ROC-AUC",

            "Sensitivity_Recall":
                "Sensitivity",

            "False_Positive_Rate":
                "FPR",
        }
    )

    a5["Dataset"] = "Action 5 ECG"

    metric_frames.append(
        a5[
            [
                c
                for c in [
                    "Dataset",
                    "Model",
                    "Accuracy",
                    "Balanced Accuracy",
                    "ROC-AUC",
                    "Sensitivity",
                    "Specificity",
                    "F1",
                    "Precision",
                    "FPR",
                ]
                if c in a5.columns
            ]
        ]
    )


if metric_frames:
    master = pd.concat(
        metric_frames,
        ignore_index=True,
        sort=False
    )


master.to_csv(
    OUT /
    "all_model_metrics_consolidated.csv",
    index=False
)

print("\n📊 Consolidated metrics:")
print(master.head(20).to_string(index=False))


# ============================================================
# FIGURE 1
# ACTION ITEM 1 REPLACEMENT:
# SENSITIVITY / SPECIFICITY / ACCURACY
# ============================================================

action1_file = (
    ROOT /
    "dataset_apnea_ecg" /
    "results" /
    "sensitivity_specificity_table.csv"
)

a1 = read_csv_if_exists(
    action1_file
)

if a1 is not None:

    apnea = a1[
        a1["Class"]
        .astype(str)
        .str.lower()
        .str.contains("apnea")
    ].copy()

    for col in [
        "Sensitivity",
        "Specificity"
    ]:
        apnea[col] = normalize_metric(
            apnea[col]
        )

    # Add approximate balanced accuracy from sens/spec
    apnea[
        "Balanced Accuracy"
    ] = (
        apnea["Sensitivity"]
        +
        apnea["Specificity"]
    ) / 2

    x = np.arange(
        len(apnea)
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.bar(
        x - width,
        apnea["Sensitivity"] * 100,
        width,
        label="Sensitivity"
    )

    ax.bar(
        x,
        apnea["Specificity"] * 100,
        width,
        label="Specificity"
    )

    ax.bar(
        x + width,
        apnea["Balanced Accuracy"] * 100,
        width,
        label="Balanced accuracy"
    )

    ax.set_title(
        "Apnea Detection Performance by Model"
    )

    ax.set_ylabel(
        "Performance (%)"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        apnea["Model"],
        rotation=20,
        ha="right"
    )

    ax.set_ylim(
        0,
        105
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUT /
        "01_sensitivity_specificity.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# FIGURE 2
# ACTION ITEM 1:
# FP / FN ERROR PROFILE
# ============================================================

if a1 is not None:

    apnea = a1[
        a1["Class"]
        .astype(str)
        .str.lower()
        .str.contains("apnea")
    ].copy()

    apnea["False_Positive_Rate"] = normalize_metric(
        apnea["False_Positive_Rate"]
    )

    apnea["False_Negative_Rate"] = normalize_metric(
        apnea["False_Negative_Rate"]
    )

    x = np.arange(
        len(apnea)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.bar(
        x - width / 2,
        apnea["False_Positive_Rate"] * 100,
        width,
        label="False positive rate"
    )

    ax.bar(
        x + width / 2,
        apnea["False_Negative_Rate"] * 100,
        width,
        label="False negative rate"
    )

    ax.set_title(
        "Error Profile by Model"
    )

    ax.set_ylabel(
        "Error rate (%)"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        apnea["Model"],
        rotation=20,
        ha="right"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUT /
        "01_false_positive_rates.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# FIGURE 3
# MULTI-DATASET ACCURACY COMPARISON
# ============================================================

if (
    not master.empty
    and "Accuracy" in master.columns
):

    plot_df = master.dropna(
        subset=["Accuracy"]
    ).copy()

    plot_df["Label"] = (
        plot_df["Dataset"]
        +
        " | "
        +
        plot_df["Model"]
    )

    # limit for readability
    plot_df = (
        plot_df
        .sort_values(
            "Accuracy"
        )
        .tail(16)
    )

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    ax.barh(
        plot_df["Label"],
        plot_df["Accuracy"] * 100
    )

    ax.set_title(
        "Model Accuracy Across OSA Datasets"
    )

    ax.set_xlabel(
        "Accuracy (%)"
    )

    ax.set_xlim(
        0,
        100
    )

    ax.grid(
        axis="x",
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        OUT /
        "03_accuracy_across_datasets.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# FIGURE 4
# MULTI-METRIC / MULTI-DATASET MODEL VIEW
# ============================================================

metric_cols = [
    c
    for c in [
        "Accuracy",
        "Balanced Accuracy",
        "ROC-AUC",
        "F1",
    ]
    if c in master.columns
]

if (
    not master.empty
    and len(metric_cols) >= 2
):

    subset = (
        master
        .dropna(
            subset=metric_cols,
            how="all"
        )
        .copy()
    )

    # summarize by dataset
    grouped = (
        subset
        .groupby(
            "Dataset"
        )[metric_cols]
        .mean()
        .reset_index()
    )

    x = np.arange(
        len(grouped)
    )

    width = 0.18

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    offsets = np.linspace(
        -width * 1.5,
        width * 1.5,
        len(metric_cols)
    )

    for offset, metric in zip(
        offsets,
        metric_cols
    ):

        ax.bar(
            x + offset,
            grouped[metric] * 100,
            width,
            label=metric
        )

    ax.set_title(
        "Average Model Performance Across OSA Datasets"
    )

    ax.set_ylabel(
        "Performance (%)"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        grouped["Dataset"],
        rotation=15
    )

    ax.set_ylim(
        0,
        105
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUT /
        "03_multidataset_metric_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# FIGURE 5
# ACTION ITEM 5:
# REPLACE REPETITIVE NN PERFORMANCE GRAPH
# ============================================================

if action5_file.exists():

    nn = pd.read_csv(
        action5_file
    )

    nn = nn.rename(
        columns={
            "Balanced_Accuracy":
                "Balanced Accuracy",

            "ROC_AUC":
                "ROC-AUC",

            "Sensitivity_Recall":
                "Sensitivity",

            "False_Positive_Rate":
                "FPR",
        }
    )

    cols = [
        c
        for c in [
            "Accuracy",
            "Balanced Accuracy",
            "ROC-AUC",
            "F1"
        ]
        if c in nn.columns
    ]

    x = np.arange(
        len(nn)
    )

    width = (
        0.75 /
        max(
            1,
            len(cols)
        )
    )

    fig, ax = plt.subplots(
        figsize=(14, 8)
    )

    for i, metric in enumerate(cols):

        offset = (
            i
            -
            (len(cols) - 1) / 2
        ) * width

        ax.bar(
            x + offset,
            nn[metric] * 100,
            width,
            label=metric
        )

    ax.set_title(
        "Neural Network and Tree Model Performance"
    )

    ax.set_ylabel(
        "Performance (%)"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        nn["Model"],
        rotation=25,
        ha="right"
    )

    ax.set_ylim(
        0,
        105
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUT /
        "05_balanced_accuracy_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# FIGURE 6
# ACTION ITEM 5:
# SENSITIVITY / SPECIFICITY / F1
# ============================================================

if action5_file.exists():

    nn = pd.read_csv(
        action5_file
    )

    nn = nn.rename(
        columns={
            "Sensitivity_Recall":
                "Sensitivity",

            "False_Positive_Rate":
                "FPR",
        }
    )

    cols = [
        c
        for c in [
            "Sensitivity",
            "Specificity",
            "F1"
        ]
        if c in nn.columns
    ]

    x = np.arange(
        len(nn)
    )

    width = (
        0.75 /
        max(
            1,
            len(cols)
        )
    )

    fig, ax = plt.subplots(
        figsize=(14, 8)
    )

    for i, metric in enumerate(cols):

        offset = (
            i
            -
            (len(cols) - 1) / 2
        ) * width

        ax.bar(
            x + offset,
            nn[metric] * 100,
            width,
            label=metric
        )

    ax.set_title(
        "Apnea Classification Quality by Model"
    )

    ax.set_ylabel(
        "Performance (%)"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        nn["Model"],
        rotation=25,
        ha="right"
    )

    ax.set_ylim(
        0,
        105
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUT /
        "05_sensitivity_specificity_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# FIGURE 7
# ACTION ITEM 5:
# ERROR + ACCURACY PROFILE
# ============================================================

if action5_file.exists():

    nn = pd.read_csv(
        action5_file
    )

    if (
        "False_Positive_Rate"
        in nn.columns
    ):

        nn[
            "False Negative Rate"
        ] = (
            1
            -
            nn[
                "Sensitivity_Recall"
            ]
        )

        x = np.arange(
            len(nn)
        )

        width = 0.25

        fig, ax = plt.subplots(
            figsize=(14, 8)
        )

        ax.bar(
            x - width,
            nn["Accuracy"] * 100,
            width,
            label="Accuracy"
        )

        ax.bar(
            x,
            nn["False_Positive_Rate"] * 100,
            width,
            label="False positive rate"
        )

        ax.bar(
            x + width,
            nn["False Negative Rate"] * 100,
            width,
            label="False negative rate"
        )

        ax.set_title(
            "Accuracy and Error Trade-offs by Model"
        )

        ax.set_ylabel(
            "Rate (%)"
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            nn["Model"],
            rotation=25,
            ha="right"
        )

        ax.grid(
            axis="y",
            alpha=0.25
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            OUT /
            "05_f1_comparison.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)


# ============================================================
# VALIDATE ALL PNG OUTPUTS
# ============================================================

expected = [
    "01_sensitivity_specificity.png",
    "01_false_positive_rates.png",
    "03_accuracy_across_datasets.png",
    "03_multidataset_metric_comparison.png",
    "05_balanced_accuracy_comparison.png",
    "05_sensitivity_specificity_comparison.png",
    "05_f1_comparison.png",
]

print("\n🔬 Validating regenerated figures...")

for filename in expected:

    path = OUT / filename

    if path.exists():

        print(
            f"✅ {filename}: "
            f"{path.stat().st_size:,} bytes"
        )

    else:

        print(
            f"⚠️ Not generated: {filename}"
        )


print("\n🎉 Figure regeneration finished.")
print("No metric values were altered.")
print("Plots were rebuilt from existing CSV results.")
