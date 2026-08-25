from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

from sklearn.model_selection import GroupShuffleSplit

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path.cwd()

DATA_FILE = (
    ROOT /
    "dataset_apnea_ecg" /
    "structured" /
    "minute_features.csv"
)

OUTPUT = ROOT / "mentor_action_items"

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(df, candidates):

    lookup = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def detect_target(df):

    return find_column(
        df,
        [
            "label",
            "y",
            "apnea_label",
            "osa_label",
            "target"
        ]
    )


def detect_subject(df):

    return find_column(
        df,
        [
            "record",
            "subject",
            "subject_id",
            "patient",
            "patient_id",
            "record_id"
        ]
    )


# ============================================================
# TARGET CONVERSION
# ============================================================

def normalize_binary_target(series):

    values = (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )

    mapping = {

        # Normal
        "N": 0,
        "NORMAL": 0,
        "0": 0,
        "0.0": 0,
        "NEGATIVE": 0,

        # Apnea
        "A": 1,
        "APNEA": 1,
        "OSA": 1,
        "1": 1,
        "1.0": 1,
        "POSITIVE": 1,
    }

    return values.map(mapping)


# ============================================================
# LOAD DATA
# ============================================================

print("\n📂 Loading apnea ECG data...")

df = pd.read_csv(DATA_FILE)

target = detect_target(df)
subject = detect_subject(df)

if target is None:
    raise ValueError(
        "❌ Could not identify target column."
    )

if subject is None:
    raise ValueError(
        "❌ Could not identify subject/record column."
    )


print("🎯 Target:", target)
print("👤 Subject:", subject)


# ============================================================
# FIND NUMERIC FEATURES
# ============================================================

features = []

for column in df.columns:

    if column in [target, subject]:
        continue

    numeric = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    if numeric.notna().mean() >= 0.80:
        features.append(column)


print(
    "🧪 Numeric features:",
    len(features)
)


# ============================================================
# CLEAN DATA
# ============================================================

data = df[
    [subject] +
    features +
    [target]
].copy()


for column in features:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


data[target] = normalize_binary_target(
    data[target]
)


data = data.dropna(
    subset=[
        subject,
        target
    ]
).copy()


data[target] = data[target].astype(int)


print("\n📊 Class distribution:")

print(
    data[target]
    .value_counts()
    .sort_index()
)


# ============================================================
# SUBJECT-INDEPENDENT SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


train_idx, test_idx = next(

    splitter.split(
        data,
        y=data[target],
        groups=data[subject]
    )

)


train = data.iloc[
    train_idx
].copy()

test = data.iloc[
    test_idx
].copy()


overlap = (
    set(train[subject])
    &
    set(test[subject])
)


print(
    "\n🔒 Subject overlap:",
    len(overlap)
)


if overlap:

    raise RuntimeError(
        "❌ Subject leakage detected."
    )


X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":
        Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]),

    "Random Forest":
        Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ]),

    "Gradient Boosting":
        Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    random_state=RANDOM_STATE
                )
            )
        ])
}


# ============================================================
# TRAIN + SAVE PROBABILITIES
# ============================================================

probabilities = {}

roc_rows = []

threshold_rows = []


for model_name, model in models.items():

    print(
        f"\n🤖 Training {model_name}..."
    )

    model.fit(
        X_train,
        y_train
    )


    probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )


    probabilities[
        model_name
    ] = probability


    # ----------------------------------------
    # ROC
    # ----------------------------------------

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probability
    )

    auc = roc_auc_score(
        y_test,
        probability
    )


    youden = (
        tpr - fpr
    )

    best_index = np.argmax(
        youden
    )

    best_threshold = thresholds[
        best_index
    ]


    print(
        f"📈 AUC = {auc:.4f}"
    )

    print(
        f"🎯 Optimal threshold = "
        f"{best_threshold:.4f}"
    )


    for fp_rate, tp_rate, threshold in zip(
        fpr,
        tpr,
        thresholds
    ):

        roc_rows.append({

            "Model":
                model_name,

            "FPR":
                fp_rate,

            "TPR_Sensitivity":
                tp_rate,

            "Threshold":
                threshold,

            "ROC_AUC":
                auc
        })


    # ----------------------------------------
    # THRESHOLD GRID
    # ----------------------------------------

    for threshold in np.arange(
        0.01,
        1.00,
        0.01
    ):

        predictions = (
            probability >= threshold
        ).astype(int)


        tn, fp, fn, tp = (
            confusion_matrix(
                y_test,
                predictions,
                labels=[0, 1]
            ).ravel()
        )


        sensitivity = (
            tp / (tp + fn)
            if tp + fn
            else np.nan
        )


        specificity = (
            tn / (tn + fp)
            if tn + fp
            else np.nan
        )


        false_positive_rate = (
            fp / (fp + tn)
            if fp + tn
            else np.nan
        )


        false_negative_rate = (
            fn / (fn + tp)
            if fn + tp
            else np.nan
        )


        threshold_rows.append({

            "Model":
                model_name,

            "Threshold":
                threshold,

            "Sensitivity":
                sensitivity,

            "Specificity":
                specificity,

            "False_Positive_Rate":
                false_positive_rate,

            "False_Negative_Rate":
                false_negative_rate,

            "Precision":
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),

            "F1":
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),

            "Balanced_Accuracy":
                balanced_accuracy_score(
                    y_test,
                    predictions
                ),

            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })


# ============================================================
# SAVE DETAILED DATA
# ============================================================

roc_df = pd.DataFrame(
    roc_rows
)

threshold_df = pd.DataFrame(
    threshold_rows
)


roc_df.to_csv(
    OUTPUT /
    "03_detailed_roc_points.csv",
    index=False
)


threshold_df.to_csv(
    OUTPUT /
    "03_detailed_threshold_analysis.csv",
    index=False
)


print(
    "\n💾 Detailed ROC and threshold "
    "tables saved."
)


# ============================================================
# FIGURE 1
# DETAILED ROC COMPARISON
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 8)
)


for model_name in models:

    subset = roc_df[
        roc_df["Model"]
        == model_name
    ]

    auc = (
        subset[
            "ROC_AUC"
        ].iloc[0]
    )


    ax.plot(
        subset["FPR"],
        subset["TPR_Sensitivity"],
        linewidth=2,
        label=(
            f"{model_name} "
            f"(AUC={auc:.3f})"
        )
    )


    youden = (
        subset[
            "TPR_Sensitivity"
        ]
        -
        subset[
            "FPR"
        ]
    )

    best_index = youden.idxmax()

    best = subset.loc[
        best_index
    ]


    ax.scatter(
        best["FPR"],
        best["TPR_Sensitivity"],
        s=80
    )


    ax.annotate(
        (
            f"{model_name}\n"
            f"Threshold="
            f"{best['Threshold']:.2f}\n"
            f"Sens="
            f"{best['TPR_Sensitivity']:.2f}\n"
            f"FPR="
            f"{best['FPR']:.2f}"
        ),

        (
            best["FPR"],
            best[
                "TPR_Sensitivity"
            ]
        ),

        xytext=(8, -5),

        textcoords="offset points",

        fontsize=8
    )


ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Chance"
)


ax.set_title(
    "Detailed ROC Comparison — Apnea Detection"
)

ax.set_xlabel(
    "False Positive Rate"
)

ax.set_ylabel(
    "Sensitivity / True Positive Rate"
)

ax.set_xlim(
    0,
    1
)

ax.set_ylim(
    0,
    1.02
)

ax.grid(
    alpha=0.3
)

ax.legend()


fig.tight_layout()


fig.savefig(
    OUTPUT /
    "03_roc_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# FIGURE 2
# THRESHOLD — ALL THREE MODELS
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 8)
)


for model_name in models:

    subset = threshold_df[
        threshold_df["Model"]
        == model_name
    ]


    ax.plot(
        subset["Threshold"],
        subset["Sensitivity"],
        linewidth=2,
        label=(
            f"{model_name} "
            "Sensitivity"
        )
    )


    ax.plot(
        subset["Threshold"],
        subset["Specificity"],
        linestyle="--",
        linewidth=2,
        label=(
            f"{model_name} "
            "Specificity"
        )
    )


ax.axvline(
    0.50,
    linestyle=":",
    linewidth=1.5,
    label="Default threshold 0.50"
)


ax.set_title(
    "Sensitivity–Specificity Trade-off "
    "Across Decision Thresholds"
)

ax.set_xlabel(
    "Decision Threshold"
)

ax.set_ylabel(
    "Rate"
)

ax.set_ylim(
    0,
    1.02
)

ax.grid(
    alpha=0.3
)

ax.legend(
    fontsize=8
)


fig.tight_layout()


fig.savefig(
    OUTPUT /
    "03_threshold_tradeoff.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# FIGURE 3
# FALSE POSITIVE RATE BY THRESHOLD
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 7)
)


for model_name in models:

    subset = threshold_df[
        threshold_df["Model"]
        == model_name
    ]


    ax.plot(
        subset["Threshold"],
        subset[
            "False_Positive_Rate"
        ],
        linewidth=2,
        label=model_name
    )


ax.axvline(
    0.50,
    linestyle=":",
    label="Default threshold 0.50"
)


ax.set_title(
    "False Positive Rate vs Decision Threshold"
)

ax.set_xlabel(
    "Decision Threshold"
)

ax.set_ylabel(
    "False Positive Rate"
)

ax.set_ylim(
    0,
    1.02
)

ax.grid(
    alpha=0.3
)

ax.legend()


fig.tight_layout()


fig.savefig(
    OUTPUT /
    "03_false_positive_thresholds.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# FIGURE 4
# BALANCED ACCURACY BY THRESHOLD
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 7)
)


for model_name in models:

    subset = threshold_df[
        threshold_df["Model"]
        == model_name
    ]


    ax.plot(
        subset["Threshold"],
        subset[
            "Balanced_Accuracy"
        ],
        linewidth=2,
        label=model_name
    )


ax.set_title(
    "Balanced Accuracy Across Decision Thresholds"
)

ax.set_xlabel(
    "Decision Threshold"
)

ax.set_ylabel(
    "Balanced Accuracy"
)

ax.set_ylim(
    0,
    1.02
)

ax.grid(
    alpha=0.3
)

ax.legend()


fig.tight_layout()


fig.savefig(
    OUTPUT /
    "03_balanced_accuracy_thresholds.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# FIGURE 5
# F1 BY THRESHOLD
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 7)
)


for model_name in models:

    subset = threshold_df[
        threshold_df["Model"]
        == model_name
    ]


    ax.plot(
        subset["Threshold"],
        subset["F1"],
        linewidth=2,
        label=model_name
    )


ax.set_title(
    "F1 Score Across Decision Thresholds"
)

ax.set_xlabel(
    "Decision Threshold"
)

ax.set_ylabel(
    "F1 Score"
)

ax.set_ylim(
    0,
    1.02
)

ax.grid(
    alpha=0.3
)

ax.legend()


fig.tight_layout()


fig.savefig(
    OUTPUT /
    "03_f1_thresholds.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# VALIDATE FILES
# ============================================================

expected = [

    "03_roc_comparison.png",

    "03_threshold_tradeoff.png",

    "03_false_positive_thresholds.png",

    "03_balanced_accuracy_thresholds.png",

    "03_f1_thresholds.png"
]


print("\n🔬 Validating PNG files...")


for filename in expected:

    path = OUTPUT / filename

    exists = path.exists()

    size = (
        path.stat().st_size
        if exists
        else 0
    )

    print(
        f"{filename}: "
        f"{size:,} bytes"
    )

    if not exists or size < 1000:

        raise RuntimeError(
            f"❌ Invalid PNG: "
            f"{filename}"
        )


print(
    "\n🎉 All detailed Action Item #3 "
    "figures generated successfully."
)
