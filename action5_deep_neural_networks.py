from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GroupShuffleSplit

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path.cwd()

DATA_FILE = (
    ROOT
    / "dataset_apnea_ecg"
    / "structured"
    / "minute_features.csv"
)

OUTPUT = ROOT / "mentor_action_items"
OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# COLUMN HELPERS
# ============================================================

def find_column(df, candidates):
    lookup = {
        str(col).lower(): col
        for col in df.columns
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
            "target",
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
            "record_id",
        ]
    )


# ============================================================
# LABEL CONVERSION
# ============================================================

def normalize_binary_target(series):
    """
    Output:
        0 = Normal / non-apnea
        1 = Apnea
    """

    values = (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )

    mapping = {
        "N": 0,
        "NORMAL": 0,
        "0": 0,
        "0.0": 0,
        "NEGATIVE": 0,
        "NON-APNEA": 0,

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

print("\n📂 Loading apnea ECG dataset...")

df = pd.read_csv(DATA_FILE)

target = detect_target(df)
subject = detect_subject(df)

if target is None:
    raise ValueError(
        f"❌ Could not find target column.\n"
        f"Columns: {df.columns.tolist()}"
    )

if subject is None:
    raise ValueError(
        f"❌ Could not find subject column.\n"
        f"Columns: {df.columns.tolist()}"
    )

print(f"🎯 Target column: {target}")
print(f"👤 Subject column: {subject}")


# ============================================================
# FIND SAFE NUMERIC FEATURES
# Prevent target leakage + remove constant features
# ============================================================

EXCLUDE_EXACT = {
    target,
    subject,
    "apnea",
    "apnea_label",
    "osa_label",
    "target",
    "label",
    "class",
    "annotation",
    "severity",
    "ahi"
}

EXCLUDE_KEYWORDS = [
    "prediction",
    "predicted",
    "ground_truth",
    "true_label",
    "event_label"
]

features = []

print("\n🔍 Screening features for leakage and useless constants...")

for column in df.columns:

    # ----------------------------------------
    # Exclude known target/identifier columns
    # ----------------------------------------

    if column in [target, subject]:
        continue

    lower_name = str(column).lower().strip()

    if lower_name in {
        str(x).lower()
        for x in EXCLUDE_EXACT
    }:
        print(
            f"🚫 Excluding leakage-risk column: {column}"
        )
        continue

    if any(
        keyword in lower_name
        for keyword in EXCLUDE_KEYWORDS
    ):
        print(
            f"🚫 Excluding suspicious column: {column}"
        )
        continue


    # ----------------------------------------
    # Convert to numeric
    # ----------------------------------------

    numeric = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    # Require at least 80% valid numeric data
    if numeric.notna().mean() < 0.80:
        continue


    # ----------------------------------------
    # Remove constant / near-useless features
    # ----------------------------------------

    unique_count = numeric.nunique(
        dropna=True
    )

    if unique_count <= 1:
        print(
            f"🗑️ Removing constant feature: {column}"
        )
        continue


    # ----------------------------------------
    # Check direct correlation with target
    # ----------------------------------------

    temp_target = normalize_binary_target(
        df[target]
    )

    valid = (
        numeric.notna()
        &
        temp_target.notna()
    )

    if valid.sum() >= 10:

        corr = np.corrcoef(
            numeric[valid],
            temp_target[valid]
        )[0, 1]

        if np.isfinite(corr) and abs(corr) >= 0.95:

            print(
                f"🚨 Removing likely leakage feature: "
                f"{column} "
                f"(corr={corr:.4f})"
            )

            continue


    # Feature passed all checks
    features.append(column)


print(
    f"\n✅ Safe numeric features detected: "
    f"{len(features)}"
)

print("\n🧪 Features being used:")

for feature in features:
    print(f" • {feature}")


# Final safety check
if "apnea" in [
    str(f).lower()
    for f in features
]:

    raise RuntimeError(
        "❌ Leakage detected: 'apnea' is still "
        "present in the feature matrix."
    )


# ============================================================
# CLEAN DATASET
# ============================================================

data = df[
    [subject] + features + [target]
].copy()

for feature in features:

    data[feature] = pd.to_numeric(
        data[feature],
        errors="coerce"
    )


print("\n🔎 Original target values:")
print(
    data[target]
    .value_counts(dropna=False)
    .head(10)
)


data[target] = normalize_binary_target(
    data[target]
)


data = data.dropna(
    subset=[subject, target]
).copy()


data[target] = data[target].astype(int)


if len(data) == 0:
    raise ValueError(
        "❌ No usable records remained after label conversion."
    )


if data[target].nunique() != 2:
    raise ValueError(
        "❌ Both normal and apnea classes are required."
    )


print("\n📊 Final class distribution:")
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


train = data.iloc[train_idx].copy()
test = data.iloc[test_idx].copy()


train_subjects = set(train[subject])
test_subjects = set(test[subject])

overlap = train_subjects & test_subjects


print("\n👤 Training subjects:", len(train_subjects))
print("👤 Testing subjects:", len(test_subjects))
print("🔒 Subject overlap:", len(overlap))


if overlap:
    raise RuntimeError(
        "❌ Subject leakage detected."
    )


X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {

    # --------------------------------------------------------
    # NEURAL NETWORK 1
    # One hidden layer
    # --------------------------------------------------------

    "NN Shallow ReLU":

        Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64,),
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    learning_rate_init=0.001,
                    max_iter=600,
                    random_state=RANDOM_STATE
                )
            ),
        ]),


    # --------------------------------------------------------
    # NEURAL NETWORK 2
    # Three hidden layers
    # --------------------------------------------------------

    "NN Deep ReLU":

        Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    learning_rate_init=0.001,
                    max_iter=800,
                    random_state=RANDOM_STATE
                )
            ),
        ]),


    # --------------------------------------------------------
    # NEURAL NETWORK 3
    # Same depth but TANH activation
    # --------------------------------------------------------

    "NN Deep Tanh":

        Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    activation="tanh",
                    solver="adam",
                    alpha=0.0001,
                    learning_rate_init=0.001,
                    max_iter=800,
                    random_state=RANDOM_STATE
                )
            ),
        ]),


    # --------------------------------------------------------
    # NEURAL NETWORK 4
    # Even deeper ReLU network
    # --------------------------------------------------------

    "NN Very Deep ReLU":

        Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(
                        256,
                        128,
                        64,
                        32
                    ),
                    activation="relu",
                    solver="adam",
                    alpha=0.0005,
                    learning_rate_init=0.0005,
                    max_iter=1000,
                    random_state=RANDOM_STATE
                )
            ),
        ]),


    # --------------------------------------------------------
    # RANDOM FOREST BASELINE
    # --------------------------------------------------------

    "Random Forest":

        Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            ),
        ]),


    # --------------------------------------------------------
    # GRADIENT BOOSTING BASELINE
    # --------------------------------------------------------

    "Gradient Boosting":

        Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    random_state=RANDOM_STATE
                )
            ),
        ]),
}


# ============================================================
# TRAINING + EVALUATION
# ============================================================

results = []
roc_data = {}


for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(f"🤖 Training {model_name}")
    print("=" * 70)


    model.fit(
        X_train,
        y_train
    )


    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )


    predictions = (
        probabilities >= 0.50
    ).astype(int)


    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()


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


    fpr_value = (
        fp / (fp + tn)
        if fp + tn
        else np.nan
    )


    auc = roc_auc_score(
        y_test,
        probabilities
    )


    row = {

        "Model":
            model_name,

        "Accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "Balanced_Accuracy":
            balanced_accuracy_score(
                y_test,
                predictions
            ),

        "Precision":
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "Sensitivity_Recall":
            sensitivity,

        "Specificity":
            specificity,

        "False_Positive_Rate":
            fpr_value,

        "F1":
            f1_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "ROC_AUC":
            auc,

        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


    results.append(row)


    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )


    roc_data[model_name] = (
        fpr,
        tpr,
        auc
    )


    # ----------------------------------------
    # Save classification report
    # ----------------------------------------

    report = pd.DataFrame(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "Normal",
                "Apnea"
            ],
            output_dict=True,
            zero_division=0
        )
    ).transpose()


    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )


    report.to_csv(
        OUTPUT /
        f"05_{safe_name}_classification_report.csv"
    )


    print(
        f"📈 ROC-AUC: {auc:.4f}"
    )

    print(
        f"🫁 Sensitivity: {sensitivity:.4f}"
    )

    print(
        f"✅ Specificity: {specificity:.4f}"
    )

    print(
        f"🚨 FPR: {fpr_value:.4f}"
    )


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    by="Balanced_Accuracy",
    ascending=False
)


results_file = (
    OUTPUT /
    "05_neural_network_model_comparison.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


print("\n" + "=" * 110)
print("📊 MODEL COMPARISON")
print("=" * 110)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# FIGURE 1 — ROC CURVES
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 8)
)


for model_name, (
    fpr,
    tpr,
    auc
) in roc_data.items():

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=(
            f"{model_name} "
            f"(AUC={auc:.3f})"
        )
    )


ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Chance"
)


ax.set_title(
    "Neural Networks vs Tree-Based Models — ROC"
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
    alpha=0.25
)

ax.legend(
    fontsize=8
)


fig.tight_layout()


roc_file = (
    OUTPUT /
    "05_neural_network_roc_comparison.png"
)


fig.savefig(
    roc_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# FIGURE 2 — BALANCED ACCURACY
# ============================================================

plot_df = results_df.sort_values(
    by="Balanced_Accuracy"
)


fig, ax = plt.subplots(
    figsize=(11, 7)
)


bars = ax.barh(
    plot_df["Model"],
    plot_df["Balanced_Accuracy"] * 100
)


ax.set_title(
    "Balanced Accuracy by Model"
)

ax.set_xlabel(
    "Balanced Accuracy (%)"
)

ax.set_xlim(
    0,
    100
)

ax.grid(
    axis="x",
    alpha=0.25
)


for bar in bars:

    width = bar.get_width()

    ax.text(
        width + 0.5,
        bar.get_y()
        + bar.get_height() / 2,
        f"{width:.1f}%",
        va="center"
    )


fig.tight_layout()


balanced_file = (
    OUTPUT /
    "05_balanced_accuracy_comparison.png"
)


fig.savefig(
    balanced_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# FIGURE 3 — SENSITIVITY / SPECIFICITY
# ============================================================

x = np.arange(
    len(results_df)
)

width = 0.35


fig, ax = plt.subplots(
    figsize=(12, 7)
)


bars1 = ax.bar(
    x - width / 2,
    results_df[
        "Sensitivity_Recall"
    ] * 100,
    width,
    label="Sensitivity"
)


bars2 = ax.bar(
    x + width / 2,
    results_df[
        "Specificity"
    ] * 100,
    width,
    label="Specificity"
)


ax.set_title(
    "Sensitivity vs Specificity by Model"
)

ax.set_ylabel(
    "Performance (%)"
)

ax.set_xticks(x)

ax.set_xticklabels(
    results_df["Model"],
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


sens_spec_file = (
    OUTPUT /
    "05_sensitivity_specificity_comparison.png"
)


fig.savefig(
    sens_spec_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# FIGURE 4 — F1
# ============================================================

plot_df = results_df.sort_values(
    by="F1"
)


fig, ax = plt.subplots(
    figsize=(11, 7)
)


bars = ax.barh(
    plot_df["Model"],
    plot_df["F1"] * 100
)


ax.set_title(
    "F1 Score by Model"
)

ax.set_xlabel(
    "F1 Score (%)"
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


f1_file = (
    OUTPUT /
    "05_f1_comparison.png"
)


fig.savefig(
    f1_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# METADATA
# ============================================================

metadata = {

    "random_state":
        RANDOM_STATE,

    "test_size":
        TEST_SIZE,

    "features_used":
        len(features),

    "training_rows":
        len(train),

    "testing_rows":
        len(test),

    "training_subjects":
        len(train_subjects),

    "testing_subjects":
        len(test_subjects),

    "subject_overlap":
        len(overlap),

    "neural_network_architectures": {

        "NN Shallow ReLU":
            [64],

        "NN Deep ReLU":
            [128, 64, 32],

        "NN Deep Tanh":
            [128, 64, 32],

        "NN Very Deep ReLU":
            [256, 128, 64, 32]
    }
}


metadata_file = (
    OUTPUT /
    "05_neural_network_metadata.json"
)


metadata_file.write_text(
    json.dumps(
        metadata,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("🔬 VALIDATION")
print("=" * 70)


print(
    "Subject overlap == 0:",
    len(overlap) == 0
)


print(
    "ROC-AUC range valid:",
    results_df[
        "ROC_AUC"
    ].between(
        0,
        1
    ).all()
)


print(
    "Balanced accuracy range valid:",
    results_df[
        "Balanced_Accuracy"
    ].between(
        0,
        1
    ).all()
)


print(
    "Specificity + FPR valid:",
    np.allclose(
        results_df[
            "Specificity"
        ]
        +
        results_df[
            "False_Positive_Rate"
        ],
        1,
        atol=1e-6
    )
)


# ============================================================
# FILE VALIDATION
# ============================================================

expected_files = [

    results_file,
    roc_file,
    balanced_file,
    sens_spec_file,
    f1_file,
    metadata_file
]


for file in expected_files:

    if not file.exists():

        raise RuntimeError(
            f"❌ Missing output file: {file}"
        )


print("\n🎉 SUCCESS!")
print(
    "Deep neural-network experiment completed."
)

print("\n💾 Results saved in:")
print(
    OUTPUT.resolve()
)
