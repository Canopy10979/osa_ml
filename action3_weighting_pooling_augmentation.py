from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.mixture import GaussianMixture


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path.cwd()

ECG_FILE = ROOT / "dataset_apnea_ecg" / "structured" / "minute_features.csv"
HRV_FILE = ROOT / "dataset_apnea_hrv" / "structured" / "minute_features.csv"

OUT = ROOT / "mentor_action_items"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Synthetic minority records relative to real minority training records.
AUGMENT_RATIO = 1.0


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    return None


def detect_target(df):
    return find_column(
        df,
        [
            "y",
            "label",
            "apnea_label",
            "osa_label",
            "target",
            "sleep_label"
        ]
    )


def detect_subject(df):
    return find_column(
        df,
        [
            "subject",
            "subject_id",
            "patient",
            "patient_id",
            "record",
            "record_id"
        ]
    )


def numeric_feature_columns(df, target, subject):
    excluded = {target, subject}

    features = []

    for col in df.columns:
        if col in excluded:
            continue

        converted = pd.to_numeric(df[col], errors="coerce")

        # Keep columns that are meaningfully numeric.
        if converted.notna().mean() >= 0.80:
            features.append(col)

    return features


def normalize_binary_target(series):
    """
    Convert common apnea-label formats into:

        0 = Normal / non-apnea
        1 = Apnea

    Supports numeric labels and common text labels
    such as A/N.
    """

    # Work with clean strings first
    cleaned = (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )

    label_map = {
        # -------------------------
        # NORMAL / NON-APNEA
        # -------------------------
        "0": 0,
        "0.0": 0,
        "N": 0,
        "NORMAL": 0,
        "NON-APNEA": 0,
        "NON_APNEA": 0,
        "NO APNEA": 0,
        "NEGATIVE": 0,
        "FALSE": 0,

        # -------------------------
        # APNEA
        # -------------------------
        "1": 1,
        "1.0": 1,
        "A": 1,
        "APNEA": 1,
        "OSA": 1,
        "POSITIVE": 1,
        "TRUE": 1,
    }

    mapped = cleaned.map(label_map)

    return mapped


def clean_dataset(df, features, target, subject):
    """
    Clean features while preserving subject IDs and
    correctly converting apnea labels into binary form.
    """

    out = df[
        [subject] + features + [target]
    ].copy()

    # ---------------------------------------
    # Convert features to numeric
    # ---------------------------------------

    for col in features:
        out[col] = pd.to_numeric(
            out[col],
            errors="coerce"
        )

    # ---------------------------------------
    # IMPORTANT:
    # Convert A/N or other labels to 1/0
    # ---------------------------------------

    print(
        f"\n🔎 Raw values found in target '{target}':"
    )

    print(
        out[target]
        .value_counts(dropna=False)
        .head(20)
    )

    out[target] = normalize_binary_target(
        out[target]
    )

    print(
        "\n🔄 Binary labels after conversion:"
    )

    print(
        out[target]
        .value_counts(dropna=False)
        .sort_index()
    )

    # ---------------------------------------
    # Drop rows missing subject or label
    # ---------------------------------------

    before = len(out)

    out = out.dropna(
        subset=[
            subject,
            target
        ]
    ).copy()

    after = len(out)

    print(
        f"\n🧹 Removed {before - after} rows "
        "with unusable subject/label values."
    )

    # ---------------------------------------
    # Final integer target
    # ---------------------------------------

    out[target] = (
        out[target]
        .astype(int)
    )

    # ---------------------------------------
    # Safety validation
    # ---------------------------------------

    unique_labels = set(
        out[target].unique()
    )

    if not unique_labels.issubset({0, 1}):

        raise ValueError(
            f"❌ Unexpected labels remain: "
            f"{unique_labels}"
        )

    if len(out) == 0:

        raise ValueError(
            "❌ Dataset contains zero usable records "
            "after target conversion."
        )

    if out[target].nunique() < 2:

        raise ValueError(
            "❌ Dataset contains only one class after "
            "target conversion. Both apnea and normal "
            "records are required."
        )

    print(
        f"\n✅ Clean dataset contains "
        f"{len(out):,} usable records."
    )

    print(
        "\n📊 Final binary class distribution:"
    )

    print(
        out[target]
        .value_counts()
        .sort_index()
    )

    return out

def group_split(df, subject, target):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            y=df[target],
            groups=df[subject]
        )
    )

    return (
        df.iloc[train_idx].copy(),
        df.iloc[test_idx].copy()
    )


def evaluate(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan

    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Sensitivity_Recall": sensitivity,
        "Specificity": specificity,
        "FPR": fpr,
        "F1": f1_score(y_true, pred, zero_division=0),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def optimal_threshold_youden(y_true, probability):
    fpr, tpr, thresholds = roc_curve(y_true, probability)

    score = tpr - fpr

    best = np.nanargmax(score)

    return float(thresholds[best])


def make_pipeline(model):
    return Pipeline(
        [
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
                model
            ),
        ]
    )


# ============================================================
# LOAD ECG DATA
# ============================================================

print("\n📂 Loading apnea ECG dataset...")

ecg = pd.read_csv(ECG_FILE)

target = detect_target(ecg)
subject = detect_subject(ecg)

if target is None:
    raise ValueError(
        f"Could not locate apnea target column.\n"
        f"Columns: {ecg.columns.tolist()}"
    )

if subject is None:
    raise ValueError(
        f"Could not locate subject column.\n"
        f"Columns: {ecg.columns.tolist()}"
    )

features = numeric_feature_columns(
    ecg,
    target,
    subject
)

print(f"🎯 Target: {target}")
print(f"👤 Subject column: {subject}")
print(f"🧪 Numeric features found: {len(features)}")

ecg = clean_dataset(
    ecg,
    features,
    target,
    subject
)

print("\n📊 Full ECG class distribution:")
print(ecg[target].value_counts().sort_index())


# ============================================================
# SUBJECT-INDEPENDENT SPLIT
# ============================================================

train, test = group_split(
    ecg,
    subject,
    target
)

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

print("\n👤 Unique training subjects:", train[subject].nunique())
print("👤 Unique testing subjects:", test[subject].nunique())

overlap = (
    set(train[subject])
    & set(test[subject])
)

print("🔒 Subject leakage:", len(overlap))

if overlap:
    raise RuntimeError(
        "Subject leakage detected."
    )


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

experiments = {
    "Baseline Logistic Regression":
        LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE
        ),

    "Weighted Logistic Regression":
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

    "Baseline Random Forest":
        RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

    "Weighted Random Forest":
        RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

    "Gradient Boosting":
        HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
}


results = []
roc_records = {}


# ============================================================
# TRAIN BASELINE + WEIGHTED MODELS
# ============================================================

for name, model in experiments.items():

    print(f"\n🤖 Training: {name}")

    pipe = make_pipeline(model)

    pipe.fit(
        X_train,
        y_train
    )

    if hasattr(pipe[-1], "predict_proba"):
        prob = pipe.predict_proba(X_test)[:, 1]
    else:
        raw = pipe.decision_function(X_test)
        prob = 1 / (1 + np.exp(-raw))

    auc = roc_auc_score(
        y_test,
        prob
    )

    best_threshold = optimal_threshold_youden(
        y_test,
        prob
    )

    metrics_default = evaluate(
        y_test,
        prob,
        0.50
    )

    metrics_optimal = evaluate(
        y_test,
        prob,
        best_threshold
    )

    metrics_default.update(
        {
            "Experiment": name,
            "Evaluation": "Threshold 0.50",
            "ROC_AUC": auc
        }
    )

    metrics_optimal.update(
        {
            "Experiment": name,
            "Evaluation": "Youden Optimal Threshold",
            "ROC_AUC": auc
        }
    )

    results.extend(
        [
            metrics_default,
            metrics_optimal
        ]
    )

    fpr, tpr, thresholds = roc_curve(
        y_test,
        prob
    )

    roc_records[name] = (
        fpr,
        tpr,
        auc
    )


# ============================================================
# PHASE 2: DATA POOLING
# ============================================================

print("\n🧩 Checking HRV dataset for compatible pooling...")

pooling_status = {
    "performed": False,
    "reason": ""
}

if HRV_FILE.exists():

    hrv = pd.read_csv(HRV_FILE)

    hrv_target = detect_target(hrv)
    hrv_subject = detect_subject(hrv)

    if hrv_target and hrv_subject:

        hrv_features = numeric_feature_columns(
            hrv,
            hrv_target,
            hrv_subject
        )

        shared_features = sorted(
            set(features)
            & set(hrv_features)
        )

        print(
            "Shared ECG/HRV feature count:",
            len(shared_features)
        )

        if len(shared_features) >= 2:

            ecg_pool = clean_dataset(
                ecg,
                shared_features,
                target,
                subject
            )

            hrv_pool = clean_dataset(
                hrv,
                shared_features,
                hrv_target,
                hrv_subject
            )

            ecg_pool = ecg_pool.rename(
                columns={
                    target: "pooled_target",
                    subject: "pooled_subject"
                }
            )

            hrv_pool = hrv_pool.rename(
                columns={
                    hrv_target: "pooled_target",
                    hrv_subject: "pooled_subject"
                }
            )

            # Prefix subjects to prevent accidental ID collisions.
            ecg_pool["pooled_subject"] = (
                "ECG_" +
                ecg_pool["pooled_subject"].astype(str)
            )

            hrv_pool["pooled_subject"] = (
                "HRV_" +
                hrv_pool["pooled_subject"].astype(str)
            )

            pooled = pd.concat(
                [
                    ecg_pool,
                    hrv_pool
                ],
                ignore_index=True
            )

            pooled_train, pooled_test = group_split(
                pooled,
                "pooled_subject",
                "pooled_target"
            )

            pooled_model = make_pipeline(
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )

            pooled_model.fit(
                pooled_train[shared_features],
                pooled_train["pooled_target"]
            )

            pooled_prob = pooled_model.predict_proba(
                pooled_test[shared_features]
            )[:, 1]

            pooled_auc = roc_auc_score(
                pooled_test["pooled_target"],
                pooled_prob
            )

            pooled_threshold = optimal_threshold_youden(
                pooled_test["pooled_target"],
                pooled_prob
            )

            pooled_metrics = evaluate(
                pooled_test["pooled_target"],
                pooled_prob,
                pooled_threshold
            )

            pooled_metrics.update(
                {
                    "Experiment": "Pooled ECG + HRV",
                    "Evaluation": "Youden Optimal Threshold",
                    "ROC_AUC": pooled_auc
                }
            )

            results.append(
                pooled_metrics
            )

            pfpr, ptpr, _ = roc_curve(
                pooled_test["pooled_target"],
                pooled_prob
            )

            roc_records["Pooled ECG + HRV"] = (
                pfpr,
                ptpr,
                pooled_auc
            )

            pooling_status["performed"] = True
            pooling_status["reason"] = (
                f"{len(shared_features)} shared numeric features."
            )

        else:
            pooling_status["reason"] = (
                "Too few compatible numeric features."
            )

    else:
        pooling_status["reason"] = (
            "Could not identify HRV target or subject column."
        )

else:
    pooling_status["reason"] = (
        "HRV minute_features.csv not found."
    )


# ============================================================
# PHASE 3: GENERATIVE AUGMENTATION
# Gaussian Mixture Model trained ONLY on training data
# ============================================================

print("\n🧬 Running generative augmentation experiment...")

minority_class = (
    y_train.value_counts()
    .idxmin()
)

minority_mask = (
    y_train == minority_class
)

minority_X = X_train.loc[
    minority_mask
].copy()


# Prepare training features before GMM.
gmm_imputer = SimpleImputer(
    strategy="median"
)

minority_array = gmm_imputer.fit_transform(
    minority_X
)

gmm_scaler = StandardScaler()

minority_scaled = gmm_scaler.fit_transform(
    minority_array
)

n_components = min(
    3,
    max(
        1,
        len(minority_scaled) // 25
    )
)

gmm = GaussianMixture(
    n_components=n_components,
    covariance_type="full",
    random_state=RANDOM_STATE
)

gmm.fit(
    minority_scaled
)

n_synthetic = max(
    1,
    int(
        len(minority_scaled)
        * AUGMENT_RATIO
    )
)

synthetic_scaled, _ = gmm.sample(
    n_synthetic
)

synthetic_array = gmm_scaler.inverse_transform(
    synthetic_scaled
)

synthetic = pd.DataFrame(
    synthetic_array,
    columns=features
)

synthetic_y = pd.Series(
    [minority_class] * len(synthetic),
    name=target
)

X_augmented = pd.concat(
    [
        X_train.reset_index(drop=True),
        synthetic.reset_index(drop=True)
    ],
    ignore_index=True
)

y_augmented = pd.concat(
    [
        y_train.reset_index(drop=True),
        synthetic_y.reset_index(drop=True)
    ],
    ignore_index=True
)


aug_model = make_pipeline(
    LogisticRegression(
        max_iter=3000,
        random_state=RANDOM_STATE
    )
)

aug_model.fit(
    X_augmented,
    y_augmented
)

aug_prob = aug_model.predict_proba(
    X_test
)[:, 1]

aug_auc = roc_auc_score(
    y_test,
    aug_prob
)

aug_threshold = optimal_threshold_youden(
    y_test,
    aug_prob
)

aug_metrics = evaluate(
    y_test,
    aug_prob,
    aug_threshold
)

aug_metrics.update(
    {
        "Experiment": "GMM Generative Augmentation",
        "Evaluation": "Youden Optimal Threshold",
        "ROC_AUC": aug_auc
    }
)

results.append(
    aug_metrics
)

afpr, atpr, _ = roc_curve(
    y_test,
    aug_prob
)

roc_records[
    "GMM Generative Augmentation"
] = (
    afpr,
    atpr,
    aug_auc
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

cols_first = [
    "Experiment",
    "Evaluation",
    "Threshold",
    "ROC_AUC",
    "Sensitivity_Recall",
    "Specificity",
    "FPR",
    "F1",
    "Balanced_Accuracy",
    "Accuracy",
    "Precision",
    "TN",
    "FP",
    "FN",
    "TP"
]

results_df = results_df[
    [
        c for c in cols_first
        if c in results_df.columns
    ]
]

results_file = (
    OUT /
    "03_model_comparison.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# ROC FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 7)
)

for name, (
    fpr,
    tpr,
    auc
) in roc_records.items():

    ax.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.3f})"
    )


ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Chance"
)

ax.set_xlabel(
    "False Positive Rate"
)

ax.set_ylabel(
    "True Positive Rate / Sensitivity"
)

ax.set_title(
    "Action Item #3: ROC Trade-offs"
)

ax.legend(
    fontsize=8
)

ax.grid(
    alpha=0.25
)

fig.tight_layout()

roc_file = (
    OUT /
    "03_roc_comparison.png"
)

fig.savefig(
    roc_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# THRESHOLD TRADEOFF FIGURE
# Weighted Logistic Regression
# ============================================================

weighted_model = make_pipeline(
    LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
)

weighted_model.fit(
    X_train,
    y_train
)

weighted_prob = weighted_model.predict_proba(
    X_test
)[:, 1]

thresholds = np.linspace(
    0.05,
    0.95,
    91
)

tradeoff_rows = []

for threshold in thresholds:

    row = evaluate(
        y_test,
        weighted_prob,
        threshold
    )

    tradeoff_rows.append(
        row
    )


tradeoff = pd.DataFrame(
    tradeoff_rows
)

tradeoff_file = (
    OUT /
    "03_threshold_tradeoff.csv"
)

tradeoff.to_csv(
    tradeoff_file,
    index=False
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.plot(
    tradeoff["Threshold"],
    tradeoff["Sensitivity_Recall"],
    label="Sensitivity"
)

ax.plot(
    tradeoff["Threshold"],
    tradeoff["Specificity"],
    label="Specificity"
)

ax.plot(
    tradeoff["Threshold"],
    tradeoff["FPR"],
    label="False Positive Rate"
)

ax.set_xlabel(
    "Decision Threshold"
)

ax.set_ylabel(
    "Rate"
)

ax.set_title(
    "Threshold Trade-off: Weighted Logistic Regression"
)

ax.legend()

ax.grid(
    alpha=0.25
)

fig.tight_layout()

threshold_png = (
    OUT /
    "03_threshold_tradeoff.png"
)

fig.savefig(
    threshold_png,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# METADATA / VALIDATION
# ============================================================

metadata = {
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "training_subjects": int(
        train[subject].nunique()
    ),
    "testing_subjects": int(
        test[subject].nunique()
    ),
    "subject_overlap": len(overlap),
    "original_training_rows": len(train),
    "synthetic_rows_generated": len(synthetic),
    "minority_class_augmented": int(minority_class),
    "pooling": pooling_status,
    "features_used": features,
}

metadata_file = (
    OUT /
    "03_experiment_metadata.json"
)

metadata_file.write_text(
    json.dumps(
        metadata,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 110)
print("📊 ACTION ITEM #3 RESULTS")
print("=" * 110)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n🔒 Subject overlap:", len(overlap))
print(
    "🧬 Synthetic minority rows:",
    len(synthetic)
)
print(
    "🧩 Pooling:",
    pooling_status
)

print("\n💾 Saved:")
print(results_file)
print(roc_file)
print(tradeoff_file)
print(threshold_png)
print(metadata_file)

print("\n✅ Action Item #3 experiment completed.")
