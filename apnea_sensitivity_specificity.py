import os
import pandas as pd
import numpy as np

RESULTS_DIR = "dataset_apnea_ecg/results"

FILES = {
    "Logistic Regression L1": "confusion_LogisticRegression_L1.csv",
    "Logistic Regression L2": "confusion_LogisticRegression_L2.csv",
    "Random Forest": "confusion_RandomForest.csv",
    "XGBoost": "confusion_XGBoost.csv",
}

def extract_confusion_matrix(path):
    df = pd.read_csv(path)

    # Try direct TN/FP/FN/TP format
    lower_cols = {c.lower(): c for c in df.columns}

    if all(x in lower_cols for x in ["tn", "fp", "fn", "tp"]):
        row = df.iloc[0]

        return (
            int(row[lower_cols["tn"]]),
            int(row[lower_cols["fp"]]),
            int(row[lower_cols["fn"]]),
            int(row[lower_cols["tp"]]),
        )

    # Try a standard 2x2 numeric confusion matrix
    numeric = df.select_dtypes(include=[np.number])

    if numeric.shape[0] >= 2 and numeric.shape[1] >= 2:
        values = numeric.iloc[:2, -2:].to_numpy()

        tn = int(values[0, 0])
        fp = int(values[0, 1])
        fn = int(values[1, 0])
        tp = int(values[1, 1])

        return tn, fp, fn, tp

    raise ValueError(
        f"Could not understand confusion matrix format in {path}\n"
        f"Columns found: {df.columns.tolist()}"
    )


rows = []
counts = []

for model, filename in FILES.items():

    path = os.path.join(RESULTS_DIR, filename)

    print(f"\n🤖 Processing {model}...")

    tn, fp, fn, tp = extract_confusion_matrix(path)

    apnea_sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    apnea_specificity = tn / (tn + fp) if (tn + fp) else np.nan
    apnea_fpr = fp / (fp + tn) if (fp + tn) else np.nan
    apnea_fnr = fn / (fn + tp) if (fn + tp) else np.nan

    # Normal class viewed as the positive class
    normal_sensitivity = tn / (tn + fp) if (tn + fp) else np.nan
    normal_specificity = tp / (tp + fn) if (tp + fn) else np.nan
    normal_fpr = fn / (fn + tp) if (fn + tp) else np.nan
    normal_fnr = fp / (fp + tn) if (fp + tn) else np.nan

    rows.append({
        "Model": model,
        "Class": "Sleep Apnea",
        "Sensitivity": apnea_sensitivity,
        "Specificity": apnea_specificity,
        "False_Positive_Rate": apnea_fpr,
        "False_Negative_Rate": apnea_fnr,
    })

    rows.append({
        "Model": model,
        "Class": "Normal",
        "Sensitivity": normal_sensitivity,
        "Specificity": normal_specificity,
        "False_Positive_Rate": normal_fpr,
        "False_Negative_Rate": normal_fnr,
    })

    counts.append({
        "Model": model,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    })


table = pd.DataFrame(rows)
count_table = pd.DataFrame(counts)

# Validation checks
table["Specificity_plus_FPR"] = (
    table["Specificity"] + table["False_Positive_Rate"]
)

table["Sensitivity_plus_FNR"] = (
    table["Sensitivity"] + table["False_Negative_Rate"]
)

table["Specificity_FPR_Valid"] = np.isclose(
    table["Specificity_plus_FPR"],
    1.0,
    atol=1e-8
)

table["Sensitivity_FNR_Valid"] = np.isclose(
    table["Sensitivity_plus_FNR"],
    1.0,
    atol=1e-8
)


# Save decimal version
table.to_csv(
    os.path.join(
        RESULTS_DIR,
        "sensitivity_specificity_table.csv"
    ),
    index=False
)

count_table.to_csv(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix_summary.csv"
    ),
    index=False
)


# Separate false-positive analysis
false_positive = table[
    [
        "Model",
        "Class",
        "False_Positive_Rate",
        "Specificity"
    ]
].copy()

false_positive.to_csv(
    os.path.join(
        RESULTS_DIR,
        "false_positive_analysis.csv"
    ),
    index=False
)


# Pretty terminal display
display = table.copy()

for col in [
    "Sensitivity",
    "Specificity",
    "False_Positive_Rate",
    "False_Negative_Rate"
]:
    display[col] = (
        display[col] * 100
    ).round(2).astype(str) + "%"


print("\n" + "=" * 110)
print("📊 SENSITIVITY / SPECIFICITY RESULTS")
print("=" * 110)

print(
    display[
        [
            "Model",
            "Class",
            "Sensitivity",
            "Specificity",
            "False_Positive_Rate",
            "False_Negative_Rate"
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 70)
print("🔢 CONFUSION MATRIX COUNTS")
print("=" * 70)

print(count_table.to_string(index=False))


print("\n" + "=" * 70)
print("✅ MATHEMATICAL VALIDATION")
print("=" * 70)

print(
    table[
        [
            "Model",
            "Class",
            "Specificity_FPR_Valid",
            "Sensitivity_FNR_Valid"
        ]
    ].to_string(index=False)
)


if (
    table["Specificity_FPR_Valid"].all()
    and table["Sensitivity_FNR_Valid"].all()
):
    print("\n✅ All sensitivity/specificity calculations passed validation.")
else:
    print("\n❌ At least one validation check failed.")


print("\n💾 Saved:")
print(" • sensitivity_specificity_table.csv")
print(" • confusion_matrix_summary.csv")
print(" • false_positive_analysis.csv")
