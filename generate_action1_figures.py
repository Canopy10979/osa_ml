from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path.cwd()

RESULTS = ROOT / "dataset_apnea_ecg" / "results"
OUTPUT = ROOT / "mentor_action_items"

OUTPUT.mkdir(parents=True, exist_ok=True)

SENS_FILE = RESULTS / "sensitivity_specificity_table.csv"
CONF_FILE = RESULTS / "confusion_matrix_summary.csv"
FPR_FILE = RESULTS / "false_positive_analysis.csv"


# ============================================================
# VALIDATE INPUT FILES
# ============================================================

required_files = [
    SENS_FILE,
    CONF_FILE,
    FPR_FILE,
]

print("\n🔎 Checking source files...")

for file in required_files:

    if not file.exists():
        raise FileNotFoundError(
            f"\n❌ Missing required file:\n{file}"
        )

    print(f"✅ Found: {file.name}")


# ============================================================
# LOAD DATA
# ============================================================

sensitivity = pd.read_csv(SENS_FILE)
confusion = pd.read_csv(CONF_FILE)
false_positive = pd.read_csv(FPR_FILE)

print("\n📊 Sensitivity/specificity columns:")
print(sensitivity.columns.tolist())

print("\n📊 Confusion matrix columns:")
print(confusion.columns.tolist())

print("\n📊 False-positive columns:")
print(false_positive.columns.tolist())


# ============================================================
# CLEAN NUMERIC VALUES
# ============================================================

metric_columns = [
    "Sensitivity",
    "Specificity",
    "False_Positive_Rate",
    "False_Negative_Rate",
]

for column in metric_columns:

    if column in sensitivity.columns:

        sensitivity[column] = pd.to_numeric(
            sensitivity[column],
            errors="coerce"
        )


if "False_Positive_Rate" in false_positive.columns:

    false_positive["False_Positive_Rate"] = pd.to_numeric(
        false_positive["False_Positive_Rate"],
        errors="coerce"
    )


# ============================================================
# FIGURE 1
# SENSITIVITY VS SPECIFICITY — APNEA CLASS
# ============================================================

apnea = sensitivity[
    sensitivity["Class"]
    .astype(str)
    .str.lower()
    .str.contains("apnea")
].copy()


if apnea.empty:
    raise ValueError(
        "❌ Could not find the Sleep Apnea class in "
        "sensitivity_specificity_table.csv"
    )


x = np.arange(len(apnea))
width = 0.36


fig, ax = plt.subplots(figsize=(11, 6))

bars1 = ax.bar(
    x - width / 2,
    apnea["Sensitivity"] * 100,
    width,
    label="Sensitivity"
)

bars2 = ax.bar(
    x + width / 2,
    apnea["Specificity"] * 100,
    width,
    label="Specificity"
)

ax.set_title(
    "Sleep Apnea Detection: Sensitivity vs Specificity"
)

ax.set_ylabel("Performance (%)")

ax.set_xlabel("Machine Learning Model")

ax.set_xticks(x)

ax.set_xticklabels(
    apnea["Model"],
    rotation=20,
    ha="right"
)

ax.set_ylim(0, 105)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)


for bars in [bars1, bars2]:

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8
        )


fig.tight_layout()

figure1 = OUTPUT / "01_sensitivity_specificity.png"

fig.savefig(
    figure1,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"\n✅ Created: {figure1}")


# ============================================================
# FIGURE 2
# FALSE POSITIVE RATE — APNEA DETECTION
# ============================================================

apnea_fpr = false_positive[
    false_positive["Class"]
    .astype(str)
    .str.lower()
    .str.contains("apnea")
].copy()


if apnea_fpr.empty:
    raise ValueError(
        "❌ Could not find apnea-class false-positive results."
    )


fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    apnea_fpr["Model"],
    apnea_fpr["False_Positive_Rate"] * 100
)

ax.set_title(
    "False Positive Rate for Sleep Apnea Detection"
)

ax.set_ylabel(
    "False Positive Rate (%)"
)

ax.set_xlabel(
    "Machine Learning Model"
)

ax.tick_params(
    axis="x",
    rotation=20
)

ax.grid(
    axis="y",
    alpha=0.25
)


for bar in bars:

    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.5,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )


fig.tight_layout()

figure2 = OUTPUT / "01_false_positive_rates.png"

fig.savefig(
    figure2,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"✅ Created: {figure2}")


# ============================================================
# FIGURE 3
# CONFUSION MATRIX COUNTS BY MODEL
# ============================================================

required_confusion_columns = [
    "Model",
    "TN",
    "FP",
    "FN",
    "TP",
]

missing = [
    column
    for column in required_confusion_columns
    if column not in confusion.columns
]

if missing:

    raise ValueError(
        "❌ Missing confusion-matrix columns: "
        + ", ".join(missing)
    )


for column in ["TN", "FP", "FN", "TP"]:

    confusion[column] = pd.to_numeric(
        confusion[column],
        errors="coerce"
    )


x = np.arange(len(confusion))
width = 0.20


fig, ax = plt.subplots(figsize=(12, 7))

bars_tn = ax.bar(
    x - 1.5 * width,
    confusion["TN"],
    width,
    label="True Negative"
)

bars_fp = ax.bar(
    x - 0.5 * width,
    confusion["FP"],
    width,
    label="False Positive"
)

bars_fn = ax.bar(
    x + 0.5 * width,
    confusion["FN"],
    width,
    label="False Negative"
)

bars_tp = ax.bar(
    x + 1.5 * width,
    confusion["TP"],
    width,
    label="True Positive"
)


ax.set_title(
    "Confusion Matrix Counts by Model"
)

ax.set_ylabel(
    "Number of Predictions"
)

ax.set_xlabel(
    "Machine Learning Model"
)

ax.set_xticks(x)

ax.set_xticklabels(
    confusion["Model"],
    rotation=20,
    ha="right"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)


fig.tight_layout()

figure3 = OUTPUT / "01_confusion_matrix_comparison.png"

fig.savefig(
    figure3,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"✅ Created: {figure3}")


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("🔬 FIGURE VALIDATION")
print("=" * 65)

generated = [
    figure1,
    figure2,
    figure3,
]

all_valid = True

for figure in generated:

    exists = figure.exists()

    size = (
        figure.stat().st_size
        if exists
        else 0
    )

    valid = (
        exists and
        size > 1000
    )

    print(
        f"{figure.name}: "
        f"exists={exists}, "
        f"size={size:,} bytes, "
        f"valid={valid}"
    )

    if not valid:
        all_valid = False


if not all_valid:

    raise RuntimeError(
        "❌ At least one PNG failed validation."
    )


print("\n🎉 SUCCESS!")
print("All three Action Item #1 figures were generated.")
print("\nSaved in:")
print(OUTPUT.resolve())
