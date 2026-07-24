import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../data/processed_dataset.csv")

# -----------------------------
# 1. Class distribution
# -----------------------------
class_counts = df["Sleep_Label"].value_counts().sort_index()

plt.figure(figsize=(6, 4))

plt.bar(
    ["Non-Apnea", "Apnea"],
    class_counts.values
)

plt.ylabel("Number of Samples")
plt.title("Sleep Apnea Class Distribution")

for index, value in enumerate(class_counts.values):
    plt.text(
        index,
        value,
        str(value),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300)
plt.close()

# -----------------------------
# 2. Heart rate distribution
# -----------------------------
plt.figure(figsize=(7, 4))

plt.hist(
    df[df["Sleep_Label"] == 0]["HR_Mean"],
    bins=20,
    alpha=0.6,
    label="Non-Apnea"
)

plt.hist(
    df[df["Sleep_Label"] == 1]["HR_Mean"],
    bins=20,
    alpha=0.6,
    label="Apnea"
)

plt.xlabel("Mean Heart Rate")
plt.ylabel("Frequency")
plt.title("Heart Rate Distribution by Class")
plt.legend()

plt.tight_layout()
plt.savefig("hr_distribution.png", dpi=300)
plt.close()

# -----------------------------
# 3. SpO2 distribution
# -----------------------------
plt.figure(figsize=(7, 4))

plt.hist(
    df[df["Sleep_Label"] == 0]["SpO2_Mean"],
    bins=20,
    alpha=0.6,
    label="Non-Apnea"
)

plt.hist(
    df[df["Sleep_Label"] == 1]["SpO2_Mean"],
    bins=20,
    alpha=0.6,
    label="Apnea"
)

plt.xlabel("Mean SpO2")
plt.ylabel("Frequency")
plt.title("SpO2 Distribution by Class")
plt.legend()

plt.tight_layout()
plt.savefig("spo2_distribution.png", dpi=300)
plt.close()

# -----------------------------
# 4. Airflow distribution
# -----------------------------
plt.figure(figsize=(7, 4))

plt.hist(
    df[df["Sleep_Label"] == 0]["Flow_Mean"],
    bins=20,
    alpha=0.6,
    label="Non-Apnea"
)

plt.hist(
    df[df["Sleep_Label"] == 1]["Flow_Mean"],
    bins=20,
    alpha=0.6,
    label="Apnea"
)

plt.xlabel("Mean Airflow")
plt.ylabel("Frequency")
plt.title("Airflow Distribution by Class")
plt.legend()

plt.tight_layout()
plt.savefig("flow_distribution.png", dpi=300)
plt.close()

print("✅ Saved class_distribution.png")
print("✅ Saved hr_distribution.png")
print("✅ Saved spo2_distribution.png")
print("✅ Saved flow_distribution.png")
