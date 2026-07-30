import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)

# Load dataset
df = pd.read_csv("../data/processed_dataset.csv")

# Features
X = df[
    [
        "HR_Mean",
        "SpO2_Mean",
        "Flow_Mean"
    ]
]

# Labels
y = df["Sleep_Label"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predict probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# Precision-Recall values
precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_scores
)

# Average Precision
ap_score = average_precision_score(
    y_test,
    y_scores
)

# Plot
plt.figure(figsize=(6,6))

plt.plot(
    recall,
    precision,
    linewidth=2,
    label=f"Average Precision = {ap_score:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "precision_recall_curve.png",
    dpi=300
)

plt.show()

print(f"\n✅ Average Precision Score: {ap_score:.4f}")
print("📁 Precision-Recall curve saved as precision_recall_curve.png")
