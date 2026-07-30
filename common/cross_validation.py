import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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

# Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Perform 5-fold cross-validation
scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n📊 Cross-Validation Accuracy Scores:")
print(scores)

print(f"\n✅ Mean Accuracy: {scores.mean():.4f}")
print(f"📉 Standard Deviation: {scores.std():.4f}")
