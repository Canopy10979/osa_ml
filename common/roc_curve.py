import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc


# Load dataset
df = pd.read_csv("../data/processed_dataset.csv")

X = df[
    [
        "HR_Mean",
        "SpO2_Mean",
        "Flow_Mean"
    ]
]

y = df["Sleep_Label"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Probability predictions
y_scores = model.predict_proba(X_test)[:, 1]


# Compute ROC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

roc_auc = auc(fpr, tpr)


# Plot
plt.figure(figsize=(6,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"Random Forest (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    linewidth=1,
    label="Random Guess"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend(loc="lower right")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "roc_curve.png",
    dpi=300
)

plt.show()


print(f"\n✅ ROC-AUC Score: {roc_auc:.4f}")
print("📁 ROC curve saved as roc_curve.png")
