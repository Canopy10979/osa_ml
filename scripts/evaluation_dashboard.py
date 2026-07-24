import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("../data/processed_dataset.csv")

X = df[["HR_Mean", "SpO2_Mean", "Flow_Mean"]]
y = df["Sleep_Label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Create figure
fig = plt.figure(figsize=(10, 6))

# Metrics text
metrics = (
    f"Accuracy : {accuracy:.3f}\n"
    f"Precision: {precision:.3f}\n"
    f"Recall   : {recall:.3f}\n"
    f"F1 Score : {f1:.3f}"
)

plt.figtext(
    0.05,
    0.65,
    metrics,
    fontsize=13,
    bbox=dict(facecolor="white")
)

# Confusion matrix
ax = plt.axes([0.45, 0.15, 0.45, 0.7])

im = ax.imshow(cm)

ax.set_title("Confusion Matrix")

ax.set_xticks([0,1])
ax.set_yticks([0,1])

ax.set_xticklabels(["Non-Apnea","Apnea"])
ax.set_yticklabels(["Non-Apnea","Apnea"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=12
        )

plt.savefig(
    "evaluation_dashboard.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("✅ evaluation_dashboard.png saved")
