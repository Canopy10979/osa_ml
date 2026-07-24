import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("../data/processed_dataset.csv")

X = df[
    [
        "HR_Mean",
        "SpO2_Mean",
        "Flow_Mean"
    ]
]

y = df["Sleep_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(6,4))
plt.bar(features, importance)

plt.title("Feature Importance")
plt.ylabel("Importance Score")

for i, v in enumerate(importance):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

plt.show()

print("\nFeature Importance:")

for feature, score in zip(features, importance):
    print(feature, ":", round(score,4))
    
