import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# Load processed dataset
df = pd.read_csv("../data/processed_dataset.csv")


# Select features
X = df[
    [
        "HR_Mean",
        "SpO2_Mean",
        "Flow_Mean"
    ]
]

# Target labels
y = df["Sleep_Label"]


# Split data
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


# Make predictions
predictions = model.predict(X_test)


# Generate classification report
report = classification_report(
    y_test,
    predictions,
    target_names=[
        "Non-Apnea",
        "Apnea"
    ]
)
print("\n📊 Classification Report:\n")
print(report)

with open("classification_report.txt", "w") as file:
    file.write(report)

print("\n✅ Classification report saved as classification_report.txt")


