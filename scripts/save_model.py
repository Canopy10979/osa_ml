import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Load processed data
df = pd.read_csv("../data/processed_dataset.csv")


# Select input features
features = [
    "HR_Mean",
    "SpO2_Mean",
    "Flow_Mean"
]

X = df[features]
y = df["Sleep_Label"]


# Use the same train/test split as before
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train the model
model.fit(X_train, y_train)


# Save the trained model
joblib.dump(model, "../models/sleep_apnea_random_forest.pkl")


# Save the feature names too
joblib.dump(features, "model_features.pkl")


print("✅ Random Forest model saved successfully!")
print("✅ Feature names saved successfully!")
print("Model file: ../models/sleep_apnea_random_forest.pkl")
print("Features file: model_features.pkl")
