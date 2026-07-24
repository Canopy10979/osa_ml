import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Load dataset
df = pd.read_csv("../data/processed_dataset.csv")

X = df[["HR_Mean", "SpO2_Mean", "Flow_Mean"]]
y = df["Sleep_Label"]

# Same tuning setup
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": [None, "balanced"]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)

grid_search.fit(X, y)

# Best trained model
best_model = grid_search.best_estimator_

# Save model
joblib.dump(best_model, "../models/optimized_sleep_apnea_model.pkl")

# Save feature order
joblib.dump(
    ["HR_Mean", "SpO2_Mean", "Flow_Mean"],
    "../models/optimized_model_features.pkl"
)

# Save best settings
with open("../models/best_model_parameters.json", "w") as file:
    json.dump(grid_search.best_params_, file, indent=4)

print("\n✅ Optimized model trained and saved")
print("📁 ../models/optimized_sleep_apnea_model.pkl")
print("📁 ../models/optimized_model_features.pkl")
print("📁 ../models/best_model_parameters.json")
print(f"\nBest cross-validated F1: {grid_search.best_score_:.4f}")
