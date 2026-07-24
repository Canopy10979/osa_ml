import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Load dataset
df = pd.read_csv("../data/processed_dataset.csv")

# Features and label
X = df[["HR_Mean", "SpO2_Mean", "Flow_Mean"]]
y = df["Sleep_Label"]

# Base model
model = RandomForestClassifier(
    random_state=42
)

# Parameters to test
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": [None, "balanced"]
}

# Stratified folds preserve the class proportions
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)

print("\n✅ Best Parameters:")
print(grid_search.best_params_)

print(f"\n✅ Best Cross-Validated F1 Score: {grid_search.best_score_:.4f}")

# Save results
results_df = pd.DataFrame(grid_search.cv_results_)
results_df.to_csv("hyperparameter_tuning_results.csv", index=False)

print("\n📁 Full tuning results saved as hyperparameter_tuning_results.csv")
