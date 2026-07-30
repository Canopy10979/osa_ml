from pathlib import Path

import pandas as pd


balanced_path = Path(
    "results/balanced_models/overall_metrics.csv"
)

block_path = Path(
    "results/block_cross_validation/block_cv_summary.csv"
)

if not balanced_path.exists():
    raise FileNotFoundError(
        f"Missing file: {balanced_path}"
    )

if not block_path.exists():
    raise FileNotFoundError(
        f"Missing file: {block_path}"
    )

balanced = pd.read_csv(balanced_path)
block = pd.read_csv(block_path)

# Account for either capitalization style.
balanced = balanced.rename(
    columns={
        "model": "Model",
        "accuracy": "Holdout_Accuracy",
        "balanced_accuracy": "Holdout_Balanced_Accuracy",
        "Accuracy": "Holdout_Accuracy",
        "Balanced_Accuracy": "Holdout_Balanced_Accuracy",
    }
)

block = block.rename(
    columns={
        "Accuracy": "Block_CV_Accuracy",
        "Balanced_Accuracy": "Block_CV_Balanced_Accuracy",
    }
)

comparison = balanced.merge(
    block[
        [
            "Model",
            "Block_CV_Accuracy",
            "Block_CV_Balanced_Accuracy",
            "Class_0_Precision",
            "Class_0_Recall",
            "Class_1_Precision",
            "Class_1_Recall",
        ]
    ],
    on="Model",
    how="left",
)

output_path = Path("results/model_comparison.csv")

comparison.to_csv(
    output_path,
    index=False,
)

print(comparison.to_string(index=False))
print(f"\nSaved to: {output_path.resolve()}")
