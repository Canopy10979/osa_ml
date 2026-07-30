import pandas as pd


results = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "Accuracy": [
            0.9617486338797814,
            0.9398907103825137,
            0.9726775956284153
        ],
        "Precision": [
            0.9611111111111111,
            0.9602272727272727,
            0.9719101123595506
        ],
        "Recall": [
            1.0,
            0.976878612716763,
            1.0
        ],
        "F1_Score": [
            0.9801699716713881,
            0.9684813753581661,
            0.9857549857549858
        ]
    }
)


results.to_csv(
    "model_evaluation_results.csv",
    index=False
)


print("✅ Model results exported successfully!")
print("\nSaved as: model_evaluation_results.csv")
print("\nResults:")
print(results)
