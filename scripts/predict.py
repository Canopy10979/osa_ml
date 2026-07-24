import joblib
import pandas as pd


# Load the saved model
model = joblib.load("../models/sleep_apnea_random_forest.pkl")


# Load the exact feature order used during training
features = joblib.load("model_features.pkl")


print("✅ Saved model loaded successfully!")


# Create multiple example sleep-data windows
sample_data = pd.DataFrame(
    [
        {
            "HR_Mean": 75.0,
            "SpO2_Mean": 95.0,
            "Flow_Mean": 0.02
        },
        {
            "HR_Mean": 82.0,
            "SpO2_Mean": 89.0,
            "Flow_Mean": 0.20
        },
        {
            "HR_Mean": 68.0,
            "SpO2_Mean": 97.0,
            "Flow_Mean": -0.03
        }
    ]
)


# Ensure the columns match the model's training order
sample_data = sample_data[features]


# Make predictions
predictions = model.predict(sample_data)


# Get prediction probabilities
probabilities = model.predict_proba(sample_data)


# Create a results table
results = sample_data.copy()

results["Predicted_Label"] = predictions
results["Prediction"] = results["Predicted_Label"].map(
    {
        0: "Non-apnea-labeled window",
        1: "Apnea-labeled window"
    }
)

results["Probability_Class_0"] = probabilities[:, 0]
results["Probability_Class_1"] = probabilities[:, 1]


# Convert probabilities into percentages for easier reading
results["Probability_Class_0_Percent"] = (
    results["Probability_Class_0"] * 100
)

results["Probability_Class_1_Percent"] = (
    results["Probability_Class_1"] * 100
)


# Display results
print("\n🧪 Prediction Results:")
print(results.to_string(index=False))


# Display each prediction separately
print("\n📋 Individual Predictions:")

for index, row in results.iterrows():
    print(f"\nSample {index + 1}")
    print(f"❤️ Mean Heart Rate: {row['HR_Mean']}")
    print(f"🫁 Mean SpO2: {row['SpO2_Mean']}")
    print(f"🌬️ Mean Airflow: {row['Flow_Mean']}")
    print(f"Predicted label: {int(row['Predicted_Label'])}")
    print(f"Prediction: {row['Prediction']}")
    print(
        "Class 0 probability: "
        f"{row['Probability_Class_0_Percent']:.2f}%"
    )
    print(
        "Class 1 probability: "
        f"{row['Probability_Class_1_Percent']:.2f}%"
    )


# Export results to a CSV file
results.to_csv(
    "sample_predictions.csv",
    index=False
)


print("\n✅ Predictions saved to sample_predictions.csv")
print(
    "⚠️ This model is a research demonstration, "
    "not a clinical diagnostic tool."
)
