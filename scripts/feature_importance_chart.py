import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load trained optimized model
model = joblib.load("../models/optimized_sleep_apnea_model.pkl")

# Feature names
features = [
    "HR_Mean",
    "SpO2_Mean",
    "Flow_Mean"
]

# Feature importances
importance = model.feature_importances_

# Create dataframe
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

# Plot
plt.figure(figsize=(7,4))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    "feature_importance_chart.png",
    dpi=300
)

plt.show()

print("✅ Feature importance chart saved as feature_importance_chart.png")
