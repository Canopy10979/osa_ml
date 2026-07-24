import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("../data/processed_dataset.csv")

# Correlation matrix
corr = df.corr(numeric_only=True)

# Plot
plt.figure(figsize=(6,5))

plt.imshow(corr, cmap="coolwarm", interpolation="nearest")

plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

# Add correlation values
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(
            j,
            i,
            f"{corr.iloc[i,j]:.2f}",
            ha="center",
            va="center",
            color="black"
        )

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "correlation_heatmap.png",
    dpi=300
)

plt.show()

print("✅ Correlation heatmap saved as correlation_heatmap.png")
