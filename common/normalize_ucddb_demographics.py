from pathlib import Path
import pandas as pd

input_path = Path(r"data\raw\ucddb\SubjectDetails.xls")
output_path = Path(r"data\normalized\ucddb_demographics.csv")

df = pd.read_excel(input_path)

# Clean column names.
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
)

# Remove completely empty rows and columns.
df = df.dropna(how="all")
df = df.dropna(axis=1, how="all")

# Add a dataset source column.
df.insert(0, "dataset", "UCDDB")

output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(df.head().to_string(index=False))
print("\nColumns:", df.columns.tolist())
print("Rows:", len(df))
print(f"\nSaved to: {output_path.resolve()}")
