from pathlib import Path
import pandas as pd

input_path = Path("data/normalized/ucddb_demographics.csv")
output_path = Path("data/normalized/unified_demographics.csv")

ucddb = pd.read_csv(input_path)

ucddb = ucddb.rename(
    columns={
        "study_number": "subject_id",
        "gender": "sex"
    }
)

required_columns = [
    "dataset",
    "subject_id",
    "age",
    "sex",
    "height_cm",
    "weight_kg",
    "bmi",
    "psg_ahi",
    "epworth_sleepiness_score",
    "study_duration_hr",
    "sleep_efficiency"
]

for column in required_columns:
    if column not in ucddb.columns:
        ucddb[column] = pd.NA

ucddb = ucddb[required_columns]

# Standardize text fields.
ucddb["dataset"] = ucddb["dataset"].astype("string").str.strip()
ucddb["subject_id"] = ucddb["subject_id"].astype("string").str.strip()
ucddb["sex"] = ucddb["sex"].astype("string").str.upper().str.strip()

# Ensure numeric fields are stored as numbers.
numeric_columns = [
    "age",
    "height_cm",
    "weight_kg",
    "bmi",
    "psg_ahi",
    "epworth_sleepiness_score",
    "study_duration_hr",
    "sleep_efficiency"
]

for column in numeric_columns:
    ucddb[column] = pd.to_numeric(ucddb[column], errors="coerce")

output_path.parent.mkdir(parents=True, exist_ok=True)
ucddb.to_csv(output_path, index=False)

print(ucddb.head().to_string(index=False))
print("\nRows:", len(ucddb))
print("Missing subject IDs:", ucddb["subject_id"].isna().sum())
print("Duplicate subject IDs:", ucddb["subject_id"].duplicated().sum())
print(f"\nSaved to: {output_path.resolve()}")
