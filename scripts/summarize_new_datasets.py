from pathlib import Path
import pandas as pd

ucddb_path = Path("data/normalized/unified_demographics.csv")
apnea_path = Path("data/normalized/apnea_ecg_labels.csv")
output_path = Path("results/dataset_normalization_summary.csv")

ucddb = pd.read_csv(ucddb_path)
apnea = pd.read_csv(apnea_path)

summary = pd.DataFrame(
    [
        {
            "dataset": "UCDDB",
            "normalized_file": str(ucddb_path),
            "rows": len(ucddb),
            "subjects": ucddb["subject_id"].nunique(),
            "includes_age": "age" in ucddb.columns,
            "includes_demographics": True,
            "includes_binary_labels": False,
        },
        {
            "dataset": "Apnea-ECG",
            "normalized_file": str(apnea_path),
            "rows": len(apnea),
            "subjects": apnea["subject_id"].nunique(),
            "includes_age": False,
            "includes_demographics": False,
            "includes_binary_labels": "sleep_label" in apnea.columns,
        },
    ]
)

output_path.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(output_path, index=False)

print(summary.to_string(index=False))
print(f"\nSaved to: {output_path.resolve()}")
