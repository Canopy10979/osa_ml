from pathlib import Path
import pandas as pd
import wfdb

input_dir = Path("data/raw/apnea_ecg")
output_path = Path("data/normalized/apnea_ecg_labels.csv")

rows = []

for apn_file in sorted(input_dir.glob("*.apn")):
    record_name = apn_file.stem

    # Skip alternate/error-corrected annotation files.
    if record_name.endswith("er") or record_name.endswith("r"):
        continue

    record_base = input_dir / record_name

    try:
        header = wfdb.rdheader(str(record_base))
        annotations = wfdb.rdann(str(record_base), "apn")
    except Exception as exc:
        print(f"Skipped {record_name}: {exc}")
        continue

    for minute_index, (sample, symbol) in enumerate(
        zip(annotations.sample, annotations.symbol)
    ):
        rows.append(
            {
                "dataset": "Apnea-ECG",
                "subject_id": record_name,
                "minute_index": minute_index,
                "sample": int(sample),
                "sampling_frequency_hz": float(header.fs),
                "original_label": symbol,
                "sleep_label": 1 if symbol == "A" else 0,
            }
        )

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No Apnea-ECG annotations were normalized.")

df = df.sort_values(
    ["subject_id", "minute_index"]
).reset_index(drop=True)

output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(df.head(10).to_string(index=False))
print("\nRows:", len(df))
print("Subjects:", df["subject_id"].nunique())
print("\nClass distribution:")
print(df["sleep_label"].value_counts().sort_index())
print(f"\nSaved to: {output_path.resolve()}")
