import pandas as pd

# Load the datasets
hr = pd.read_csv("50_HR.csv")
spo2 = pd.read_csv("50_SpO2.csv")
flow = pd.read_csv("50_Flow_DR.csv")
sleep = pd.read_csv("50_sleep_stage.csv")

datasets = {
    "Heart Rate": hr,
    "SpO2": spo2,
    "Flow": flow,
    "Sleep Stage": sleep
}

for name, df in datasets.items():
    print("\n==============================")
    print(name)
    print("==============================")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nShape (rows, columns):")
    print(df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())
