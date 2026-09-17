"""
prepare_cardio_data.py

Cleans the Kaggle "Cardiovascular Disease Dataset" (Sulianova, cardio_train.csv,
70,000 rows, semicolon-separated) into the format expected by the training
pipeline: a CSV with a 'target' column and clean numeric features.

Run from the project root:
    venv\\Scripts\\python.exe prepare_cardio_data.py
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = "data/cardio_train.csv"
OUT_PATH = "data/clinical_prepared.csv"


def main():
    df = pd.read_csv(RAW_PATH, sep=";")
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # 1. Drop identifier column
    df = df.drop(columns=["id"])

    # 2. Age is stored in days in this dataset -> convert to years
    df["age"] = (df["age"] / 365.25).round(1)

    # 3. Engineer BMI = weight(kg) / height(m)^2
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

    # 4. Remove physiologically impossible / clearly erroneous rows
    #    (this dataset is known to contain data-entry errors in ap_hi/ap_lo)
    before = len(df)
    df = df[
        (df["ap_hi"] >= 80) & (df["ap_hi"] <= 250) &
        (df["ap_lo"] >= 40) & (df["ap_lo"] <= 200) &
        (df["ap_hi"] > df["ap_lo"]) &
        (df["height"] >= 120) & (df["height"] <= 220) &
        (df["weight"] >= 30) & (df["weight"] <= 200)
    ].reset_index(drop=True)
    removed = before - len(df)
    print(f"Removed {removed} rows with physiologically implausible values "
          f"({removed / before:.1%} of data)")

    # 5. Rename target column to the name the pipeline expects
    df = df.rename(columns={"cardio": "target"})

    # 6. Reorder so target is last (cosmetic, not required)
    cols = [c for c in df.columns if c != "target"] + ["target"]
    df = df[cols]

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved cleaned dataset to: {OUT_PATH}")
    print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target balance: {df['target'].value_counts(normalize=True).to_dict()}")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
