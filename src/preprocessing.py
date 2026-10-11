"""
End-to-end preprocessing pipeline for the Kaggle Medical Students Dataset.

Usage:
    python src/preprocessing.py --input data/medical_students_dataset.csv \
                                --output data/medical_students_clean.csv

Steps:
    1. Load and inspect the raw data
    2. Handle missing values (drop rows without an ID, impute the rest)
    3. Remove duplicate records
    4. Detect and cap outliers with the IQR rule
    5. Fix data types
    6. Engineer features (recomputed BMI, BMI category)
    7. Encode categorical variables
    8. Scale numeric features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

NUMERIC_COLS = [
    "Age", "Height", "Weight", "BMI", "Temperature",
    "Heart Rate", "Blood Pressure", "Cholesterol",
]
NOMINAL_COLS = ["Gender", "Blood Type"]
BINARY_COLS = ["Diabetes", "Smoking"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV."""
    return pd.read_csv(path)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no Student ID, then impute numeric columns with the
    median and categorical columns with the mode.

    Dropping every row with any missing value would discard most of the
    dataset because missing values are spread across all columns.
    """
    df = df.copy()
    if "Student ID" in df.columns:
        df = df.dropna(subset=["Student ID"])

    for col in [c for c in NUMERIC_COLS if c in df.columns]:
        df[col] = df[col].fillna(df[col].median())
    for col in [c for c in NOMINAL_COLS + BINARY_COLS if c in df.columns]:
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates and repeated Student IDs (keep the first)."""
    df = df.drop_duplicates()
    if "Student ID" in df.columns:
        df = df.drop_duplicates(subset=["Student ID"], keep="first")
    return df.reset_index(drop=True)


def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Return the lower and upper IQR fences for a numeric series."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def cap_outliers(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Clip values outside the IQR fences instead of dropping rows."""
    df = df.copy()
    for col in cols or [c for c in NUMERIC_COLS if c in df.columns]:
        low, high = iqr_bounds(df[col])
        df[col] = df[col].clip(lower=low, upper=high)
    return df


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to sensible types."""
    df = df.copy()
    if "Student ID" in df.columns:
        df["Student ID"] = df["Student ID"].astype(int)
    df["Age"] = df["Age"].round().astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute BMI from height (cm) and weight (kg) and bucket it into
    WHO categories."""
    df = df.copy()
    height_m = df["Height"] / 100
    df["BMI_Computed"] = (df["Weight"] / height_m**2).round(2)
    df["BMI_Category"] = pd.cut(
        df["BMI_Computed"],
        bins=[0, 18.5, 25, 30, np.inf],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
        right=False,
    )
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Binary Yes/No columns -> 0/1; nominal columns -> one-hot;
    BMI category (ordered) -> ordinal integers."""
    df = df.copy()
    for col in BINARY_COLS:
        df[col] = df[col].map({"No": 0, "Yes": 1}).astype(int)
    if "BMI_Category" in df.columns:
        df["BMI_Category_Code"] = df["BMI_Category"].cat.codes
    df = pd.get_dummies(df, columns=NOMINAL_COLS, dtype=int)
    return df


def scale_features(df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    """Scale continuous columns with z-score ('standard') or min-max."""
    df = df.copy()
    cols = [c for c in NUMERIC_COLS + ["BMI_Computed"] if c in df.columns]
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def run_pipeline(df: pd.DataFrame, scale: str = "standard") -> pd.DataFrame:
    """Run every step in order and return a model-ready DataFrame."""
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = cap_outliers(df)
    df = fix_types(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    df = scale_features(df, method=scale)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--input", default="data/medical_students_dataset.csv")
    parser.add_argument("--output", default="data/medical_students_clean.csv")
    parser.add_argument("--scale", choices=["standard", "minmax"], default="standard")
    args = parser.parse_args()

    raw = load_data(args.input)
    print(f"Raw data:   {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    clean = run_pipeline(raw, scale=args.scale)
    print(f"Clean data: {clean.shape[0]:,} rows x {clean.shape[1]} columns")
    clean.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
