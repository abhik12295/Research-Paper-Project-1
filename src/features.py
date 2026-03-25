from __future__ import annotations
from pathlib import Path
from typing import Dict, List, cast
import numpy as np
import pandas as pd
from src.config import PROCESSED_DATA_DIR

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def to_month_start(series: pd.Series) -> pd.Series:
    return cast(pd.Series, pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp())

# Time Features
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()
    dt = pd.to_datetime(df["date"])
    df["month"] = dt.dt.month

    # cyclic encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.sin(2 * np.pi * df["month"] / 12.0)

    # trend
    df["t"] = np.arange(len(df))
    return df

# Lag Features
def add_lags(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    df = df.sort_values("date").copy().set_index("date")
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df.reset_index()

# Rolling features (leakage safe)
def add_rolling_features(df: pd.DataFrame, col: str, windows: List[int]) -> pd.DataFrame:
    df = df.sort_values("date").copy().set_index("date")
    shifted = df[col].shift(1)
    for w in windows:
        df[f"{col}_rollmean{w}"] = shifted.rolling(w).mean()
        df[f"{col}_rollstd{w}"] = shifted.rolling(w).std()
    return df.reset_index()

# build supervised dataset
def build_supervised(df: pd.DataFrame, target_col:str) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    df["y_next"] = df[target_col].shift(-1)
    df = df.dropna(subset=["y_next"]).reset_index(drop=True)
    return df

# full feature pipeline
def build_feature_dataset(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    print("\n [STEP] Adding time features....")
    df = add_time_features(df)

    print("[STEP] Identifying numeric columns...")
    numeric_cols = [
        c for c in df.columns
        if c!= "date" and pd.api.types.is_numeric_dtype(df[c])
    ]

    core_cols = [
        c for c in numeric_cols
        if c not in ["month", "month_sin", "month_cost", "t"]
    ]

    print(f"[INFO] Core feature columns: {core_cols}")

    print(f"\n[STEP] Adding lag features...")
    df = add_lags(df, cols=core_cols, lags= [1,2,3,6, 12])

    print("\n[STEP] Adding rolling features...")
    if target_col in df.columns:
        df = add_rolling_features(df, target_col, windows=[3, 6, 12])
    else:
        raise ValueError(f"Target column {target_col} not found.")
    
    print("\n[STEP] Building supervised dataset...")
    df = build_supervised(df, target_col)
    return df

# Validation
def validate_feature_dataset(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        raise ValueError("Feature dataset is empty.")
    
    if "y_next" not in df.columns:
        raise ValueError("Missing target column 'y_next'.")
    
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found.")
    
    summary = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "start_date": df["date"].min(),
        "end_date": df["date"].max(),
        "null_counts": df.isna().to_dict(),
    }

    return summary

def save_feature_dataset(df: pd.DataFrame) -> Path:
    ensure_dir(PROCESSED_DATA_DIR)
    output_path = PROCESSED_DATA_DIR / "model_dataset.csv"
    df.to_csv(output_path, index=False)
    return output_path

# output run using main_feature_run.py