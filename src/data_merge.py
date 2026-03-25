from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import pandas as pd
from src.config import PROCESSED_DATA_DIR
from typing import cast

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def to_month_start(series: pd.Series) -> pd.Series:
    return cast(pd.Series, pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp())

def load_local_fred_series(csv_path: Path, value_column_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if list(df.columns) != ["DATE", value_column_name.upper()] and list(df.columns)!= ["DATE", value_column_name]:
        if len(df.columns) !=2:
            raise ValueError(f"Unexpected FRED schema in {csv_path.name}: {df.columns.tolist()}")
    date_col = df.columns[0]
    value_col = df.columns[1]

    df = df.rename(columns={
        date_col: "date",
        value_col: value_column_name.lower()
    })

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_column_name.lower()] = pd.to_numeric(df[value_column_name.lower()], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = to_month_start(df["date"])
    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"])
        .reset_index(drop=True)
    )
    return df

def valdidate_fred_series(df: pd.DataFrame, series_name: str) -> Dict[str, object]:
    expected_cols = ["date", series_name]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{series_name}: missing expected columns {missing}")
    
    if df.empty:
        raise ValueError(f"{series_name}: dataframe is empty")
    
    if df["date"].duplicated().any():
        raise ValueError(f"{series_name}: duplicate dates found")
    
    if not df["date"].is_monotonic_increasing:
        raise ValueError(f"{series_name}: dates are not sorted ascending")
    
    return {
        "series_name": series_name,
        "row_count": len(df),
        "start_date": df["date"].min(),
        "end_date": df["date"].max(),
        "null_count": int(df[series_name].isna().sum())
    }

def load_fred_bundle(raw_data_dir: Path) -> pd.DataFrame:
    series_files = {
        "pcu484484": raw_data_dir / "fred_pcu484484.csv",
        "wpu057303": raw_data_dir / "fred_wpu057303.csv",
        "ces4348400001": raw_data_dir / "fred_ces4348400001.csv",
    }
    loaded: List[pd.DataFrame] = []

    for series_name, path in series_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing FRED file: {path}")
        df = load_local_fred_series(path, value_column_name=series_name)
        valdidate_fred_series(df, series_name)
        loaded.append(df)
    merged = loaded[0]
    for df in loaded[1:]:
        merged = merged.merge(df, on="date", how="outer")
    
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged

def merge_base_with_fred(base_df: pd.DataFrame, fred_df: pd.DataFrame) -> pd.DataFrame:
    merged = (
        base_df.merge(fred_df, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
        )
    return merged

def validate_merged_panel(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        raise ValueError("Merged panel is empty")
    
    if df["date"].duplicated().any():
        raise ValueError("Merged panel contains duplicate dates")
    
    if not df["date"].is_monotonic_increasing:
        raise ValueError("Merged panel dates are not sorted ascending")
    
    return {
        "row_count"   : len(df),
        "column_count": len(df.columns),
        "start_date"  : df['date'].min(),
        "end_date"    : df['date'].max(),
        "null_counts" : df.isna().sum().to_dict(),
        "columns"     : df.columns.tolist(),
    }

def save_processed_dataset(df: pd.DataFrame, filename: str) -> Path:
    ensure_dir(PROCESSED_DATA_DIR)
    output_path = PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    return output_path