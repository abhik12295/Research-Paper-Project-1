from __future__ import annotations
from pathlib import Path
from typing import Dict
import pandas as pd
from src.config import BASE_DATASET_PATH, PROCESSED_DATA_DIR
from typing import cast

def ensure_dir(path: Path) -> None:
    path.mkdir(parents = True, exist_ok= True)

def to_month_start(series: pd.Series) -> pd.Series:
    return cast(pd.Series, pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp())

def inspect_excel_workbook(xlsx_path: Path) -> Dict[str, list]:
    excel_file = pd.ExcelFile(xlsx_path)
    return {
        "sheet_names" : excel_file.sheet_names
    }

def load_bts_figure4_base(xlsx_path: Path) -> pd.DataFrame:
    cass = pd.read_excel(xlsx_path, sheet_name="CASS", header=10)
    tsi = pd.read_excel(xlsx_path, sheet_name="TSI", header=3)

    cass = cass.rename(columns={
        "observation_date": "date",
        "FRGSHPUSM649NCIS": "cass_shipments",
        "FRGEXPUSM649NCIS": "cass_expenditures",
    })

    #TSI-Freight
    tsi = tsi.rename(columns={
        "Date": "date",
        "TSI-Freight": "tsi_freight",
    })

    required_cass_cols = ["date", "cass_shipments", "cass_expenditures"]
    required_tsi_cols = ["date", "tsi_freight"]

    missing_cass = [c for c in required_cass_cols if c not in cass.columns]
    missing_tsi = [c for c in required_tsi_cols if c not in tsi.columns]

    if missing_cass:
        raise ValueError(f"Missing expected CASS columns: {missing_cass}")
    if missing_tsi:
        raise ValueError(f"Missing expected TSI columns: {missing_tsi}")

    cass = cass[required_cass_cols].copy()
    tsi = tsi[required_tsi_cols].copy()

    cass["date"] = pd.to_datetime(cass["date"], errors = "coerce")
    tsi["date"] = pd.to_datetime(tsi["date"], errors = "coerce")

    cass["cass_shipments"] = pd.to_numeric(cass["cass_shipments"], errors="coerce")
    cass["cass_expenditures"] = pd.to_numeric(cass["cass_expenditures"], errors="coerce")
    tsi["tsi_freight"] = pd.to_numeric(tsi["tsi_freight"], errors="coerce")

    cass = cass.dropna(subset=["date"])
    tsi = tsi.dropna(subset=["date"])

    cass["date"] = to_month_start(cass["date"])
    tsi["date"] = to_month_start(tsi["date"])

    base = (
        cass.merge(tsi, on="date", how="inner")
        .sort_values("date")
        .drop_duplicates(subset=["date"])
        .reset_index(drop=True)
    )

    return base

def validate_base_dataset(df: pd.DataFrame) -> Dict[str, object]:
    expected_cols = ["date", "cass_shipments", "cass_expenditures", "tsi_freight"]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Base dataset missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("Base Dataset is empty.")
    if df["date"].isna().any():
        raise ValueError("Base dataset contains null dates.")
    
    if df["date"].duplicated().any():
        dupes = df[df["date"].duplicated(keep=False)].sort_values("date")
        raise ValueError(f"Duplicate monthly dates found:\n {dupes}")
    
    if not df["date"].is_monotonic_increasing:
        raise ValueError("Base dataset dates are not sorted ascending.")
    
    summary = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "start_date": df["date"].min(),
        "end_date": df["date"].max(),
        "null_counts": df.isna().sum().to_dict(),
        "columns": df.columns.tolist(),
    }
    return summary

def save_base_dataset(df: pd.DataFrame, output_path: Path=BASE_DATASET_PATH) -> Path:
    ensure_dir(PROCESSED_DATA_DIR)
    df.to_csv(output_path, index=False)
    return output_path