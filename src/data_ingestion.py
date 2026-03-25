from __future__ import annotations
from pathlib import Path
import requests
import pandas as pd

from config import (
    RAW_DATA_DIR,
    BTS_FIGURE4_XLSX_URL,
    BORDER_CROSSINGS_CSV_URL,
    BTS_MONTHLY_TEU_CSV_URL,
    FRED_CSV_URL,
    FRED_SERIES_DEFAULT,
)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents = True, exist_ok=True)

def download_file(url:str, destination:Path, timeout: int=120, overwrite: bool=False) -> Path:
    ensure_dir(destination.parent)
    if destination.exists() and not overwrite:
        print(f"[SKIP] File already exists: {destination}")
        return destination
    
    print(f"[DOWNLOAD] {url}")
    response = requests.get(url, timeout=timeout)
    # response.raise_for_status()
    destination.write_bytes(response.content)
    return destination

def fetch_base_bts_file(overwrite: bool = False) -> Path:
    destination = RAW_DATA_DIR / "Figure4_1.xlsx"
    return download_file(BTS_FIGURE4_XLSX_URL, destination, overwrite=overwrite)

def fetch_border_crossings_file(overwrite: bool = False) -> Path:
    destination = RAW_DATA_DIR / "border_crossings.csv"
    return download_file(BORDER_CROSSINGS_CSV_URL, destination, overwrite=overwrite)

def fetch_bts_monthly_teu_file(overwrite: bool = False) -> Path:
    destination = RAW_DATA_DIR / "bts_monthly_teu.csv"
    return download_file(BTS_MONTHLY_TEU_CSV_URL, destination, overwrite=overwrite)

def fetch_fred_series(series_id: str, overwrite: bool = False) -> Path:
    destination = RAW_DATA_DIR / f"fred_{series_id.lower()}.csv"
    url = FRED_CSV_URL.format(series_id=series_id)
    return download_file(url, destination, overwrite=overwrite)

def fetch_fred_bundle(series_ids: list[str] | None = None, overwrite: bool = False) -> list[Path]:
    if series_ids is None:
        series_ids = FRED_SERIES_DEFAULT
    paths = []
    for series_id in series_ids:
        path = fetch_fred_series(series_id, overwrite=overwrite)
        paths.append(path)
    return paths

def preview_file(path: Path, n:int=5) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return df.head(n)

