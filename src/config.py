from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

#https://www.bts.gov/sites/bts.dot.gov/files/Figure4_1.xlsx
BTS_FIGURE4_XLSX_URL = "https://www.bts.gov/sites/bts.dot.gov/files/Figure4_1.xlsx"
BORDER_CROSSINGS_CSV_URL = "https://data.transportation.gov/api/views/keg4-3bc2/rows.csv?accessType=DOWNLOAD"
BTS_MONTHLY_TEU_CSV_URL = "https://data.bts.gov/api/views/rd72-aq8r/rows.csv?accessType=DOWNLOAD"
NOAA_STORMS_DIR_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

FRED_SERIES_DEFAULT = [
    "PCU484484",
    "WPU057303",
    "CES4348400001",
]

BASE_DATASET_PATH = PROCESSED_DATA_DIR / "base_freight_index.csv"

#https://cmt3.research.microsoft.com/iGET2026/Track/1/Submission/Create