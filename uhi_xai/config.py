"""Paths and constants. Override the data and output locations with environment variables:

    UHI_DATA     path to the sample workbook (default: data/Data.xlsx)
    UHI_WEKA     path to the Weka 2023 prediction table (default: data/weka_predictions_2023.csv)
    UHI_RESULTS  output folder for tables, pickles and figures (default: results/)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("UHI_DATA", ROOT / "data" / "Data.xlsx"))
WEKA = Path(os.environ.get("UHI_WEKA", ROOT / "data" / "weka_predictions_2023.csv"))
RESULTS = Path(os.environ.get("UHI_RESULTS", ROOT / "results"))
FIGURES = RESULTS / "figures"

YEARS = [1993, 2003, 2013, 2023]
F4 = ["NDVI", "NDBI", "NDWI", "UI"]                 # non-redundant indices used for explanation
F6 = ["NDVI", "NDBI", "NDWI", "UI", "BI", "PV"]     # original six-index benchmark
COLUMNS = ["NDVI", "NDBI", "NDWI", "UI", "BI", "PV", "LST", "RSEI", "UTFVI"]
SEED = 0


def ensure_dirs():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
