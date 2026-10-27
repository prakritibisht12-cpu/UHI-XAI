"""Loading the sample workbook and year-wise harmonisation."""
import openpyxl
import pandas as pd

from .config import COLUMNS, DATA


def load_samples(path=DATA):
    """Read the sample workbook.

    Layout (see data/README.md): row 1 holds the year above each block, row 2 the column names
    (NDVI, NDBI, NDWI, UI, BI, PV, LST, RSEI, UTFVI), and each year occupies one block of columns.
    Returns a long table with one row per sample and a `Year` column.
    """
    ws = openpyxl.load_workbook(path, data_only=True, read_only=True).active
    rows = list(ws.iter_rows(values_only=True))
    years, header = rows[0], rows[1]
    blocks = []
    for start, year in enumerate(years):
        if year is None:
            continue
        names = list(header[start:start + len(COLUMNS)])
        if names != COLUMNS:
            raise ValueError(f"Unexpected columns for {year}: {names}")
        d = pd.DataFrame([r[start:start + len(COLUMNS)] for r in rows[2:]], columns=COLUMNS).dropna(how="all")
        d = d.astype(float)
        d["Year"] = int(year)
        d["SampleID"] = range(1, len(d) + 1)
        blocks.append(d)
    return pd.concat(blocks, ignore_index=True)


def harmonise(df, cols=("NDVI", "NDBI", "NDWI", "UI", "BI", "PV")):
    """Add within-year z-scores (<col>_z) and the within-year LST anomaly (dLST).

    Standardisation uses only the index values of each year; the target year's LST is never used
    to transform its predictors.
    """
    d = df.copy()
    for c in list(cols) + ["LST"]:
        g = d.groupby("Year")[c]
        d[c + "_z"] = (d[c] - g.transform("mean")) / g.transform("std")
    d["dLST"] = d.LST - d.groupby("Year").LST.transform("mean")
    return d
