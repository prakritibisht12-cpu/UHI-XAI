# Data

The data are **not included** in this repository. They are available from the authors on reasonable request until the associated paper is published.

## `Data.xlsx`: sample points (required)

One worksheet with 164 sample points per year (656 in total) for 1993, 2003, 2013 and 2023. The values were extracted from Landsat-derived rasters at randomly located points.

| Row | Content |
|---|---|
| 1 | The year (1993, 2003, 2013, 2023) in the first cell of each year's block |
| 2 | Column names for each block: `NDVI, NDBI, NDWI, UI, BI, PV, LST, RSEI, UTFVI` |
| 3 onward | One sample per row |

The year blocks sit side by side and are separated by one empty column. Variables:

- **NDVI, NDBI, NDWI, UI, BI:** spectral indices. NDWI is the green–NIR form. BI = NDBI − NDVI.
- **PV:** proportionate vegetation, as supplied. It is used only for the PV audit and the six-index benchmark. Script 03 shows why it is not comparable between years.
- **LST:** land surface temperature in °C (Bisht et al., 2024).
- **RSEI:** the un-inverted PC1 score (RSEI₀). Higher values mean poorer ecological condition.
- **UTFVI:** the Urban Thermal Field Variance Index.

## `weka_predictions_2023.csv`: Weka predictions (optional)

These are the 2023 test-set predictions of the Weka models, which were trained on 1993–2013 with six indices. The file needs four columns:

```
observed,M5P,SVR_PUK,GPR_PUK
33.777,32.554,33.529,34.073
...
```

If this file is missing, script 04 and the Fig. 10 / Fig. S6 panels are skipped.

## Other locations

You can keep the data elsewhere. Point the scripts to it with environment variables:

```bash
export UHI_DATA=/path/to/Data.xlsx
export UHI_WEKA=/path/to/weka_predictions_2023.csv
export UHI_RESULTS=/path/to/output
```
