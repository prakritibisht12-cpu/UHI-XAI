# UHI-XAI

**Explainable, temporally validated machine learning of land surface temperature (LST) in Dehradun, India, 1993–2023**

This repository holds the analysis code for the manuscript:

> Bisht, P., Deswal, S., Pal, M. *Modelling thermal field variation and its relationship with land use/land cover and land use indices using statistical and machine learning approaches: a case study of Dehradun, India.* Manuscript in preparation.

It covers every statistical, machine-learning and explainability analysis in the paper. The Landsat processing, LULC classification and map production were done in ArcGIS and are not included. The data are not distributed with the code (see [`data/README.md`](data/README.md)).

## What the analysis does

LST and four spectral indices (NDVI, NDBI, NDWI, UI) were sampled at 164 random points in each of four Landsat years (1993, 2003, 2013, 2023). The code then works through six steps:

1. **Screens the predictors.**
   - It shows that BI = NDBI − NDVI exactly.
   - It shows that the supplied PV equals (NDVI − c)², where the offset c changes from year to year.
   - It uses adversarial validation to measure how far the predictors shift between years.
2. **Reproduces the original benchmark.** This is a six-index model trained on 1993–2013 and tested on 2023. The script also evaluates the Weka predictions and re-implements SVR with the Pearson VII (PUK) kernel in Python.
3. **Tests temporal transferability** with leave-one-year-out (LOYO) validation, comparing raw indices that predict absolute LST with year-standardised indices that predict the within-year LST anomaly. The models are MLR and XGBoost.
4. **Explains the harmonised model** with TreeSHAP values, SHAP interaction values, partial dependence and ICE plots, and accumulated local effects (ALE). It also computes permutation importance for single indices and for groups of indices.
5. **Quantifies uncertainty.** It bootstraps the SHAP importance shares (B = 300) and builds temporal split-conformal 90% prediction intervals.
6. **Runs descriptive analyses:** correlations, two-index within-year MLR, and LST contrasts across within-year index terciles.

SHAP values are reported as *model-attributed* contributions, not causal effects. No extra algorithms were added beyond those in the paper.

## Key results

| Analysis | Result |
|---|---|
| Six-index benchmark (Weka, 2023 test, n = 164) | SVR-PUK R = 0.713, RMSE = 1.443 °C. Without PV, the Python re-implementation drops to R = 0.586. |
| Adversarial validation AUC (2023 vs 1993–2013) | Raw six indices 1.00; raw four indices 0.86; year-standardised 0.62 |
| LOYO with raw indices | R = 0.12–0.71 |
| LOYO with year-standardised indices (MLR / XGBoost) | R = 0.70–0.80, RMSE = 0.89–1.75 °C. This is close to the within-year 10-fold CV score (mean R 0.733). |
| Global mean \|SHAP\| | NDBI 0.50, UI 0.37, NDVI 0.36, NDWI 0.20 °C |
| Grouped permutation, increase in RMSE | Built-up (NDBI + UI) 0.51–0.63 °C; vegetation (NDVI + NDWI) 0.19–0.29 °C |
| Conformal 90% intervals | Half-width about ±2.1 °C (MLR); coverage 0.76–1.00 by year |

## How to run

```bash
git clone https://github.com/prakritibisht12-cpu/dehradun-uhi-xai.git
cd dehradun-uhi-xai
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put Data.xlsx (and optionally weka_predictions_2023.csv) in data/
bash run_all.sh                                         # or run the scripts one by one
```

The full run takes about 5 minutes on a laptop; the bootstrap step (07) takes most of that time. Tables and pickles are written to `results/` and figures to `results/figures/` as PDF and 600-dpi PNG.

| Script | Output |
|---|---|
| `01_descriptive_and_correlation.py` | Sample statistics, Pearson correlations |
| `02_mlr_within_year.py` | Two-index MLR models |
| `03_predictor_screening.py` | BI identity, PV audit, adversarial AUC |
| `04_benchmark_2023.py` | Weka metrics, Python SVR-PUK, KernelSHAP |
| `05_loyo_transferability.py` | LOYO metrics and predictions, within-year CV, conformal intervals |
| `06_explainability.py` | TreeSHAP, interactions, per-year SHAP, permutation importance |
| `07_bootstrap_shap_shares.py` | Bootstrap SHAP shares |
| `08_sample_contrasts.py` | LST distribution, tercile contrasts |
| `09_figures.py` | All figures |

## Repository layout

```
uhi_xai/        config (paths, constants), data loading and harmonisation, models and metrics
scripts/        numbered analysis scripts 01–09
data/           (not tracked) put Data.xlsx here; see data/README.md
results/        (not tracked) created when the scripts run
```

## Citation

If you use this code, please cite it using [`CITATION.cff`](CITATION.cff). Once the paper is published, please cite the paper as well.

## License

The code is released under the MIT License. The data are not covered by this licence.
