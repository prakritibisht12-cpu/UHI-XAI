"""Predictor screening: BI identity, PV construction (PV = (NDVI - c)^2) and adversarial validation."""
import _common  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict

from uhi_xai.config import F4, F6, RESULTS, ensure_dirs
from uhi_xai.data import harmonise, load_samples

ensure_dirs()
A = load_samples()
D = harmonise(A)

print("max |BI - (NDBI - NDVI)| =", float(np.abs(A.BI - (A.NDBI - A.NDVI)).max()))

pv = []
for y, d in A.groupby("Year"):
    best = None
    for sign in (-1, 1):                      # c = NDVI -/+ sqrt(PV); keep the branch that is constant
        c = d.NDVI - sign * np.sqrt(d.PV)
        if best is None or c.std() < best[1]:
            best = (c.median(), c.std())
    c = best[0]
    pv.append(dict(Year=y, c=c, max_abs_residual=float(np.abs(d.PV - (d.NDVI - c) ** 2).max()),
                   PV_min=d.PV.min(), PV_max=d.PV.max()))
pv = pd.DataFrame(pv)
pv.to_csv(RESULTS / "pv_audit.csv", index=False)
print(pv.round(4).to_string())

adv = []
yb = (D.Year == 2023).astype(int).values
for cols, label in [(F6, "Raw, six indices"), (F4, "Raw, four indices"), ([c + "_z" for c in F4], "Year-standardised, four indices")]:
    p = cross_val_predict(GradientBoostingClassifier(random_state=0), D[cols].values, yb,
                          cv=KFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    adv.append(dict(Features=label, AUC=roc_auc_score(yb, p)))
adv = pd.DataFrame(adv)
adv.to_csv(RESULTS / "adversarial_validation.csv", index=False)
print(adv.round(3).to_string())
