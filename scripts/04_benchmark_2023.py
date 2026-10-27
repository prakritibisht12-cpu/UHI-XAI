"""Six-index temporal benchmark: train on 1993-2013, test on 2023 (n = 164).

(a) Metrics of the original Weka models from their exported predictions (if data/weka_predictions_2023.csv exists).
(b) Python re-implementation of Weka SMOreg/PUK (C = 4, omega = 0.48, sigma = 1) with six, five (no PV)
    and four (non-redundant) indices.
(c) KernelSHAP of the six-index SVR-PUK on the 2023 test set (Fig. S7b).
"""
import _common  # noqa: F401
import pickle

import numpy as np
import pandas as pd
import shap

from uhi_xai.config import F4, F6, RESULTS, WEKA, ensure_dirs
from uhi_xai.data import load_samples
from uhi_xai.models import metrics, svr_puk

ensure_dirs()
A = load_samples()
tr, te = A[A.Year < 2023], A[A.Year == 2023]

if WEKA.exists():
    W = pd.read_csv(WEKA)
    rows = []
    for m in ["M5P", "SVR_PUK", "GPR_PUK"]:
        d = metrics(W.observed, W[m])
        d["Model"] = m
        rows.append(d)
    W = pd.DataFrame(rows)
    W.to_csv(RESULTS / "table_7_weka_2023.csv", index=False)
    print(W.round(3).to_string())
else:
    print(f"[skip] {WEKA} not found - Weka metrics not computed (see data/README.md).")

rows, preds = [], {}
for name, cols in [("Six indices", F6), ("Without PV", ["NDVI", "NDBI", "NDWI", "UI", "BI"]), ("Four indices", F4)]:
    m = svr_puk().fit(tr[cols].values, tr.LST.values)
    p = m.predict(te[cols].values)
    d = metrics(te.LST.values, p)
    d["FeatureSet"] = name
    rows.append(d)
    preds[name] = p
    if name == "Six indices":
        svr6 = m
S = pd.DataFrame(rows)
S.to_csv(RESULTS / "svr_puk_python_2023.csv", index=False)
print(S.round(3).to_string())

bg = shap.kmeans(tr[F6].values, 30)
ex = shap.KernelExplainer(svr6.predict, bg)
sv = ex.shap_values(te[F6].values, nsamples=400, silent=True)
imp = pd.Series(np.abs(sv).mean(0), index=F6, name="mean_abs_SHAP")
imp.to_csv(RESULTS / "svr_puk_kernelshap_2023.csv")
print(imp.round(3).to_string())
pickle.dump(dict(shap=sv, X=te[F6].values), open(RESULTS / "svr_puk_kernelshap.pkl", "wb"))
