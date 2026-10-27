"""Explainable ML of the harmonised XGBoost model (Figs. 12-13, Figs. S10-S12).

SHAP (TreeSHAP) values and interactions for a model fitted to all four years (year-standardised indices ->
LST anomaly), year-specific SHAP importance, and permutation / grouped permutation importance on held-out years.
SHAP values are model-attributed contributions, not causal effects.
"""
import _common  # noqa: F401
import pickle

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.inspection import permutation_importance

from uhi_xai.config import F4, RESULTS, YEARS, ensure_dirs
from uhi_xai.data import harmonise, load_samples
from uhi_xai.models import xgb_model

ensure_dirs()
D = harmonise(load_samples())
Z = [c + "_z" for c in F4]

xg = xgb_model(max_depth=3, min_child_weight=5).fit(D[Z].values, D.dLST.values)
te = shap.TreeExplainer(xg)
S, SI = te.shap_values(D[Z].values), te.shap_interaction_values(D[Z].values)
imp = pd.Series(np.abs(S).mean(0), index=F4, name="mean_abs_SHAP")
print(imp.round(3).to_string())
inter = pd.DataFrame(np.abs(SI).mean(0), index=F4, columns=F4)
print(inter.round(3).to_string())
imp.to_csv(RESULTS / "shap_global.csv")
inter.to_csv(RESULTS / "shap_interactions.csv")
pickle.dump(dict(model=xg, S=S, SI=SI, X=D[Z].values, Xraw=D[F4].values, year=D.Year.values,
                 base=te.expected_value), open(RESULTS / "shap_pooled.pkl", "wb"))

# year-specific models on raw indices
py = {}
for y in YEARS:
    d = D[D.Year == y]
    m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, min_child_weight=5,
                         random_state=0, n_jobs=2).fit(d[F4].values, d.LST.values)
    py[y] = np.abs(shap.TreeExplainer(m).shap_values(d[F4].values)).mean(0)
py = pd.DataFrame(py, index=F4).T
py.to_csv(RESULTS / "shap_per_year.csv")
print(py.round(3).to_string())

# permutation and grouped permutation importance on each held-out year
groups = {"Vegetation/moisture (NDVI, NDWI)": ["NDVI_z", "NDWI_z"], "Built-up (NDBI, UI)": ["NDBI_z", "UI_z"]}
rng = np.random.default_rng(0)
pim, gim = [], []
for y in YEARS:
    trn, tst = D[D.Year != y], D[D.Year == y]
    m = xgb_model(max_depth=3, min_child_weight=5).fit(trn[Z].values, trn.dLST.values)
    r = permutation_importance(m, tst[Z].values, tst.dLST.values, n_repeats=30, random_state=0,
                               scoring="neg_root_mean_squared_error")
    pim.append(pd.Series(r.importances_mean, index=F4, name=y))
    base = np.sqrt(((m.predict(tst[Z].values) - tst.dLST.values) ** 2).mean())
    for g, cs in groups.items():
        inc = []
        for _ in range(30):
            Xp = tst[Z].copy()
            idx = rng.permutation(len(Xp))
            for c in cs:
                Xp[c] = Xp[c].values[idx]
            inc.append(np.sqrt(((m.predict(Xp.values) - tst.dLST.values) ** 2).mean()) - base)
        gim.append(dict(Year=y, Group=g, dRMSE=np.mean(inc)))
pd.concat(pim, axis=1).to_csv(RESULTS / "permutation_importance.csv")
gim = pd.DataFrame(gim)
gim.to_csv(RESULTS / "grouped_permutation_importance.csv", index=False)
print(gim.pivot(index="Year", columns="Group", values="dRMSE").round(3).to_string())
