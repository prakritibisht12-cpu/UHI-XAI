"""Bootstrap (B = 300) 95% intervals of each index's share of total mean |SHAP| in year-specific models
(Supplementary Table S7, Fig. S13), with a linear-SHAP (MLR) counterpart."""
import _common  # noqa: F401
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.linear_model import LinearRegression

from uhi_xai.config import F4, RESULTS, ensure_dirs
from uhi_xai.data import load_samples

ensure_dirs()
A = load_samples()
B = 300
rng = np.random.default_rng(42)


def xgbm():
    return xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, min_child_weight=5,
                            random_state=0, n_jobs=2)


rows, lin = [], []
for y, d in A.groupby("Year"):
    X, t, n = d[F4].values, d.LST.values, len(d)
    for b in range(B + 1):                     # b = 0 is the full-sample estimate
        idx = np.arange(n) if b == 0 else rng.integers(0, n, n)
        s = np.abs(shap.TreeExplainer(xgbm().fit(X[idx], t[idx])).shap_values(X)).mean(0)
        rows.append(dict(Year=y, b=b, **dict(zip(F4, s / s.sum()))))
        lm = LinearRegression().fit(X[idx], t[idx])
        c = np.abs(lm.coef_) * np.abs(X - X.mean(0)).mean(0)
        lin.append(dict(Year=y, b=b, **dict(zip(F4, c / c.sum()))))
R, L = pd.DataFrame(rows), pd.DataFrame(lin)
R.to_csv(RESULTS / "bootstrap_shap_shares.csv", index=False)
L.to_csv(RESULTS / "bootstrap_linear_shares.csv", index=False)


def summary(T, name):
    out = []
    for y, g in T.groupby("Year"):
        p, bs = g[g.b == 0].iloc[0], g[g.b > 0]
        for f in F4 + ["Built-up (NDBI+UI)"]:
            v = (bs.NDBI + bs.UI) if f.startswith("Built") else bs[f]
            pv = (p.NDBI + p.UI) if f.startswith("Built") else p[f]
            out.append(dict(Model=name, Year=y, Index=f, Share=pv, CI_low=v.quantile(.025), CI_high=v.quantile(.975)))
    return pd.DataFrame(out)


S = pd.concat([summary(R, "XGBoost TreeSHAP"), summary(L, "Linear SHAP (MLR)")])
S.to_csv(RESULTS / "table_S7_bootstrap_summary.csv", index=False)
print(S[S.Model == "XGBoost TreeSHAP"].round(2).to_string())
top = R[R.b > 0][F4].idxmax(axis=1)
print(pd.crosstab(R[R.b > 0].Year, top, normalize="index").round(2).to_string())
