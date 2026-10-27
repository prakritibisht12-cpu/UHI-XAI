"""Two-index within-year MLR models (Supplementary Table S6, Fig. S5). In-sample fit, n = 164 per year."""
import _common  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from uhi_xai.config import RESULTS, ensure_dirs
from uhi_xai.data import load_samples

ensure_dirs()
A = load_samples()
pairs = [("NDVI", "NDWI"), ("NDVI", "NDBI"), ("NDVI", "UI"), ("NDWI", "UI"), ("NDWI", "NDBI")]
rows, preds = [], []
for y, d in A.groupby("Year"):
    for a, b in pairs:
        X, t = d[[a, b]].values, d.LST.values
        m = LinearRegression().fit(X, t)
        p = m.predict(X)
        e = p - t
        rows.append(dict(Year=y, Indices=f"{a}–{b}", b0=m.intercept_, b1=m.coef_[0], b2=m.coef_[1],
                         R=np.corrcoef(p, t)[0, 1], R2=1 - (e ** 2).sum() / ((t - t.mean()) ** 2).sum(),
                         RMSE=np.sqrt((e ** 2).mean()), MAE=np.abs(e).mean(), n=len(t)))
        if (a, b) == ("NDVI", "NDBI"):
            preds.append(pd.DataFrame(dict(Year=y, Observed=t, Fitted=p)))
T = pd.DataFrame(rows)
assert (T.RMSE >= T.MAE).all() and np.allclose(T.R ** 2, T.R2)
T.to_csv(RESULTS / "table_S6_mlr.csv", index=False)
pd.concat(preds).to_csv(RESULTS / "mlr_ndvi_ndbi_fitted.csv", index=False)
print(T.round(3).to_string())
