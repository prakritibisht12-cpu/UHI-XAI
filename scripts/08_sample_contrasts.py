"""LST distribution by year (Fig. 7) and LST anomaly by within-year NDBI/NDVI tercile (Fig. 9).
Terciles are classes of index intensity, NOT LULC classes (no LULC label is stored per sample)."""
import _common  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats

from uhi_xai.config import RESULTS, ensure_dirs
from uhi_xai.data import harmonise, load_samples

ensure_dirs()
D = harmonise(load_samples())
s = D.groupby("Year").agg(n=("LST", "size"), LST_mean=("LST", "mean"), LST_sd=("LST", "std"),
                          LST_median=("LST", "median"), LST_min=("LST", "min"), LST_max=("LST", "max"),
                          anomaly_sd=("dLST", "std"))
s.to_csv(RESULTS / "lst_distribution_by_year.csv")
print(s.round(2).to_string())

rows = []
for v in ["NDBI", "NDVI"]:
    for y, d in D.groupby("Year"):
        d = d.assign(cls=pd.qcut(d[v], 3, labels=["Low", "Middle", "High"]))
        for c, g in d.groupby("cls", observed=True):
            m, sd, n = g.dLST.mean(), g.dLST.std(), len(g)
            h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
            rows.append(dict(Index=v, Year=y, Tercile=c, n=n, index_min=g[v].min(), index_max=g[v].max(),
                             anomaly_mean=m, anomaly_sd=sd, ci_low=m - h, ci_high=m + h))
T = pd.DataFrame(rows)
T.to_csv(RESULTS / "lst_anomaly_by_tercile.csv", index=False)
gap = T.pivot_table(index=["Index", "Year"], columns="Tercile", values="anomaly_mean", observed=True)
gap["High_minus_Low"] = gap["High"] - gap["Low"]
print(gap.round(2).to_string())
