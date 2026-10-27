"""Sample statistics of the indices (Table S5), Pearson correlations (Fig. 8) and collinearity checks."""
import _common  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from uhi_xai.config import RESULTS, ensure_dirs
from uhi_xai.data import load_samples

ensure_dirs()
A = load_samples()
print(A.groupby("Year").size().rename("n").to_string())

rows = []
for y, d in A.groupby("Year"):
    for c in ["NDVI", "NDBI", "NDWI", "UI", "BI", "LST"]:
        s = d[c]
        rows.append(dict(Variable=c, Year=y, Max=s.max(), Min=s.min(), Mean=s.mean(), SD=s.std(ddof=1), n=len(s)))
pd.DataFrame(rows).to_csv(RESULTS / "table_S5_sample_statistics.csv", index=False)

V = ["NDVI", "NDBI", "NDWI", "UI", "BI", "RSEI", "UTFVI", "LST"]
corr = []
for y, d in A.groupby("Year"):
    for i, a in enumerate(V):
        for b in V[i + 1:]:
            r, p = pearsonr(d[a], d[b])
            corr.append(dict(Year=y, Var1=a, Var2=b, r=r, p=p))
C = pd.DataFrame(corr)
C.to_csv(RESULTS / "correlations.csv", index=False)
print(C[C.Var2 == "LST"].pivot(index="Var1", columns="Year", values="r").round(2).to_string())

for y, d in A.groupby("Year"):
    print(y, "r(NDBI,UI) = %.3f  r(NDVI,NDWI) = %.3f" % (d.NDBI.corr(d.UI), d.NDVI.corr(d.NDWI)))
