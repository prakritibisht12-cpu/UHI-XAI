"""Leave-one-year-out (LOYO) temporal transferability (Table 8, Fig. 11, Figs. S8-S9).

Set-ups: raw indices -> absolute LST, and year-standardised indices -> within-year LST anomaly.
Models: MLR (linear reference) and XGBoost (tuned by LOYO grid search inside the training years).
Also: within-year 10-fold CV reference and temporal split-conformal prediction intervals.
"""
import _common  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

from uhi_xai.config import F4, RESULTS, YEARS, ensure_dirs
from uhi_xai.data import harmonise, load_samples
from uhi_xai.models import fit_xgb_tuned, metrics, mlr, xgb_model

ensure_dirs()
D = harmonise(load_samples())
Z = [c + "_z" for c in F4]
SETUPS = {"Raw indices -> absolute LST": (F4, "LST", False),
          "Year-standardised indices -> LST anomaly": (Z, "dLST", True)}

met, pred = [], []
for setup, (X, target, anomaly) in SETUPS.items():
    for ty in YEARS:
        tr, te = D[D.Year != ty], D[D.Year == ty]
        for name in ["MLR", "XGBoost"]:
            if name == "MLR":
                m = mlr().fit(tr[X].values, tr[target].values)
            else:
                m, _ = fit_xgb_tuned(tr[X].values, tr[target].values, tr.Year.values)
            p = m.predict(te[X].values)
            p_abs = p + te.LST.mean() if anomaly else p          # anomaly RMSE == absolute RMSE given the scene mean
            d = metrics(te.LST.values, p_abs)
            d.update(Setup=setup, TestYear=ty, Model=name)
            met.append(d)
            pred.append(pd.DataFrame(dict(Setup=setup, TestYear=ty, Model=name, SampleID=te.SampleID.values,
                                          LST_obs=te.LST.values, LST_pred=p_abs,
                                          LSTanom_obs=te.dLST.values, LSTanom_pred=p_abs - te.LST.mean())))
M = pd.DataFrame(met)
M.to_csv(RESULTS / "table_8_loyo.csv", index=False)
pd.concat(pred).to_csv(RESULTS / "loyo_predictions.csv", index=False)
print(M.pivot_table(index=["Setup", "Model"], columns="TestYear", values=["R", "RMSE"]).round(3).to_string())

# within-year 10-fold CV reference (no temporal transfer)
ref = []
for y in YEARS:
    d = D[D.Year == y]
    for name, est in [("MLR", mlr()), ("XGBoost", xgb_model(max_depth=3, min_child_weight=5, reg_lambda=10))]:
        p = cross_val_predict(est, d[F4].values, d.LST.values, cv=KFold(10, shuffle=True, random_state=0))
        r = metrics(d.LST.values, p)
        r.update(Year=y, Model=name)
        ref.append(r)
ref = pd.DataFrame(ref)
ref.to_csv(RESULTS / "within_year_cv.csv", index=False)
print(ref.groupby("Model")[["R", "RMSE"]].mean().round(3).to_string())

# temporal split-conformal intervals (harmonised set-up)
def est_for(name):
    return mlr() if name == "MLR" else xgb_model(max_depth=3, min_child_weight=5)

conf = []
for name in ["MLR", "XGBoost"]:
    for y in YEARS:
        others = [v for v in YEARS if v != y]
        cal = []
        for c in others:                                      # LOYO residuals among the three calibration years
            trn, cl = D[D.Year.isin([o for o in others if o != c])], D[D.Year == c]
            e = est_for(name).fit(trn[Z].values, trn.dLST.values)
            cal += list(np.abs(e.predict(cl[Z].values) - cl.dLST.values))
        cal = np.array(cal)
        n = len(cal)
        trn, tst = D[D.Year != y], D[D.Year == y]
        p = est_for(name).fit(trn[Z].values, trn.dLST.values).predict(tst[Z].values)
        for a in (0.1, 0.2):
            q = np.quantile(cal, min(1, np.ceil((n + 1) * (1 - a)) / n))
            conf.append(dict(Model=name, TestYear=y, Nominal=1 - a, HalfWidth=q,
                             Coverage=(np.abs(p - tst.dLST.values) <= q).mean()))
conf = pd.DataFrame(conf)
conf.to_csv(RESULTS / "conformal_intervals.csv", index=False)
print(conf[conf.Nominal == 0.9].round(3).to_string())
