"""Publication figures from the result files of scripts 01-08 (vector PDF + 600-dpi PNG).

Map figures (LULC, index, UHI/UHS, UTFVI/RSEI and Gi* maps) were produced in ArcGIS and are not part of
this repository. Figures that need the Weka prediction table are skipped if it is absent.
"""
import _common  # noqa: F401
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.inspection import partial_dependence
from statsmodels.nonparametric.smoothers_lowess import lowess

from uhi_xai.config import F4, F6, FIGURES, RESULTS, WEKA, YEARS, ensure_dirs
from uhi_xai.data import harmonise, load_samples

ensure_dirs()
MM = 1 / 25.4
W2, W15, W1 = 190 * MM, 140 * MM, 90 * MM
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 600})
INK, MUTED, REF, GREY, BLUE = "#000000", "#4d4d4d", "#7f7f7f", "#9a9a9a", "#0072B2"
YC = {1993: "#0072B2", 2003: "#D55E00", 2013: "#009E73", 2023: "#E69F00"}   # Okabe-Ito
YM = {1993: "o", 2003: "s", 2013: "^", 2023: "D"}
DIV = LinearSegmentedColormap.from_list("div", ["#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B"])
DIVS = LinearSegmentedColormap.from_list("divs", ["#2166AC", "#80B1D3", "#C8C8C8", "#F09A73", "#B2182B"])
SCENE_MEAN = {1993: 32.16, 2003: 27.19, 2013: 35.71, 2023: 35.87}   # whole-scene LST means (Bisht et al., 2024)


def lab(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURES / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", name)


A = load_samples()
D = harmonise(A)

# ---- Fig. 7: LST distribution, absolute and anomaly
fig, axs = plt.subplots(1, 2, figsize=(W15, 62 * MM))
for a, col, yl in [(axs[0], "LST", "LST (°C)"), (axs[1], "dLST", "LST anomaly (°C)")]:
    data = [D.loc[D.Year == y, col].values for y in YEARS]
    vp = a.violinplot(data, positions=range(4), widths=0.75, showextrema=False)
    for b, y in zip(vp["bodies"], YEARS):
        b.set_facecolor(YC[y]); b.set_edgecolor("none"); b.set_alpha(0.35)
    a.boxplot(data, positions=range(4), widths=0.18, patch_artist=True, showfliers=False,
              boxprops=dict(facecolor="white", edgecolor=INK, lw=0.6), medianprops=dict(color=INK, lw=0.9),
              whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    a.set_xticks(range(4)); a.set_xticklabels(YEARS); a.set_xlabel("Year"); a.set_ylabel(yl)
axs[0].scatter(range(4), [SCENE_MEAN[y] for y in YEARS], marker="D", s=14, color=INK, zorder=5, label="Scene mean LST")
axs[0].legend(loc="upper left", fontsize=7)
axs[1].axhline(0, color=REF, lw=0.6, ls="--")
lab(axs[0], "(a)", x=-0.14); lab(axs[1], "(b)", x=-0.14)
fig.tight_layout(w_pad=2.0); save(fig, "Fig07_LST_distribution_absolute_anomaly")

# ---- Fig. 8: correlation heatmaps
V = ["NDVI", "NDBI", "NDWI", "UI", "BI", "RSEI", "UTFVI", "LST"]
VL = ["NDVI", "NDBI", "NDWI", "UI", "BI", "RSEI$_0$", "UTFVI", "LST"]
fig, axs = plt.subplots(2, 2, figsize=(W2, 165 * MM))
for k, (a, (y, d)) in enumerate(zip(axs.flat, A.groupby("Year"))):
    M = d[V].corr().values
    Mm = np.ma.array(M, mask=np.triu(np.ones_like(M, bool), 0))[1:, :-1]
    im = a.imshow(Mm, cmap=DIV, vmin=-1, vmax=1)
    for i in range(1, len(V)):
        for j in range(i):
            a.text(j, i - 1, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(M[i, j]) > 0.8 else INK)
    a.set_xticks(range(len(V) - 1)); a.set_xticklabels(VL[:-1], rotation=45, ha="right")
    a.set_yticks(range(len(V) - 1)); a.set_yticklabels(VL[1:])
    a.spines[:].set_visible(False); a.tick_params(length=0); a.set_title(f"{y} (n = {len(d)})", loc="left", pad=3)
    lab(a, f"({'abcd'[k]})", x=-0.12)
fig.tight_layout(rect=(0, 0, 0.92, 1), h_pad=1.5)
cb = fig.colorbar(im, cax=fig.add_axes([0.935, 0.3, 0.012, 0.4])); cb.set_label("Pearson r")
save(fig, "Fig08_correlation_heatmaps")

# ---- Fig. 9: LST anomaly by tercile
T = pd.read_csv(RESULTS / "lst_anomaly_by_tercile.csv")
fig, axs = plt.subplots(1, 2, figsize=(W15, 64 * MM), sharey=True)
off = dict(zip(YEARS, [-0.24, -0.08, 0.08, 0.24]))
for a, v, t in [(axs[0], "NDBI", "(a)"), (axs[1], "NDVI", "(b)")]:
    for y in YEARS:
        d = T[(T.Index == v) & (T.Year == y)].set_index("Tercile").loc[["Low", "Middle", "High"]]
        a.errorbar(np.arange(3) + off[y], d.anomaly_mean, yerr=[d.anomaly_mean - d.ci_low, d.ci_high - d.anomaly_mean],
                   fmt=YM[y], color=YC[y], ms=4, mew=0.6, mec=INK, elinewidth=0.8, capsize=1.8, label=str(y))
    a.axhline(0, color=REF, lw=0.6, ls="--")
    a.set_xticks(range(3)); a.set_xticklabels(["Low", "Middle", "High"]); a.set_xlabel(f"Within-year {v} tercile")
    lab(a, t, x=-0.10)
axs[0].set_ylabel("LST anomaly (°C), mean ± 95% CI")
axs[1].legend(loc="upper right", ncol=2, fontsize=7)
fig.tight_layout(w_pad=1.5); save(fig, "Fig09_LST_anomaly_by_index_tercile")

# ---- Fig. 10 and Fig. S6: Weka benchmark (needs the Weka prediction table)
if WEKA.exists():
    W = pd.read_csv(WEKA)
    obs = W.observed.values
    mods = [("M5P model tree", W.M5P.values, "#0072B2"), ("SVR-PUK", W.SVR_PUK.values, "#D55E00"),
            ("GPR-PUK", W.GPR_PUK.values, "#009E73")]
    fig, axs = plt.subplots(2, 2, figsize=(W15, W15 * 0.98))
    lo, hi = np.floor(min(obs.min(), *(p.min() for _, p, _ in mods))) - 1, np.ceil(max(obs.max(), *(p.max() for _, p, _ in mods))) + 1
    for k, (n, p, c) in enumerate(mods):
        a = axs.flat[k]; e = p - obs
        a.plot([lo, hi], [lo, hi], color=REF, lw=0.7, ls="--")
        a.scatter(obs, p, s=9, color=c, edgecolor="white", linewidth=0.3)
        a.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal", xlabel="Observed LST (°C)", ylabel="Predicted LST (°C)")
        a.text(0.04, 0.97, f"{n}\nR = {np.corrcoef(obs, p)[0, 1]:.3f}\nRMSE = {np.sqrt((e ** 2).mean()):.3f} °C\nMAE = {abs(e).mean():.3f} °C",
               transform=a.transAxes, va="top", fontsize=7.5)
        lab(a, f"({'abc'[k]})")
    a = axs.flat[3]
    bp = a.boxplot([p - obs for _, p, _ in mods], widths=0.5, patch_artist=True, medianprops=dict(color=INK))
    for b, (_, _, c) in zip(bp["boxes"], mods):
        b.set(facecolor=c, alpha=0.85)
    a.axhline(0, color=REF, lw=0.7, ls="--"); a.set_xticks([1, 2, 3]); a.set_xticklabels(["M5P", "SVR-PUK", "GPR-PUK"])
    a.set_ylabel("Prediction error (°C)"); a.set_box_aspect(1); lab(a, "(d)")
    fig.tight_layout(); save(fig, "Fig10_ML_benchmark_2023_test")

    fig, axs = plt.subplots(1, 3, figsize=(W2, 62 * MM), sharey=True)
    for k, (a, (n, p, c)) in enumerate(zip(axs, mods)):
        e = p - obs; lw_ = lowess(e, obs, frac=0.5)
        a.axhline(0, color=REF, lw=0.6, ls="--"); a.scatter(obs, e, s=8, color=c, lw=0.2, edgecolor="white")
        a.plot(lw_[:, 0], lw_[:, 1], color=INK, lw=1.0)
        a.text(0.04, 0.96, f"{n}\nbias = {e.mean():.3f} °C".replace("-", "−"), transform=a.transAxes, va="top", fontsize=7.2)
        a.set_xlabel("Observed LST (°C)"); lab(a, f"({'abc'[k]})", x=-0.04)
    axs[0].set_ylabel("Residual, predicted − observed (°C)")
    fig.tight_layout(w_pad=0.8); save(fig, "FigS06_Weka_2023_residuals")

# ---- Fig. 11: LOYO transferability
R = pd.read_csv(RESULTS / "table_8_loyo.csv")
Wy = pd.read_csv(RESULTS / "within_year_cv.csv"); Wy = Wy[Wy.Model == "MLR"].set_index("Year").loc[YEARS]
RAW, HAR = "Raw indices -> absolute LST", "Year-standardised indices -> LST anomaly"
ST = {RAW: dict(c=GREY, fill=False), HAR: dict(c=BLUE, fill=True)}
MK = {"MLR": ("o", -0.09), "XGBoost": ("s", 0.09)}


def draw(a, met):
    for setup, st in ST.items():
        for m, (mk, dx) in MK.items():
            d = R[(R.Setup == setup) & (R.Model == m)].set_index("TestYear").loc[YEARS]
            a.plot(np.arange(4) + dx, d[met].values, color=st["c"], lw=0.7, alpha=0.8, zorder=2)
            a.scatter(np.arange(4) + dx, d[met].values, marker=mk, s=22, zorder=3, linewidths=0.9,
                      facecolors=st["c"] if st["fill"] else "white", edgecolors=st["c"])
    a.scatter(np.arange(4), Wy[met].values, marker="_", s=260, color=INK, linewidths=1.1, zorder=4)
    a.set_xticks(range(4)); a.set_xticklabels(YEARS); a.set_xlim(-0.45, 3.45)


fig = plt.figure(figsize=(W2, 76 * MM))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.6], hspace=0.08, wspace=0.28, left=0.07, right=0.99, top=0.95, bottom=0.25)
a0 = fig.add_subplot(gs[:, 0]); draw(a0, "R"); a0.set_ylim(0, 1); a0.set_ylabel("Correlation coefficient (R)"); a0.set_xlabel("Held-out year")
lab(a0, "(a)", x=-0.075, y=1.03)
au = fig.add_subplot(gs[0, 1]); al = fig.add_subplot(gs[1, 1], sharex=au)
draw(au, "RMSE"); draw(al, "RMSE")
rmax = R.RMSE.max()
au.set_ylim(max(3.5, R[R.RMSE > 3.2].RMSE.min() - 1) if (R.RMSE > 3.2).any() else 3.5, rmax + 0.9); al.set_ylim(0, 3.2)
au.spines["bottom"].set_visible(False); au.tick_params(axis="x", bottom=False, labelbottom=False)
au.plot([-0.012, 0.012], [-0.02, 0.02], transform=au.transAxes, color=INK, clip_on=False, lw=0.6)
al.plot([-0.012, 0.012], [0.985, 1.015], transform=al.transAxes, color=INK, clip_on=False, lw=0.6)
al.set_xlabel("Held-out year"); fig.text(0.535, 0.60, "RMSE (°C)", rotation=90, va="center", ha="center")
lab(au, "(b)", x=-0.075, y=1.06)
h = [Line2D([], [], color=GREY, marker="o", mfc="white", mec=GREY, lw=0.7, ms=5, label="Raw indices → absolute LST, MLR"),
     Line2D([], [], color=GREY, marker="s", mfc="white", mec=GREY, lw=0.7, ms=5, label="Raw indices → absolute LST, XGBoost"),
     Line2D([], [], color=BLUE, marker="o", mfc=BLUE, mec=BLUE, lw=0.7, ms=5, label="Year-standardised → LST anomaly, MLR"),
     Line2D([], [], color=BLUE, marker="s", mfc=BLUE, mec=BLUE, lw=0.7, ms=5, label="Year-standardised → LST anomaly, XGBoost"),
     Line2D([], [], color=INK, marker="_", ms=11, mew=1.1, lw=0, label="Within-year CV, MLR (reference)")]
fig.legend(handles=h, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=7.2)
save(fig, "Fig11_LOYO_temporal_transferability")

# ---- Fig. 12 and 13: SHAP
P = pickle.load(open(RESULTS / "shap_pooled.pkl", "rb"))
S, Xv, xg = P["S"], P["X"], P["model"]
imp = np.abs(S).mean(0); o = np.argsort(imp)
fig, axs = plt.subplots(1, 2, figsize=(W2, 72 * MM), gridspec_kw=dict(width_ratios=[1.65, 1]))
rng = np.random.default_rng(0); a = axs[0]
for k, j in enumerate(o):
    v = S[:, j]; hist, edges = np.histogram(v, bins=60); idx = np.clip(np.digitize(v, edges) - 1, 0, 59)
    yj = k + (rng.random(len(v)) - .5) * 0.75 * hist[idx] / hist.max()
    sc = a.scatter(v, yj, c=np.clip(Xv[:, j], -2.5, 2.5), cmap=DIVS, vmin=-2.5, vmax=2.5, s=3, lw=0)
a.set_yticks(range(4)); a.set_yticklabels([F4[j] for j in o]); a.axvline(0, color=REF, lw=0.6)
a.set_xlabel("SHAP value (contribution to LST anomaly, °C)"); lab(a, "(a)", x=-0.10)
cb = fig.colorbar(sc, ax=a, pad=0.015, aspect=25, fraction=0.04); cb.set_label("Index value (year-standardised z-score)", fontsize=7.5)
a = axs[1]; a.barh(range(4), imp[o], color=BLUE, height=0.55)
for k, j in enumerate(o):
    a.text(imp[j] + 0.008, k, f"{imp[j]:.2f}", va="center", fontsize=7.5)
a.set_yticks(range(4)); a.set_yticklabels([F4[j] for j in o]); a.set_xlim(0, imp.max() * 1.2); a.set_xlabel("Mean |SHAP| (°C)")
lab(a, "(b)", x=-0.10)
fig.tight_layout(w_pad=2); save(fig, "Fig12_SHAP_global_importance")

fig, axs = plt.subplots(2, 2, figsize=(W2, 150 * MM))
for a, j, jc, t in zip(axs.flat, [0, 1, 2, 3], [1, 0, 0, 1], "abcd"):
    sc = a.scatter(Xv[:, j], S[:, j], c=np.clip(Xv[:, jc], -2.5, 2.5), cmap=DIVS, vmin=-2.5, vmax=2.5, s=4, lw=0)
    lw_ = lowess(S[:, j], Xv[:, j], frac=0.4); a.plot(lw_[:, 0], lw_[:, 1], color=INK, lw=1.1)
    zc = lw_[np.where(np.diff(np.sign(lw_[:, 1])) != 0)[0], 0]
    if len(zc):
        a.axvline(zc[0], color=MUTED, ls=":", lw=0.8)
    a.axhline(0, color=REF, lw=0.5)
    a.set_xlabel(f"{F4[j]} (year-standardised z-score)"); a.set_ylabel(f"SHAP value for {F4[j]} (°C)")
    a.text(0.98, 0.04, f"colour: {F4[jc]}", transform=a.transAxes, ha="right", fontsize=7, color=MUTED)
    lab(a, f"({t})", x=-0.12)
fig.tight_layout(rect=(0, 0, 0.93, 1), h_pad=1.5, w_pad=2.5)
cb = fig.colorbar(sc, cax=fig.add_axes([0.945, 0.3, 0.012, 0.4])); cb.set_label("Colour index value (z-score)", fontsize=7.5)
save(fig, "Fig13_SHAP_dependence")

# ---- Fig. S2: PV audit
pv = pd.read_csv(RESULTS / "pv_audit.csv").set_index("Year")
fig, a = plt.subplots(figsize=(W15, 75 * MM)); xx = np.linspace(A.NDVI.min() - 0.02, A.NDVI.max() + 0.02, 200)
for y, d in A.groupby("Year"):
    c = pv.loc[y, "c"]
    a.scatter(d.NDVI, d.PV, s=5, marker=YM[y], color=YC[y], lw=0)
    a.plot(xx, (xx - c) ** 2, color=YC[y], lw=0.8, label=f"{y}: PV = (NDVI {'+' if c < 0 else '−'} {abs(c):.3f})²")
a.set(xlabel="NDVI", ylabel="PV (as supplied)"); a.legend(fontsize=7)
fig.tight_layout(); save(fig, "FigS02_PV_audit")

# ---- Fig. S5: MLR fitted
F = pd.read_csv(RESULTS / "mlr_ndvi_ndbi_fitted.csv"); T6 = pd.read_csv(RESULTS / "table_S6_mlr.csv")
fig, axs = plt.subplots(2, 2, figsize=(W15, W15 * 0.98))
for k, (a, y) in enumerate(zip(axs.flat, YEARS)):
    d = F[F.Year == y]; t = T6[(T6.Year == y) & (T6.Indices == "NDVI–NDBI")].iloc[0]
    lo, hi = np.floor(min(d.Observed.min(), d.Fitted.min())) - 0.5, np.ceil(max(d.Observed.max(), d.Fitted.max())) + 0.5
    a.plot([lo, hi], [lo, hi], color=REF, lw=0.7, ls="--")
    a.scatter(d.Observed, d.Fitted, s=8, marker=YM[y], color=YC[y], edgecolor="white", linewidth=0.3)
    a.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal", xlabel="Observed LST (°C)", ylabel="Fitted LST (°C)")
    a.text(0.04, 0.97, f"{y}\nR = {t.R:.3f}\nRMSE = {t.RMSE:.3f} °C", transform=a.transAxes, va="top", fontsize=7.5)
    lab(a, f"({'abcd'[k]})")
fig.tight_layout(); save(fig, "FigS05_MLR_NDVI_NDBI_within_year")

# ---- Fig. S7: SVR-PUK diagnosis
K = pickle.load(open(RESULTS / "svr_puk_kernelshap.pkl", "rb"))
fig, axs = plt.subplots(1, 2, figsize=(W2, 72 * MM))
a = axs[0]
for y in YEARS:
    a.hist(A[A.Year == y].PV, bins=np.linspace(0, 1, 31), histtype="step", lw=1.1, color=YC[y], label=str(y))
a.set_xlabel("PV (as supplied)"); a.set_ylabel("Number of samples"); a.legend(); lab(a, "(a)", x=-0.08)
a = axs[1]; im6 = np.abs(K["shap"]).mean(0); o6 = np.argsort(im6)
a.barh(range(6), im6[o6], height=0.55, color=["#D55E00" if F6[j] in ("PV", "BI") else BLUE for j in o6])
for k, j in enumerate(o6):
    a.text(im6[j] + 0.02, k, f"{im6[j]:.2f}", va="center", fontsize=7.5)
a.set_yticks(range(6)); a.set_yticklabels([F6[j] for j in o6]); a.set_xlabel("Mean |SHAP| on the 2023 test set (°C)")
lab(a, "(b)", x=-0.10); fig.tight_layout(w_pad=2); save(fig, "FigS07_SVR_PUK_diagnostic")

# ---- Fig. S8: LOYO observed vs predicted (harmonised)
Pr = pd.read_csv(RESULTS / "loyo_predictions.csv"); Pr = Pr[Pr.Setup == HAR]
fig, axs = plt.subplots(2, 4, figsize=(W2, 100 * MM), sharex=True, sharey=True); lim = (-7.5, 7.5)
for i, m in enumerate(["MLR", "XGBoost"]):
    for j, y in enumerate(YEARS):
        a = axs[i, j]; d = Pr[(Pr.Model == m) & (Pr.TestYear == y)]; e = d.LSTanom_pred - d.LSTanom_obs
        a.plot(lim, lim, color=REF, lw=0.6, ls="--")
        a.scatter(d.LSTanom_obs, d.LSTanom_pred, s=6, color=YC[y], marker=YM[y], lw=0.2, edgecolor="white")
        a.text(0.04, 0.96, f"{m}, {y}\nR = {np.corrcoef(d.LSTanom_obs, d.LSTanom_pred)[0, 1]:.3f}\nRMSE = {np.sqrt((e ** 2).mean()):.3f} °C",
               transform=a.transAxes, va="top", fontsize=6.8)
        a.set(xlim=lim, ylim=lim, aspect="equal")
        if i == 1: a.set_xlabel("Observed anomaly (°C)")
        if j == 0: a.set_ylabel("Predicted anomaly (°C)")
        lab(a, f"({'abcdefgh'[4 * i + j]})", x=-0.05)
fig.tight_layout(h_pad=0.6, w_pad=0.4); save(fig, "FigS08_LOYO_observed_vs_predicted")

# ---- Fig. S9: conformal coverage
C = pd.read_csv(RESULTS / "conformal_intervals.csv")
fig, a = plt.subplots(figsize=(W1, 65 * MM))
for k, (m, c) in enumerate(zip(["MLR", "XGBoost"], [BLUE, "#D55E00"])):
    d = C[(C.Model == m) & (C.Nominal == 0.9)]
    a.bar(np.arange(4) + (k - .5) * 0.36, d.Coverage, 0.34, color=c, label=f"{m} (mean half-width ±{d.HalfWidth.mean():.1f} °C)")
a.axhline(0.9, color=INK, ls="--", lw=0.7); a.set_xticks(range(4)); a.set_xticklabels(YEARS); a.set_ylim(0.5, 1.02)
a.set_xlabel("Held-out year"); a.set_ylabel("Empirical coverage"); fig.legend(loc="lower center", fontsize=7)
fig.tight_layout(rect=(0, 0.13, 1, 1)); save(fig, "FigS09_conformal_coverage")

# ---- Fig. S10: per-year importance and grouped permutation
Py = pd.read_csv(RESULTS / "shap_per_year.csv", index_col=0)
G = pd.read_csv(RESULTS / "grouped_permutation_importance.csv").pivot(index="Year", columns="Group", values="dRMSE")
fig, axs = plt.subplots(1, 2, figsize=(W2, 72 * MM)); a = axs[0]
for k, y in enumerate(Py.index):
    a.bar(np.arange(4) + (k - 1.5) * 0.2, Py.loc[y].values, 0.18, color=YC[int(y)], label=str(y))
a.set_xticks(range(4)); a.set_xticklabels(F4); a.set_ylabel("Mean |SHAP| (°C)"); a.legend(ncol=4, loc="upper center"); lab(a, "(a)", x=-0.08)
a = axs[1]
for k, (g, c) in enumerate([("Built-up (NDBI, UI)", "#D55E00"), ("Vegetation/moisture (NDVI, NDWI)", "#009E73")]):
    a.bar(np.arange(4) + (k - .5) * 0.36, G[g].values, 0.34, color=c, label=g)
a.set_xticks(range(4)); a.set_xticklabels(G.index); a.set_xlabel("Held-out year"); a.set_ylabel("Increase in RMSE when permuted (°C)")
a.legend(loc="upper left"); lab(a, "(b)", x=-0.08)
fig.tight_layout(w_pad=2); save(fig, "FigS10_year_specific_and_grouped_importance")


# ---- Fig. S11: ALE vs PDP
def ale(model, X, j, bins=20):
    q = np.unique(np.quantile(X[:, j], np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(q, X[:, j], side="right") - 1, 0, len(q) - 2)
    eff, cnt = np.zeros(len(q) - 1), np.zeros(len(q) - 1)
    for b in range(len(q) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        lo_, hi_ = X[m].copy(), X[m].copy(); lo_[:, j] = q[b]; hi_[:, j] = q[b + 1]
        eff[b] = (model.predict(hi_) - model.predict(lo_)).mean(); cnt[b] = m.sum()
    acc = np.concatenate([[0], np.cumsum(eff)]); mid = (acc[1:] + acc[:-1]) / 2
    return q, acc - np.sum(mid * cnt) / cnt.sum()


fig, axs = plt.subplots(1, 4, figsize=(W2, 55 * MM), sharey=True)
for a, j in zip(axs, range(4)):
    q, ac = ale(xg, Xv, j); a.plot(q, ac, color="#D55E00", lw=1.2, label="ALE")
    pdp = partial_dependence(xg, Xv, [j], grid_resolution=40, percentiles=(0, 1)); pa = pdp["average"][0]
    a.plot(pdp["grid_values"][0], pa - pa.mean(), color=BLUE, lw=0.9, ls="--", label="PDP (centred)")
    a.axhline(0, color=REF, lw=0.5); a.set_xlabel(f"{F4[j]} (z-score)"); lab(a, f"({'abcd'[j]})", x=-0.06)
axs[0].set_ylabel("Effect on LST anomaly (°C)"); axs[0].legend(loc="upper right", fontsize=7)
fig.tight_layout(w_pad=1.0); save(fig, "FigS11_ALE_vs_PDP")

# ---- Fig. S12: PDP + ICE and two-way PDP
fig = plt.figure(figsize=(W2, 150 * MM)); gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.25], hspace=0.45, wspace=0.45)
a0 = None
for j in range(4):
    a = fig.add_subplot(gs[0, j], sharey=a0); a0 = a0 or a
    pdp = partial_dependence(xg, Xv, [j], kind="both", grid_resolution=40, percentiles=(0.02, 0.98))
    gv, ice = pdp["grid_values"][0], pdp["individual"][0]
    for i in np.random.default_rng(1).choice(len(ice), 80, replace=False):
        a.plot(gv, ice[i], color="#9ECAE1", lw=0.3, alpha=0.8)
    a.plot(gv, pdp["average"][0], color="#08519C", lw=1.3); a.set_xlabel(f"{F4[j]} (z-score)")
    if j == 0:
        a.set_ylabel("Partial dependence of\nLST anomaly (°C)")
    lab(a, f"({'abcd'[j]})", x=-0.08)
a = fig.add_subplot(gs[1, 1:3]); p2 = partial_dependence(xg, Xv, [(0, 1)], grid_resolution=30, percentiles=(0.02, 0.98))
G0, G1 = np.meshgrid(p2["grid_values"][0], p2["grid_values"][1], indexing="ij"); Zs = p2["average"][0]
lim2 = np.ceil(np.abs(Zs).max() * 2) / 2
cs = a.contourf(G0, G1, Zs, levels=np.arange(-lim2, lim2 + 0.01, 0.25), cmap=DIV); a.contour(G0, G1, Zs, levels=[0], colors=INK, linewidths=0.7)
a.set_xlabel("NDVI (z-score)"); a.set_ylabel("NDBI (z-score)"); lab(a, "(e)", x=-0.06)
fig.colorbar(cs, ax=a, pad=0.02).set_label("Partial dependence (°C)", fontsize=7.5)
save(fig, "FigS12_PDP_ICE_twoway")

# ---- Fig. S13: bootstrap shares
Sm = pd.read_csv(RESULTS / "table_S7_bootstrap_summary.csv"); Sm = Sm[Sm.Model == "XGBoost TreeSHAP"]
fig, axs = plt.subplots(1, 2, figsize=(W2, 72 * MM), gridspec_kw=dict(width_ratios=[1.7, 1])); a = axs[0]
for k, y in enumerate(YEARS):
    d = Sm[Sm.Year == y].set_index("Index").loc[F4]
    a.errorbar(np.arange(4) + (k - 1.5) * 0.17, d.Share, yerr=[d.Share - d.CI_low, d.CI_high - d.Share], fmt=YM[y], ms=3.5,
               color=YC[y], elinewidth=0.8, capsize=1.5, label=str(y))
a.set_xticks(range(4)); a.set_xticklabels(F4); a.set_ylabel("Share of total mean |SHAP|"); a.set_ylim(0, 0.8)
a.legend(ncol=4, loc="upper center"); lab(a, "(a)", x=-0.06)
a = axs[1]; d = Sm[Sm.Index == "Built-up (NDBI+UI)"].set_index("Year").loc[YEARS]
a.errorbar(range(4), d.Share, yerr=[d.Share - d.CI_low, d.CI_high - d.Share], fmt="o", ms=4, color="#D55E00", capsize=2)
a.axhline(0.5, color=REF, ls="--", lw=0.6); a.set_xticks(range(4)); a.set_xticklabels(YEARS); a.set_ylim(0, 1)
a.set_xlabel("Year"); a.set_ylabel("Built-up share (NDBI + UI)"); lab(a, "(b)", x=-0.12)
fig.tight_layout(w_pad=2); save(fig, "FigS13_bootstrap_SHAP_shares")
