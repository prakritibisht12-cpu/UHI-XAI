"""Models and metrics used in the paper (no additional algorithms)."""
import numpy as np
import xgboost as xgb
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

XGB_GRID = {"max_depth": [2, 3, 5], "min_child_weight": [1, 5], "reg_lambda": [1, 10]}


class PUK:
    """Pearson VII universal kernel (Üstün et al., 2006), as in Weka's PolyKernel family `PUK`."""

    def __init__(self, omega=0.48, sigma=1.0):
        self.omega, self.sigma = omega, sigma

    def __call__(self, X, Y):
        d = np.sqrt(np.maximum(euclidean_distances(X, Y, squared=True), 0))
        return 1.0 / (1 + (2 * d * np.sqrt(2 ** (1 / self.omega) - 1) / self.sigma) ** 2) ** self.omega


def svr_puk(C=4.0, epsilon=0.001, omega=0.48, sigma=1.0):
    """Python re-implementation of the Weka SMOreg/PUK model (inputs and target min-max normalised)."""
    reg = make_pipeline(MinMaxScaler(), SVR(kernel=PUK(omega, sigma), C=C, epsilon=epsilon))
    return TransformedTargetRegressor(regressor=reg, transformer=MinMaxScaler())


def mlr():
    return make_pipeline(StandardScaler(), LinearRegression())


def xgb_model(**kw):
    p = dict(n_estimators=400, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=2)
    p.update(kw)
    return xgb.XGBRegressor(**p)


def fit_xgb_tuned(X, y, groups):
    """XGBoost tuned by grid search with leave-one-year-out CV inside the training years only."""
    gs = GridSearchCV(xgb_model(), XGB_GRID, cv=LeaveOneGroupOut(), scoring="neg_root_mean_squared_error", n_jobs=1)
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_, gs.best_params_


def metrics(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    e = pred - obs
    r = np.corrcoef(obs, pred)[0, 1]
    return dict(R=r, R2=r * r, RMSE=np.sqrt((e ** 2).mean()), MAE=np.abs(e).mean(), Bias=e.mean(),
                NSE=1 - (e ** 2).sum() / ((obs - obs.mean()) ** 2).sum(), n=len(obs))
