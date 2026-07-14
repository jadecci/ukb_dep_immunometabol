from importlib.resources import files
from pathlib import Path

from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

import ukb_dep_immunometabol


def load_resource(file_name: str) -> Path:
    resource_file = files(ukb_dep_immunometabol) / "data" / file_name
    return Path(str(resource_file))


def vip_score(model: PLSRegression) -> np.ndarray:
    vips = np.empty((model.x_rotations_.shape[0],))
    s = np.diag(
        model.x_scores_.T @ model.x_scores_ @ model.y_loadings_.T
        @ model.y_loadings_).reshape(model.x_rotations_.shape[1], -1)
    for j in range(model.x_rotations_.shape[0]):
        w = (model.x_rotations_[j] / np.linalg.norm(model.x_rotations_)) ** 2
        vips[j] = np.squeeze(np.sqrt(
            model.x_rotations_.shape[0] * (s.T @ w) / np.sum(s)))

    return vips


def pls_regression(
        x: pd.DataFrame, y: pd.DataFrame,
        covar: pd.DataFrame) -> tuple[float, float, np.ndarray, np.ndarray]:
    # Standardise features and regress out covariates from target
    # PLSRegression should scale both x and y by default, but just to be safe
    x_std = StandardScaler().fit_transform(x)
    y_resid = y - np.dot(covar, np.linalg.lstsq(covar, y, rcond=-1)[0])

    # PLS regression
    model = PLSRegression(n_components=1, copy=True)
    model.fit(x_std, y_resid)
    res = pearsonr(model.x_scores_.reshape(-1), y_resid)
    vips = vip_score(model)

    # Permutation testing with 1000 repeats
    n_repeat = 1000
    vips_null = np.empty((n_repeat, x.shape[1]))
    r_null = np.empty((n_repeat,))
    for repeat in range(n_repeat):
        y_perm = np.random.permutation(y_resid)
        model_perm = PLSRegression(n_components=1, copy=True)
        model_perm.fit(x_std, y_perm)
        vips_null[repeat, :] = vip_score(model_perm)
        r_null[repeat] = pearsonr(model_perm.x_scores_.reshape(-1), y_perm)[0]

    # One-sided P-value
    r_p = (1 + np.sum(np.abs(r_null) >= np.abs(res.statistic))) / (n_repeat + 1)
    vips_p = (1 + np.sum(vips_null >= vips, axis=0)) / (n_repeat + 1)

    return res.statistic, r_p, vips, vips_p
