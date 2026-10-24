"""Shared machinery for the feature search: experiments, models, scoring."""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core import training_mask
from data import TARGETS, TRAIN_END
from power import INTERVAL_HOURS, PowerCurve

warnings.filterwarnings("ignore")
WINDOW_DAYS = 30
PC = PowerCurve("T1")

LGB_PARAMS = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    n_jobs=1, verbose=-1,
)


def experiments(wide: pd.DataFrame, stride_days: float = 1, days: int = 3):
    """Fake 3-day outages in Jul/Aug, each with the 30 days before it as training."""
    idx = wide.index
    first = idx[0] + pd.Timedelta(days=WINDOW_DAYS)
    last_start = TRAIN_END - pd.Timedelta(days=days) + pd.Timedelta(minutes=10)
    exps = []
    for start in pd.date_range(first, last_start, freq=pd.Timedelta(days=stride_days)):
        end = start + pd.Timedelta(days=days) - pd.Timedelta(minutes=10)
        test = (idx >= start) & (idx <= end)
        if wide.loc[test, "wind_speed"].notna().mean().min() < 0.5:
            continue
        train = training_mask(idx, start, WINDOW_DAYS)
        for t in TARGETS:
            y = wide[("wind_speed", t)].to_numpy()
            tr = np.where(train & ~np.isnan(y))[0]
            te = np.where(test & ~np.isnan(y))[0]
            exps.append({"id": f"{start:%m%d}_{t}", "target": t, "tr": tr, "te": te, "y": y})
    return exps


def make_model(kind: str, params: dict | None = None):
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=(params or {}).get("alpha", 1.0)))
    if kind == "lgbm":
        return lgb.LGBMRegressor(**{**LGB_PARAMS, **(params or {})})
    raise ValueError(kind)


def metrics(yt: np.ndarray, yp: np.ndarray) -> dict:
    pt, pp = PC.power_kw(yt), PC.power_kw(yp)
    et, ep = pt.sum() * INTERVAL_HOURS, pp.sum() * INTERVAL_HOURS
    return {
        "ws_rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
        "ws_mae": float(np.mean(np.abs(yp - yt))),
        "p_rmse": float(np.sqrt(np.mean((pp - pt) ** 2))),
        "p_mae": float(np.mean(np.abs(pp - pt))),
        "e_true": float(et),
        "e_pred": float(ep),
    }


def run_config(X: np.ndarray, exps: list[dict], kind: str, params: dict | None = None, return_pred=False) -> pd.DataFrame:
    rows, preds = [], {}
    for e in exps:
        m = make_model(kind, params)
        m.fit(X[e["tr"]], e["y"][e["tr"]])
        yp = np.clip(m.predict(X[e["te"]]), 0, None)
        rows.append({"id": e["id"], "target": e["target"], **metrics(e["y"][e["te"]], yp)})
        if return_pred:
            preds[e["id"]] = yp
    df = pd.DataFrame(rows)
    return (df, preds) if return_pred else df


def summary(df: pd.DataFrame) -> dict:
    return {
        "ws_rmse": df.ws_rmse.mean(),
        "ws_mae": df.ws_mae.mean(),
        "p_rmse": df.p_rmse.mean(),
        "p_mae": df.p_mae.mean(),
        "abs_e_dev_pct": (100 * (df.e_pred - df.e_true).abs() / df.e_true).mean(),
        "total_e_dev_pct": 100 * (df.e_pred.sum() - df.e_true.sum()) / df.e_true.sum(),
    }


def paired(df: pd.DataFrame, base: pd.DataFrame, col: str = "p_rmse") -> dict:
    """How a config compares with a baseline on the SAME experiments."""
    d = df.set_index("id")[col] - base.set_index("id")[col]
    return {
        "delta": d.mean(),
        "delta_se": d.std(ddof=1) / np.sqrt(len(d)),
        "win_rate": float((d < 0).mean()),
    }
