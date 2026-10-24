"""The judges' metrics: wind speed RMSE/MAE, power RMSE/MAE, summed energy deviation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from power import INTERVAL_HOURS, PowerCurve


def score(pc: PowerCurve, y_true: pd.Series, y_pred: pd.Series) -> dict:
    m = (y_true.notna() & y_pred.notna()).to_numpy()
    yt, yp = y_true.to_numpy()[m], y_pred.to_numpy()[m]
    pt, pp = pc.power_kw(yt), pc.power_kw(yp)
    e_true = pt.sum() * INTERVAL_HOURS
    e_pred = pp.sum() * INTERVAL_HOURS
    return {
        "n": int(m.sum()),
        "ws_rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
        "ws_mae": float(np.mean(np.abs(yp - yt))),
        "ws_bias": float(np.mean(yp - yt)),
        "p_rmse_kw": float(np.sqrt(np.mean((pp - pt) ** 2))),
        "p_mae_kw": float(np.mean(np.abs(pp - pt))),
        "e_true_kwh": float(e_true),
        "e_pred_kwh": float(e_pred),
        "e_dev_pct": float(100 * (e_pred - e_true) / e_true) if e_true > 0 else np.nan,
    }
