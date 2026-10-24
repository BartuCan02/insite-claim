"""Power curve interpolation and energy accounting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import load_power_curve

INTERVAL_HOURS = 10 / 60


class PowerCurve:
    def __init__(self, turbine: str = "T1", frame: pd.DataFrame | None = None):
        pc = load_power_curve(turbine) if frame is None else frame.sort_values("wind_speed")
        self.ws = pc["wind_speed"].to_numpy(dtype=float)
        self.p = pc["power"].to_numpy(dtype=float)
        self.cut_out = self.ws[-1]

    def power_kw(self, wind_speed) -> np.ndarray:
        ws = np.asarray(wind_speed, dtype=float)
        p = np.interp(ws, self.ws, self.p, left=0.0, right=0.0)
        p = np.where(ws > self.cut_out, 0.0, p)
        return np.where(np.isnan(ws), np.nan, p)

    def energy_kwh(self, wind_speed) -> float:
        """Energy over the series, treating NaN wind speed as 0 contribution."""
        return float(np.nansum(self.power_kw(wind_speed)) * INTERVAL_HOURS)


def energy_metrics(pc: PowerCurve, y_true: pd.Series, y_pred: pd.Series) -> dict:
    mask = y_true.notna()
    e_true = pc.energy_kwh(y_true[mask])
    e_pred = pc.energy_kwh(y_pred[mask])
    p_true = pc.power_kw(y_true[mask])
    p_pred = pc.power_kw(y_pred[mask])
    return {
        "e_true_mwh": e_true / 1000,
        "e_pred_mwh": e_pred / 1000,
        "e_err_pct": 100 * (e_pred - e_true) / e_true if e_true > 0 else np.nan,
        "p_rmse_kw": float(np.sqrt(np.nanmean((p_pred - p_true) ** 2))),
    }
