"""Direction-dependent travel time between parks.

Wind needs time to travel from one park to another, and how long depends on where it
comes from. For each wind-direction sector, the lag that best lines up each reference
park's wind with the target is learned on the training rows only (hour-scale changes,
slow trends removed). The feature is that park's wind read at that lag."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fe import build_all


def _hp(s: pd.Series) -> pd.Series:
    return s - s.rolling(36, center=True, min_periods=12).mean()


def advect_features(wide: pd.DataFrame, refs: list[str], tr: np.ndarray, y: np.ndarray,
                    sectors: int = 4, max_lag: int = 12, min_rows: int = 120) -> np.ndarray:
    F = build_all(wide, refs, shifts=(1, 3, 6), rolls=(3, 6, 18))
    parks = sorted({r[0] for r in refs})
    ws = {p: F[[f"ws__{r}__now" for r in refs if r[0] == p]].mean(axis=1) for p in parks}
    sin = np.nanmean(np.vstack([F[f"dir__{p}__sin"] for p in parks]), axis=0)
    cos = np.nanmean(np.vstack([F[f"dir__{p}__cos"] for p in parks]), axis=0)
    ang = (np.rad2deg(np.arctan2(sin, cos)) + 360) % 360
    sec = np.floor(((ang + 180 / sectors) % 360) / (360 / sectors))
    n = len(wide)
    ytr = pd.Series(np.nan, index=wide.index)
    ytr.iloc[tr] = y[tr]
    yh = _hp(ytr).to_numpy()
    trmask = np.zeros(n, bool)
    trmask[tr] = True
    cols = []
    for p in parks:
        s = ws[p]
        sh = _hp(s)
        lag_used = np.zeros(n, int)
        for k in range(sectors):
            m = trmask & (sec == k)
            best, bc = 0, -1.0
            if m.sum() >= min_rows:
                for lag in range(-max_lag, max_lag + 1):
                    v = sh.shift(-lag).to_numpy()
                    ok = m & ~np.isnan(v) & ~np.isnan(yh)
                    if ok.sum() < 60:
                        continue
                    cc = np.corrcoef(v[ok], yh[ok])[0, 1]
                    if cc > bc:
                        best, bc = lag, cc
            lag_used[sec == k] = best
        idx = np.clip(np.arange(n) + lag_used, 0, n - 1)
        cols.append(s.to_numpy()[idx])
    return np.column_stack(cols)
