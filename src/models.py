"""Prediction models. Each takes an explicit reference list, so nothing is
tied to a particular park."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from core import park_of


def circular_mean_deg(deg: pd.DataFrame) -> pd.Series:
    rad = np.deg2rad(deg)
    s = np.sin(rad).mean(axis=1, skipna=True)
    c = np.cos(rad).mean(axis=1, skipna=True)
    return (np.rad2deg(np.arctan2(s, c)) + 360) % 360


class RefFeatures:
    """Reference wind speeds with small gaps filled from the other references."""

    def __init__(self, refs: list[str]):
        self.refs = refs

    def fit(self, wide: pd.DataFrame):
        self.means = wide["wind_speed"][self.refs].mean()
        return self

    def transform(self, wide: pd.DataFrame) -> pd.DataFrame:
        ws = wide["wind_speed"][self.refs]
        norm = ws / self.means
        fleet = norm.mean(axis=1, skipna=True)
        out = pd.DataFrame(index=wide.index)
        for r in self.refs:
            out[f"ws_{r}"] = ws[r].fillna(fleet * self.means[r])
        out["ws_fleet"] = fleet
        for p in sorted({park_of(r) for r in self.refs}):
            members = [r for r in self.refs if park_of(r) == p]
            out[f"ws_park_{p}"] = norm[members].mean(axis=1, skipna=True)
        out["wd_ref"] = circular_mean_deg(wide["wind_direction"][self.refs])
        return out

    @property
    def ws_cols(self):
        return [f"ws_{r}" for r in self.refs]


class FleetRatio:
    name = "fleet_ratio"

    def __init__(self, refs):
        self.refs = refs

    def fit(self, train, target):
        self.f = RefFeatures(self.refs).fit(train)
        X = self.f.transform(train)
        y = train[("wind_speed", target)]
        m = y.notna() & X.ws_fleet.notna()
        self.scale = float(y[m].sum() / X.ws_fleet[m].sum())
        return self

    def predict(self, test):
        return self.f.transform(test).ws_fleet * self.scale


class LinearRefs:
    name = "linear"

    def __init__(self, refs, alpha=1.0):
        self.refs, self.alpha = refs, alpha

    def fit(self, train, target):
        self.f = RefFeatures(self.refs).fit(train)
        X = self.f.transform(train)[self.f.ws_cols]
        y = train[("wind_speed", target)]
        m = y.notna() & X.notna().all(axis=1)
        self.model = Ridge(alpha=self.alpha).fit(X[m], y[m])
        return self

    def predict(self, test):
        X = self.f.transform(test)[self.f.ws_cols]
        pred = pd.Series(np.nan, index=test.index)
        m = X.notna().all(axis=1)
        pred[m] = self.model.predict(X[m])
        return pred.clip(lower=0)


class SectorLinear:
    """One linear fit per wind-direction sector; global fit as fallback for thin sectors."""

    name = "sector_linear"

    def __init__(self, refs, n_sectors=8, min_rows=150, alpha=1.0):
        self.refs, self.n, self.min_rows, self.alpha = refs, n_sectors, min_rows, alpha

    def _sector(self, wd):
        width = 360 / self.n
        return np.floor(((wd + width / 2) % 360) / width)

    def fit(self, train, target):
        self.f = RefFeatures(self.refs).fit(train)
        F = self.f.transform(train)
        cols = self.f.ws_cols
        y = train[("wind_speed", target)]
        m = y.notna() & F[cols].notna().all(axis=1) & F.wd_ref.notna()
        X, y, sec = F.loc[m, cols], y[m], self._sector(F.wd_ref[m])
        self.global_model = Ridge(alpha=self.alpha).fit(X, y)
        self.models = {}
        for s in range(self.n):
            ms = (sec == s).to_numpy()
            if ms.sum() >= self.min_rows:
                self.models[s] = Ridge(alpha=self.alpha).fit(X[ms], y[ms])
        return self

    def predict(self, test):
        F = self.f.transform(test)
        cols = self.f.ws_cols
        pred = pd.Series(np.nan, index=test.index)
        ok = F[cols].notna().all(axis=1).to_numpy()
        pred[ok] = self.global_model.predict(F.loc[ok, cols])
        sec = self._sector(F.wd_ref).to_numpy()
        for s, model in self.models.items():
            ms = ok & (sec == s)
            if ms.any():
                pred[ms] = model.predict(F.loc[ms, cols])
        return pred.clip(lower=0)
