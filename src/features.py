"""Reference-only features (nothing from the target park is used)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import PARKS, REFERENCES


def circular_mean_deg(deg: pd.DataFrame) -> pd.Series:
    rad = np.deg2rad(deg)
    s = np.sin(rad).mean(axis=1, skipna=True)
    c = np.cos(rad).mean(axis=1, skipna=True)
    return (np.rad2deg(np.arctan2(s, c)) + 360) % 360


class ReferenceFeatures:
    """Fills gaps in reference wind speeds with a normalised park mean and
    derives park-level aggregates. Scaling constants are learned on train."""

    def fit(self, train: pd.DataFrame):
        ws = train["wind_speed"][REFERENCES]
        self.ref_means = ws.mean()
        return self

    def transform(self, wide: pd.DataFrame) -> pd.DataFrame:
        ws = wide["wind_speed"][REFERENCES]
        norm = ws / self.ref_means  # each ref expressed relative to its own mean
        fleet_norm = norm.mean(axis=1, skipna=True)  # NaN only when all refs missing
        filled = ws.copy()
        for r in REFERENCES:
            filled[r] = filled[r].fillna(fleet_norm * self.ref_means[r])
        out = pd.DataFrame(index=wide.index)
        for r in REFERENCES:
            out[f"ws_{r}"] = filled[r]
        out["ws_fleet"] = fleet_norm
        out["ws_H"] = norm[PARKS["H"]].mean(axis=1, skipna=True)
        out["ws_O"] = norm[PARKS["O"]].mean(axis=1, skipna=True)
        wd = circular_mean_deg(wide["wind_direction"][REFERENCES])
        out["wd_ref"] = wd
        out["wd_sin"] = np.sin(np.deg2rad(wd))
        out["wd_cos"] = np.cos(np.deg2rad(wd))
        out["temp_ref"] = wide["ambient_temperature"][REFERENCES].mean(axis=1, skipna=True)
        out["hour"] = wide.index.hour
        return out
