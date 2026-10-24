"""Step 3: air density. Denser air pushes harder on the blades, so the same wind
makes more power. Standard practice (IEC 61400-12) folds this into the wind speed:
v_eff = v * (rho / rho_ref)^(1/3), then uses the normal power curve.

During the outage the target park's own temperature and pressure are missing too,
so density is estimated from the reference parks plus the offset between the parks
learned in the 30 training days (height above sea level differs between parks)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import park_of

R_DRY = 287.05  # J/(kg K)
RHO_REF = 1.225  # kg/m3, standard air the power curve is assumed to refer to


def air_density(p_hpa, t_c):
    return p_hpa * 100 / (R_DRY * (t_c + 273.15))


def _proxy(wide: pd.DataFrame, refs: list[str]) -> pd.Series:
    p = wide["air_pressure"][refs].mean(axis=1, skipna=True)  # O park has no real pressure -> NaN, skipped
    t = wide["ambient_temperature"][refs].mean(axis=1, skipna=True)
    return air_density(p, t)


def estimate_density(wide: pd.DataFrame, target: str, refs: list[str], tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    own = [c for c in wide["air_pressure"].columns if park_of(c) == park_of(target)]
    rho_target = air_density(wide["air_pressure"][own].mean(axis=1), wide["ambient_temperature"][own].mean(axis=1))
    with_pressure = [r for r in refs if wide["air_pressure"][r].notna().any()]
    if not with_pressure:  # no usable reference pressure: fall back to the target's training average
        return np.full(len(te), float(rho_target.iloc[tr].mean()))
    proxy = _proxy(wide, with_pressure)
    offset = float((rho_target.iloc[tr] - proxy.iloc[tr]).mean())
    est = (proxy.iloc[te] + offset).to_numpy()
    return np.where(np.isnan(est), float(rho_target.iloc[tr].mean()), est)


def wind_factor(rho: np.ndarray) -> np.ndarray:
    return (rho / RHO_REF) ** (1 / 3)
