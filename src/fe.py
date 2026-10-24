"""Candidate features for the feature search.

Every feature is computed from reference turbines only (never the target park),
on the full 10-minute grid, so time shifts are exact. Reference data exists
during the outage, so shifts in both directions (earlier and later) are allowed:
this is gap filling after the fact, not forecasting.

Column names are  <group>__<owner>__<detail>  so the search can select by group
and by reference turbine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import park_of

DEFAULT_SHIFTS = (1, 3, 6)  # steps of 10 min: 10 min, 30 min, 1 h
DEFAULT_ROLLS = (3, 6, 18)  # centred windows: 30 min, 1 h, 3 h

GROUPS = ["shift", "roll", "dir", "temp", "pres", "turb", "time", "spread"]


def circ_mean(deg: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    rad = np.deg2rad(deg)
    return np.sin(rad).mean(axis=1, skipna=True), np.cos(rad).mean(axis=1, skipna=True)


def filled_ws(wide: pd.DataFrame, refs: list[str]) -> pd.DataFrame:
    """Reference wind speeds; a missing value is filled from the other references
    (ratio-to-own-mean averaged across the references that have data)."""
    ws = wide["wind_speed"][refs]
    means = ws.mean()
    fleet = (ws / means).mean(axis=1, skipna=True)
    return ws.apply(lambda c: c.fillna(fleet * means[c.name]))


def build_all(wide: pd.DataFrame, refs: list[str], shifts=DEFAULT_SHIFTS, rolls=DEFAULT_ROLLS, shift_level="park", directions=("m", "p")) -> pd.DataFrame:
    parks = sorted({park_of(r) for r in refs})
    members = {p: [r for r in refs if park_of(r) == p] for p in parks}
    ws = filled_ws(wide, refs)
    park_ws = pd.DataFrame({p: ws[members[p]].mean(axis=1) for p in parks})
    out = {}

    for r in refs:
        out[f"ws__{r}__now"] = ws[r]

    src = park_ws if shift_level == "park" else ws
    for name in src.columns:
        for k in shifts:
            if "m" in directions:
                out[f"shift__{name}__m{k}"] = src[name].shift(k)  # value k steps earlier
            if "p" in directions:
                out[f"shift__{name}__p{k}"] = src[name].shift(-k)  # value k steps later

    for p in parks:
        for w in rolls:
            out[f"roll__{p}__mean{w}"] = park_ws[p].rolling(w, center=True, min_periods=1).mean()
            out[f"roll__{p}__std{w}"] = park_ws[p].rolling(w, center=True, min_periods=2).std()

    for p in parks:
        s, c = circ_mean(wide["wind_direction"][members[p]])
        out[f"dir__{p}__sin"], out[f"dir__{p}__cos"] = s, c

    for p in parks:
        t = wide["ambient_temperature"][members[p]].mean(axis=1, skipna=True)
        out[f"temp__{p}__now"] = t
        out[f"temp__{p}__d3h"] = t - t.shift(18)  # warming/cooling over 3 h (day/night stability)
    if len(parks) > 1:
        tp = [out[f"temp__{p}__now"] for p in parks]
        out["temp__diff__now"] = tp[0] - tp[1]

    pres_parks = [p for p in parks if wide["air_pressure"][members[p]].notna().any().any()]
    for p in pres_parks:
        pr = wide["air_pressure"][members[p]].mean(axis=1, skipna=True)
        out[f"pres__{p}__now"] = pr
        out[f"pres__{p}__d3h"] = pr - pr.shift(18)  # falling pressure = front coming

    turb_parks = [p for p in parks if wide["wind_speed_stddev"][members[p]].notna().any().any()]
    for p in turb_parks:
        sd = wide["wind_speed_stddev"][members[p]].mean(axis=1, skipna=True)
        out[f"turb__{p}__std"] = sd
        out[f"turb__{p}__ti"] = sd / park_ws[p].clip(lower=1.0)  # turbulence intensity

    hour = wide.index.hour + wide.index.minute / 60
    out["time__all__hsin"] = pd.Series(np.sin(2 * np.pi * hour / 24), index=wide.index)
    out["time__all__hcos"] = pd.Series(np.cos(2 * np.pi * hour / 24), index=wide.index)

    norm = ws / ws.mean()
    out["spread__all__std"] = norm.std(axis=1)  # do the references agree with each other?

    return pd.DataFrame(out, index=wide.index)


def select(F: pd.DataFrame, groups: set[str], refs: list[str] | None = None) -> list[str]:
    """Always include current wind speed of the chosen refs, plus the chosen groups."""
    cols = []
    for c in F.columns:
        g, owner, _ = c.split("__")
        if g == "ws":
            if refs is None or owner in refs:
                cols.append(c)
        elif g in groups:
            cols.append(c)
    return cols
