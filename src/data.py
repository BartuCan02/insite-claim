"""Load the raw BayWa r.e. hackathon Excel files into one tidy table."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TURBINES = ["T1", "T2", "T3", "H1", "H2", "H3", "H4", "H5", "O1", "O2"]
TARGETS = ["T1", "T2", "T3"]
REFERENCES = ["H1", "H2", "H3", "H4", "H5", "O1", "O2"]
PARKS = {"T": TARGETS, "H": ["H1", "H2", "H3", "H4", "H5"], "O": ["O1", "O2"]}
VARS = ["wind_speed", "wind_direction", "ambient_temperature", "air_pressure", "wind_speed_stddev"]

HIDDEN = (pd.Timestamp("2026-09-01 00:00"), pd.Timestamp("2026-09-03 23:50"))
SHOWN = (pd.Timestamp("2026-09-04 00:00"), pd.Timestamp("2026-09-09 23:50"))
TRAIN_END = pd.Timestamp("2026-08-31 23:50")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def load_long(raw_dir: Path = RAW) -> pd.DataFrame:
    frames = []
    for name in TURBINES:
        df = pd.read_excel(raw_dir / f"{name}_data.xlsx")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    # O park has no stddev and a constant placeholder pressure: mark both as missing
    o_mask = long["turbine_name"].isin(PARKS["O"])
    long.loc[o_mask, "wind_speed_stddev"] = np.nan
    long.loc[o_mask, "air_pressure"] = np.nan
    return long


def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Columns are a MultiIndex (variable, turbine); index is the 10-min timestamp."""
    wide = long.pivot(index="timestamp", columns="turbine_name", values=VARS)
    wide = wide.sort_index()
    # 3-4 Aug are absent from every file; reindex so they appear as NaN rows
    full_index = pd.date_range(wide.index.min(), wide.index.max(), freq="10min")
    return wide.reindex(full_index).rename_axis("timestamp")


def load_wide(force: bool = False) -> pd.DataFrame:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    cache = PROCESSED / "wide.parquet"
    if cache.exists() and not force:
        wide = pd.read_parquet(cache)
        wide.columns = pd.MultiIndex.from_tuples([tuple(c.split("|")) for c in wide.columns])
        return wide
    wide = to_wide(load_long())
    flat = wide.copy()
    flat.columns = [f"{v}|{t}" for v, t in wide.columns]
    flat.to_parquet(cache)
    return wide


def load_power_curve(turbine: str = "T1") -> pd.DataFrame:
    pc = pd.read_excel(RAW / f"{turbine}_power_curve.xlsx")
    return pc[["wind_speed", "power"]].sort_values("wind_speed").reset_index(drop=True)


if __name__ == "__main__":
    w = load_wide(force=True)
    print(w.shape)
    print(w["wind_speed"].describe().round(2).to_string())
