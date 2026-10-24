"""Generic outage logic: works for any target turbine and any gap.

Turbine names are a park letter plus a number (T1, H3, O2). References must
come from a different park than the target.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=10)


def park_of(turbine: str) -> str:
    return turbine[0]


@dataclass(frozen=True)
class Gap:
    start: pd.Timestamp
    end: pd.Timestamp  # inclusive, last missing timestamp

    @property
    def n_steps(self) -> int:
        return int((self.end - self.start) / STEP) + 1

    @property
    def hours(self) -> float:
        return self.n_steps / 6


def detect_gaps(ws: pd.Series, min_steps: int = 6) -> list[Gap]:
    """Runs of consecutive NaN in a regular 10-minute series, at least min_steps long."""
    missing = ws.isna().to_numpy()
    gaps, i, n = [], 0, len(missing)
    while i < n:
        if not missing[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and missing[j + 1]:
            j += 1
        if j - i + 1 >= min_steps:
            gaps.append(Gap(ws.index[i], ws.index[j]))
        i = j + 1
    return gaps


def longest_gap(ws: pd.Series, min_steps: int = 6) -> Gap | None:
    gaps = detect_gaps(ws, min_steps)
    return max(gaps, key=lambda g: g.n_steps) if gaps else None


def training_mask(index: pd.DatetimeIndex, outage_start: pd.Timestamp, days: int = 30) -> np.ndarray:
    """The `days` before the outage, outage itself excluded."""
    return (index >= outage_start - pd.Timedelta(days=days)) & (index < outage_start)


def candidate_references(turbines: list[str], target: str) -> list[str]:
    return [t for t in turbines if park_of(t) != park_of(target)]


def rank_references(ws: pd.DataFrame, target: str, train: np.ndarray, outage: np.ndarray | None = None) -> pd.DataFrame:
    """Correlation of each candidate with the target over the training window.

    coverage_outage = share of the outage where the reference has data
    (a reference that is itself down during the outage is useless).
    """
    rows = []
    y = ws.loc[train, target]
    for ref in candidate_references(list(ws.columns), target):
        x = ws.loc[train, ref]
        both = x.notna() & y.notna()
        rows.append(
            {
                "reference": ref,
                "park": park_of(ref),
                "corr": float(np.corrcoef(x[both], y[both])[0, 1]) if both.sum() > 10 else np.nan,
                "n_overlap": int(both.sum()),
                "coverage_outage": float(ws.loc[outage, ref].notna().mean()) if outage is not None else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("corr", ascending=False).reset_index(drop=True)


def select_references(ranking: pd.DataFrame, k: int | None = None, min_corr: float = 0.0, min_coverage: float = 0.5) -> list[str]:
    usable = ranking[ranking["corr"].notna()]
    ok = usable[(usable["corr"] >= min_corr) & ~(usable["coverage_outage"] < min_coverage)]
    if ok.empty:  # nothing passes the filters: fall back to the best-correlated reference
        ok = usable.head(1)
    refs = ok["reference"].tolist()
    return refs[:k] if k else refs
