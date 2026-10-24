import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core import candidate_references, detect_gaps, longest_gap, rank_references, select_references, training_mask  # noqa: E402

IDX = pd.date_range("2026-07-01", periods=200, freq="10min")


def series_with_nans(positions):
    s = pd.Series(np.arange(len(IDX), dtype=float), index=IDX)
    s.iloc[positions] = np.nan
    return s


def test_detect_gaps_ignores_short_runs_and_finds_long_ones():
    s = series_with_nans([5, 6] + list(range(50, 80)) + list(range(120, 127)))
    gaps = detect_gaps(s, min_steps=6)
    assert [(g.start, g.n_steps) for g in gaps] == [(IDX[50], 30), (IDX[120], 7)]


def test_gap_touching_series_end_is_found():
    s = series_with_nans(list(range(190, 200)))
    g = longest_gap(s)
    assert g.start == IDX[190] and g.end == IDX[199] and g.hours == 10 / 6


def test_longest_gap_none_when_complete():
    assert longest_gap(series_with_nans([])) is None


def test_training_mask_is_strictly_before_outage():
    m = training_mask(IDX, IDX[100], days=0.5)  # 12 h = 72 steps
    assert m.sum() == 72 and not m[100] and m[99] and not m[27] and m[28]


def test_references_never_come_from_target_park():
    assert candidate_references(["T1", "T2", "H1", "O2"], "T1") == ["H1", "O2"]


def test_rank_and_select_prefers_correlated_and_drops_unavailable():
    rng = np.random.default_rng(0)
    y = rng.normal(5, 2, len(IDX))
    ws = pd.DataFrame(
        {"T1": y, "H1": y + rng.normal(0, 0.1, len(IDX)), "H2": rng.normal(5, 2, len(IDX)), "O1": y * 1.1},
        index=IDX,
    )
    outage = np.zeros(len(IDX), bool)
    outage[150:] = True
    ws.loc[outage, "O1"] = np.nan  # O1 is down during the outage itself
    train = training_mask(IDX, IDX[150], days=1)
    r = rank_references(ws, "T1", train, outage)
    assert set(r.reference) == {"H1", "H2", "O1"}
    assert select_references(r, min_corr=0.5) == ["H1"]


def test_select_falls_back_to_best_reference_when_none_pass():
    r = pd.DataFrame({"reference": ["H1", "O1"], "corr": [0.6, 0.5], "coverage_outage": [0.1, 0.0]})
    assert select_references(r) == ["H1"]
