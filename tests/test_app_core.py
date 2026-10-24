import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import app_core as ac  # noqa: E402

IDX = pd.date_range("2026-07-01", periods=24 * 6 * 40, freq="10min")


def csv_bytes(name, ws, sep=",", decimal=".", pressure=None):
    df = pd.DataFrame({"turbine_name": name, "timestamp": IDX.strftime("%Y-%m-%d %H:%M:%S"), "wind_speed": ws,
                       "air_pressure": pressure if pressure is not None else 1000 + np.sin(np.arange(len(IDX)))})
    return df.to_csv(index=False, sep=sep, decimal=decimal).encode()


def fake_park():
    rng = np.random.default_rng(0)
    base = 6 + np.sin(np.arange(len(IDX)) / 50) * 2
    files = []
    for name in ["T1", "T2", "H1", "O1"]:
        ws = base + rng.normal(0, 0.5, len(IDX))
        if name.startswith("T"):
            ws[-500:-68] = np.nan  # 3-day outage
        files.append((f"{name}_data.csv", csv_bytes(name, ws, pressure=np.full(len(IDX), 990.0) if name == "O1" else None)))
    return files


def test_reads_german_csv_with_semicolons_and_decimal_commas():
    ws = 5.5 + (np.arange(len(IDX)) % 3) * 0.25
    b = csv_bytes("H1", ws, sep=";", decimal=",")
    long = ac.parse_wind_files([("H1.csv", b)])
    assert np.allclose(long.wind_speed, ws)


def test_constant_column_is_treated_as_missing():
    long = ac.parse_wind_files(fake_park())
    wide = ac.to_wide(long)
    assert wide[("air_pressure", "O1")].isna().all() and wide[("air_pressure", "H1")].notna().any()


def test_detects_shared_outage_of_affected_park():
    wide = ac.to_wide(ac.parse_wind_files(fake_park()))
    d = ac.detect_outage(wide)
    assert d["targets"] == ["T1", "T2"] and d["parks"] == ["T"]
    assert d["start"] == IDX[-500] and d["end"] == IDX[-69]


def test_test_window_falls_back_to_days_after_outage():
    wide = ac.to_wide(ac.parse_wind_files(fake_park()))
    d = ac.detect_outage(wide)
    a, b = ac.detect_test_window(wide, pd.Series(dtype=bool), d["targets"], d["end"])
    assert a == IDX[-68] and b == IDX[-1]
