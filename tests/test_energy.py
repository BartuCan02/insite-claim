import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy import Wobble, expected_power  # noqa: E402
from power import PowerCurve  # noqa: E402

PC = PowerCurve("T1")


def wobble(sd):
    rng = np.random.default_rng(0)
    pred = rng.uniform(2, 12, 2000)
    return Wobble().fit(pred, pred + rng.normal(0, sd, 2000))


def test_no_wobble_means_plain_power_curve():
    pred = np.array([4.0, 6.0, 8.0])
    w = wobble(1e-6)
    assert np.allclose(expected_power(PC, pred, w, "gauss_global"), PC.power_kw(pred), atol=1)


def test_wobble_raises_power_on_the_steep_part_of_the_curve():
    pred = np.array([6.0])
    assert expected_power(PC, pred, wobble(1.5), "gauss_global")[0] > PC.power_kw(pred)[0] * 1.05


def test_wobble_lowers_power_at_rated_plateau():
    pred = np.array([12.0])  # already at 4500 kW: wobble can only take power away
    assert expected_power(PC, pred, wobble(1.5), "gauss_global")[0] < 4500


def test_matches_worked_example_from_explanation():
    # real wind 4 and 8 average 1387 kW, the smooth 6 m/s guess gives 1054 kW
    assert (PC.power_kw(np.array([4.0]))[0] + PC.power_kw(np.array([8.0]))[0]) / 2 == 1387
    assert PC.power_kw(np.array([6.0]))[0] == 1054
