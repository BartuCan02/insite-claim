import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from density import RHO_REF, air_density, wind_factor  # noqa: E402


def test_standard_air_is_1225():
    assert abs(air_density(1013.25, 15.0) - RHO_REF) < 0.001


def test_thinner_air_lowers_effective_wind():
    assert wind_factor(np.array([1.19]))[0] < 1 and wind_factor(np.array([RHO_REF]))[0] == 1
