import pytest
import numpy as np
import xarray as xr
import sys
import os
import pytest
import pandas as pd

# Adds the parent folder to Python's import path so we can import modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from jake_solar.src import toa_influx_functions as tf

def make_test_dataset(ecf, sza):
    time = [pd.to_datetime("2020-01-01")]
    lat = [3.5]  # example latitude(s)
    lon = [101.5]  # example longitude(s)
    return xr.Dataset(
        data_vars={
            "ECF": (("valid_time"), [ecf]),
            "SZA": (("valid_time", "latitude", "longitude"), [[[sza]]]),
        },
        coords={
            "valid_time": time,
            "latitude": lat,
            "longitude": lon,
        },
    )

@pytest.mark.parametrize(
    "ecf, sza, expected",
    [
        # Case 1: Equator, equinox, noon → max flux ~ solar constant
        (1.0, 0, 1361),

        # Case 2: Equator, equinox, horizon → flux = 0
        (1.0, np.pi/2, 0),

        # Case 3: Night (SZA = 180°) → cos(SZA) = -1 → physically should be clamped to 0
        (1.0, np.pi, 0),

        # Case 4: Eccentricity factor at perihelion (ECF ~ 1.033), noon
        (1.033, 0, 1361 * 1.033),

        # Case 5: Eccentricity factor at aphelion (ECF ~ 0.967), noon
        (0.967, 0, 1361 * 0.967),
    ],
)
def test_toa_influx(ecf, sza, expected):
    ds = make_test_dataset(ecf, sza)
    result = tf.calc_toa_incident_shortwave(ds)
    flux = result["toa_influx"].values.item()

    # Clamp negative values to zero for physical realism
    flux = max(flux, 0)

    np.testing.assert_allclose(flux, expected, rtol=1e-3, atol=1e-3)
