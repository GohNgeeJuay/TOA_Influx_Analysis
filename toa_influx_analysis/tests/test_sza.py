import numpy as np
import xarray as xr
import sys
import os
import pytest
import pandas as pd

# Adds the parent folder to Python's import path so we can import modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from jake_solar.src import toa_influx_functions as tf


def make_test_dataset(valid_time, latitude, sda, hour_angle):
    return xr.Dataset(
        data_vars={
            "SDA": (("valid_time"), [sda]),             # in radians
            "hour_angle": (("longitude", "valid_time"), [[hour_angle]]) # in radians
        },
        coords={
            "latitude": ("latitude", [latitude]),  # already radians
            "longitude": ("longitude", [0]),       # dummy longitude
            "valid_time": ("valid_time", [valid_time])
        }
    )

@pytest.mark.parametrize(
    "valid_time, latitude, sda, hour_angle, expected_sza",
    [
        # Sun directly overhead at equator on equinox noon.
        (pd.to_datetime("2020-03-21 12:00:00"), 0, 0, 0, 0),

        # Sun on the horizon at equator 6 hours before noon (hour angle = -90°).
        (pd.to_datetime("2020-03-21 06:00:00"), 0, 0, -np.pi/2, np.pi/2),

        # Sun fully below the horizon at equator at midnight (hour angle = ±180°).
        (pd.to_datetime("2020-03-21 00:00:00"), 0, 0, np.pi, np.pi),

        # Sun directly overhead at Tropic of Cancer during June solstice noon.
        (pd.to_datetime("2020-06-21 12:00:00"), 23.45, np.deg2rad(23.45), 0, 0),

        # Arctic Circle at June solstice noon: SZA equals |latitude − declination|.
        (pd.to_datetime("2020-06-21 12:00:00"), 66.55, np.deg2rad(23.45), 0, np.deg2rad(66.55 - 23.45)),

        # Mid-latitudes on equinox noon: SZA equals the latitude
        (pd.to_datetime("2020-03-21 12:00:00"), 45, 0, 0, np.deg2rad(45)),
    ],
)

def test_sza(valid_time, latitude, sda, hour_angle, expected_sza):
    ds = make_test_dataset(valid_time, latitude, sda, hour_angle)
    result = tf.calc_solar_zenith_angle(ds)
    sza_rad = result["SZA"].values.item()

    np.testing.assert_allclose(sza_rad, expected_sza, atol=1e-7)
