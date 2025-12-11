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

def make_test_dataset(day, longitude, hour):
    time = pd.to_datetime(f"2020-01-{day:02d} {hour:02d}:00:00")
    x = xr.Dataset(
        # Just create a stub array of 0's with only 1 val. 
        # Latitude is not used but we keep it to mimic the Era5land dataset. 
        # The valid_time also is also not needed (since there is only 1 value), but we keep it to mimic the dataset.
        data_vars={
            "ssrd": (("valid_time", "latitude", "longitude"), np.zeros((1, 1, 1)))
        },
        coords={
            "longitude": ("longitude", [longitude]),
            "valid_time": ("valid_time", [time]),
            "day_of_year": ("day_of_year", [time.dayofyear])
        }
        )
    return x

@pytest.mark.parametrize(
    #This is hardcoded to be in Msia timezone. Need to remove the -8. 
    "day, longitude, hour, expected",
    [
        (21, 120, 12-8, 0.0),            # Case 1: Solar noon → hour angle ~ 0
        (21, 120, 10-8, -np.pi/6),       # Case 2: Morning (10:00) → negative HRA. -30 degrees in radians
        (21, 120, 14-8, np.pi/6),        # Case 3: Afternoon (14:00) → positive HRA. +30 degrees in radians
        (20, 120, 22, -np.pi/2),         # Case 4: Sunrise approx → hour angle near -90° (π/2). -90 degrees in radians
        (21, 120, 18-8, np.pi/2),        # Case 5: Sunset approx → hour angle near +90° (π/2). +90 degrees in radians
    ],
)

def test_hour_angle(day, longitude, hour, expected):
    ds = make_test_dataset(day, longitude, hour)
    result = tf.calc_hour_angle(ds)
    hour_angle = result["hour_angle"].values

    np.testing.assert_allclose(
        hour_angle,
        expected,
        atol=np.deg2rad(5),  # allow 5° tolerance due to equation of time, time zone effects
    )

