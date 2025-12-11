import numpy as np
import xarray as xr
import sys
import os
import pytest

# Adds the parent folder to Python's import path so we can import modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from jake_solar.src import toa_influx_functions as tf

def make_test_dataset(type_test, days):
    if type_test != "range":
        return xr.Dataset({"day_of_year": ("valid_time", days)})
    else:
        data = np.arange(days[0], days[1], 1)
        return xr.Dataset({"day_of_year": ("valid_time", data)})


@pytest.mark.parametrize(
    "type_test, days, expected_range, approx_value",
    [
        ( "single", [81], (None, None), 0),            # Around the March 22 equinox, SDA should be 0 radians
        ( "single", [173], (None, None), 0.409),       # Around the June 21 equinox, the SDA should be 23.45 degrees or 0.409 radians
        ( "single", [266], (None, None), 0),           # Around the September 23 equinox, SDA should be 0 radians
        ( "single", [357], (None, None), -0.409),      # Around the December 23 equinox, SDA should be -23.45 or -0.409 radians
        ( "range", [1,366], (-0.409, 0.409 ), None),   # Range of all values should be between -0.409 radians and 0.409 radians
        
    ],
)
def test_sda(type_test, days, expected_range, approx_value):
    ds = make_test_dataset(type_test,days)
    result = tf.calc_solar_declination_angle(ds)
    sda = result["SDA"].values

    atol = 0.03
    if type_test == "single":
        np.testing.assert_allclose(sda, approx_value, atol=atol)


    elif type_test == "range":
        assert np.all((sda >= expected_range[0] - atol) & (sda <= expected_range[1] + atol))
