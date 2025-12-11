import pytest
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
        ( "single", [3], (None, None), 1.033),       # perihelion ~ max
        ( "single", [185], (None, None), 0.967),     # aphelion ~ min
        ( "single", [81], (None, None), 1.0),     
        ( "single", [173], (None, None), 0.967),     
        ( "single", [266], (None, None), 1.0),     
        ( "single", [355], (None, None), 1.03),     
        ( "range", [1, 366], (0.967, 1.033), None),   # full year range test
    ],
)
def test_ecf(type_test, days, expected_range, approx_value):
    ds = make_test_dataset(type_test,days)
    result = tf.calc_eccentricty_correlation_factor(ds)
    ecf = result["ECF"].values

    atol = 1e-2
    if type_test == "single":
        np.testing.assert_allclose(ecf, approx_value, atol=atol)

    elif type_test == "range":
        assert np.all((ecf >= expected_range[0] - atol) & (ecf <= expected_range[1] + atol))
