import numpy as np
import xarray as xr
import pandas as pd



def convert_to_radians(dataset : xr.Dataset) ->  xr.Dataset:

    dataset["latitude_rad"] = np.deg2rad(dataset["latitude"])
    dataset["longitude_rad"] = np.deg2rad(dataset["longitude"])

    return dataset


def calc_day_of_year(dataset : xr.Dataset) ->  xr.Dataset:

    dataset["day_of_year"] = dataset["valid_time"].dt.dayofyear
    return dataset


def calc_eccentricty_correlation_factor(dataset : xr.Dataset) ->  xr.Dataset:

    dataset["ECF"] = xr.DataArray(
        1 + 0.0333 * np.cos( np.deg2rad( (360*dataset["day_of_year"])/365)),
        dims=["valid_time"],
        coords={"valid_time": dataset["valid_time"]}
    )
    return dataset



def calc_hour_angle(dataset : xr.Dataset) ->  xr.Dataset:
    """
    At LST = 12, HRA = 0° → solar noon
    At LST < 12, HRA is negative → morning (Sun east of meridian)
    At LST > 12, HRA is positive → afternoon (Sun west of meridian)
    """
    
    #Reference for calculating local solar time: https://www.pveducation.org/pvcdrom/properties-of-sunlight/solar-time
    #TODO the timezone has to be dynamic if we are calculating the hour angle for regions outside GMT+8
    time_utc = dataset["valid_time"]
    #Tz_localize add the UTC, tz_convert converts the time into a specific time zone. 
    time_local = pd.to_datetime(time_utc).tz_localize("UTC").tz_convert("Asia/Kuala_Lumpur")

    time_zone = 8 #Need to pass in with the Asia/Kuala Lumpur
    # #calculate local standard time meridian (LSTM)
    LSTM = 15*time_zone
    
    #calculate local time in hours
    dataset["LT"] = xr.DataArray(
        time_local.hour + time_local.minute/60 + time_local.second/3600,
        dims=["valid_time"],
        coords={"valid_time": dataset["valid_time"]}
    )

    #Calculate parameter B
    B_param = 360/365*(dataset["day_of_year"]-81)
    dataset["B_param_rad"] = xr.DataArray(
        np.deg2rad(B_param),
        dims=["valid_time"],
        coords={"valid_time": dataset["valid_time"]}
    )
    
    #calculate equation of time (EOT)
    dataset["EOT"] = xr.DataArray(
        9.87*np.sin(2*dataset["B_param_rad"]) - 7.53*np.cos(dataset["B_param_rad"]) - 1.5*np.sin(dataset["B_param_rad"]),
        dims=["valid_time"],
        coords={"valid_time": dataset["valid_time"]}
    )

    dataset["TC"] = 4 * (dataset["longitude"] - LSTM) + dataset["EOT"]     

    #calculate local solar time (LST)
    dataset["LST"] = xr.DataArray(
        dataset["LT"] + (dataset["TC"]/60), 
        dims = ["valid_time", "longitude"],
        coords = {"valid_time": dataset["valid_time"], "longitude": dataset["longitude"]}
    )
    
    #calculate hour angle (HRA). Convert the final result into radians. Previous step should not convert into radians.
    dataset["hour_angle"] = xr.DataArray(
        np.deg2rad(15*(dataset["LST"] - 12)),
        dims = ["valid_time", "longitude"],
        coords = {"valid_time": dataset["valid_time"], "longitude": dataset["longitude"]}
    )

    return dataset


def calc_solar_declination_angle(dataset : xr.Dataset) ->  xr.Dataset:

    sda_deg = 23.45 * np.sin(np.deg2rad((360/365) * (dataset["day_of_year"] - 81)))
    dataset["SDA"] = xr.DataArray(
        np.deg2rad(sda_deg),
        dims=["valid_time"],
        coords={"valid_time": dataset["valid_time"]}
    )

    return dataset



def calc_solar_zenith_angle(dataset: xr.Dataset) -> xr.Dataset:
    
    lat = np.deg2rad(dataset["latitude"])  
    sda = dataset["SDA"]
    ha = dataset["hour_angle"]

    cos_sza = (
        np.sin(lat) * np.sin(sda)
        + np.cos(lat) * np.cos(sda) * np.cos(ha)
    )

    cos_sza = cos_sza.clip(-1, 1)
    dataset["SZA"] = np.arccos(cos_sza)
    return dataset



def calc_toa_incident_shortwave(dataset : xr.Dataset) ->  xr.Dataset:

    influx = 1361 * dataset["ECF"] * np.cos(dataset["SZA"])
    dataset["toa_influx"] = influx.clip(min=0)

    return dataset



