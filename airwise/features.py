"""Feature building for AirWise India.

This module repeats, step by step, what the Databricks notebooks did in training:

    raw sensor hours -> cleaned sensor hours -> city-hourly series -> model features

It uses pandas only and never touches the network, so every step can be tested.
If you change a rule here, the model's inputs change, so change the training
notebooks in the same way (and rerun the parity test in tests/).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

CITIES = ["Delhi", "Hyderabad", "Kolkata"]
HORIZON_H = 24
LAGS = [1, 2, 3, 6, 12, 24, 48, 144, 168]
MAX_FILL_H = 3  # gaps of up to this many hours are filled by straight-line interpolation

# Order matters: LightGBM reads DataFrame columns by position.
FEATURES = [
    "pm25", "lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "lag_24", "lag_48", "lag_144", "lag_168",
    "roll_mean_6", "roll_mean_24", "roll_std_24", "diff_1", "diff_6", "precip_24h",
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "wind_speed_10m", "precipitation",
    "surface_pressure", "cloud_cover", "wind_dir_sin", "wind_dir_cos", "hour_sin", "hour_cos",
    "dow", "city_code",
    "fc2_temp", "fc2_rh", "fc2_wind", "fc2_prec", "fc2_pres",
    "fc2_dtemp", "fc2_dwind", "fc2_dpres", "fc2_drh",
]

WEATHER_COLUMNS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "wind_speed_10m",
    "wind_direction_10m", "precipitation", "surface_pressure", "cloud_cover",
]

# forecast variable -> short name used in the fc2_* feature columns
FORECAST_VARS = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "rh",
    "wind_speed_10m": "wind",
    "precipitation": "prec",
    "surface_pressure": "pres",
}


def lower_median(values) -> float:
    """Median that picks the lower middle value when the count is even.

    Training used Spark's percentile_approx(x, 0.5). On small groups it returns the
    element at rank ceil(0.5 * n), which is the lower median. pandas' median would
    average the two middle values and give slightly different numbers.
    """
    v = np.asarray(values, dtype=float)
    v = np.sort(v[~np.isnan(v)])
    if len(v) == 0:
        return float("nan")
    return float(v[math.ceil(0.5 * len(v)) - 1])


def clean_sensor_hours(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw OpenAQ sensor hours -> one clean reading per sensor per clock hour (Indian time).

    raw columns: city, sensor_id, hour_start_utc, pm25
    Rules (same as training):
      * keep only rows whose UTC start minute is 30 (one reading per hour; the old
        overlapping half-hour windows are dropped)
      * drop duplicate (sensor, time) rows
      * convert to Indian time (UTC+5:30), so :30 UTC becomes :00 IST
      * zero and values of 999 or more are treated as missing (instrument faults)
    """
    df = raw.copy()
    t = pd.to_datetime(df["hour_start_utc"], utc=True)
    df["hour_start_utc"] = t
    df = df[t.dt.minute == 30].drop_duplicates(["sensor_id", "hour_start_utc"]).copy()
    df["hour_ist"] = df["hour_start_utc"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    ok = (df["pm25"] > 0) & (df["pm25"] < 999)
    df["pm25_clean"] = df["pm25"].where(ok)
    return df[["city", "sensor_id", "hour_ist", "pm25", "pm25_clean"]].reset_index(drop=True)


def aggregate_city_hourly(clean: pd.DataFrame, excluded_sensors) -> pd.DataFrame:
    """Cleaned sensor hours -> one value per city per hour.

    Steps (same as training):
      1. sensors flagged as excluded never contribute
      2. for each city-hour, take the median of the remaining sensors
      3. a reading of 100 or more is dropped when it is more than 10x that median
         (and at least 5 sensors reported that hour): isolated spikes are faults
      4. the city value is the median of what is left
    Returns columns: city, hour_ist, pm25_city_median, n_sensors
    """
    d = clean.copy()
    d["sensor_excluded"] = d["sensor_id"].isin(set(excluded_sensors))
    good = d[~d["sensor_excluded"] & d["pm25_clean"].notna()]
    med = (good.groupby(["city", "hour_ist"])["pm25_clean"]
               .agg(city_median=lower_median, n_sensors="size").reset_index())
    d = d.merge(med, on=["city", "hour_ist"], how="left")
    ratio = d["pm25_clean"] / d["city_median"]
    big = (d["n_sensors"] >= 5) & (d["pm25_clean"] >= 100)
    d["pm25_final"] = d["pm25_clean"]
    d.loc[d["sensor_excluded"], "pm25_final"] = np.nan
    d.loc[big & (ratio > 10), "pm25_final"] = np.nan
    out = (d[d["pm25_final"].notna()].groupby(["city", "hour_ist"])["pm25_final"]
             .agg(pm25_city_median=lower_median, n_sensors="size").reset_index())
    return out


def build_model_series(gold: pd.DataFrame) -> pd.DataFrame:
    """City-hourly values -> complete hourly series per city, split into unbroken segments.

    * a full hourly grid is built from the first to the last hour of each city
    * gaps of MAX_FILL_H hours or fewer are filled by straight-line interpolation
    * longer gaps stay empty, and every unbroken stretch gets its own segment_id
    Returns columns: city, hour_ist, pm25, observed, filled, n_sensors, segment_id
    """
    parts = []
    for city, d in gold.groupby("city"):
        d = d.sort_values("hour_ist").set_index("hour_ist")
        d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="h"))
        d.index.name = "hour_ist"
        d["city"] = city
        raw = d["pm25_city_median"]
        isna = raw.isna()
        run_id = (isna != isna.shift()).cumsum()
        short = isna & (isna.groupby(run_id).transform("size") <= MAX_FILL_H)
        d["pm25"] = raw.where(~short, raw.interpolate(limit_area="inside"))
        d["observed"] = ~isna
        d["filled"] = short & d["pm25"].notna()
        d["segment_id"] = d["pm25"].isna().cumsum()
        parts.append(d[d["pm25"].notna()].reset_index()[
            ["city", "hour_ist", "pm25", "observed", "filled", "n_sensors", "segment_id"]])
    return pd.concat(parts, ignore_index=True)


def build_features(series: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Series + current weather -> lag, rolling and calendar features.

    weather columns: city, hour_ist and every name in WEATHER_COLUMNS (observed weather).
    Rows without weather are dropped before the lags are computed, as in training.
    Lags are taken inside each (city, segment_id), so they never cross a long gap.
    The 'target' column is NaN for the newest hours, which is expected when forecasting.
    """
    d = (series.merge(weather[["city", "hour_ist"] + WEATHER_COLUMNS], on=["city", "hour_ist"], how="left")
               .dropna(subset=["temperature_2m"])
               .sort_values(["city", "segment_id", "hour_ist"]).reset_index(drop=True))
    grp = d.groupby(["city", "segment_id"])
    for k in LAGS:
        d[f"lag_{k}"] = grp["pm25"].shift(k)
    d["roll_mean_6"] = grp["pm25"].transform(lambda s: s.rolling(6).mean())
    d["roll_mean_24"] = grp["pm25"].transform(lambda s: s.rolling(24).mean())
    d["roll_std_24"] = grp["pm25"].transform(lambda s: s.rolling(24).std())
    d["diff_1"] = d["pm25"] - d["lag_1"]
    d["diff_6"] = d["pm25"] - d["lag_6"]
    d["precip_24h"] = grp["precipitation"].transform(lambda s: s.rolling(24, min_periods=1).sum())
    h = d["hour_ist"].dt.hour
    d["hour_sin"] = np.sin(2 * np.pi * h / 24)
    d["hour_cos"] = np.cos(2 * np.pi * h / 24)
    d["dow"] = d["hour_ist"].dt.dayofweek
    d["month"] = d["hour_ist"].dt.month
    wd = np.deg2rad(d["wind_direction_10m"])
    d["wind_dir_sin"] = np.sin(wd)
    d["wind_dir_cos"] = np.cos(wd)
    d["target"] = grp["pm25"].shift(-HORIZON_H)
    d["target_time"] = d["hour_ist"] + pd.Timedelta(hours=HORIZON_H)
    return d


def prepare_forecast_day2(raw_forecast: pd.DataFrame) -> pd.DataFrame:
    """Open-Meteo 'previous runs' output -> columns city, target_time, fc2_*.

    raw_forecast columns: city, hour_ist (the hour the forecast is FOR) and
    '<variable>_previous_day2' for each variable in FORECAST_VARS.
    """
    cols = {f"{v}_previous_day2": f"fc2_{s}" for v, s in FORECAST_VARS.items()}
    return (raw_forecast[["city", "hour_ist"] + list(cols)]
            .rename(columns=cols).rename(columns={"hour_ist": "target_time"}))


def add_forecast_features(features: pd.DataFrame, forecast_day2: pd.DataFrame) -> pd.DataFrame:
    """Attach the day-2 forecast for the target hour and its difference from current weather."""
    f = features.merge(forecast_day2, on=["city", "target_time"], how="left")
    f["fc2_dtemp"] = f["fc2_temp"] - f["temperature_2m"]
    f["fc2_dwind"] = f["fc2_wind"] - f["wind_speed_10m"]
    f["fc2_dpres"] = f["fc2_pres"] - f["surface_pressure"]
    f["fc2_drh"] = f["fc2_rh"] - f["relative_humidity_2m"]
    f["city_code"] = pd.Categorical(f["city"], categories=CITIES)
    return f


def latest_feature_rows(features: pd.DataFrame, now_ist, max_age_hours: float = 6.0) -> pd.DataFrame:
    """The newest row per city that has every model input, plus how old it is.

    A city is missing from the result when no row is complete, for example when
    less than a week of unbroken history exists. Callers must handle that case.
    """
    rows = []
    for city, g in features.groupby("city"):
        g = g.sort_values("hour_ist")
        g = g[g[FEATURES].notna().all(axis=1)]
        if g.empty:
            continue
        last = g.iloc[-1].copy()
        last["age_hours"] = (pd.Timestamp(now_ist) - last["hour_ist"]) / pd.Timedelta(hours=1)
        last["is_stale"] = bool(last["age_hours"] > max_age_hours)
        rows.append(last)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["city_code"] = pd.Categorical(out["city"], categories=CITIES)
    return out
