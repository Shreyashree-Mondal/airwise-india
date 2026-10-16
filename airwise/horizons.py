"""Training and forecasting table for the 5-day hourly model.

One row = one starting hour (the "origin") and one number of hours ahead, h = 1..120.
The model learns the CHANGE in PM2.5 between the origin hour and hour origin + h:

    forecast(origin + h) = pm25(origin) + model(features, h)

Everything an origin row knows was available at the origin hour. The weather forecast for the target
hour is the one issued ceil(h / 24) + 1 days before that hour, which is always at least 24 hours before
the origin (a safety margin for publication delay).

Uses the same origin features as the 24-hour model (airwise.features.build_features).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from airwise.features import CITIES, FEATURES, FORECAST_VARS

HORIZON_MAX = 120
LEAD_CHOICES = (2, 3, 4, 5, 6)

ORIGIN_COLS = [c for c in FEATURES if c != "city_code" and not c.startswith("fc2_")]
FC_COLS = ["fc_temp", "fc_rh", "fc_wind", "fc_prec", "fc_pres", "fc_dtemp", "fc_dwind", "fc_dpres", "fc_drh"]
H_FEATURES = ORIGIN_COLS + ["city_code", "h", "tgt_hour_sin", "tgt_hour_cos", "tgt_dow"] + FC_COLS
BASELINES = ["base_persist", "base_seasonal", "base_mean24"]


def lead_days(h: int) -> int:
    """Days of lead for the weather forecast used at h hours ahead: 1-24 -> 2, 25-48 -> 3, ... 97-120 -> 6."""
    return -(-int(h) // 24) + 1


def complete_origins(features: pd.DataFrame) -> pd.DataFrame:
    """Rows of build_features output that have every origin feature (a week of unbroken history)."""
    ok = features[ORIGIN_COLS].notna().all(axis=1)
    return features[ok]


def make_horizon_table(features: pd.DataFrame, series: pd.DataFrame, leads: pd.DataFrame,
                       horizons=range(1, HORIZON_MAX + 1), origin_step: int = 1,
                       with_target: bool = True) -> pd.DataFrame:
    """Build the long table described in the module docstring.

    features: output of features.build_features (origin features, observed weather at the origin)
    series:   output of features.build_model_series (city, hour_ist, pm25), used to look up targets
    leads:    city, hour_ist and '<var>_previous_day<K>' columns for K in LEAD_CHOICES (forecast weather)
    origin_step: keep every n-th origin hour per city (saves memory; 1 keeps all)
    with_target: False for live forecasting (targets are unknown, so no rows are dropped for lacking one)
    """
    base = complete_origins(features).sort_values(["city", "hour_ist"])
    if origin_step > 1:
        base = base[base.groupby("city").cumcount() % origin_step == 0]
    base = base[["city", "segment_id", "hour_ist"] + ORIGIN_COLS].reset_index(drop=True)

    look = series[["city", "hour_ist", "pm25"]]
    parts = []
    for h in horizons:
        k = lead_days(h)
        d = base.copy()
        d["h"] = int(h)
        d["target_time"] = d["hour_ist"] + pd.Timedelta(hours=int(h))

        cols = {f"{v}_previous_day{k}": f"fc_{s}" for v, s in FORECAST_VARS.items()}
        fc = leads[["city", "hour_ist"] + list(cols)].rename(columns=cols).rename(columns={"hour_ist": "target_time"})
        d = d.merge(fc, on=["city", "target_time"], how="left")
        d["fc_dtemp"] = d["fc_temp"] - d["temperature_2m"]
        d["fc_dwind"] = d["fc_wind"] - d["wind_speed_10m"]
        d["fc_dpres"] = d["fc_pres"] - d["surface_pressure"]
        d["fc_drh"] = d["fc_rh"] - d["relative_humidity_2m"]

        hr = d["target_time"].dt.hour
        d["tgt_hour_sin"] = np.sin(2 * np.pi * hr / 24)
        d["tgt_hour_cos"] = np.cos(2 * np.pi * hr / 24)
        d["tgt_dow"] = d["target_time"].dt.dayofweek

        # baselines known at the origin
        d["base_persist"] = d["pm25"]
        d["base_mean24"] = d["roll_mean_24"]
        same_clock = d["hour_ist"] + pd.Timedelta(hours=int(h) - 24 * (-(-int(h) // 24)))  # same hour of day, in the last 24 h
        sn = look.rename(columns={"hour_ist": "_t", "pm25": "base_seasonal"})
        d["_t"] = same_clock
        d = d.merge(sn, left_on=["city", "_t"], right_on=["city", "_t"], how="left").drop(columns="_t")

        if with_target:
            tg = look.rename(columns={"hour_ist": "target_time", "pm25": "target_pm25"})
            d = d.merge(tg, on=["city", "target_time"], how="inner")
            d["y"] = d["target_pm25"] - d["pm25"]
        parts.append(d)

    out = pd.concat(parts, ignore_index=True)
    out["city_code"] = pd.Categorical(out["city"], categories=CITIES)
    first = ["city", "segment_id", "hour_ist", "target_time", "h"]
    rest = [c for c in out.columns if c not in first]
    return out[first + rest]


def to_float32(table: pd.DataFrame) -> pd.DataFrame:
    """Halve the memory of the numeric columns (training does not need float64)."""
    out = table.copy()
    for c in out.columns:
        if out[c].dtype == "float64":
            out[c] = out[c].astype("float32")
    return out
