"""One 5-day forecast run: fetch live data, forecast 120 hours ahead, write the files the app reads.

    python -m airwise.five_day

Reads the OpenAQ key from the environment variable OPENAQ_API_KEY.

How the forecast is made
    * Delhi and Kolkata: a LightGBM model (models/horizon_model_v4.txt) predicts a log-ratio against the average
      of the last 24 hours, for every hour ahead h = 1..120.
    * Hyderabad: the model did not beat a simple baseline, so the forecast is the seasonal baseline (the value at
      the same hour of day in the most recent 24 hours).
    * Ranges: forecast x multipliers per city and day ahead (models/horizon_calibration.json), learned from how far
      past forecasts were off. Hourly and daily-average ranges have their own multipliers.

Statuses in data/latest.json, per city
    ok                    fresh data, forecast for the days ahead
    stale                 newest reading is older than MAX_AGE_HOURS; days still ahead are shown
    outdated              so old that no 24-hour window is still ahead
    insufficient_history  less than a week of unbroken data (gap of more than 3 hours)
    weather_unavailable   weather or forecast weather could not be fetched
    no_data               no sensor readings could be fetched
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from airwise import features as F
from airwise import horizons as H
from airwise.data_sources import (CITY_COORDS, HISTORY_DAYS, OPEN_METEO_PREVIOUS_RUNS_URL, FetchError,
                                  _hourly_frame, _request, fetch_all_sensors, fetch_observed_weather, load_sensors)

MODEL_CITIES = ("Delhi", "Kolkata")      # everyone else uses the seasonal baseline
MAX_AGE_HOURS = 6.0
MIN_HOURS_PER_DAY = 18                   # a 24-hour window needs this many forecast hours, and this many still ahead
EXTRA = ["seas", "mean24", "seas_minus_cur", "seas_minus_mean24", "cur_minus_mean24"]
FEATS = H.H_FEATURES + EXTRA

LIMITS = [
    "Forecasts show a typical value; on polluted days the real value is often higher.",
    "Ranges covered roughly 8 in 10 outcomes in past tests overall, but less in some stretches, autumn especially.",
    "Hyderabad uses a simple baseline, because the model did not beat it.",
    "The model's edge over simple baselines is modest and was measured on about one year of data.",
]


class ModelError(Exception):
    """The saved model files do not match this code."""


def add_extra(table: pd.DataFrame) -> pd.DataFrame:
    """The five extra inputs used in training (all known at the starting hour)."""
    out = table.copy()
    out["seas"] = out["base_seasonal"]
    out["mean24"] = out["base_mean24"]
    out["seas_minus_cur"] = out["base_seasonal"] - out["pm25"]
    out["seas_minus_mean24"] = out["base_seasonal"] - out["base_mean24"]
    out["cur_minus_mean24"] = out["pm25"] - out["base_mean24"]
    return out


class FiveDayModel:
    def __init__(self, models_dir):
        d = Path(models_dir)
        self.meta = json.loads((d / "horizon_meta.json").read_text())
        self.calib = json.loads((d / "horizon_calibration.json").read_text())
        if self.meta["features"] != FEATS:
            raise ModelError("feature list in models/horizon_meta.json differs from airwise/five_day.py")
        self.booster = lgb.Booster(model_file=str(d / "horizon_model_v4.txt"))

    def point_forecast(self, table: pd.DataFrame) -> pd.DataFrame:
        """table: output of horizons.make_horizon_table. Returns city, h, target_time, fc (ug/m3), method."""
        if table.empty:
            return pd.DataFrame(columns=["city", "h", "target_time", "fc", "method"])
        x = add_extra(table)
        numeric = [c for c in FEATS if c != "city_code"]
        m = x[numeric].apply(pd.to_numeric, errors="raise").astype(float)
        m["city_code"] = pd.Categorical(x["city"].astype(str).to_numpy(), categories=F.CITIES)
        m = m[FEATS]
        if m.isna().any().any():
            raise ModelError("model inputs contain missing values")
        out = self.booster.predict(m)
        model_fc = np.clip(np.expm1(np.log1p(x["base_mean24"].to_numpy(float)) + out), 0, None)
        use_model = x["city"].isin(MODEL_CITIES).to_numpy()
        fc = np.where(use_model, model_fc, x["base_seasonal"].to_numpy(float))
        return pd.DataFrame({"city": x["city"].to_numpy(), "h": x["h"].to_numpy(), "target_time": x["target_time"].to_numpy(),
                             "fc": fc, "method": np.where(use_model, "model", "seasonal_baseline")})

    def hourly_ranges(self, fc: pd.DataFrame) -> pd.DataFrame:
        out = fc.copy()
        day = (out["h"] - 1) // 24 + 1
        lo, hi = [], []
        for city, d in zip(out["city"], day):
            q = self.calib["hourly"][city][str(int(d))]
            lo.append(q[0])
            hi.append(q[1])
        out["low"], out["high"] = out["fc"] * np.array(lo), out["fc"] * np.array(hi)
        return out

    def daily_summary(self, fc: pd.DataFrame, now_ist) -> pd.DataFrame:
        """One row per city and 24-hour window that is still (mostly) ahead of now."""
        rows = []
        for city, g in fc.groupby("city"):
            g = g.assign(day=(g["h"] - 1) // 24 + 1)
            for d, w in g.groupby("day"):
                future = int((pd.to_datetime(w["target_time"]) > pd.Timestamp(now_ist)).sum())
                if len(w) < MIN_HOURS_PER_DAY or future < MIN_HOURS_PER_DAY:
                    continue
                mean = float(w["fc"].mean())
                q = self.calib["daily"][city][str(int(d))]
                rows.append({"city": city, "day": int(d), "window_start": pd.to_datetime(w["target_time"]).min(),
                             "window_end": pd.to_datetime(w["target_time"]).max(), "hours": len(w),
                             "forecast": mean, "low": mean * q[0], "high": mean * q[1]})
        return pd.DataFrame(rows, columns=["city", "day", "window_start", "window_end", "hours", "forecast", "low", "high"])


def fetch_leads(city, past_days=HISTORY_DAYS, forecast_days=7, session=None, sleep=time.sleep):
    """Weather forecasts issued 2..6 days ahead, for every hour. Columns: city, hour_ist, <var>_previous_day<K>."""
    import requests
    session = session or requests.Session()
    lat, lon = CITY_COORDS[city]
    cols = [f"{v}_previous_day{k}" for v in F.FORECAST_VARS for k in H.LEAD_CHOICES]
    r = _request(session, OPEN_METEO_PREVIOUS_RUNS_URL,
                 params={"latitude": lat, "longitude": lon, "hourly": ",".join(cols), "past_days": past_days,
                         "forecast_days": forecast_days, "timezone": "Asia/Kolkata"}, sleep=sleep)
    df = _hourly_frame(r.json(), cols, f"forecast weather {city}")
    df["city"] = city
    return df[["city", "hour_ist"] + cols]


def _iso(ts):
    return None if pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M")


def _r(x, nd=1):
    return None if pd.isna(x) else round(float(x), nd)


def run(sensors_path, models_dir, out_dir, api_key, now_utc=None, session=None, sleep=time.sleep, pause=1.1,
        max_age_hours=MAX_AGE_HOURS):
    now_utc = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    now_ist = now_utc.tz_convert("Asia/Kolkata").tz_localize(None)

    model = FiveDayModel(models_dir)
    sensors = load_sensors(sensors_path)
    excluded = set(sensors.loc[sensors["excluded"], "sensor_id"])
    raw, failures = fetch_all_sensors(sensors, now_utc, api_key, session=session, pause=pause, sleep=sleep)
    failures = [(str(sid), msg) for sid, msg in failures]

    weather_frames, lead_frames, weather_ok = [], [], {}
    for city in F.CITIES:
        try:
            weather_frames.append(fetch_observed_weather(city, now_ist, session=session, sleep=sleep))
            lead_frames.append(fetch_leads(city, session=session, sleep=sleep))
            weather_ok[city] = True
        except FetchError as ex:
            weather_ok[city] = False
            failures.append((f"weather:{city}", str(ex)))

    origin_age, hourly, daily = {}, pd.DataFrame(), pd.DataFrame()
    if len(raw) and weather_frames:
        clean = F.clean_sensor_hours(raw)
        gold = F.aggregate_city_hourly(clean, excluded)
        if len(gold):
            series = F.build_model_series(gold)
            feats = F.build_features(series, pd.concat(weather_frames, ignore_index=True))
            newest = feats.loc[feats.groupby("city")["hour_ist"].idxmax()]
            origins = H.complete_origins(newest)
            for _, r in origins.iterrows():
                origin_age[r["city"]] = (now_ist - r["hour_ist"]) / pd.Timedelta(hours=1)
            if len(origins):
                table = H.make_horizon_table(origins, series, pd.concat(lead_frames, ignore_index=True), with_target=False)
                table = table.dropna(subset=H.FC_COLS)   # hours without forecast weather get no forecast
                fc = model.point_forecast(table)
                hourly = model.hourly_ranges(fc)
                daily = model.daily_summary(fc, now_ist)
                newest_info = origins.set_index("city")

    requested = sensors[~sensors["excluded"]].groupby("city").size().to_dict()
    with_data = (raw.dropna(subset=["pm25"]).groupby("city")["sensor_id"].nunique().to_dict() if len(raw) else {})
    cities = {}
    for city in F.CITIES:
        info = {"method": "model" if city in MODEL_CITIES else "seasonal_baseline",
                "range_level": model.calib["levels"].get(city),
                "sensors_requested": int(requested.get(city, 0)), "sensors_with_data": int(with_data.get(city, 0))}
        city_daily = daily[daily["city"] == city] if len(daily) else daily
        if with_data.get(city, 0) == 0:
            info["status"] = "no_data"
        elif not weather_ok.get(city, False):
            info["status"] = "weather_unavailable"
        elif city not in origin_age:
            info["status"] = "insufficient_history"
        else:
            o = newest_info.loc[city]
            info.update({"as_of_ist": _iso(o["hour_ist"]), "age_hours": _r(origin_age[city]),
                         "current_pm25": _r(o["pm25"]),
                         "sensors_in_latest_hour": int(o["n_sensors"]) if pd.notna(o["n_sensors"]) else None})
            has_forecast = len(hourly) > 0 and bool((hourly["city"] == city).any())
            if not has_forecast:
                info["status"] = "weather_unavailable"          # the origin was fine, but no hour had forecast weather
            elif len(city_daily) == 0:
                info["status"] = "outdated"
            else:
                city_hourly = hourly[(hourly["city"] == city) & (pd.to_datetime(hourly["target_time"]) > now_ist)]
                info["status"] = "stale" if origin_age[city] > max_age_hours else "ok"
                info["hourly"] = [{"t": _iso(r.target_time), "forecast": _r(r.fc), "low": _r(r.low), "high": _r(r.high)}
                                  for r in city_hourly.sort_values("target_time").itertuples()]
                info["daily"] = [{"day": int(r.day), "from": _iso(r.window_start), "to": _iso(r.window_end),
                                  "hours": int(r.hours), "forecast": _r(r.forecast), "low": _r(r.low), "high": _r(r.high)}
                                 for r in city_daily.sort_values("day").itertuples()]
        cities[city] = info

    result = {"schema": 2, "generated_at_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "horizon_hours": H.HORIZON_MAX,
              "model_trained_through": model.meta.get("train_origins_to"), "cities": cities,
              "fetch_failures": [{"what": w, "message": m} for w, m in failures], "limits": LIMITS}
    _write_outputs(Path(out_dir), result)
    return result


def _write_outputs(out_dir: Path, result: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "latest.json.tmp"
    tmp.write_text(json.dumps(result, indent=1))
    os.replace(tmp, out_dir / "latest.json")
    rows = []
    for city, info in result["cities"].items():
        if info.get("status") in ("ok", "stale"):
            for d in info["daily"]:
                rows.append({"generated_at_utc": result["generated_at_utc"], "city": city, "as_of_ist": info["as_of_ist"],
                             "age_hours": info["age_hours"], "day": d["day"], "window_start": d["from"], "window_end": d["to"],
                             "forecast": d["forecast"], "low": d["low"], "high": d["high"], "status": info["status"],
                             "method": info["method"]})
    if not rows:
        return
    path = out_dir / "forecast_log_5day.csv"
    new = pd.DataFrame(rows)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.drop_duplicates(["city", "as_of_ist", "day"], keep="last").to_csv(path, index=False)


def main():
    key = (os.environ.get("OPENAQ_API_KEY") or "").strip()
    if not key:
        print("OPENAQ_API_KEY is not set", file=sys.stderr)
        return 1
    result = run("config/sensors.csv", "models", "data", key)
    for city, info in result["cities"].items():
        days = [(d["day"], d["forecast"]) for d in info.get("daily", [])]
        print(city, info["status"], days)
    if result["fetch_failures"]:
        print(len(result["fetch_failures"]), "fetch problem(s); see data/latest.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
