"""Live data fetching for AirWise India: OpenAQ sensor readings and Open-Meteo weather.

Everything that touches the network lives here, so the rest of the code can be tested
without internet. Functions take a `session` (anything with a requests-style .get) and
a `sleep` function so tests can replace both.

Rules this module follows:
  * the OpenAQ API key is passed in from outside and never appears in error messages
  * temporary failures (timeouts, 408, 429, 5xx) are retried; other errors fail at once
  * one failing sensor does not stop the run; failures are returned for the caller to judge
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from airwise.features import FORECAST_VARS, WEATHER_COLUMNS

OPENAQ_URL = "https://api.openaq.org/v3"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# City-centre coordinates used for the training weather (keep identical).
CITY_COORDS = {"Delhi": (28.61, 77.21), "Hyderabad": (17.38, 78.48), "Kolkata": (22.57, 88.36)}

HISTORY_DAYS = 16  # more than the 168 hours (7 days) the longest lag needs, plus a margin


class FetchError(Exception):
    """A request failed for good (retries used up, or an error that retrying cannot fix)."""


def _request(session, url, params=None, headers=None, attempts=6, sleep=time.sleep):
    last = "no response"
    for attempt in range(1, attempts + 1):
        try:
            r = session.get(url, params=params, headers=headers, timeout=90)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as ex:
            last = type(ex).__name__
            sleep(5 * attempt)
            continue
        if r.status_code == 429:
            last = "429 rate limited"
            try:
                wait = int(r.headers.get("Retry-After", 60))
            except (TypeError, ValueError):
                wait = 60
            sleep(wait)
            continue
        if r.status_code == 408 or r.status_code >= 500:
            last = str(r.status_code)
            sleep(5 * attempt)
            continue
        if r.status_code != 200:
            raise FetchError(f"HTTP {r.status_code} from {url}: {str(r.text)[:200]}")
        return r
    raise FetchError(f"gave up after {attempts} attempts ({last}) for {url}")


def load_sensors(path: str) -> pd.DataFrame:
    """config/sensors.csv -> DataFrame with columns city, location_id, name, sensor_id, excluded."""
    df = pd.read_csv(path)
    df["excluded"] = df["excluded"].astype(str).str.lower().isin(["true", "1"])
    df["sensor_id"] = df["sensor_id"].astype(int)
    return df


def fetch_sensor_hours(sensor_id, start_utc, end_utc, api_key, session=None, pause=1.1,
                       sleep=time.sleep, limit=1000) -> pd.DataFrame:
    """Hourly PM2.5 for one sensor. Columns: sensor_id, hour_start_utc, pm25."""
    session = session or requests.Session()
    rows, page = [], 1
    while True:
        r = _request(session, f"{OPENAQ_URL}/sensors/{int(sensor_id)}/hours",
                     params={"datetime_from": pd.Timestamp(start_utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "datetime_to": pd.Timestamp(end_utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "limit": limit, "page": page},
                     headers={"X-API-Key": api_key}, sleep=sleep)
        results = r.json().get("results", [])
        for x in results:
            value = x.get("value")
            rows.append((int(sensor_id), x["period"]["datetimeFrom"]["utc"],
                         np.nan if value is None else float(value)))
        sleep(pause)
        if len(results) < limit:
            break
        page += 1
    df = pd.DataFrame(rows, columns=["sensor_id", "hour_start_utc", "pm25"])
    df["hour_start_utc"] = pd.to_datetime(df["hour_start_utc"], utc=True)
    return df


def fetch_all_sensors(sensors: pd.DataFrame, now_utc, api_key, days=HISTORY_DAYS, session=None,
                      pause=1.1, sleep=time.sleep):
    """Fetch every non-excluded sensor. Returns (raw_df, failures).

    raw_df columns: city, sensor_id, hour_start_utc, pm25
    failures: list of (sensor_id, message). Excluded sensors are skipped: they never
    contribute to the city value, so fetching them would only waste requests.
    """
    session = session or requests.Session()
    now_utc = pd.Timestamp(now_utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    start, end = now_utc - pd.Timedelta(days=days), now_utc + pd.Timedelta(hours=1)
    frames, failures = [], []
    for row in sensors[~sensors["excluded"]].itertuples():
        try:
            df = fetch_sensor_hours(row.sensor_id, start, end, api_key, session=session,
                                    pause=pause, sleep=sleep)
        except FetchError as ex:
            failures.append((int(row.sensor_id), str(ex)))
            continue
        df["city"] = row.city
        frames.append(df)
    if frames:
        raw = pd.concat(frames, ignore_index=True)[["city", "sensor_id", "hour_start_utc", "pm25"]]
    else:
        raw = pd.DataFrame(columns=["city", "sensor_id", "hour_start_utc", "pm25"])
    return raw, failures


def _hourly_frame(payload, needed, label):
    hourly = payload.get("hourly") or {}
    missing = [c for c in ["time"] + list(needed) if c not in hourly]
    if missing:
        raise FetchError(f"{label}: response is missing {missing}")
    df = pd.DataFrame({c: hourly[c] for c in ["time"] + list(needed)})
    df["hour_ist"] = pd.to_datetime(df.pop("time"))
    return df


def fetch_observed_weather(city, now_ist, past_days=HISTORY_DAYS, session=None, sleep=time.sleep):
    """Weather up to the current hour. Columns: city, hour_ist, WEATHER_COLUMNS.

    Hours after now_ist are dropped on purpose: the model must only see the past.
    """
    session = session or requests.Session()
    lat, lon = CITY_COORDS[city]
    r = _request(session, OPEN_METEO_FORECAST_URL,
                 params={"latitude": lat, "longitude": lon, "hourly": ",".join(WEATHER_COLUMNS),
                         "past_days": past_days, "forecast_days": 1, "timezone": "Asia/Kolkata"},
                 sleep=sleep)
    df = _hourly_frame(r.json(), WEATHER_COLUMNS, f"observed weather {city}")
    df = df[df["hour_ist"] <= pd.Timestamp(now_ist)].copy()
    df["city"] = city
    return df[["city", "hour_ist"] + WEATHER_COLUMNS]


def fetch_forecast_day2(city, past_days=HISTORY_DAYS, session=None, sleep=time.sleep):
    """The forecast issued two days ahead, for each hour. Columns: city, hour_ist, <var>_previous_day2."""
    session = session or requests.Session()
    lat, lon = CITY_COORDS[city]
    cols = [f"{v}_previous_day2" for v in FORECAST_VARS]
    r = _request(session, OPEN_METEO_PREVIOUS_RUNS_URL,
                 params={"latitude": lat, "longitude": lon, "hourly": ",".join(cols),
                         "past_days": past_days, "forecast_days": 3, "timezone": "Asia/Kolkata"},
                 sleep=sleep)
    df = _hourly_frame(r.json(), cols, f"day-2 forecast {city}")
    df["city"] = city
    return df[["city", "hour_ist"] + cols]
