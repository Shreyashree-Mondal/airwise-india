"""One forecast run: fetch live data, build the model inputs, forecast, write the files the app reads.

    python -m airwise.pipeline

Reads the OpenAQ key from the environment variable OPENAQ_API_KEY (never from a file).
Writes, inside data/:
    latest.json         what the app shows
    forecast_log.csv    one row per city per run (to compare forecasts with outcomes later)
    inputs_log.csv      the exact model inputs used (to spot differences from training later)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from airwise import features as F
from airwise.data_sources import (FetchError, fetch_all_sensors, fetch_forecast_day2,
                                  fetch_observed_weather, load_sensors)
from airwise.model import ForecastModels

# How old the newest complete hour may be before the forecast is marked stale.
# This is a first guess; adjust it after looking at the real delay of the data feed.
MAX_AGE_HOURS = 6.0


def _iso(ts):
    return None if pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M")


def _num(x, nd=1):
    return None if pd.isna(x) else round(float(x), nd)


def run(sensors_path, models_dir, out_dir, api_key, now_utc=None, session=None,
        sleep=time.sleep, pause=1.1, max_age_hours=MAX_AGE_HOURS):
    now_utc = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    now_ist = now_utc.tz_convert("Asia/Kolkata").tz_localize(None)

    models = ForecastModels(models_dir)
    sensors = load_sensors(sensors_path)
    excluded = set(sensors.loc[sensors["excluded"], "sensor_id"])

    raw, failures = fetch_all_sensors(sensors, now_utc, api_key, session=session, pause=pause, sleep=sleep)
    failures = [(str(sid), msg) for sid, msg in failures]

    weather_frames, forecast_frames, weather_ok = [], [], {}
    for city in F.CITIES:
        try:
            weather_frames.append(fetch_observed_weather(city, now_ist, session=session, sleep=sleep))
            forecast_frames.append(fetch_forecast_day2(city, session=session, sleep=sleep))
            weather_ok[city] = True
        except FetchError as ex:
            weather_ok[city] = False
            failures.append((f"weather:{city}", str(ex)))

    rows = pd.DataFrame()
    if len(raw) and weather_frames:
        clean = F.clean_sensor_hours(raw)
        gold = F.aggregate_city_hourly(clean, excluded)
        if len(gold):
            series = F.build_model_series(gold)
            weather = pd.concat(weather_frames, ignore_index=True)
            forecast = F.prepare_forecast_day2(pd.concat(forecast_frames, ignore_index=True))
            feats = F.add_forecast_features(F.build_features(series, weather), forecast)
            rows = F.latest_feature_rows(feats, now_ist, max_age_hours=max_age_hours)
    preds = models.predict(rows)

    requested = sensors[~sensors["excluded"]].groupby("city").size().to_dict()
    with_data = (raw.dropna(subset=["pm25"]).groupby("city")["sensor_id"].nunique().to_dict()
                 if len(raw) else {})

    cities = {}
    for city in F.CITIES:
        info = {"sensors_requested": int(requested.get(city, 0)),
                "sensors_with_data": int(with_data.get(city, 0))}
        row = rows[rows["city"] == city] if len(rows) else rows
        if with_data.get(city, 0) == 0:
            info["status"] = "no_data"
        elif not weather_ok.get(city, False):
            info["status"] = "weather_unavailable"
        elif row.empty:
            info["status"] = "insufficient_history"
        else:
            r, p = row.iloc[0], preds[preds["city"] == city].iloc[0]
            info.update({
                "status": "stale" if bool(r["is_stale"]) else "ok",
                "as_of_ist": _iso(r["hour_ist"]),
                "target_ist": _iso(r["target_time"]),
                "age_hours": _num(r["age_hours"]),
                "current_pm25": _num(r["pm25"]),
                "forecast_pm25": _num(p["pred"]),
                "range_low": _num(p["lo"]),
                "range_high": _num(p["hi"]),
                "sensors_in_latest_hour": int(r["n_sensors"]) if pd.notna(r["n_sensors"]) else None,
            })
        cities[city] = info

    result = {
        "generated_at_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_hours": F.HORIZON_H,
        "interval_nominal": models.meta["interval"].get("nominal"),
        "model_trained_through": models.meta.get("train_to"),
        "cities": cities,
        "fetch_failures": [{"what": w, "message": m} for w, m in failures],
        "limits": models.meta.get("limits", []),
    }
    _write_outputs(Path(out_dir), result, rows, preds, now_utc)
    return result


def _write_outputs(out_dir: Path, result: dict, rows: pd.DataFrame, preds: pd.DataFrame, now_utc):
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "latest.json.tmp"
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, out_dir / "latest.json")
    if rows.empty:
        return
    good_cities = [c for c, i in result["cities"].items() if i["status"] in ("ok", "stale")]
    ok = rows[rows["city"].isin(good_cities)]
    if ok.empty:
        return
    gen = result["generated_at_utc"]
    p = preds.set_index("city")
    log = pd.DataFrame({
        "generated_at_utc": gen, "city": ok["city"].to_numpy(),
        "as_of_ist": [_iso(x) for x in ok["hour_ist"]], "target_ist": [_iso(x) for x in ok["target_time"]],
        "current_pm25": ok["pm25"].astype(float).to_numpy(),
        "forecast_pm25": p.loc[ok["city"], "pred"].to_numpy(),
        "range_low": p.loc[ok["city"], "lo"].to_numpy(), "range_high": p.loc[ok["city"], "hi"].to_numpy(),
        "status": [result["cities"][c]["status"] for c in ok["city"]]})
    inputs = ok[["city", "hour_ist"] + [c for c in F.FEATURES if c != "city_code"]].copy()
    inputs.insert(0, "generated_at_utc", gen)
    inputs["hour_ist"] = [_iso(x) for x in inputs["hour_ist"]]
    for name, new, keys in (("forecast_log.csv", log, ["city", "as_of_ist"]),
                            ("inputs_log.csv", inputs, ["city", "hour_ist"])):
        path = out_dir / name
        if path.exists():
            new = pd.concat([pd.read_csv(path), new], ignore_index=True)
        new.drop_duplicates(keys, keep="last").to_csv(path, index=False)


def main():
    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        print("OPENAQ_API_KEY is not set", file=sys.stderr)
        return 1
    result = run("config/sensors.csv", "models", "data", key)
    for city, info in result["cities"].items():
        print(city, info["status"], info.get("forecast_pm25"))
    if result["fetch_failures"]:
        print(len(result["fetch_failures"]), "fetch problem(s); see data/latest.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
