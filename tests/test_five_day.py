"""Tests for airwise/five_day.py: the 5-day forecast run, with a fake world and small dummy models."""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from airwise import features as F
from airwise import five_day as D
from airwise import horizons as H

SECRET = "TEST-KEY-DO-NOT-LEAK"
NOW = pd.Timestamp("2026-10-15T12:00:00Z")          # 17:30 in India
NOW_IST = pd.Timestamp("2026-10-15T17:30")
NO_SLEEP = lambda s: None
SENSOR_CITY = {11: "Delhi", 12: "Delhi", 21: "Hyderabad", 22: "Hyderabad", 31: "Kolkata", 32: "Kolkata", 99: "Delhi"}
BASE = {"Delhi": 80.0, "Hyderabad": 30.0, "Kolkata": 50.0}
LO, HI = 0.8, 1.3                                     # dummy hourly range multipliers
DLO, DHI = 0.9, 1.2                                   # dummy daily range multipliers (different on purpose)


# ------------------------------------------------------------------ fake world
class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._p, self.headers, self.text = status, payload or {}, {}, text

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(url)
        return self.handler(url, params, headers)


def make_world(scenario="healthy"):
    def openaq(url, params):
        sid = int(url.split("/sensors/")[1].split("/")[0])
        if scenario == "all_fail":
            return FakeResponse(503)
        city = SENSOR_CITY[sid]
        start, end = pd.Timestamp(params["datetime_from"]), pd.Timestamp(params["datetime_to"])
        t = pd.date_range(start.floor("h") + pd.Timedelta(minutes=30), min(end, NOW - pd.Timedelta(hours=1)), freq="h")
        t = t[t >= start]
        if scenario == "kolkata_outage" and city == "Kolkata":
            t = t[(t < NOW - pd.Timedelta(hours=60)) | (t > NOW - pd.Timedelta(hours=54))]
        if scenario == "feed_outage40":
            t = t[t <= NOW - pd.Timedelta(hours=40)]
        if scenario == "feed_outage150":
            t = t[t <= NOW - pd.Timedelta(hours=150)]
        vals = BASE[city] + 10 * np.sin(2 * np.pi * (t.hour + 0.5) / 24) + (sid % 5)
        return FakeResponse(200, {"results": [{"value": float(v), "period": {"datetimeFrom": {
            "utc": x.strftime("%Y-%m-%dT%H:%M:%SZ")}}} for x, v in zip(t, vals)]})

    def open_meteo(url, params):
        if scenario == "delhi_weather_fail" and abs(params["latitude"] - 28.61) < 0.01:
            return FakeResponse(500)
        today = NOW_IST.normalize()
        times = pd.date_range(today - pd.Timedelta(days=params["past_days"]),
                              today + pd.Timedelta(days=params["forecast_days"]) - pd.Timedelta(hours=1), freq="h")
        if scenario == "short_leads" and "previous-runs" in url:
            times = times[times <= NOW_IST + pd.Timedelta(hours=60)]
        out = {"time": [x.strftime("%Y-%m-%dT%H:%M") for x in times]}
        n = len(times)
        for v in params["hourly"].split(","):
            if v.startswith("relative_humidity"):
                out[v] = [int(60 + 20 * np.sin(i / 7)) for i in range(n)]
            elif v.startswith("precipitation"):
                out[v] = [0.0] * n
            elif v.startswith("wind_direction"):
                out[v] = [float((i * 7) % 360) for i in range(n)]
            elif v.startswith("surface_pressure"):
                out[v] = [1000 + 3 * np.sin(i / 11) for i in range(n)]
            elif v.startswith("cloud_cover"):
                out[v] = [int(50 + 40 * np.sin(i / 5)) for i in range(n)]
            else:
                out[v] = [20 + 5 * np.sin(i / 9) for i in range(n)]
        return FakeResponse(200, {"hourly": out})

    return FakeSession(lambda url, params, headers: openaq(url, params) if "openaq" in url else open_meteo(url, params))


def make_models(dirpath):
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({c: rng.normal(size=n) for c in D.FEATS if c != "city_code"})
    X["city_code"] = pd.Categorical(rng.choice(F.CITIES, n), categories=F.CITIES)
    y = rng.normal(scale=0.2, size=n)
    booster = lgb.train({"objective": "regression", "min_data_in_leaf": 10, "num_leaves": 15, "verbose": -1},
                        lgb.Dataset(X[D.FEATS], label=y), num_boost_round=20)
    booster.save_model(str(dirpath / "horizon_model_v4.txt"))
    hourly = {c: {str(d): [LO, HI] for d in range(1, 6)} for c in F.CITIES}
    daily = {c: {str(d): [DLO, DHI] for d in range(1, 6)} for c in F.CITIES}
    calib = {"levels": {"Delhi": 0.8, "Kolkata": 0.9, "Hyderabad": 0.9}, "hourly": hourly, "daily": daily}
    (dirpath / "horizon_calibration.json").write_text(json.dumps(calib))
    (dirpath / "horizon_meta.json").write_text(json.dumps({"features": D.FEATS, "train_origins_to": "2026-10-02 13:00:00",
                                                           "trees": 20}))
    return dirpath


@pytest.fixture
def env(tmp_path):
    models = make_models(tmp_path)
    sensors = tmp_path / "sensors.csv"
    sensors.write_text("city,location_id,name,sensor_id,excluded\n" + "\n".join(
        f"{c},{i},s{i},{sid},{'True' if sid == 99 else 'False'}" for i, (sid, c) in enumerate(SENSOR_CITY.items())))
    return {"models": models, "sensors": sensors, "out": tmp_path / "data"}


def go(env, scenario="healthy", **kw):
    return D.run(env["sensors"], env["models"], env["out"], SECRET, now_utc=NOW, session=make_world(scenario),
                 sleep=NO_SLEEP, pause=0, **kw)


# ------------------------------------------- a small table to test the maths directly
START = pd.Timestamp("2025-03-01 00:00")


def horizon_table(n_hours=500):
    rng = np.random.default_rng(1)
    gold = []
    for ci, city in enumerate(F.CITIES):
        idx = np.arange(n_hours)
        gold.append(pd.DataFrame({"city": city, "hour_ist": START + pd.to_timedelta(idx, unit="h"),
                                  "pm25_city_median": 40 + 15 * np.sin(idx / 24 * 2 * np.pi) + 10 * ci, "n_sensors": 10.0}))
    series = F.build_model_series(pd.concat(gold, ignore_index=True))
    span = pd.date_range(START, periods=n_hours + 240, freq="h")
    weather, leads = [], []
    for city in F.CITIES:
        n = len(span)
        weather.append(pd.DataFrame({"city": city, "hour_ist": span, "temperature_2m": 25 + rng.normal(0, 2, n),
                                     "relative_humidity_2m": rng.integers(30, 90, n), "dew_point_2m": 15 + rng.normal(0, 1, n),
                                     "wind_speed_10m": np.abs(rng.normal(8, 3, n)), "wind_direction_10m": rng.uniform(0, 360, n),
                                     "precipitation": np.zeros(n), "surface_pressure": 1000 + rng.normal(0, 3, n),
                                     "cloud_cover": rng.integers(0, 100, n)}))
        d = {"city": city, "hour_ist": span}
        for k in H.LEAD_CHOICES:
            d[f"temperature_2m_previous_day{k}"] = 25 + rng.normal(0, 2, n)
            d[f"relative_humidity_2m_previous_day{k}"] = rng.integers(30, 90, n)
            d[f"wind_speed_10m_previous_day{k}"] = np.abs(rng.normal(8, 3, n))
            d[f"precipitation_previous_day{k}"] = np.zeros(n)
            d[f"surface_pressure_previous_day{k}"] = 1000 + rng.normal(0, 3, n)
        leads.append(pd.DataFrame(d))
    feats = F.build_features(series, pd.concat(weather, ignore_index=True))
    newest = feats.loc[feats.groupby("city")["hour_ist"].idxmax()]
    return H.make_horizon_table(newest, series, pd.concat(leads, ignore_index=True), with_target=False)


@pytest.fixture(scope="module")
def table():
    return horizon_table()


# ------------------------------------------------------------------- the maths
def test_model_cities_use_the_log_ratio_model_and_hyderabad_the_recent_average(env, table):
    m = D.FiveDayModel(env["models"])
    fc = m.point_forecast(table)
    x = D.add_extra(table)
    hy = fc["city"] == "Hyderabad"
    assert (fc.loc[hy, "method"] == "recent_average").all() and (fc.loc[~hy, "method"] == "model").all()
    assert np.allclose(fc.loc[hy, "fc"].to_numpy(), x.loc[hy.to_numpy(), "base_mean24"].to_numpy())
    assert fc.loc[hy, "fc"].nunique() == 1                      # held flat across all 120 hours
    # Delhi and Kolkata: forecast = expm1(log1p(mean24) + model output), never below zero
    xm = x[~hy.to_numpy()]
    M = xm[[c for c in D.FEATS if c != "city_code"]].astype(float)
    M["city_code"] = pd.Categorical(xm["city"].to_numpy(), categories=F.CITIES)
    raw = m.booster.predict(M[D.FEATS])
    expected = np.clip(np.expm1(np.log1p(xm["base_mean24"].to_numpy(float)) + raw), 0, None)
    assert np.allclose(fc.loc[~hy, "fc"].to_numpy(), expected)
    assert (fc["fc"] >= 0).all() and len(fc) == len(table)


def test_hourly_ranges_apply_the_multipliers_for_the_right_day(env, table):
    m = D.FiveDayModel(env["models"])
    calib = json.loads((env["models"] / "horizon_calibration.json").read_text())
    calib["hourly"]["Delhi"]["3"] = [0.5, 2.0]
    m.calib = calib
    r = m.hourly_ranges(m.point_forecast(table))
    day3 = r[(r["city"] == "Delhi") & ((r["h"] - 1) // 24 + 1 == 3)]
    assert np.allclose(day3["low"], day3["fc"] * 0.5) and np.allclose(day3["high"], day3["fc"] * 2.0)
    other = r[(r["city"] == "Delhi") & ((r["h"] - 1) // 24 + 1 == 1)]
    assert np.allclose(other["low"], other["fc"] * LO) and np.allclose(other["high"], other["fc"] * HI)


def test_daily_summary_is_the_mean_of_the_hourly_forecasts_with_daily_multipliers(env, table):
    m = D.FiveDayModel(env["models"])
    fc = m.point_forecast(table)
    origin = pd.to_datetime(table["hour_ist"]).iloc[0]
    d = m.daily_summary(fc, origin + pd.Timedelta(minutes=30))
    assert sorted(d["day"].unique()) == [1, 2, 3, 4, 5] and (d["hours"] == 24).all()
    row = d[(d["city"] == "Kolkata") & (d["day"] == 2)].iloc[0]
    hours = fc[(fc["city"] == "Kolkata") & (fc["h"] >= 25) & (fc["h"] <= 48)]["fc"]
    assert np.isclose(row["forecast"], hours.mean())
    assert np.isclose(row["low"], hours.mean() * DLO) and np.isclose(row["high"], hours.mean() * DHI)


def test_daily_windows_that_are_mostly_past_are_dropped(env, table):
    m = D.FiveDayModel(env["models"])
    fc = m.point_forecast(table)
    origin = pd.to_datetime(table["hour_ist"]).iloc[0]
    d = m.daily_summary(fc, origin + pd.Timedelta(hours=41, minutes=30))   # hours 1..41 are past
    assert sorted(d["day"].unique()) == [3, 4, 5]
    assert m.daily_summary(fc, origin + pd.Timedelta(hours=200)).empty


def test_mismatched_feature_list_is_refused(env):
    meta = json.loads((env["models"] / "horizon_meta.json").read_text())
    meta["features"] = meta["features"][::-1]
    (env["models"] / "horizon_meta.json").write_text(json.dumps(meta))
    with pytest.raises(D.ModelError):
        D.FiveDayModel(env["models"])


# ------------------------------------------------------------------- the run
def test_healthy_run_gives_five_days_for_every_city(env):
    res = go(env)
    assert res["schema"] == 2 and res["fetch_failures"] == []
    for city in F.CITIES:
        c = res["cities"][city]
        assert c["status"] == "ok", (city, c)
        assert [d["day"] for d in c["daily"]] == [1, 2, 3, 4, 5]
        assert len(c["hourly"]) >= 118 and all(pd.Timestamp(h["t"]) > NOW_IST for h in c["hourly"])
        assert all(d["low"] <= d["forecast"] <= d["high"] for d in c["daily"])
        assert all(h["low"] <= h["forecast"] <= h["high"] for h in c["hourly"])
        assert c["as_of_ist"] == "2026-10-15T16:00" and c["age_hours"] == 1.5
    assert res["cities"]["Delhi"]["method"] == "model" and res["cities"]["Hyderabad"]["method"] == "recent_average"
    on_disk = json.loads((env["out"] / "latest.json").read_text())
    assert on_disk["cities"]["Delhi"]["status"] == "ok" and on_disk["limits"] == D.LIMITS


def test_data_two_days_old_still_shows_the_days_that_remain(env):
    res = go(env, "feed_outage40")
    for city in F.CITIES:
        c = res["cities"][city]
        assert c["status"] == "stale" and c["age_hours"] > 40, (city, c)
        assert [d["day"] for d in c["daily"]] == [3, 4, 5]
        assert len(c["hourly"]) == 80 and pd.Timestamp(c["hourly"][0]["t"]) > NOW_IST


def test_data_far_too_old_is_outdated_with_no_forecast(env):
    res = go(env, "feed_outage150")
    for city in F.CITIES:
        c = res["cities"][city]
        assert c["status"] == "outdated" and c["age_hours"] > 140 and "daily" not in c and "hourly" not in c
    assert not (env["out"] / "forecast_log_5day.csv").exists()


def test_recent_gap_means_insufficient_history_for_that_city_only(env):
    res = go(env, "kolkata_outage")
    assert res["cities"]["Kolkata"]["status"] == "insufficient_history" and "daily" not in res["cities"]["Kolkata"]
    assert res["cities"]["Delhi"]["status"] == "ok" and res["cities"]["Hyderabad"]["status"] == "ok"


def test_weather_failure_is_reported_for_that_city(env):
    res = go(env, "delhi_weather_fail")
    assert res["cities"]["Delhi"]["status"] == "weather_unavailable" and res["cities"]["Kolkata"]["status"] == "ok"
    assert any(f["what"] == "weather:Delhi" for f in res["fetch_failures"])


def test_forecast_weather_that_stops_early_shortens_the_forecast(env):
    res = go(env, "short_leads")
    for city in F.CITIES:
        c = res["cities"][city]
        assert c["status"] == "ok" and [d["day"] for d in c["daily"]] == [1, 2], (city, c)
        assert max(pd.Timestamp(h["t"]) for h in c["hourly"]) <= NOW_IST + pd.Timedelta(hours=61)


def test_total_data_outage_does_not_crash(env):
    res = go(env, "all_fail")
    assert all(res["cities"][c]["status"] == "no_data" for c in F.CITIES)
    assert (env["out"] / "latest.json").exists() and not (env["out"] / "forecast_log_5day.csv").exists()


def test_excluded_sensor_is_never_requested(env):
    sess = make_world()
    D.run(env["sensors"], env["models"], env["out"], SECRET, now_utc=NOW, session=sess, sleep=NO_SLEEP, pause=0)
    assert not any("/sensors/99/" in u for u in sess.calls)


def test_log_has_one_row_per_city_and_day_and_reruns_do_not_duplicate(env):
    go(env)
    go(env)
    log = pd.read_csv(env["out"] / "forecast_log_5day.csv")
    assert len(log) == 15 and set(log["city"]) == set(F.CITIES) and sorted(log["day"].unique()) == [1, 2, 3, 4, 5]


def test_api_key_is_never_written_to_disk(env):
    go(env)
    go(env, "all_fail")
    for p in env["out"].rglob("*"):
        if p.is_file():
            assert SECRET not in p.read_text()


def test_main_needs_the_key_and_strips_it(monkeypatch):
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)
    assert D.main() == 1
    seen = {}
    monkeypatch.setattr(D, "run", lambda s, m, o, key, **kw: seen.update(key=key) or {"cities": {}, "fetch_failures": []})
    monkeypatch.setenv("OPENAQ_API_KEY", "  abc123\n")
    assert D.main() == 0 and seen["key"] == "abc123"
