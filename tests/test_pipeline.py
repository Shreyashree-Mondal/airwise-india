"""Tests for airwise/model.py and airwise/pipeline.py.

A fake world answers every web request, so the whole run (fetch -> clean -> features ->
forecast -> files) is exercised without internet. The models here are small dummy models
with the real feature list: they test the plumbing, not forecast quality.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from airwise import features as F
from airwise import pipeline as P
from airwise.model import ForecastModels, ModelError

SECRET = "TEST-KEY-DO-NOT-LEAK"
NOW = pd.Timestamp("2026-10-15T12:00:00Z")
NO_SLEEP = lambda s: None
SENSOR_CITY = {11: "Delhi", 12: "Delhi", 21: "Hyderabad", 22: "Hyderabad", 31: "Kolkata", 32: "Kolkata", 99: "Delhi"}
BASE = {"Delhi": 80.0, "Hyderabad": 30.0, "Kolkata": 50.0}


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
        last = NOW - pd.Timedelta(hours=1)  # a reading is complete one hour after its start
        t = pd.date_range(start.floor("h") + pd.Timedelta(minutes=30), min(end, last), freq="h")
        t = t[t >= start]
        if scenario == "kolkata_outage" and city == "Kolkata":
            t = t[(t < NOW - pd.Timedelta(hours=60)) | (t > NOW - pd.Timedelta(hours=54))]
        if scenario == "hyderabad_stale" and city == "Hyderabad":
            t = t[t <= NOW - pd.Timedelta(hours=12)]
        vals = BASE[city] + 10 * np.sin(2 * np.pi * (t.hour + 0.5) / 24) + (sid % 5)
        results = [{"value": float(v), "period": {"datetimeFrom": {"utc": x.strftime("%Y-%m-%dT%H:%M:%SZ")}}}
                   for x, v in zip(t, vals)]
        return FakeResponse(200, {"results": results})

    def open_meteo(url, params):
        if scenario == "delhi_weather_fail" and abs(params["latitude"] - 28.61) < 0.01:
            return FakeResponse(500)
        today = NOW.tz_convert("Asia/Kolkata").tz_localize(None).normalize()
        times = pd.date_range(today - pd.Timedelta(days=params["past_days"]),
                              today + pd.Timedelta(days=params["forecast_days"]) - pd.Timedelta(hours=1), freq="h")
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

    def handler(url, params, headers):
        return openaq(url, params) if "openaq" in url else open_meteo(url, params)

    return FakeSession(handler)


def make_models(dirpath, qn=None):
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({c: rng.normal(size=n) for c in F.FEATURES if c != "city_code"})
    X["city_code"] = pd.Categorical(rng.choice(F.CITIES, n), categories=F.CITIES)
    y = rng.normal(size=n)
    for name, extra in {"point": {"objective": "regression"},
                        "q10": {"objective": "quantile", "alpha": 0.1},
                        "q90": {"objective": "quantile", "alpha": 0.9}}.items():
        params = {"min_data_in_leaf": 10, "num_leaves": 15, "verbose": -1, **extra}
        booster = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=15)
        booster.save_model(str(dirpath / f"model_{name}.txt"))
    meta = {"horizon_hours": 24, "features": F.FEATURES, "cities": F.CITIES, "train_to": "2026-10-02",
            "interval": {"nominal": 0.8, "width_offset": 1.0,
                         "qn_by_city": qn or {"Delhi": 0.17, "Hyderabad": 0.02, "Kolkata": 0.03}},
            "limits": ["test limit"]}
    (dirpath / "meta.json").write_text(json.dumps(meta))
    return dirpath


@pytest.fixture
def env(tmp_path):
    models = make_models(tmp_path)
    sensors = tmp_path / "sensors.csv"
    sensors.write_text("city,location_id,name,sensor_id,excluded\n" + "\n".join(
        f"{c},{i},s{i},{sid},{'True' if sid == 99 else 'False'}" for i, (sid, c) in enumerate(SENSOR_CITY.items())))
    return {"models": models, "sensors": sensors, "out": tmp_path / "data"}


def go(env, scenario="healthy", **kw):
    return P.run(env["sensors"], env["models"], env["out"], SECRET, now_utc=NOW,
                 session=make_world(scenario), sleep=NO_SLEEP, pause=0, **kw)


# ----------------------------------------------------------------------- model
def test_interval_rule_matches_training_rule(env):
    models = ForecastModels(env["models"])
    rng = np.random.default_rng(5)
    rows = pd.DataFrame({c: rng.normal(size=6) for c in F.FEATURES if c != "city_code"})
    rows["pm25"] = [10, 20, 30, 60, 90, 120.0]
    rows["city"] = ["Delhi", "Delhi", "Hyderabad", "Hyderabad", "Kolkata", "Kolkata"]
    out = models.predict(rows)
    X = ForecastModels._matrix(rows)
    base = X["pm25"].to_numpy()
    a = base + models.boosters["q10"].predict(X)
    b = base + models.boosters["q90"].predict(X)
    lo0, hi0 = np.clip(np.minimum(a, b), 0, None), np.clip(np.maximum(a, b), 0, None)
    w = hi0 - lo0 + 1.0
    qn = rows["city"].map({"Delhi": 0.17, "Hyderabad": 0.02, "Kolkata": 0.03}).to_numpy()
    np.testing.assert_allclose(out["lo"], np.clip(lo0 - qn * w, 0, None))
    np.testing.assert_allclose(out["hi"], hi0 + qn * w)
    assert (out["lo"] <= out["hi"]).all() and (out["pred"] >= 0).all() and (out["lo"] >= 0).all()


def test_forecasts_and_ranges_never_go_below_zero(env):
    models = ForecastModels(env["models"])
    rng = np.random.default_rng(11)
    n = 60
    rows = pd.DataFrame({c: rng.normal(scale=3, size=n) for c in F.FEATURES if c != "city_code"})
    rows["pm25"] = 0.0  # clean air: model changes can push the raw forecast below zero
    rows["city"] = (F.CITIES * n)[:n]
    raw = models.boosters["point"].predict(ForecastModels._matrix(rows))
    assert (raw < 0).any(), "test setup should produce some negative raw outputs"
    out = models.predict(rows)
    assert (out["pred"] >= 0).all() and (out["lo"] >= 0).all() and (out["hi"] >= out["lo"]).all()


def test_larger_calibration_constant_widens_the_range(tmp_path):
    narrow_dir, wide_dir = tmp_path / "narrow", tmp_path / "wide"
    narrow_dir.mkdir()
    wide_dir.mkdir()
    narrow = ForecastModels(make_models(narrow_dir))
    wide = ForecastModels(make_models(wide_dir, qn={"Delhi": 1.0, "Hyderabad": 1.0, "Kolkata": 1.0}))
    rng = np.random.default_rng(1)
    rows = pd.DataFrame({c: rng.normal(size=3) for c in F.FEATURES if c != "city_code"})
    rows["pm25"] = [40.0, 40.0, 40.0]
    rows["city"] = F.CITIES
    n, w = narrow.predict(rows), wide.predict(rows)
    assert ((w["hi"] - w["lo"]) > (n["hi"] - n["lo"])).all()


def test_mismatched_feature_list_is_refused(env):
    meta = json.loads((env["models"] / "meta.json").read_text())
    meta["features"] = meta["features"][::-1]
    (env["models"] / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ModelError):
        ForecastModels(env["models"])


# -------------------------------------------------------------------- pipeline
def test_healthy_run_forecasts_every_city(env):
    res = go(env)
    assert res["fetch_failures"] == []
    for city in F.CITIES:
        c = res["cities"][city]
        assert c["status"] == "ok", (city, c)
        assert c["as_of_ist"] == "2026-10-15T16:00" and c["target_ist"] == "2026-10-16T16:00"
        assert c["age_hours"] == 1.5
        assert c["sensors_requested"] == 2 and c["sensors_with_data"] == 2
        assert c["range_low"] <= c["range_high"] and c["forecast_pm25"] >= 0
    on_disk = json.loads((env["out"] / "latest.json").read_text())
    assert on_disk["cities"]["Delhi"]["status"] == "ok" and on_disk["limits"] == ["test limit"]


def test_excluded_sensor_is_never_requested(env):
    sess = make_world()
    P.run(env["sensors"], env["models"], env["out"], SECRET, now_utc=NOW, session=sess, sleep=NO_SLEEP, pause=0)
    assert not any("/sensors/99/" in u for u in sess.calls)


def test_logs_are_written_and_reruns_do_not_duplicate(env):
    go(env)
    go(env)
    log = pd.read_csv(env["out"] / "forecast_log.csv")
    inputs = pd.read_csv(env["out"] / "inputs_log.csv")
    assert len(log) == 3 and set(log["city"]) == set(F.CITIES)
    assert len(inputs) == 3
    assert [c for c in F.FEATURES if c != "city_code"][0] in inputs.columns


def test_api_key_is_never_written_to_disk(env):
    go(env)
    go(env, "all_fail")
    for p in env["out"].rglob("*"):
        if p.is_file():
            assert SECRET not in p.read_text()


def test_recent_gap_means_insufficient_history_for_that_city_only(env):
    res = go(env, "kolkata_outage")
    assert res["cities"]["Kolkata"]["status"] == "insufficient_history"
    assert "forecast_pm25" not in res["cities"]["Kolkata"]
    assert res["cities"]["Delhi"]["status"] == "ok" and res["cities"]["Hyderabad"]["status"] == "ok"
    log = pd.read_csv(env["out"] / "forecast_log.csv")
    assert "Kolkata" not in set(log["city"])


def test_old_data_is_marked_stale(env):
    res = go(env, "hyderabad_stale")
    c = res["cities"]["Hyderabad"]
    assert c["status"] == "stale" and c["age_hours"] > 6 and "forecast_pm25" in c
    assert res["cities"]["Delhi"]["status"] == "ok"


def test_weather_failure_is_reported_for_that_city(env):
    res = go(env, "delhi_weather_fail")
    assert res["cities"]["Delhi"]["status"] == "weather_unavailable"
    assert res["cities"]["Kolkata"]["status"] == "ok"
    assert any(f["what"] == "weather:Delhi" for f in res["fetch_failures"])


def test_total_data_outage_does_not_crash(env):
    res = go(env, "all_fail")
    assert all(res["cities"][c]["status"] == "no_data" for c in F.CITIES)
    assert len(res["fetch_failures"]) >= 6
    assert (env["out"] / "latest.json").exists()
    assert not (env["out"] / "forecast_log.csv").exists()


def test_main_requires_the_api_key(monkeypatch):
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)
    assert P.main() == 1
