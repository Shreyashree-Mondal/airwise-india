"""Tests for airwise/data_sources.py. No real network access: a fake session answers requests."""
import numpy as np
import pandas as pd
import pytest
import requests

from airwise import data_sources as D
from airwise.features import FORECAST_VARS, WEATHER_COLUMNS

SECRET = "TEST-KEY-DO-NOT-LEAK"
NO_SLEEP = lambda s: None


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, text=""):
        self.status_code, self._payload, self.headers, self.text = status, payload or {}, headers or {}, text

    def json(self):
        return self._payload


class FakeSession:
    """Answers each call from a queue (or a function), and records every call."""

    def __init__(self, responses=None, handler=None):
        self.responses, self.handler, self.calls = list(responses or []), handler, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        if self.handler:
            return self.handler(url, params, headers)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def aq_results(n, start="2026-10-01T00:30:00Z", value=50.0):
    t0 = pd.Timestamp(start)
    return {"results": [{"value": value + i, "period": {"datetimeFrom": {"utc": (t0 + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")}}}
                        for i in range(n)]}


# ------------------------------------------------------------ retry behaviour
def test_retries_on_server_error_then_succeeds():
    s = FakeSession([FakeResponse(500), FakeResponse(408), FakeResponse(200, aq_results(3))])
    df = D.fetch_sensor_hours(1, "2026-10-01", "2026-10-02", SECRET, session=s, sleep=NO_SLEEP)
    assert len(df) == 3 and len(s.calls) == 3


def test_rate_limit_waits_retry_after_seconds():
    waits = []
    s = FakeSession([FakeResponse(429, headers={"Retry-After": "17"}), FakeResponse(200, aq_results(1))])
    D.fetch_sensor_hours(1, "2026-10-01", "2026-10-02", SECRET, session=s, sleep=waits.append)
    assert 17 in waits


def test_timeouts_are_retried():
    s = FakeSession([requests.exceptions.Timeout(), requests.exceptions.ConnectionError(), FakeResponse(200, aq_results(2))])
    assert len(D.fetch_sensor_hours(1, "2026-10-01", "2026-10-02", SECRET, session=s, sleep=NO_SLEEP)) == 2


def test_gives_up_after_attempts_and_never_leaks_the_key():
    s = FakeSession(handler=lambda *a: FakeResponse(503))
    with pytest.raises(D.FetchError) as e:
        D.fetch_sensor_hours(1, "2026-10-01", "2026-10-02", SECRET, session=s, sleep=NO_SLEEP)
    assert "gave up" in str(e.value) and SECRET not in str(e.value)


def test_client_error_fails_at_once_without_retry_and_without_key():
    s = FakeSession([FakeResponse(401, text="unauthorized")])
    with pytest.raises(D.FetchError) as e:
        D.fetch_sensor_hours(1, "2026-10-01", "2026-10-02", SECRET, session=s, sleep=NO_SLEEP)
    assert len(s.calls) == 1 and "401" in str(e.value) and SECRET not in str(e.value)


# ------------------------------------------------------------------ OpenAQ
def test_pagination_collects_every_page_and_parses_values():
    first = aq_results(4, "2026-10-01T00:30:00Z")
    first["results"][1]["value"] = None  # a missing value must become NaN
    s = FakeSession([FakeResponse(200, first), FakeResponse(200, aq_results(2, "2026-10-01T04:30:00Z"))])
    df = D.fetch_sensor_hours(7, "2026-10-01", "2026-10-02", SECRET, session=s, sleep=NO_SLEEP, limit=4)
    assert len(df) == 6 and s.calls[1]["params"]["page"] == 2
    assert df["pm25"].isna().sum() == 1
    assert (df["sensor_id"] == 7).all()
    assert str(df["hour_start_utc"].dt.tz) == "UTC"
    assert s.calls[0]["headers"] == {"X-API-Key": SECRET}


def test_fetch_all_skips_excluded_and_collects_failures():
    sensors = pd.DataFrame({"city": ["Delhi", "Delhi", "Kolkata", "Kolkata"], "location_id": [1, 2, 3, 4],
                            "name": list("abcd"), "sensor_id": [11, 12, 13, 14],
                            "excluded": [False, True, False, False]})

    def handler(url, params, headers):
        if "/sensors/13/" in url:
            return FakeResponse(503)
        return FakeResponse(200, aq_results(3))

    s = FakeSession(handler=handler)
    raw, failures = D.fetch_all_sensors(sensors, "2026-10-15T12:00:00Z", SECRET, session=s, pause=0, sleep=NO_SLEEP)
    assert set(raw["sensor_id"]) == {11, 14}  # 12 excluded, 13 failed
    assert [f[0] for f in failures] == [13] and SECRET not in failures[0][1]
    assert not any("/sensors/12/" in c["url"] for c in s.calls)
    assert set(raw.columns) == {"city", "sensor_id", "hour_start_utc", "pm25"}


def test_fetch_all_window_covers_history_days():
    sensors = pd.DataFrame({"city": ["Delhi"], "location_id": [1], "name": ["a"], "sensor_id": [11], "excluded": [False]})
    s = FakeSession(handler=lambda *a: FakeResponse(200, aq_results(1)))
    D.fetch_all_sensors(sensors, "2026-10-15T12:00:00Z", SECRET, session=s, pause=0, sleep=NO_SLEEP)
    p = s.calls[0]["params"]
    assert p["datetime_from"] == "2026-09-29T12:00:00Z" and p["datetime_to"] == "2026-10-15T13:00:00Z"
    assert D.HISTORY_DAYS * 24 > 168


def test_load_sensors_reads_flags(tmp_path):
    p = tmp_path / "sensors.csv"
    p.write_text("city,location_id,name,sensor_id,excluded\nDelhi,1,A,11,False\nDelhi,2,B,12,True\n")
    df = D.load_sensors(str(p))
    assert df["excluded"].tolist() == [False, True] and df["sensor_id"].dtype.kind == "i"


# ----------------------------------------------------------------- Open-Meteo
def om_payload(variables, start="2026-10-14T00:00", hours=48):
    t = pd.date_range(start, periods=hours, freq="h")
    out = {"time": [x.strftime("%Y-%m-%dT%H:%M") for x in t]}
    for v in variables:
        out[v] = list(np.arange(hours, dtype=float))
    return {"hourly": out}


def test_observed_weather_drops_future_hours_and_asks_for_ist():
    s = FakeSession([FakeResponse(200, om_payload(WEATHER_COLUMNS))])
    df = D.fetch_observed_weather("Delhi", "2026-10-14T23:00", session=s, sleep=NO_SLEEP)
    assert df["hour_ist"].max() == pd.Timestamp("2026-10-14T23:00")
    assert list(df.columns) == ["city", "hour_ist"] + WEATHER_COLUMNS
    assert (df["city"] == "Delhi").all()
    p = s.calls[0]["params"]
    assert p["timezone"] == "Asia/Kolkata" and (p["latitude"], p["longitude"]) == D.CITY_COORDS["Delhi"]


def test_forecast_day2_returns_previous_day2_columns():
    cols = [f"{v}_previous_day2" for v in FORECAST_VARS]
    s = FakeSession([FakeResponse(200, om_payload(cols))])
    df = D.fetch_forecast_day2("Kolkata", session=s, sleep=NO_SLEEP)
    assert list(df.columns) == ["city", "hour_ist"] + cols and len(df) == 48
    assert s.calls[0]["url"] == D.OPEN_METEO_PREVIOUS_RUNS_URL
    assert s.calls[0]["params"]["forecast_days"] == 3


def test_missing_variable_in_response_is_an_error():
    cols = [f"{v}_previous_day2" for v in FORECAST_VARS][:-1]
    s = FakeSession([FakeResponse(200, om_payload(cols))])
    with pytest.raises(D.FetchError):
        D.fetch_forecast_day2("Delhi", session=s, sleep=NO_SLEEP)


def test_output_plugs_into_feature_code():
    """Fetched frames must be accepted by the feature module as they are."""
    from airwise import features as F
    cols = [f"{v}_previous_day2" for v in FORECAST_VARS]
    fc = D.fetch_forecast_day2("Delhi", session=FakeSession([FakeResponse(200, om_payload(cols))]), sleep=NO_SLEEP)
    out = F.prepare_forecast_day2(fc)
    assert {"city", "target_time", "fc2_temp", "fc2_rh", "fc2_wind", "fc2_prec", "fc2_pres"} <= set(out.columns)
