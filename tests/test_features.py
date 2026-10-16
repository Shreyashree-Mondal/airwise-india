"""Tests for airwise/features.py.

Part 1 checks the cleaning rules on small hand-made cases.
Part 2 checks that the feature code gives the same numbers as the original training
notebook code (copied below as 'reference') on synthetic data with gaps.
A final test compares the feature list with models/meta.json when that file exists.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airwise import features as F


# ---------------------------------------------------------------- part 1: rules
def test_lower_median_picks_lower_middle():
    assert F.lower_median([1, 2, 3, 4]) == 2
    assert F.lower_median([1, 2, 3]) == 2
    assert F.lower_median([5]) == 5
    assert F.lower_median([4, 1, 3, 2]) == 2
    assert np.isnan(F.lower_median([]))
    assert F.lower_median([np.nan, 1, 3]) == 1


def test_clean_keeps_only_half_hour_rows_and_converts_to_ist():
    raw = pd.DataFrame({
        "city": ["Delhi"] * 5,
        "sensor_id": [1, 1, 1, 1, 1],
        "hour_start_utc": ["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z", "2026-01-01T00:30:00Z",
                           "2026-01-01T01:30:00Z", "2026-01-01T02:30:00Z"],
        "pm25": [10.0, 20.0, 20.0, 0.0, 999.0],
    })
    out = F.clean_sensor_hours(raw)
    # :00 row dropped, duplicate :30 row dropped -> 3 rows left
    assert len(out) == 3
    assert out["hour_ist"].iloc[0] == pd.Timestamp("2026-01-01 06:00")  # 00:30 UTC = 06:00 IST
    assert out["hour_ist"].dt.minute.eq(0).all()
    assert out["pm25_clean"].iloc[0] == 20.0
    assert np.isnan(out["pm25_clean"].iloc[1])  # zero -> missing
    assert np.isnan(out["pm25_clean"].iloc[2])  # 999 -> missing
    assert out["pm25"].iloc[1] == 0.0  # raw value is kept


def _clean_rows(values, hour="2026-01-01 06:00", city="Delhi", first_sensor=1):
    return pd.DataFrame({
        "city": city,
        "sensor_id": range(first_sensor, first_sensor + len(values)),
        "hour_ist": pd.Timestamp(hour),
        "pm25": values,
        "pm25_clean": values,
    })


def test_spike_removed_when_more_than_10x_median_and_enough_sensors():
    clean = _clean_rows([20, 22, 24, 26, 28, 500.0])
    out = F.aggregate_city_hourly(clean, excluded_sensors=[])
    assert out["n_sensors"].iloc[0] == 5
    assert out["pm25_city_median"].iloc[0] == 24  # median of 20,22,24,26,28


def test_spike_kept_when_fewer_than_5_sensors():
    clean = _clean_rows([20, 22, 24, 500.0])
    out = F.aggregate_city_hourly(clean, excluded_sensors=[])
    assert out["n_sensors"].iloc[0] == 4  # the 500 reading was kept
    assert out["pm25_city_median"].iloc[0] == 22  # lower median of 20,22,24,500


def test_high_reading_kept_when_not_more_than_10x_median():
    clean = _clean_rows([100, 110, 120, 130, 140, 1000 - 1.0 - 800])  # 199, within 10x
    out = F.aggregate_city_hourly(clean, excluded_sensors=[])
    assert out["n_sensors"].iloc[0] == 6


def test_excluded_sensor_never_contributes():
    clean = _clean_rows([20, 22, 24, 26, 28, 900.0])
    out = F.aggregate_city_hourly(clean, excluded_sensors=[6])
    assert out["n_sensors"].iloc[0] == 5
    assert out["pm25_city_median"].iloc[0] == 24


def test_missing_readings_do_not_count():
    clean = _clean_rows([20, np.nan, 24, np.nan, 28])
    out = F.aggregate_city_hourly(clean, excluded_sensors=[])
    assert out["n_sensors"].iloc[0] == 3


def _series_from(values, start="2026-01-01 00:00"):
    idx = pd.date_range(start, periods=len(values), freq="h")
    return pd.DataFrame({"city": "Delhi", "hour_ist": idx,
                         "pm25_city_median": values, "n_sensors": 10.0}).dropna(subset=["pm25_city_median"])


def test_short_gap_filled_long_gap_splits_segments():
    nan = np.nan
    vals = [10, 20, nan, nan, 50, 60, nan, nan, nan, nan, 100, 110]
    s = F.build_model_series(_series_from(vals))
    # the 2-hour gap is filled (30, 40), the 4-hour gap is not
    assert len(s) == 12 - 4
    filled = s[s["filled"]]
    assert list(filled["pm25"]) == [30.0, 40.0]
    assert s["segment_id"].nunique() == 2
    first_seg = s[s["segment_id"] == s["segment_id"].min()]
    assert len(first_seg) == 6


def test_gap_of_exactly_three_hours_is_filled():
    nan = np.nan
    s = F.build_model_series(_series_from([10, nan, nan, nan, 50]))
    assert len(s) == 5 and s["segment_id"].nunique() == 1


# ----------------------------------------------- part 2: parity with notebook code
CITIES = F.CITIES


def make_gold(n_hours=2400, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for i, city in enumerate(CITIES):
        idx = pd.date_range("2025-03-01", periods=n_hours, freq="h")
        vals = 40 + 20 * np.sin(np.arange(n_hours) / 24 * 2 * np.pi) + rng.normal(0, 5, n_hours)
        df = pd.DataFrame({"city": city, "hour_ist": idx, "pm25_city_median": vals,
                           "n_sensors": rng.integers(5, 40, n_hours).astype(float)})
        drop = set()
        for start, length in [(100 + i, 2), (500, 3 + (i % 2)), (900 + 3 * i, 4), (1300, 30), (1900, 1)]:
            drop.update(range(start, start + length))
        parts.append(df[~df.index.isin(drop)])
    return pd.concat(parts, ignore_index=True)


def make_weather(gold, extra_hours=48, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for city in CITIES:
        idx = pd.date_range(gold["hour_ist"].min(), gold["hour_ist"].max() + pd.Timedelta(hours=extra_hours), freq="h")
        n = len(idx)
        rows.append(pd.DataFrame({
            "city": city, "hour_ist": idx,
            "temperature_2m": 25 + rng.normal(0, 3, n),
            "relative_humidity_2m": rng.integers(30, 100, n),
            "dew_point_2m": 15 + rng.normal(0, 2, n),
            "wind_speed_10m": np.abs(rng.normal(8, 4, n)),
            "wind_direction_10m": rng.uniform(0, 360, n),
            "precipitation": np.where(rng.random(n) < 0.1, rng.random(n) * 5, 0.0),
            "surface_pressure": 1000 + rng.normal(0, 4, n),
            "cloud_cover": rng.integers(0, 100, n),
        }))
    return pd.concat(rows, ignore_index=True)


def make_forecast_raw(weather, seed=2):
    rng = np.random.default_rng(seed)
    out = weather[["city", "hour_ist"]].copy()
    n = len(out)
    out["temperature_2m_previous_day2"] = 25 + rng.normal(0, 3, n)
    out["relative_humidity_2m_previous_day2"] = rng.integers(30, 100, n)
    out["wind_speed_10m_previous_day2"] = np.abs(rng.normal(8, 4, n))
    out["precipitation_previous_day2"] = np.where(rng.random(n) < 0.1, rng.random(n) * 5, 0.0)
    out["surface_pressure_previous_day2"] = 1000 + rng.normal(0, 4, n)
    return out


# ---- reference: the training notebooks' code, kept as it was run on Databricks
def ref_model_series(g):
    parts = []
    for city, d in g.groupby("city"):
        d = d.sort_values("hour_ist").set_index("hour_ist")
        d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="h"))
        d.index.name = "hour_ist"
        d["city"] = city
        raw = d["pm25_city_median"]
        isna = raw.isna()
        run_id = (isna != isna.shift()).cumsum()
        short = isna & (isna.groupby(run_id).transform("size") <= 3)
        d["pm25"] = raw.where(~short, raw.interpolate(limit_area="inside"))
        d["observed"] = ~isna
        d["filled"] = short & d["pm25"].notna()
        d["segment_id"] = d["pm25"].isna().cumsum()
        parts.append(d[d["pm25"].notna()].reset_index()[
            ["city", "hour_ist", "pm25", "observed", "filled", "n_sensors", "segment_id"]])
    return pd.concat(parts, ignore_index=True)


def ref_features(m, wx):
    d = (m.merge(wx, on=["city", "hour_ist"], how="left")
           .dropna(subset=["temperature_2m"])
           .sort_values(["city", "segment_id", "hour_ist"]).reset_index(drop=True))
    grp = d.groupby(["city", "segment_id"])
    for k in [1, 2, 3, 6, 12, 24, 48, 144, 168]:
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
    d["target"] = grp["pm25"].shift(-24)
    d["target_time"] = d["hour_ist"] + pd.Timedelta(hours=24)
    feat = d[d.target.notna() & d.lag_168.notna() & d.roll_mean_24.notna()].copy()
    return feat


def ref_fc2(f, fc):
    W = {"temperature_2m": "temp", "relative_humidity_2m": "rh", "wind_speed_10m": "wind",
         "precipitation": "prec", "surface_pressure": "pres"}
    cols = {f"{v}_previous_day2": f"fc2_{s}" for v, s in W.items()}
    part = (fc[["city", "hour_ist"] + list(cols)].rename(columns=cols)
              .rename(columns={"hour_ist": "target_time"}))
    f = f.merge(part, on=["city", "target_time"], how="left").dropna(subset=["fc2_temp"])
    f["fc2_dtemp"] = f["fc2_temp"] - f["temperature_2m"]
    f["fc2_dwind"] = f["fc2_wind"] - f["wind_speed_10m"]
    f["fc2_dpres"] = f["fc2_pres"] - f["surface_pressure"]
    f["fc2_drh"] = f["fc2_rh"] - f["relative_humidity_2m"]
    f["city_code"] = f["city"].astype("category")
    return f


@pytest.fixture(scope="module")
def synthetic():
    gold = make_gold()
    weather = make_weather(gold)
    fc_raw = make_forecast_raw(weather)
    return gold, weather, fc_raw


def test_model_series_matches_reference(synthetic):
    gold, _, _ = synthetic
    pd.testing.assert_frame_equal(F.build_model_series(gold), ref_model_series(gold), check_dtype=False)


def test_features_match_reference(synthetic):
    gold, weather, fc_raw = synthetic
    series = F.build_model_series(gold)
    mine = F.add_forecast_features(F.build_features(series, weather), F.prepare_forecast_day2(fc_raw))
    ref = ref_fc2(ref_features(ref_model_series(gold), weather), fc_raw)
    assert len(ref) > 1000  # make sure the comparison is not empty

    keys = ["city", "hour_ist"]
    compare = F.FEATURES + ["target"]
    a = ref.set_index(keys)[compare].sort_index()
    b = mine.set_index(keys).loc[a.index, compare].sort_index()
    a, b = a.copy(), b.copy()
    a["city_code"] = a["city_code"].astype(str)
    b["city_code"] = b["city_code"].astype(str)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, rtol=1e-12, atol=1e-12)


def test_lags_never_cross_a_long_gap(synthetic):
    gold, weather, _ = synthetic
    series = F.build_model_series(gold)
    feats = F.build_features(series, weather)
    for (city, seg), g in feats.groupby(["city", "segment_id"]):
        g = g.sort_values("hour_ist")
        # the first row of every segment has no lags at all
        assert np.isnan(g["lag_1"].iloc[0])
        # a row with lag_168 must have at least 168 unbroken hours behind it
        has = g[g["lag_168"].notna()]
        if len(has):
            span = (has["hour_ist"].iloc[0] - g["hour_ist"].iloc[0]) / pd.Timedelta(hours=1)
            assert span >= 168


def test_latest_rows_flags_stale_and_skips_short_history(synthetic):
    gold, weather, fc_raw = synthetic
    series = F.build_model_series(gold)
    feats = F.add_forecast_features(F.build_features(series, weather), F.prepare_forecast_day2(fc_raw))
    now = feats["hour_ist"].max() + pd.Timedelta(hours=1)
    rows = F.latest_feature_rows(feats, now, max_age_hours=6)
    assert set(rows["city"]) <= set(CITIES)
    assert (rows["age_hours"] >= 1).all()
    stale = F.latest_feature_rows(feats, now + pd.Timedelta(hours=48), max_age_hours=6)
    assert stale["is_stale"].all()
    # a city with less than a week of history yields no usable row
    short_series = series[series["hour_ist"] > series["hour_ist"].max() - pd.Timedelta(hours=100)]
    short = F.add_forecast_features(F.build_features(short_series, weather), F.prepare_forecast_day2(fc_raw))
    assert F.latest_feature_rows(short, now).empty


def test_feature_list_matches_training_meta():
    meta_path = Path(__file__).resolve().parents[1] / "models" / "meta.json"
    if not meta_path.exists():
        pytest.skip("models/meta.json not found")
    meta = json.loads(meta_path.read_text())
    assert F.FEATURES == meta["features"]
    assert F.CITIES == meta["cities"]
