"""Tests for airwise/horizons.py: row alignment, forecast-lead choice, leakage and gaps."""
import numpy as np
import pandas as pd
import pytest

from airwise import features as F
from airwise import horizons as H

START = pd.Timestamp("2025-03-01 00:00")


def hours_since_start(t):
    return ((t - START) / pd.Timedelta(hours=1)).astype(int)


def make_world(n_hours=700, value=None, gap=None, seed=0):
    """Series for three cities plus matching weather and forecast-lead tables.

    value(idx_array, city_index) -> pm25 values. gap=(start, length) removes hours from every city.
    """
    rng = np.random.default_rng(seed)
    value = value or (lambda idx, c: 40.0 + idx * 0.0)
    gold = []
    for ci, city in enumerate(F.CITIES):
        idx = np.arange(n_hours)
        t = START + pd.to_timedelta(idx, unit="h")
        df = pd.DataFrame({"city": city, "hour_ist": t, "pm25_city_median": value(idx, ci), "n_sensors": 10.0})
        if gap:
            df = df[~((idx >= gap[0]) & (idx < gap[0] + gap[1]))]
        gold.append(df)
    series = F.build_model_series(pd.concat(gold, ignore_index=True))

    span = pd.date_range(START, periods=n_hours + 240, freq="h")  # weather and leads extend beyond the series
    weather, leads = [], []
    for city in F.CITIES:
        n = len(span)
        weather.append(pd.DataFrame({
            "city": city, "hour_ist": span, "temperature_2m": 25 + rng.normal(0, 2, n),
            "relative_humidity_2m": rng.integers(30, 90, n), "dew_point_2m": 15 + rng.normal(0, 1, n),
            "wind_speed_10m": np.abs(rng.normal(8, 3, n)), "wind_direction_10m": rng.uniform(0, 360, n),
            "precipitation": np.zeros(n), "surface_pressure": 1000 + rng.normal(0, 3, n),
            "cloud_cover": rng.integers(0, 100, n)}))
        ld = {"city": city, "hour_ist": span}
        for k in H.LEAD_CHOICES:
            ld[f"temperature_2m_previous_day{k}"] = k * 1000.0 + hours_since_start(span)  # encodes lead and time
            ld[f"relative_humidity_2m_previous_day{k}"] = rng.integers(30, 90, n)
            ld[f"wind_speed_10m_previous_day{k}"] = np.abs(rng.normal(8, 3, n))
            ld[f"precipitation_previous_day{k}"] = np.zeros(n)
            ld[f"surface_pressure_previous_day{k}"] = 1000 + rng.normal(0, 3, n)
        leads.append(pd.DataFrame(ld))
    weather, leads = pd.concat(weather, ignore_index=True), pd.concat(leads, ignore_index=True)
    feats = F.build_features(series, weather)
    return series, weather, leads, feats


def test_lead_days_mapping():
    expect = {1: 2, 24: 2, 25: 3, 48: 3, 49: 4, 72: 4, 73: 5, 96: 5, 97: 6, 120: 6}
    assert {h: H.lead_days(h) for h in expect} == expect


def test_target_is_exactly_h_hours_after_the_origin():
    series, _, leads, feats = make_world(value=lambda idx, c: idx.astype(float))  # pm25 equals the hour number
    t = H.make_horizon_table(feats, series, leads, horizons=range(1, 121))
    assert len(t) > 1000
    assert ((t["target_time"] - t["hour_ist"]) / pd.Timedelta(hours=1) == t["h"]).all()
    assert np.allclose(t["target_pm25"], t["pm25"] + t["h"])
    assert np.allclose(t["y"], t["h"])


def test_forecast_weather_uses_the_right_lead_for_the_target_hour():
    series, _, leads, feats = make_world()
    t = H.make_horizon_table(feats, series, leads, horizons=[1, 24, 25, 48, 49, 96, 97, 120])
    for h, g in t.groupby("h"):
        expected = H.lead_days(h) * 1000.0 + hours_since_start(g["target_time"])
        assert np.allclose(g["fc_temp"], expected), h


def test_forecast_used_was_issued_at_least_a_day_before_the_origin():
    series, _, leads, feats = make_world()
    t = H.make_horizon_table(feats, series, leads)
    issued = t["target_time"] - pd.to_timedelta(t["h"].map(H.lead_days) * 24, unit="h")
    assert (issued <= t["hour_ist"] - pd.Timedelta(hours=24)).all()


def test_seasonal_baseline_is_the_same_clock_hour_from_the_last_day():
    # a perfectly repeating daily pattern: the seasonal baseline must equal the target exactly
    series, _, leads, feats = make_world(value=lambda idx, c: 10.0 + (idx % 24))
    t = H.make_horizon_table(feats, series, leads)
    assert np.allclose(t["base_seasonal"], t["target_pm25"])
    assert not np.allclose(t["base_persist"], t["target_pm25"])


def test_targets_across_a_gap_are_kept_but_never_inside_it():
    gap = (450, 30)
    series, _, leads, feats = make_world(n_hours=1100, gap=gap)
    t = H.make_horizon_table(feats, series, leads)
    g0, g1 = START + pd.Timedelta(hours=gap[0]), START + pd.Timedelta(hours=gap[0] + gap[1])
    assert not ((t["target_time"] >= g0) & (t["target_time"] < g1)).any()
    assert ((t["hour_ist"] < g0) & (t["target_time"] >= g1)).any()  # origin before the gap, target after it


def test_inference_mode_keeps_every_horizon_and_has_no_target():
    series, _, leads, feats = make_world()
    newest = feats.loc[feats.groupby("city")["hour_ist"].idxmax()]
    t = H.make_horizon_table(newest, series, leads, with_target=False)
    assert len(t) == 3 * H.HORIZON_MAX
    assert sorted(t["h"].unique()) == list(range(1, 121))
    assert "y" not in t.columns and "target_pm25" not in t.columns


def test_origin_step_thins_the_origins():
    series, _, leads, feats = make_world()
    full = H.make_horizon_table(feats, series, leads, horizons=[24])
    thin = H.make_horizon_table(feats, series, leads, horizons=[24], origin_step=3)
    assert 0.30 < len(thin) / len(full) < 0.37


def test_table_has_every_model_column_and_origin_values_match_the_features():
    series, _, leads, feats = make_world(value=lambda idx, c: 30 + 10 * np.sin(idx / 5) + c)
    t = H.make_horizon_table(feats, series, leads, horizons=[1, 50, 120])
    assert set(H.H_FEATURES) <= set(t.columns) and set(H.BASELINES) <= set(t.columns)
    assert str(t["city_code"].dtype) == "category" and list(t["city_code"].cat.categories) == F.CITIES
    a = t[t["h"] == 50].set_index(["city", "hour_ist"])[H.ORIGIN_COLS].sort_index()
    b = feats.set_index(["city", "hour_ist"]).loc[a.index, H.ORIGIN_COLS].sort_index()
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_h24_target_matches_the_24_hour_model_target():
    series, _, leads, feats = make_world(value=lambda idx, c: 30 + 10 * np.sin(idx / 5) + c)
    t = H.make_horizon_table(feats, series, leads, horizons=[24]).set_index(["city", "hour_ist"])
    f = feats.dropna(subset=["target"]).set_index(["city", "hour_ist"])
    common = f.index.intersection(t.index)
    assert len(common) > 300
    assert np.allclose(t.loc[common, "target_pm25"], f.loc[common, "target"])


def test_float32_conversion_keeps_categories_and_times():
    series, _, leads, feats = make_world()
    t = H.make_horizon_table(feats, series, leads, horizons=[1, 2])
    s = H.to_float32(t)
    assert s["pm25"].dtype == "float32" and str(s["city_code"].dtype) == "category"
    assert str(s["hour_ist"].dtype).startswith("datetime64")


def test_forecast_minus_current_weather_columns():
    series, _, leads, feats = make_world()
    t = H.make_horizon_table(feats, series, leads, horizons=[5, 70])
    assert np.allclose(t["fc_dtemp"], t["fc_temp"] - t["temperature_2m"])
    assert np.allclose(t["fc_dwind"], t["fc_wind"] - t["wind_speed_10m"])
    assert np.allclose(t["fc_dpres"], t["fc_pres"] - t["surface_pressure"])
    assert np.allclose(t["fc_drh"], t["fc_rh"] - t["relative_humidity_2m"])
