"""Tests for airwise/horizon_eval.py: day grouping, best baseline, daily windows, categories and intervals."""
import numpy as np
import pandas as pd
import pytest

from airwise import horizon_eval as E

START = pd.Timestamp("2025-06-01 00:00")


def make_preds(model_offset=0.0, base_offsets=(5.0, 6.0, 7.0), cities=("Delhi", "Kolkata"), n_origins=300, seed=0):
    """Predictions for every hour 1..120 from 300 starting hours; model and baselines are the truth plus an offset."""
    rng = np.random.default_rng(seed)
    rows = []
    for city in cities:
        for o in range(n_origins):
            origin = START + pd.Timedelta(hours=3 * o)
            h = np.arange(1, 121)
            truth = 60 + 25 * np.sin((o * 3 + h) / 24 * 2 * np.pi) + rng.normal(0, 3, 120)
            df = pd.DataFrame({"city": city, "hour_ist": origin, "h": h, "target_pm25": truth})
            df["pred"] = truth + model_offset
            for c, off in zip(E.BASELINES, base_offsets):
                df[c] = truth + off
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def test_category_code_edges_match_cpcb_bands():
    got = E.category_code([0, 30, 30.4, 30.5, 31, 60, 60.4, 60.5, 90, 90.5, 120, 120.5, 250, 250.4, 250.5, 900])
    assert list(got) == [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5]


def test_day_ahead_grouping():
    d = E.add_day(pd.DataFrame({"h": [1, 24, 25, 48, 49, 120]}))
    assert list(d["day"]) == [1, 1, 2, 2, 3, 5]


def test_perfect_model_scores_100_percent_everywhere():
    df = make_preds(model_offset=0.0)
    h = E.hourly_scores(df, n_boot=200)
    assert set(h["day"]) == {1, 2, 3, 4, 5} and {"Delhi", "Kolkata", "All"} == set(h["city"])
    assert np.allclose(h["skill_pct"], 100.0) and np.allclose(h["model_MAE"], 0.0)
    assert np.allclose(h["best_base_MAE"], 5.0)           # the closest baseline is off by 5
    assert (h["best_baseline"] == "base_persist").all()


def test_model_equal_to_best_baseline_scores_zero():
    df = make_preds(model_offset=5.0)                      # exactly as bad as the best baseline
    h = E.hourly_scores(df, n_boot=200)
    assert np.allclose(h["skill_pct"], 0.0)


def test_worse_model_scores_negative_and_interval_is_ordered():
    df = make_preds(model_offset=10.0)                     # twice the error of the best baseline
    h = E.hourly_scores(df, n_boot=300)
    assert (h["skill_pct"] < -90).all()
    assert (h["ci_low"] <= h["ci_high"]).all()


def test_best_baseline_is_chosen_per_city_and_day():
    df = make_preds()
    # for Kolkata only, make the seasonal baseline the closest one
    m = df["city"] == "Kolkata"
    df.loc[m, "base_seasonal"] = df.loc[m, "target_pm25"] + 1.0
    h = E.hourly_scores(df, n_boot=100).set_index(["city", "day"])
    assert (h.loc["Kolkata", "best_baseline"] == "base_seasonal").all()
    assert (h.loc["Delhi", "best_baseline"] == "base_persist").all()


def test_interval_contains_the_point_estimate_for_a_noisy_model():
    rng = np.random.default_rng(1)
    df = make_preds(model_offset=0.0)
    df["pred"] = df["target_pm25"] + rng.normal(0, 4, len(df))     # noisy model, baselines off by 5-7
    h = E.hourly_scores(df, n_boot=500)
    assert (h["ci_low"] <= h["skill_pct"] + 1e-9).all() and (h["skill_pct"] <= h["ci_high"] + 1e-9).all()
    assert (h["ci_high"] - h["ci_low"] > 0).all()


def test_results_are_repeatable():
    df = make_preds(model_offset=2.0)
    a = E.hourly_scores(df, n_boot=200, seed=3)
    b = E.hourly_scores(df, n_boot=200, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_daily_windows_need_enough_measured_hours():
    df = make_preds(cities=("Delhi",), n_origins=2)
    o = df["hour_ist"].min()
    # day 1 of the first origin keeps only 17 hours; of the second origin 18 hours
    drop1 = (df["hour_ist"] == o) & (df["h"] > 17) & (df["h"] <= 24)
    second = df["hour_ist"].unique()[1]
    drop2 = (df["hour_ist"] == second) & (df["h"] > 18) & (df["h"] <= 24)
    a = E.daily_means(df[~(drop1 | drop2)])
    day1 = a[a["day"] == 1].set_index("hour_ist")
    assert o not in day1.index and second in day1.index
    assert day1.loc[second, "n_hours"] == 18


def test_daily_mean_uses_only_the_measured_hours_for_model_and_baselines():
    df = make_preds(cities=("Delhi",), n_origins=1)
    df = df[~((df["h"] > 20) & (df["h"] <= 24))].copy()            # 20 hours measured on day 1
    a = E.daily_means(df)
    r = a[a["day"] == 1].iloc[0]
    day1 = df[df["h"] <= 24]
    assert r["n_hours"] == 20
    assert np.isclose(r["target"], day1["target_pm25"].mean())
    assert np.isclose(r["pred"], day1["pred"].mean())
    assert np.isclose(r["base_persist"], day1["base_persist"].mean())


def test_daily_scores_perfect_model():
    df = make_preds(model_offset=0.0)
    d = E.daily_scores(df, n_boot=100)
    assert np.allclose(d["skill_pct"], 100.0) and (d["rows"] > 100).all()


def test_category_scores():
    df = make_preds(model_offset=0.0)
    c = E.category_scores(df)
    assert np.allclose(c["model_correct"], 1.0) and np.allclose(c["model_within_one"], 1.0)
    assert (c["baseline_correct"] <= 1.0).all() and (c["days_scored"] > 100).all()
    # a model that is always 40 ug/m3 too high must miss most categories
    bad = E.category_scores(make_preds(model_offset=40.0))
    assert (bad["model_correct"] < 0.6).all()


def test_all_cities_row_pools_the_cities():
    df = make_preds()
    h = E.hourly_scores(df, n_boot=100)
    n_city = h[h["city"] != "All"].groupby("day")["rows"].sum()
    n_all = h[h["city"] == "All"].set_index("day")["rows"]
    assert (n_city.values == n_all.values).all()


def test_a_different_model_column_can_be_scored():
    df = make_preds(model_offset=0.0)
    df["pred_v2"] = df["target_pm25"] + 5.0
    h = E.hourly_scores(df, model_col="pred_v2", n_boot=100)
    assert np.allclose(h["skill_pct"], 0.0)
    assert np.allclose(E.daily_scores(df, model_col="pred_v2", n_boot=100)["skill_pct"], 0.0)
    assert (E.category_scores(df, model_col="pred_v2")["model_correct"] <= 1.0).all()


def test_missing_hours_in_the_middle_of_a_day_count_correctly():
    df = make_preds(cities=("Delhi",), n_origins=2)
    origins = list(df["hour_ist"].unique())
    hole6 = (df["hour_ist"] == origins[0]) & (df["h"] >= 5) & (df["h"] <= 10)    # 6 hours missing -> 18 left
    hole7 = (df["hour_ist"] == origins[1]) & (df["h"] >= 5) & (df["h"] <= 11)    # 7 hours missing -> 17 left
    a = E.daily_means(df[~(hole6 | hole7)])
    day1 = a[a["day"] == 1].set_index("hour_ist")
    assert origins[0] in day1.index and day1.loc[origins[0], "n_hours"] == 18
    assert origins[1] not in day1.index
