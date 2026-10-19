"""Tests for airwise/display.py (plain functions) and streamlit_app.py (run headlessly)."""
import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from airwise import display as D

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "streamlit_app.py")
CITIES = ["Delhi", "Hyderabad", "Kolkata"]


# ------------------------------------------------------------------- display
@pytest.mark.parametrize("value,expected", [
    (0, "Good"), (30, "Good"), (30.4, "Good"), (30.5, "Satisfactory"), (60, "Satisfactory"), (60.5, "Moderate"),
    (90.4, "Moderate"), (91, "Poor"), (120, "Poor"), (121, "Very poor"), (250, "Very poor"), (251, "Severe"), (900, "Severe")])
def test_category_bands_follow_cpcb_table(value, expected):
    assert D.category(value) == expected


def test_labels_and_ranges():
    assert D.category(None) is None and D.label(None) == "" and D.label(45) == "🟡 Satisfactory"
    assert D.range_labels(31, 59) == "🟡 Satisfactory"
    assert D.range_labels(45, 123) == "🟡 Satisfactory to 🟣 Very poor"
    for _, name in D.BANDS:
        assert name in D.GUIDANCE and name in D.ICONS
    assert "Severe" in D.GUIDANCE and "Severe" in D.ICONS


def test_fmt_age_and_period_text():
    assert D.fmt_age(1) == "1 hour" and D.fmt_age(5.2) == "5 hours" and D.fmt_age(None) == "unknown age"
    assert D.fmt_age(59.1) == "59 hours (2 days 11 hours)"
    assert D.period_text("2026-10-18T00:00", "2026-10-18T23:00") == "Sun 18 Oct, 00:00 to 23:00"
    assert D.period_text("2026-10-17T22:00", "2026-10-18T21:00") == "Sat 17 Oct 22:00 to Sun 18 Oct 21:00"


def test_skill_formatting_and_clear_days():
    assert D.fmt_skill(4.7, 0.9, 8.6) == "+4.7%* (+0.9 to +8.6)"
    assert D.fmt_skill(4.7, 0.0, 8.6) == "+4.7% (+0.0 to +8.6)"          # touching zero is not a clear gain
    assert D.fmt_skill(-3.0, -5.0, -1.0) == "-3.0%* (-5.0 to -1.0)"
    rows = [{"city": "Delhi", "day": 1, "ci_low": -1}, {"city": "Delhi", "day": 3, "ci_low": 2},
            {"city": "Delhi", "day": 2, "ci_low": 0.5}, {"city": "Kolkata", "day": 1, "ci_low": 3}]
    assert D.clear_days(rows, "Delhi") == [2, 3] and D.clear_days(rows, "Hyderabad") == []


def test_city_notice_by_status():
    assert D.city_notice({"status": "ok"}) is None
    level, text = D.city_notice({"status": "stale", "age_hours": 84.3})
    assert level == "warning" and "84 hours (3 days 12 hours)" in text
    level, text = D.city_notice({"status": "outdated", "age_hours": 160})
    assert level == "error" and "No forecast is shown" in text
    for s in ("insufficient_history", "weather_unavailable", "no_data"):
        assert D.city_notice({"status": s}) is not None
    assert "mystery" in D.city_notice({"status": "mystery"})[1]


def test_feed_banner_only_when_every_city_is_down():
    down = {c: {"status": "stale", "age_hours": 84.3, "as_of_ist": "2026-10-13T23:00"} for c in "ABC"}
    assert "paused" in D.feed_banner(down) and "2026-10-13 23:00 IST" in D.feed_banner(down)
    assert D.feed_banner(dict(down, A={"status": "ok", "age_hours": 1.5, "as_of_ist": "2026-10-16T08:00"})) is None
    assert D.feed_banner({}) is None


# --------------------------------------------------------------------- fixtures
def city_info(status, method="model", level=0.8, windows=(1, 2, 3, 4, 5), age=1.5, origin="2026-10-16T08:00"):
    info = {"status": status, "method": method, "range_level": level, "sensors_requested": 9, "sensors_with_data": 9}
    if status in ("ok", "stale", "outdated"):
        info.update({"as_of_ist": origin, "age_hours": age, "current_pm25": 116.0, "sensors_in_latest_hour": 30})
    if status in ("ok", "stale"):
        o = pd.Timestamp(origin)
        info["daily"] = [{"day": d, "from": (o + pd.Timedelta(hours=24 * (d - 1) + 1)).strftime("%Y-%m-%dT%H:%M"),
                          "to": (o + pd.Timedelta(hours=24 * d)).strftime("%Y-%m-%dT%H:%M"), "hours": 24,
                          "forecast": 90.0 - 8 * d, "low": 60.0 - 5 * d, "high": 130.0 - 8 * d} for d in windows]
        first = o + pd.Timedelta(hours=24 * (min(windows) - 1) + 1)
        info["hourly"] = [{"t": (first + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M"), "forecast": 80.0 + i % 5,
                           "low": 50.0, "high": 120.0} for i in range(24 * len(windows))]
    return info


def payload(**cities):
    return {"schema": 2, "generated_at_utc": "2026-10-16T09:00:00Z", "horizon_hours": 120,
            "cities": {c: cities.get(c, city_info("no_data")) for c in CITIES},
            "fetch_failures": [], "limits": ["a test limit"]}


EVIDENCE = {
    "hourly_skill": [{"city": c, "day": d, "model_MAE": 1, "best_base_MAE": 1, "best_baseline": "base_mean24",
                      "skill_pct": 5.0, "ci_low": 1.0 if (c != "Hyderabad" and d >= 2) else -1.0, "ci_high": 9.0, "rows": 10}
                     for c in CITIES + ["All"] for d in range(1, 6)],
    "daily_skill": [{"city": c, "day": d, "model_MAE": 1, "best_base_MAE": 1, "best_baseline": "base_mean24",
                     "skill_pct": 2.0, "ci_low": -3.0, "ci_high": 7.0, "rows": 10} for c in CITIES + ["All"] for d in range(1, 6)],
    "category": [{"city": c, "day": d, "days_scored": 10, "model_correct": 0.7, "baseline_correct": 0.69,
                  "best_baseline": "base_mean24", "model_within_one": 0.97} for c in CITIES + ["All"] for d in range(1, 6)],
    "coverage_daily": {c: {str(d): 0.8 for d in range(1, 6)} for c in CITIES},
    "coverage_hourly": {c: {str(d): 0.8 for d in range(1, 6)} for c in CITIES},
    "coverage_by_block_daily": {"1": 0.6, "2": 0.85, "3": 0.9, "7": 0.73}, "blocks": "test"}


def replay_csv(path, n_origins=3):
    rows = []
    for city in CITIES:
        for i in range(n_origins):
            o = pd.Timestamp("2026-09-01 23:00") + pd.Timedelta(days=i)
            for d in range(1, 6):
                s = o + pd.Timedelta(hours=24 * (d - 1) + 1)
                rows.append({"city": city, "origin": o.strftime("%Y-%m-%d %H:%M"), "day": d,
                             "window_start": s.strftime("%Y-%m-%d %H:%M"), "window_end": (s + pd.Timedelta(hours=23)).strftime("%Y-%m-%d %H:%M"),
                             "forecast": 60.0 + d, "low": 40.0, "high": 90.0, "actual": 55.0 + 10 * d, "hours_measured": 24,
                             "origin_pm25": 50.0, "method": "model"})
    pd.DataFrame(rows).to_csv(path, index=False)


def run_app(tmp_path, monkeypatch, latest=None, evidence=EVIDENCE, replay=True):
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    if latest is not None:
        (data / "latest.json").write_text(json.dumps(latest))
    if evidence is not None:
        (models / "horizon_evidence.json").write_text(json.dumps(evidence))
    if replay:
        replay_csv(data / "replay_daily.csv")
    monkeypatch.setenv("AIRWISE_DATA_DIR", str(data))
    monkeypatch.setenv("AIRWISE_MODELS_DIR", str(models))
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def text_of(at):
    return " ".join(m.value for m in at.markdown)


# ------------------------------------------------------------------------- app
def test_healthy_forecast_shows_five_periods_with_ranges_and_guidance(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch, payload(Delhi=city_info("ok"), Hyderabad=city_info("ok", "recent_average", 0.9),
                                               Kolkata=city_info("ok", level=0.9)))
    assert not at.warning and not at.error
    table = at.dataframe[0].value
    assert len(table) == 5 and list(table.columns)[:3] == ["Period (IST)", "Typical average (µg/m³)", "Likely range (µg/m³)"]
    assert table["Typical average (µg/m³)"].tolist()[0] == 82 and table["Likely range (µg/m³)"].tolist()[0] == "55 to 122"
    assert at.metric[0].label.startswith("Last reading") and at.metric[0].value == "116 µg/m³"
    caps = " ".join(c.value for c in at.caption)
    assert "machine-learning model" in caps and "about 80%" in caps
    assert D.GUIDANCE["Moderate"] in text_of(at)                        # 82 is Moderate


def test_hyderabad_is_described_as_the_flat_average(tmp_path, monkeypatch):
    p = payload(Delhi=city_info("ok"), Hyderabad=city_info("ok", "recent_average", 0.9), Kolkata=city_info("ok"))
    data = tmp_path / "data"
    data.mkdir()
    (data / "latest.json").write_text(json.dumps(p))
    monkeypatch.setenv("AIRWISE_DATA_DIR", str(data))
    monkeypatch.setenv("AIRWISE_MODELS_DIR", str(tmp_path / "none"))
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.radio[0].set_value("Hyderabad").run()
    assert not at.exception
    caps = " ".join(c.value for c in at.caption)
    assert "held flat" in caps and "about 90%" in caps


def test_stale_data_shows_the_banner_and_only_the_remaining_periods(tmp_path, monkeypatch):
    stale = {c: city_info("stale", windows=(3, 4, 5), age=41.5, origin="2026-10-14T01:00") for c in CITIES}
    at = run_app(tmp_path, monkeypatch, payload(**stale))
    assert any("paused" in w.value for w in at.warning)
    assert len(at.dataframe[0].value) == 3
    assert any("Last reading (2026-10-14 01:00 IST)" in m.label for m in at.metric)


def test_outdated_data_shows_no_forecast_but_says_so(tmp_path, monkeypatch):
    out = {c: city_info("outdated", age=160, origin="2026-10-09T01:00") for c in CITIES}
    at = run_app(tmp_path, monkeypatch, payload(**out))
    assert any("paused" in w.value for w in at.warning)
    assert any("No forecast is shown" in e.value for e in at.error)
    assert at.metric[0].label.startswith("Last reading")
    assert "Typical daily average" not in text_of(at)


def test_each_city_gets_its_own_message(tmp_path, monkeypatch):
    p = payload(Delhi=city_info("ok"), Hyderabad=city_info("weather_unavailable"), Kolkata=city_info("insufficient_history"))
    at = run_app(tmp_path, monkeypatch, p)
    assert not at.warning
    at.radio[0].set_value("Kolkata").run()
    assert any("unbroken" in i.value for i in at.info)
    at.radio[0].set_value("Hyderabad").run()
    assert any("weather source" in w.value for w in at.warning)


def test_missing_forecast_file_is_explained_not_a_crash(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch, latest=None)
    assert any("No forecast file yet" in i.value for i in at.info)


def test_replay_tab_scores_a_past_forecast(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch, payload(Delhi=city_info("ok")))
    labels = [m.label for m in at.metric]
    assert "Actual average inside the range" in labels
    inside = [m for m in at.metric if m.label == "Actual average inside the range"][0]
    assert inside.value == "3 of 5 days"                                # actuals 65, 75, 85, 95, 105 against a 40-90 range
    at.selectbox[0].set_value("Kolkata").run()
    assert not at.exception and len(at.selectbox) == 2


def test_replay_tab_explains_when_the_file_is_missing(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch, payload(Delhi=city_info("ok")), replay=False)
    assert any("Replay data is not available" in i.value for i in at.info)


def test_evaluation_tab_states_the_honest_result_from_the_evidence_file(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch, payload(Delhi=city_info("ok")))
    text = text_of(at)
    assert "hard to beat" in text and "Hyderabad** uses the simple rule itself" in text
    assert "hourly forecasts for day 2, 3, 4, 5" in text               # clear gains from the fixture
    assert "positive on most days but not statistically clear" in text   # daily averages: intervals include zero
    assert "Sep-Oct 2025 60%" in text and "a test limit" in text


def test_missing_evidence_file_is_explained(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch, payload(Delhi=city_info("ok")), evidence=None)
    assert any("Evaluation results are not available" in i.value for i in at.info)


def test_the_real_repo_files_render_without_errors(monkeypatch):
    needed = [ROOT / "data" / "latest.json", ROOT / "models" / "horizon_evidence.json", ROOT / "data" / "replay_daily.csv"]
    if not all(p.exists() for p in needed):
        pytest.skip("repo data files not present")
    monkeypatch.delenv("AIRWISE_DATA_DIR", raising=False)
    monkeypatch.delenv("AIRWISE_MODELS_DIR", raising=False)
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]
    for city in CITIES:
        at.radio[0].set_value(city).run()
        assert not at.exception
    text = text_of(at)
    assert "Delhi** (model)" in text and "Kolkata** (model)" in text
