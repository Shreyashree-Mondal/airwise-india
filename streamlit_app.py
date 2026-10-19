"""AirWise India: a 5-day PM2.5 outlook for Delhi, Hyderabad and Kolkata.

Reads files written by the scheduled forecast job and the training notebooks:
    data/latest.json            the current forecast (written by `python -m airwise.five_day`)
    data/replay_daily.csv       past forecasts next to what actually happened
    models/horizon_evidence.json  test results of the method
"""
import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from airwise.display import (BANDS, GUIDANCE, ICONS, category, city_notice, clear_days, feed_banner, fmt_age,
                             fmt_skill, label, period_text, range_labels, refresh_view)

REPO_URL = "https://github.com/Shreyashree-Mondal/airwise-india"
CITIES = ["Delhi", "Hyderabad", "Kolkata"]
ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("AIRWISE_DATA_DIR") or ROOT / "data")
MODELS_DIR = Path(os.environ.get("AIRWISE_MODELS_DIR") or ROOT / "models")
BLOCK_NAMES = {"1": "Sep-Oct 2025", "2": "Nov-Dec 2025", "3": "Jan-Feb 2026", "4": "Mar-Apr 2026",
               "5": "May-Jun 2026", "6": "Jul-Aug 2026", "7": "Sep-Oct 2026"}
THRESHOLDS = [30, 60, 90, 120]

st.set_page_config(page_title="AirWise India", page_icon="🌫️", layout="wide")


def now_ist():
    """Current time in India (naive). AIRWISE_NOW_IST can override it, which the tests use."""
    override = os.environ.get("AIRWISE_NOW_IST")
    return pd.Timestamp(override) if override else pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def load_replay():
    try:
        df = pd.read_csv(DATA_DIR / "replay_daily.csv")
    except (OSError, ValueError):
        return None
    need = {"city", "origin", "day", "window_start", "window_end", "forecast", "low", "high", "actual"}
    return df if need <= set(df.columns) and len(df) else None


def show(level, text):
    {"error": st.error, "warning": st.warning, "info": st.info}.get(level, st.info)(text)


def band_chart(frame, x, line_cols, low="low", high="high", x_title="Time (IST)", x_type="T"):
    """A line (or lines) with a shaded range, and dashed lines at the CPCB band edges."""
    base = alt.Chart(frame).encode(x=alt.X(f"{x}:{x_type}", title=x_title))
    ribbon = base.mark_area(opacity=0.25).encode(y=alt.Y(f"{low}:Q", title="PM2.5 (µg/m³)"), y2=f"{high}:Q")
    layers = [ribbon]
    for col, name in line_cols:
        layers.append(base.mark_line(point=True).encode(y=f"{col}:Q", color=alt.value(name)))
    rules = alt.Chart(pd.DataFrame({"y": THRESHOLDS})).mark_rule(strokeDash=[4, 4], opacity=0.35).encode(y="y:Q")
    return alt.layer(*layers, rules)


latest = load_json(DATA_DIR / "latest.json")
evidence = load_json(MODELS_DIR / "horizon_evidence.json")
replay = load_replay()

st.title("AirWise India")
st.caption("A 5-day outlook for PM2.5 in Delhi, Hyderabad and Kolkata, with honest ranges and an honest scorecard. "
           "A portfolio project: not an official forecast and not medical advice.")

tab_outlook, tab_replay, tab_eval, tab_about = st.tabs(["Outlook", "Replay", "How good is it?", "About"])

# ------------------------------------------------------------------ Outlook
with tab_outlook:
    if latest is None:
        st.info("No forecast file yet. The scheduled job has not produced one.")
    else:
        NOW = now_ist()
        cities = {c: refresh_view(i, NOW) for c, i in latest.get("cities", {}).items()}
        banner = feed_banner(cities)
        if banner:
            st.warning(banner)
        city = st.radio("City", CITIES, horizontal=True)
        info = cities.get(city, {"status": "no_data"})
        st.subheader(city)
        level = info.get("range_level")
        how = ("Forecast by a machine-learning model." if info.get("method") == "model"
               else "Forecast is the average of the last 24 hours, held flat: the model did not beat it here.")
        if level:
            how += f" Ranges aim to cover about {round(level * 100)}% of outcomes."
        st.caption(how)
        notice = city_notice(info)
        if notice and not (banner and info.get("status") == "stale"):
            show(*notice)
        if info.get("current_pm25") is not None:
            st.metric(f"Last reading ({info['as_of_ist'].replace('T', ' ')} IST)", f"{info['current_pm25']:.0f} µg/m³")
            st.caption(f"{label(info['current_pm25'])} · reading is {fmt_age(info.get('age_hours'))} old")
        if info.get("hidden_periods"):
            st.caption(f"{info['hidden_periods']} earlier forecast period(s) are already in the past and are hidden.")
        if info.get("daily"):
            st.markdown("**Typical daily average for each 24-hour period**")
            st.dataframe(pd.DataFrame([{
                "Period (IST)": period_text(d["from"], d["to"]),
                "Typical average (µg/m³)": round(d["forecast"]),
                "Likely range (µg/m³)": f"{d['low']:.0f} to {d['high']:.0f}",
                "Category": label(d["forecast"]),
                "Range spans": range_labels(d["low"], d["high"])} for d in info["daily"]]), hide_index=True)
            first = info["daily"][0]
            st.write(f"For {period_text(first['from'], first['to']).split(',')[0]}: {GUIDANCE[category(first['forecast'])]}")
        if info.get("hourly"):
            h = pd.DataFrame(info["hourly"])
            h["t"] = pd.to_datetime(h["t"])
            st.markdown("**Hour by hour**")
            st.altair_chart(band_chart(h, "t", [("forecast", "#1f77b4")]))
            st.caption("Line: typical value. Shaded band: likely range. Dashed lines: CPCB category edges "
                       "(30, 60, 90 and 120 µg/m³).")
        st.divider()
        st.caption(
            "Forecasts show a typical value; on polluted days the real value is often higher. Each period is the 24 hours "
            "after the newest reading, so it may not line up with a calendar day. Category names use the CPCB 24-hour PM2.5 "
            "bands as a guide. Suggestions are general information written for this project, not CPCB wording and not "
            f"medical advice. Forecast file written {pd.Timestamp(latest['generated_at_utc']).strftime('%Y-%m-%d %H:%M UTC')} "
            f"({fmt_age((NOW - pd.Timedelta(hours=5.5) - pd.Timestamp(latest['generated_at_utc']).tz_localize(None)) / pd.Timedelta(hours=1))} ago).")
        if latest.get("fetch_failures"):
            with st.expander("Data problems on the last run"):
                st.write(latest["fetch_failures"])

# ------------------------------------------------------------------- Replay
with tab_replay:
    st.subheader("What the forecasts said, and what happened")
    if replay is None:
        st.info("Replay data is not available yet.")
    else:
        st.write("Pick a city and a past starting point. These are genuine out-of-sample forecasts: each one was made "
                 "by a model that had not seen those days.")
        rcity = st.selectbox("City", CITIES, key="replay_city")
        sub = replay[replay["city"] == rcity]
        origins = sorted(sub["origin"].unique(), reverse=True)
        pick = st.selectbox("Forecast made at (IST)", origins)
        one = sub[sub["origin"] == pick].sort_values("day").copy()
        one["window_start"] = pd.to_datetime(one["window_start"])
        inside = int(((one["actual"] >= one["low"]) & (one["actual"] <= one["high"])).sum())
        c1, c2 = st.columns(2)
        c1.metric("Actual average inside the range", f"{inside} of {len(one)} days")
        c2.metric("Average miss", f"{(one['actual'] - one['forecast']).abs().mean():.1f} µg/m³")
        st.altair_chart(band_chart(one, "window_start", [("forecast", "#1f77b4"), ("actual", "#d62728")],
                                   x_title="Start of 24-hour period (IST)"))
        st.caption("Blue: forecast. Red: what was measured. Shaded band: the range shown at the time.")
        st.dataframe(pd.DataFrame({
            "Day ahead": one["day"], "Period (IST)": [period_text(a, b) for a, b in zip(one["window_start"], one["window_end"])],
            "Forecast": one["forecast"].round(0), "Range": [f"{a:.0f} to {b:.0f}" for a, b in zip(one["low"], one["high"])],
            "Actual": one["actual"].round(0)}), hide_index=True)
        allin = ((sub["actual"] >= sub["low"]) & (sub["actual"] <= sub["high"])).mean()
        st.write(f"Across all {len(sub):,} past forecast days for {rcity}, the actual daily average fell inside the range "
                 f"{allin:.0%} of the time. The ranges were fitted on these same forecasts, so that figure is a little "
                 "flattering. The next tab has the fairer, earlier-data-only numbers.")

# --------------------------------------------------------------- Evaluation
with tab_eval:
    st.subheader("Is it better than simple rules?")
    if evidence is None:
        st.info("Evaluation results are not available.")
    else:
        hourly, daily = evidence["hourly_skill"], evidence["daily_skill"]
        st.write("Air pollution lingers from one day to the next, so simple rules like 'the average of the last 24 hours' "
                 "are hard to beat. Every score below compares the method against the best simple rule, over eight rolling "
                 "tests from July 2025 to early October 2026, each on two months the model had not seen.")
        for city in ("Delhi", "Kolkata"):
            h, d = clear_days(hourly, city), clear_days(daily, city)
            st.write(f"- **{city}** (model): clearly better than the best simple rule on hourly forecasts for day "
                     f"{', '.join(map(str, h)) if h else 'none'}; on daily averages the gain is "
                     f"{'clear on day ' + ', '.join(map(str, d)) if d else 'positive on most days but not statistically clear'}.")
        st.write("- **Hyderabad** uses the simple rule itself (the average of the last 24 hours, held flat), because the "
                 "model did not beat it. So no skill is claimed there.")

        def grid(rows, cities):
            t = pd.DataFrame([r for r in rows if r["city"] in cities])
            t["cell"] = [fmt_skill(r.skill_pct, r.ci_low, r.ci_high) for r in t.itertuples()]
            return t.pivot(index="city", columns="day", values="cell").rename(columns=lambda d: f"Day {d}")

        st.markdown("**Hourly forecasts: error compared with the best simple rule** (positive = better; * = clear gain)")
        st.dataframe(grid(hourly, CITIES))
        st.markdown("**Daily averages: error compared with the best simple rule**")
        st.dataframe(grid(daily, CITIES))
        st.caption("Percentage reduction in mean absolute error, with a 95% interval from resampling whole weeks.")
        cat = pd.DataFrame(evidence["category"])
        cat = cat[cat["city"].isin(CITIES)]
        st.markdown("**How often the daily CPCB category is right**")
        st.dataframe(cat.assign(**{"exactly right": (cat["model_correct"] * 100).round(0).astype(int).astype(str) + "%",
                                   "within one category": (cat["model_within_one"] * 100).round(0).astype(int).astype(str) + "%"})
                     .pivot(index="city", columns="day", values="exactly right").rename(columns=lambda d: f"Day {d}"))
        st.markdown("**How often the actual daily average fell inside the range**")
        st.dataframe(pd.DataFrame(evidence["coverage_daily"]).T.round(2).rename(columns=lambda d: f"Day {d}"))
        blocks = {BLOCK_NAMES.get(k, k): v for k, v in evidence["coverage_by_block_daily"].items()}
        st.write("By test period (all cities): " + ", ".join(f"{k} {v:.0%}" for k, v in blocks.items()) + ".")
        st.caption("Ranges aim for about 80% (Delhi) or 90% (Kolkata and Hyderabad) and were learned only from earlier "
                   "periods. They cover well on average but not in every season; autumn has been the weak spot.")
    st.subheader("Known limits")
    for item in (latest or {}).get("limits", []):
        st.write(f"- {item}")
    st.write("- After any gap in the readings longer than three hours, the model needs about a week of clean data "
             "before it can forecast again.")
    st.write("- The data feed has gone quiet for days at a time before. The app then shows only what is still ahead, "
             "or nothing, instead of guessing.")

# -------------------------------------------------------------------- About
with tab_about:
    st.subheader("What this is")
    st.write("A forecasting pipeline and app built on real monitoring data. It forecasts hourly PM2.5 for the next five days "
             "in Delhi, Hyderabad and Kolkata, shows each 24-hour average with a range, and reports how well the method "
             "did in past tests.")
    st.subheader("Where the data comes from")
    st.write("- Pollution readings: OpenAQ, which republishes data from India's CPCB and state pollution control boards.")
    st.write("- Weather and weather forecasts: Open-Meteo.")
    st.write("- Readings from a few faulty sensors, and isolated spikes far above neighbouring stations, are removed before "
             "the city value (a median across stations) is computed.")
    st.subheader("How the forecast is made")
    st.write("Delhi and Kolkata use a gradient-boosted model that predicts how the next days compare with the average of the "
             "last 24 hours, using recent readings, current weather and weather forecasts. Hyderabad uses the simple "
             "average of the last 24 hours. A scheduled job refreshes the forecast and this page reads the result.")
    st.write(f"Source code, tests and the full evaluation: {REPO_URL}")
    st.subheader("Category bands")
    st.write("CPCB 24-hour PM2.5 bands (µg/m³): " + ", ".join(f"{ICONS[n]} {n} up to {u}" for u, n in BANDS)
             + f", {ICONS['Severe']} Severe above 250. An hourly value is not an official AQI.")
    st.caption("If this page has been idle, it can take a moment to wake up.")
