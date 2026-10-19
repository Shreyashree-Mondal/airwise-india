"""Plain helpers that turn pipeline output into text for the app. No Streamlit, no heavy imports."""
from __future__ import annotations

import math

import pandas as pd

# CPCB (India) PM2.5 category bands for 24-hour averages, in ug/m3. Source: CPCB National AQI table.
# Values are rounded to whole numbers first, as the official table uses integers.
BANDS = [(30, "Good"), (60, "Satisfactory"), (90, "Moderate"), (120, "Poor"), (250, "Very poor")]
ICONS = {"Good": "🟢", "Satisfactory": "🟡", "Moderate": "🟠", "Poor": "🔴", "Very poor": "🟣", "Severe": "🟤"}

# General, plain-language suggestions written for this app. They are NOT official CPCB wording
# and not medical advice.
GUIDANCE = {
    "Good": "Air is fine for normal outdoor activity.",
    "Satisfactory": "Fine for most people. Unusually sensitive people may notice mild discomfort.",
    "Moderate": "People with breathing or heart conditions, older adults and children may want to take it easier outdoors.",
    "Poor": "Consider shorter or lighter outdoor exercise, especially for children, older adults and anyone with breathing or heart conditions.",
    "Very poor": "Consider limiting time outdoors and avoiding hard exercise outside, especially for sensitive groups.",
    "Severe": "Consider avoiding outdoor exertion where you can. Sensitive groups should be especially careful.",
}

STATUS_TEXT = {
    "insufficient_history": ("info", "Not enough unbroken recent data for a forecast. The model needs a week of readings "
                                      "without a gap longer than three hours."),
    "weather_unavailable": ("warning", "The weather source could not be reached on the last run, so there is no forecast."),
    "no_data": ("warning", "No sensor readings could be fetched on the last run, so there is no forecast."),
}


def category(pm25):
    """CPCB band name for a PM2.5 value, or None when the value is missing."""
    if pm25 is None or (isinstance(pm25, float) and math.isnan(pm25)):
        return None
    v = math.floor(float(pm25) + 0.5)
    for upper, name in BANDS:
        if v <= upper:
            return name
    return "Severe"


def label(pm25):
    """'🟠 Moderate' style label, or an empty string when the value is missing."""
    c = category(pm25)
    return "" if c is None else f"{ICONS[c]} {c}"


def range_labels(low, high):
    """'🟡 Satisfactory' when a range sits inside one band, else '🟡 Satisfactory to 🟠 Moderate'."""
    a, b = label(low), label(high)
    return a if a == b else f"{a} to {b}"


def fmt_age(hours):
    """59.1 -> '59 hours (2 days 11 hours)'; 5.2 -> '5 hours'."""
    if hours is None:
        return "unknown age"
    h = int(round(float(hours)))
    if h < 48:
        return f"{h} hour{'s' if h != 1 else ''}"
    return f"{h} hours ({h // 24} days {h % 24} hours)"


def period_text(start, end):
    """'Sun 18 Oct, 00:00 to 23:00' for one-day windows, a longer form otherwise. Inputs like '2026-10-18T00:00'."""
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    if a.date() == b.date():
        return f"{a:%a %d %b}, {a:%H:%M} to {b:%H:%M}"
    return f"{a:%a %d %b %H:%M} to {b:%a %d %b %H:%M}"


def city_notice(info):
    """(level, text) to show above a city's numbers, or None when nothing needs saying."""
    status = info.get("status")
    if status == "ok":
        return None
    if status == "stale":
        return ("warning", f"The newest reading is {fmt_age(info.get('age_hours'))} old, so these forecasts start from "
                           "old data. Only the periods still ahead are shown, and they are less certain than usual.")
    if status == "outdated":
        return ("error", f"The newest reading is {fmt_age(info.get('age_hours'))} old, so every forecast period is already "
                         "in the past. No forecast is shown; the last reading is below.")
    return STATUS_TEXT.get(status, ("info", f"Status: {status}"))


def feed_banner(cities):
    """A page-level warning when the data feed looks down for every city, else None."""
    infos = list(cities.values())
    if infos and all(i.get("status") in ("stale", "outdated", "no_data") for i in infos):
        ages = [i["age_hours"] for i in infos if i.get("age_hours") is not None]
        newest = max((i["as_of_ist"] for i in infos if i.get("as_of_ist")), default=None)
        when = f" The newest reading is from {newest.replace('T', ' ')} IST ({fmt_age(min(ages))} ago)." if newest and ages else ""
        return ("The public data feed behind this app appears to be paused for all three cities." + when +
                " Forecasts update automatically when fresh readings arrive.")
    return None


def fmt_skill(skill, lo, hi):
    """'+4.7% (0.0 to +8.6)' with a star when the interval excludes zero."""
    star = "*" if (lo > 0 or hi < 0) else ""
    return f"{skill:+.1f}%{star} ({lo:+.1f} to {hi:+.1f})"


def clear_days(rows, city):
    """Days ahead on which the 95% interval of the skill is entirely above zero, for one city."""
    return sorted(int(r["day"]) for r in rows if r["city"] == city and r["ci_low"] > 0)
