"""Scores for the 5-day hourly forecasts, by day ahead.

Input: a table of predictions with columns
    city, hour_ist (the starting hour), h (hours ahead, 1..120), target_pm25 (what was measured),
    the model column (default 'pred') and the baselines base_persist, base_seasonal, base_mean24.

Three views, each by city and day ahead (day 1 = hours 1-24 ahead, ... day 5 = hours 97-120):
    hourly_scores   error of the hourly forecasts against the best simple baseline
    daily_scores    error of the daily AVERAGE (the quantity CPCB's categories are defined for)
    category_scores how often the forecast daily average lands in the right CPCB category

'Skill' is the percentage by which the model's mean absolute error is lower than the best baseline's.
The best baseline is picked on the same rows, which is slightly generous to the baseline.
Intervals come from resampling whole weeks of starting hours, because neighbouring hours are not independent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASELINES = ["base_persist", "base_seasonal", "base_mean24"]
MIN_HOURS_PER_DAY = 18  # a day counts only if at least this many of its 24 hours were measured
CATEGORY_EDGES = [30.5, 60.5, 90.5, 120.5, 250.5]  # CPCB PM2.5 bands after rounding to whole numbers
CATEGORY_NAMES = ["Good", "Satisfactory", "Moderate", "Poor", "Very poor", "Severe"]


def category_code(values) -> np.ndarray:
    """0 = Good ... 5 = Severe for PM2.5 values (ug/m3)."""
    return np.digitize(np.asarray(values, dtype=float), CATEGORY_EDGES)


def add_day(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["day"] = (out["h"] - 1) // 24 + 1
    return out


def _week_blocks(times: pd.Series) -> pd.Series:
    return ((times - times.min()) / pd.Timedelta(days=7)).astype(int)


def _skill_ci(err_model: pd.Series, err_base: pd.Series, blocks: pd.Series, n_boot: int, rng):
    """(skill %, low, high). Resamples whole blocks (weeks) of starting hours."""
    s_m = err_model.groupby(blocks.values).sum().to_numpy()
    s_b = err_base.groupby(blocks.values).sum().to_numpy()
    point = (1 - s_m.sum() / s_b.sum()) * 100
    idx = rng.integers(0, len(s_m), size=(n_boot, len(s_m)))
    draws = (1 - s_m[idx].sum(axis=1) / s_b[idx].sum(axis=1)) * 100
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _score_groups(frame, model_col, truth_col, base_cols, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in list(frame.groupby(["city", "day"])) + list(frame.assign(city="All").groupby(["city", "day"])):
        maes = {c: (g[truth_col] - g[c]).abs().mean() for c in base_cols}
        best = min(maes, key=maes.get)
        e_m = (g[truth_col] - g[model_col]).abs()
        e_b = (g[truth_col] - g[best]).abs()
        skill, lo, hi = _skill_ci(e_m, e_b, g["blk"], n_boot, rng)
        rows.append({"city": key[0], "day": key[1], "rows": len(g), "model_MAE": e_m.mean(), "best_base_MAE": e_b.mean(),
                     "best_baseline": best, "skill_pct": skill, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows).sort_values(["city", "day"]).reset_index(drop=True)


def hourly_scores(df: pd.DataFrame, model_col: str = "pred", n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    d = add_day(df)
    d["blk"] = _week_blocks(d["hour_ist"])
    return _score_groups(d, model_col, "target_pm25", BASELINES, n_boot, seed)


def daily_means(df: pd.DataFrame, model_col: str = "pred") -> pd.DataFrame:
    """One row per starting hour and day ahead: the mean over the hours that were measured.

    The model and the baselines are averaged over the same hours as the measurement, so the comparison is fair.
    Windows with fewer than MIN_HOURS_PER_DAY measured hours are dropped.
    """
    d = add_day(df)
    cols = {"target": ("target_pm25", "mean"), "n_hours": ("h", "size"), "pred": (model_col, "mean")}
    cols.update({c: (c, "mean") for c in BASELINES})
    agg = d.groupby(["city", "hour_ist", "day"]).agg(**cols).reset_index()
    agg = agg[agg["n_hours"] >= MIN_HOURS_PER_DAY].copy()
    agg["blk"] = _week_blocks(agg["hour_ist"])
    return agg


def daily_scores(df: pd.DataFrame, model_col: str = "pred", n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    return _score_groups(daily_means(df, model_col), "pred", "target", BASELINES, n_boot, seed)


def category_scores(df: pd.DataFrame, model_col: str = "pred") -> pd.DataFrame:
    """Share of daily averages whose CPCB category was predicted correctly, model vs best baseline."""
    a = daily_means(df, model_col)
    rows = []
    for key, g in list(a.groupby(["city", "day"])) + list(a.assign(city="All").groupby(["city", "day"])):
        maes = {c: (g["target"] - g[c]).abs().mean() for c in BASELINES}
        best = min(maes, key=maes.get)
        truth = category_code(g["target"])
        rows.append({"city": key[0], "day": key[1], "days_scored": len(g),
                     "model_correct": float((category_code(g["pred"]) == truth).mean()),
                     "baseline_correct": float((category_code(g[best]) == truth).mean()), "best_baseline": best,
                     "model_within_one": float((abs(category_code(g["pred"]) - truth) <= 1).mean())})
    return pd.DataFrame(rows).sort_values(["city", "day"]).reset_index(drop=True)
