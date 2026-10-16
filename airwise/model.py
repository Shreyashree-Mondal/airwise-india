"""Loads the three saved LightGBM models and turns feature rows into forecasts with a range.

The models predict the CHANGE in PM2.5 over the next 24 hours (a "delta" model):
    forecast = current PM2.5 + model output
The range comes from two quantile models (10% and 90%) widened by a calibration
constant per city, using the exact rule stored in models/meta.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from airwise.features import CITIES, FEATURES


class ModelError(Exception):
    """The saved model files do not match this code."""


class ForecastModels:
    def __init__(self, models_dir):
        d = Path(models_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        if self.meta["features"] != FEATURES:
            raise ModelError("feature list in models/meta.json differs from airwise/features.py")
        if self.meta["cities"] != CITIES:
            raise ModelError("city list in models/meta.json differs from airwise/features.py")
        self.boosters = {name: lgb.Booster(model_file=str(d / f"model_{name}.txt"))
                         for name in ("point", "q10", "q90")}
        interval = self.meta["interval"]
        self.qn = interval["qn_by_city"]
        self.offset = float(interval.get("width_offset", 1.0))

    @staticmethod
    def _matrix(rows: pd.DataFrame) -> pd.DataFrame:
        """Model inputs in the training column order, with plain numeric types."""
        numeric = [c for c in FEATURES if c != "city_code"]
        X = rows[numeric].apply(pd.to_numeric, errors="raise").astype(float)
        X["city_code"] = pd.Categorical(rows["city"].astype(str).to_numpy(), categories=CITIES)
        if X.isna().any().any():
            raise ModelError("model inputs contain missing values")
        return X[FEATURES]

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """rows: output of features.latest_feature_rows. Returns city, pred, lo, hi (ug/m3)."""
        if rows.empty:
            return pd.DataFrame(columns=["city", "pred", "lo", "hi"])
        X = self._matrix(rows)
        base = X["pm25"].to_numpy()
        pred = np.clip(base + self.boosters["point"].predict(X), 0, None)
        a = base + self.boosters["q10"].predict(X)
        b = base + self.boosters["q90"].predict(X)
        lo0 = np.clip(np.minimum(a, b), 0, None)
        hi0 = np.clip(np.maximum(a, b), 0, None)
        width = hi0 - lo0 + self.offset
        qn = rows["city"].map(self.qn).to_numpy(dtype=float)
        lo = np.clip(lo0 - qn * width, 0, None)
        hi = hi0 + qn * width
        return pd.DataFrame({"city": rows["city"].to_numpy(), "pred": pred, "lo": lo, "hi": hi})
