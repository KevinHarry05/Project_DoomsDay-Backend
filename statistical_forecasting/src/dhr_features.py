"""
Dynamic Harmonic Regression (DHR) exogenous features.

DHR's whole idea: represent seasonality with a small number of smooth Fourier
terms in a regression, so the ARIMA component only has to explain what those
harmonics leave behind - short-run autocorrelation - rather than needing an
expensive seasonal ARIMA term at period 24 or 168 on hourly data.

LEAKAGE NOTE
------------
Every regressor here is a function of the CALENDAR TIMESTAMP being forecast,
not of the demand series. A calendar is known in advance regardless of
horizon, so there is nothing to leak - this is the same reasoning already
applied to LightGBM's target-time calendar block, and it is the reason DHR
regressors can be extended arbitrarily far into the future without touching
any observed demand value.
"""
from __future__ import annotations

import logging

import holidays as holidays_pkg
import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


def _fourier_block(t_hours: np.ndarray, period: float, order: int, tag: str) -> pd.DataFrame:
    out = {}
    for k in range(1, max(order, 0) + 1):
        angle = 2.0 * np.pi * k * t_hours / period
        out[f"fourier_{tag}_sin_{k}"] = np.sin(angle)
        out[f"fourier_{tag}_cos_{k}"] = np.cos(angle)
    return pd.DataFrame(out)


def _holiday_flags(idx: pd.DatetimeIndex, country: str, adjacency: bool) -> pd.DataFrame:
    years = sorted({int(y) for y in idx.year.unique()})
    try:
        cal = holidays_pkg.country_holidays(country, years=years)
    except Exception as exc:  # pragma: no cover
        logger.warning("Holiday calendar unavailable for %s (%s); using zeros", country, exc)
        out = pd.DataFrame({"is_holiday": np.zeros(len(idx), dtype=int)}, index=idx)
        if adjacency:
            out["is_day_before_holiday"] = 0
            out["is_day_after_holiday"] = 0
        return out

    local = idx.tz_convert("America/New_York").normalize()
    dates = local.date
    holiday_set = set(cal.keys())
    out = pd.DataFrame(index=idx)
    out["is_holiday"] = np.fromiter((d in holiday_set for d in dates), dtype=int, count=len(idx))
    if adjacency:
        one_day = pd.Timedelta(days=1)
        next_dates = (local + one_day).date
        prev_dates = (local - one_day).date
        out["is_day_before_holiday"] = np.fromiter(
            (d in holiday_set for d in next_dates), dtype=int, count=len(idx)
        )
        out["is_day_after_holiday"] = np.fromiter(
            (d in holiday_set for d in prev_dates), dtype=int, count=len(idx)
        )
    return out


def build_exog(index: pd.DatetimeIndex, cfg: Config, t0: pd.Timestamp) -> pd.DataFrame:
    """Exogenous regressor matrix for DHR, evaluated at `index` timestamps.

    `t0` anchors the linear trend term so it is comparable across a train fit
    and later out-of-sample extension (hours elapsed since the SAME origin,
    not since whatever the first row of a given slice happens to be).
    """
    idx = pd.DatetimeIndex(index)
    t_hours = (idx - t0).total_seconds().to_numpy() / 3600.0

    orders = cfg.get("dhr.fourier_orders", {}) or {}
    blocks = [
        _fourier_block(t_hours, 24.0, int(orders.get("daily", 4)), "daily"),
        _fourier_block(idx.dayofweek.to_numpy() * 24 + idx.hour.to_numpy(), 168.0,
                       int(orders.get("weekly", 3)), "weekly"),
        _fourier_block(t_hours, 8766.0, int(orders.get("yearly", 6)), "yearly"),
    ]
    exog = pd.concat(blocks, axis=1)
    exog.index = idx

    if cfg.get("dhr.add_linear_trend", True):
        exog["trend"] = t_hours

    if cfg.get("dhr.add_holiday_regressor", True):
        holidays_df = _holiday_flags(
            idx, cfg.get("dhr.holiday_country", "US"), cfg.get("dhr.add_holiday_adjacency", True)
        )
        exog = pd.concat([exog, holidays_df], axis=1)

    # NOTE: deliberately no separate is_weekend flag here. It is collinear with
    # the weekly Fourier block (which already spans period-168 patterns,
    # weekday/weekend included) - adding it caused the fitted coefficients to
    # become ill-conditioned, producing a spurious multi-hundred-MW jump in the
    # regression level at every single Saturday-00:00 boundary (found via a
    # diagnostic that isolated the harmonic-regression component from the ARIMA
    # forecast and traced a 24h-ahead forecast blowup to exactly that flag).
    # The whole point of DHR is that Fourier terms replace explicit day-type
    # dummies; adding one back on top defeats that and destabilizes the fit.
    return exog.astype(float)
