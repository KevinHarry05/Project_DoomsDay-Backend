"""
Feature engineering for direct multi-horizon demand forecasting.

--------------------------------------------------------------------------
THE LEAKAGE RULE THAT GOVERNS THIS ENTIRE MODULE
--------------------------------------------------------------------------
A forecast made at ORIGIN time t for TARGET time t+h may use:

  (a) demand observations at or before t          -> "origin features"
  (b) calendar/holiday attributes of t+h          -> "target-time features"

It may NOT use any demand observation after t. Every function below is written
so that (a) is mechanically incapable of reaching past t.

Concretely: origin features are computed on the demand series indexed by t, and
the target is produced by shifting demand BACKWARD (`shift(-h)`). The feature
matrix itself is never shifted. This is the direct multi-horizon formulation -
one independent model per horizon - which also avoids the error accumulation
you get from recursively feeding a model its own predictions.

Calendar attributes of t+h are legitimately known at t (a calendar is not a
measurement), which is why hour-of-target and holiday-of-target are the single
most valuable features in the set.

--------------------------------------------------------------------------
WHAT IS DELIBERATELY ABSENT
--------------------------------------------------------------------------
No weather features. The cleaned dataset contains no temperature column, and
temperature is the dominant exogenous driver of electricity demand. Adding a
weather join is the highest-value future improvement; fabricating one now would
be inventing data the source does not have.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import holidays as holidays_pkg
import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)

TARGET_COL = "y"
ORIGIN_TS_COL = "forecast_timestamp"
TARGET_TS_COL = "target_timestamp"

# Columns that describe the row rather than predict it. Never fed to the model.
META_COLUMNS = [ORIGIN_TS_COL, TARGET_TS_COL, "horizon_hours", "region_code", TARGET_COL]


# ---------------------------------------------------------------------------
# (a) ORIGIN FEATURES - demand history, strictly <= t
# ---------------------------------------------------------------------------
def build_origin_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Demand-history features indexed by forecast origin t.

    `lag_0` is demand AT t. That is correct and not leakage: at the moment we
    stand at t and forecast t+h, the reading for t is already observed.
    """
    series = df.set_index("timestamp_utc")["demand_mw"].astype(float).sort_index()
    feats = pd.DataFrame(index=series.index)

    # --- raw lags ---------------------------------------------------------
    for lag in cfg["features.lags"]:
        feats[f"lag_{lag}"] = series.shift(lag)

    # --- rolling windows: all windows END at t (inclusive) ----------------
    stats = cfg["features.rolling_stats"]
    for window in cfg["features.rolling_windows"]:
        roll = series.rolling(window=window, min_periods=max(2, window // 4))
        if "mean" in stats:
            feats[f"roll_mean_{window}"] = roll.mean()
        if "std" in stats:
            feats[f"roll_std_{window}"] = roll.std()
        if "min" in stats:
            feats[f"roll_min_{window}"] = roll.min()
        if "max" in stats:
            feats[f"roll_max_{window}"] = roll.max()

    # --- exponentially weighted levels ------------------------------------
    for halflife in cfg.get("features.ewm_halflives", []):
        feats[f"ewm_hl{halflife}"] = series.ewm(halflife=halflife, min_periods=2).mean()

    # --- differences / momentum -------------------------------------------
    for d in cfg["features.diff_lags"]:
        feats[f"diff_{d}"] = series - series.shift(d)
        with np.errstate(divide="ignore", invalid="ignore"):
            feats[f"pct_change_{d}"] = (series - series.shift(d)) / series.shift(d).replace(0, np.nan)

    # --- normalised position within recent range --------------------------
    # Where does the current reading sit inside the last day / week?
    for window in (24, 168):
        lo = feats.get(f"roll_min_{window}")
        hi = feats.get(f"roll_max_{window}")
        if lo is not None and hi is not None:
            span = (hi - lo).replace(0, np.nan)
            feats[f"pos_in_range_{window}"] = (series - lo) / span

    # --- volatility ratio: short-term vs long-term dispersion -------------
    if "roll_std_24" in feats and "roll_std_168" in feats:
        feats["vol_ratio_24_168"] = feats["roll_std_24"] / feats["roll_std_168"].replace(0, np.nan)

    # --- level ratios ------------------------------------------------------
    if "roll_mean_24" in feats and "roll_mean_168" in feats:
        feats["mean_ratio_24_168"] = feats["roll_mean_24"] / feats["roll_mean_168"].replace(0, np.nan)

    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats.index.name = ORIGIN_TS_COL
    return feats


# ---------------------------------------------------------------------------
# (b) TARGET-TIME FEATURES - calendar attributes of t+h
# ---------------------------------------------------------------------------
def build_calendar_features(timestamps: pd.DatetimeIndex, cfg: Config) -> pd.DataFrame:
    """Calendar / holiday attributes for a set of timestamps.

    Applied to TARGET timestamps. Deterministic from the clock, therefore known
    at forecast origin regardless of horizon.
    """
    idx = pd.DatetimeIndex(timestamps)
    cal = pd.DataFrame(index=idx)

    cal["hour"] = idx.hour
    cal["dayofweek"] = idx.dayofweek
    cal["day"] = idx.day
    cal["month"] = idx.month
    cal["quarter"] = idx.quarter
    cal["year"] = idx.year
    cal["dayofyear"] = idx.dayofyear
    cal["weekofyear"] = idx.isocalendar().week.to_numpy().astype(int)
    cal["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    cal["is_month_start"] = idx.is_month_start.astype(int)
    cal["is_month_end"] = idx.is_month_end.astype(int)

    # Interaction: the daily load shape differs sharply between weekday/weekend.
    cal["hour_x_weekend"] = cal["hour"] + 24 * cal["is_weekend"]
    cal["hour_x_dow"] = cal["hour"] + 24 * cal["dayofweek"]

    # Monotone trend term so the model can express slow structural drift.
    cal["time_index"] = (idx - idx.min()).total_seconds() / 3600.0

    if cfg.get("features.add_fourier_terms", True):
        orders = cfg.get("features.fourier_orders", {}) or {}
        _add_fourier(cal, idx.hour.to_numpy(), 24, int(orders.get("daily", 3)), "daily")
        _add_fourier(cal, (idx.dayofweek.to_numpy() * 24 + idx.hour.to_numpy()), 168,
                     int(orders.get("weekly", 2)), "weekly")
        _add_fourier(cal, idx.dayofyear.to_numpy(), 365.25, int(orders.get("yearly", 3)), "yearly")

    country = cfg.get("features.holiday_country", "US")
    cal = _add_holiday_features(cal, idx, country, cfg.get("features.add_holiday_adjacency", True))

    cal.index.name = TARGET_TS_COL
    return cal


def _add_fourier(frame: pd.DataFrame, values: np.ndarray, period: float, order: int, tag: str) -> None:
    """Smooth periodic encodings. Give the trees a continuous handle on cycles
    so they are not forced to spend splits carving up integer hour buckets."""
    for k in range(1, max(order, 0) + 1):
        angle = 2.0 * np.pi * k * values / period
        frame[f"fourier_{tag}_sin_{k}"] = np.sin(angle)
        frame[f"fourier_{tag}_cos_{k}"] = np.cos(angle)


def _add_holiday_features(
    cal: pd.DataFrame, idx: pd.DatetimeIndex, country: str, adjacency: bool
) -> pd.DataFrame:
    years = sorted({int(y) for y in idx.year.unique()})
    try:
        cal_holidays = holidays_pkg.country_holidays(country, years=years)
    except Exception as exc:  # pragma: no cover - unknown country code
        logger.warning("Holiday calendar unavailable for %s (%s); using zeros", country, exc)
        cal["is_holiday"] = 0
        if adjacency:
            cal["is_day_before_holiday"] = 0
            cal["is_day_after_holiday"] = 0
        return cal

    # PJM is a US grid operator; holiday effects follow US local calendar dates.
    local_dates = idx.tz_convert("America/New_York").normalize()
    date_vals = local_dates.date

    holiday_set = set(cal_holidays.keys())
    cal["is_holiday"] = np.fromiter((d in holiday_set for d in date_vals), dtype=int, count=len(idx))

    if adjacency:
        one_day = pd.Timedelta(days=1)
        prev_dates = (local_dates - one_day).date
        next_dates = (local_dates + one_day).date
        cal["is_day_before_holiday"] = np.fromiter(
            (d in holiday_set for d in next_dates), dtype=int, count=len(idx)
        )
        cal["is_day_after_holiday"] = np.fromiter(
            (d in holiday_set for d in prev_dates), dtype=int, count=len(idx)
        )
    return cal


# ---------------------------------------------------------------------------
# ASSEMBLY - join (a) and (b) into a supervised matrix for one horizon
# ---------------------------------------------------------------------------
def build_supervised_frame(
    df: pd.DataFrame,
    cfg: Config,
    horizon_hours: int,
    origin_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the (X, y) frame for one region and one horizon.

    Row semantics:
        forecast_timestamp = t          (origin; all demand features are <= t)
        target_timestamp   = t + h
        y                  = demand at t + h
    """
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be > 0")

    region_code = df["region_code"].iloc[0]
    if origin_features is None:
        origin_features = build_origin_features(df, cfg)

    series = df.set_index("timestamp_utc")["demand_mw"].astype(float).sort_index()

    frame = origin_features.copy()
    frame[TARGET_TS_COL] = frame.index + pd.Timedelta(hours=horizon_hours)
    # shift(-h) pulls the FUTURE value into the row as the label. The feature
    # block above is untouched, so no future information enters X.
    frame[TARGET_COL] = series.shift(-horizon_hours).to_numpy()

    calendar = build_calendar_features(pd.DatetimeIndex(frame[TARGET_TS_COL]), cfg)
    calendar = calendar.reset_index(drop=True)
    frame = frame.reset_index()  # forecast_timestamp becomes a column
    frame = pd.concat([frame, calendar], axis=1)

    frame["horizon_hours"] = int(horizon_hours)
    frame["region_code"] = region_code

    # A row is usable only if the label exists. Rows with NaN features survive -
    # LightGBM routes NaN natively and dropping them would silently discard the
    # earliest years of every region.
    before = len(frame)
    frame = frame[frame[TARGET_COL].notna()].reset_index(drop=True)
    logger.debug(
        "region=%s h=%d supervised rows %d -> %d (dropped %d with missing label)",
        region_code, horizon_hours, before, len(frame), before - len(frame),
    )
    return frame


def feature_columns(frame: pd.DataFrame) -> List[str]:
    """Model input columns = everything except metadata/label."""
    return [c for c in frame.columns if c not in META_COLUMNS]


def build_all_horizons(
    df: pd.DataFrame, cfg: Config, horizons: List[int] | None = None
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    """Origin features are horizon-independent, so compute them once and reuse."""
    horizons = horizons or cfg.horizons
    origin = build_origin_features(df, cfg)
    frames = {h: build_supervised_frame(df, cfg, h, origin_features=origin) for h in horizons}
    return frames, origin


# ---------------------------------------------------------------------------
# SELF-CHECK - executable proof that the feature block cannot see the future
# ---------------------------------------------------------------------------
def assert_no_future_leakage(df: pd.DataFrame, cfg: Config, horizon_hours: int,
                             n_probes: int = 25, seed: int = 0) -> Dict[str, object]:
    """Perturbation test for temporal leakage.

    Method: corrupt the demand series strictly AFTER a cut instant, rebuild
    features, and confirm that no feature row with origin <= cut changed. If any
    did, a feature is reading forward in time.
    """
    rng = np.random.default_rng(seed)
    baseline = build_supervised_frame(df, cfg, horizon_hours)
    feat_cols = feature_columns(baseline)
    demand_feats = [c for c in feat_cols if c.startswith(
        ("lag_", "roll_", "ewm_", "diff_", "pct_change_", "pos_in_range_", "vol_", "mean_ratio_")
    )]

    n = len(df)
    cut_positions = sorted(rng.choice(np.arange(int(n * 0.3), int(n * 0.9)),
                                      size=min(n_probes, max(1, int(n * 0.6) - 1)),
                                      replace=False).tolist())

    failures: List[Dict[str, object]] = []
    for pos in cut_positions[:n_probes]:
        cut_ts = df["timestamp_utc"].iloc[pos]
        corrupted = df.copy()
        mask = corrupted["timestamp_utc"] > cut_ts
        corrupted.loc[mask, "demand_mw"] = corrupted.loc[mask, "demand_mw"] * 1000.0 + 1e6

        probe = build_supervised_frame(corrupted, cfg, horizon_hours)
        merged = baseline[[ORIGIN_TS_COL] + demand_feats].merge(
            probe[[ORIGIN_TS_COL] + demand_feats], on=ORIGIN_TS_COL, suffixes=("_base", "_probe")
        )
        merged = merged[merged[ORIGIN_TS_COL] <= cut_ts]
        if merged.empty:
            continue
        for col in demand_feats:
            a = merged[f"{col}_base"].to_numpy(dtype=float)
            b = merged[f"{col}_probe"].to_numpy(dtype=float)
            both_nan = np.isnan(a) & np.isnan(b)
            differs = ~both_nan & ~np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)
            if differs.any():
                failures.append({"cut": str(cut_ts), "feature": col, "n_diff": int(differs.sum())})

    result = {
        "horizon_hours": horizon_hours,
        "probes": len(cut_positions[:n_probes]),
        "features_checked": len(demand_feats),
        "leaking_features": sorted({f["feature"] for f in failures}),
        "passed": not failures,
    }
    if failures:
        logger.error("LEAKAGE DETECTED: %s", result["leaking_features"])
    else:
        logger.info(
            "Leakage self-check PASSED (h=%dh, %d probes, %d demand features)",
            horizon_hours, result["probes"], len(demand_feats),
        )
    return result
