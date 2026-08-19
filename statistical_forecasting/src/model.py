"""
DHR + ARIMA training and rolling-origin forecasting.

STRATEGY - GENUINELY DIFFERENT FROM LIGHTGBM, ON PURPOSE
---------------------------------------------------------
LightGBM trains one independent booster per horizon because a tree ensemble has
no native notion of "forecast N steps ahead from here" - direct multi-horizon is
the correct answer for that model family.

A state-space model (SARIMAX) is the opposite: it IS a generative model of the
series, so it produces a genuine multi-step forecast path from a single fit.
Building 12 separate ARIMA fits per region to match LightGBM's per-horizon
structure would not be more correct - it would just be paying MLE fitting cost
12 times over for something one fit already gives you.

So DHR+ARIMA fits ONCE per region and is then advanced through the held-out
period on a cadence (default: every 24h), producing a fresh multi-step forecast
from each cadence point using `results.append(..., refit=False)` - which
updates the Kalman filter STATE with newly-observed data but re-estimates
NOTHING. This mirrors how a statistical forecasting service is actually run in
production (state refreshed on a schedule, parameters re-estimated far less
often) and matches the project's own "give me the next 24 hours" framing.

WHY A CADENCE, NOT EVERY HOUR
------------------------------
Rolling the origin forward one hour at a time across ~36,000 held-out hours
would mean 36,000 append+forecast cycles per region for marginal additional
insight over a daily cadence - the forecast distribution barely changes between
origin 14:00 and origin 15:00 on the same day. 24-hour cadence is the honest,
documented trade-off, not a shortcut hidden from the numbers: it does mean
DHR+ARIMA produces far fewer forecast rows than LightGBM for the same region,
and that difference is recorded in the run's metadata rather than papered over.

NO LEAKAGE
----------
`results.append()` only ever receives observations up to the current origin.
The exogenous regressors for the FORECAST horizon are pure calendar features
(see dhr_features.py) - deterministic functions of the target timestamp, known
regardless of horizon. Nothing about the demand series beyond the origin is
touched until that block is appended, one cadence step later.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import Config
from .contracts import ModelRunContract
from .dhr_features import build_exog
from .evaluation import evaluate_split
from .utils import Timer

logger = logging.getLogger(__name__)

SPLIT_COL = "split_name"


@dataclass
class HorizonSeries:
    horizon_hours: int
    predictions: pd.DataFrame   # forecast_timestamp, target_timestamp, actual, predicted, split
    metrics: Dict[str, Dict[str, float]]


@dataclass
class RegionResult:
    region_code: str
    run: ModelRunContract
    horizons: Dict[int, HorizonSeries] = field(default_factory=dict)
    n_origins: int = 0
    fit_seconds: float = 0.0
    status: str = "SUCCESS"
    failure_reason: Optional[str] = None

    @property
    def succeeded_horizons(self) -> List[int]:
        return sorted(self.horizons.keys()) if self.status == "SUCCESS" else []

    @property
    def failed_horizons(self) -> List[int]:
        return [] if self.status == "SUCCESS" else sorted(self.horizons.keys())


def _chronological_bounds(index: pd.DatetimeIndex, cfg: Config):
    n = len(index)
    train_frac = float(cfg.get("split.train_frac", 0.70))
    val_frac = float(cfg.get("split.val_frac", 0.15))
    train_end = index[max(1, int(n * train_frac) - 1)]
    val_end = index[min(int(n * (train_frac + val_frac)) - 1, n - 2)]
    return train_end, val_end


def train_region(region_df: pd.DataFrame, cfg: Config, horizons: Optional[List[int]] = None) -> RegionResult:
    region_code = region_df["region_code"].iloc[0]
    horizons = sorted(horizons or cfg.horizons)
    max_h = max(int(cfg.get("forecast.max_horizon_hours", 24)), max(horizons))
    cadence = int(cfg.get("forecast.origin_cadence_hours", 24))

    logger.info("=" * 78)
    logger.info("REGION %s | DHR+ARIMA | horizons=%s | cadence=%dh", region_code, horizons, cadence)
    logger.info("=" * 78)

    series = region_df.set_index("timestamp_utc")["demand_mw"].astype(float).sort_index()
    # ARIMA cannot fit through NaN. Grid gaps are rare (see data_loader) - a
    # short linear interpolation is a defensible, clearly-logged fill for a
    # handful of missing hours; it is never used as a "real" observation
    # anywhere else (LightGBM leaves these as native NaN, since trees handle
    # them; SARIMAX cannot).
    n_missing = int(series.isna().sum())
    if n_missing:
        logger.info("  Interpolating %d missing hours (%.3f%%) for ARIMA fitting",
                    n_missing, 100 * n_missing / len(series))
    series = series.interpolate(method="time").ffill().bfill()

    train_end, val_end = _chronological_bounds(series.index, cfg)
    train_y = series[series.index <= train_end]
    t0 = train_y.index[0]

    full_exog = build_exog(series.index, cfg, t0=t0)
    train_exog = full_exog.loc[train_y.index]

    order = tuple(cfg.get("arima.order", [2, 0, 2]))
    try:
        with Timer() as fit_timer, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.tsa.SARIMAX(
                train_y, exog=train_exog, order=order,
                trend=cfg.get("arima.trend", None),
                enforce_stationarity=bool(cfg.get("arima.enforce_stationarity", False)),
                enforce_invertibility=bool(cfg.get("arima.enforce_invertibility", False)),
            )
            results = model.fit(
                method=cfg.get("arima.fit_method", "lbfgs"),
                maxiter=int(cfg.get("arima.fit_maxiter", 60)),
                disp=False,
            )
        logger.info("  Fit complete in %.1fs | order=%s | aic=%.1f | train_rows=%d",
                    fit_timer.seconds, order, results.aic, len(train_y))
    except Exception as exc:  # noqa: BLE001
        logger.exception("  DHR+ARIMA fit FAILED for %s", region_code)
        run = _failed_run_contract(cfg, region_code, horizons, exc)
        return RegionResult(region_code=region_code, run=run, status="FAILED",
                            failure_reason=f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Rolling-origin forecasting through val + test, state-only updates.
    # ------------------------------------------------------------------
    holdout = series[series.index > train_end]
    origins = holdout.index[::cadence]  # cadence-spaced timestamps within holdout, ascending

    rows: List[Dict[str, object]] = []
    current = results
    cursor = train_end
    n_origins = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for origin in origins:
            future_idx = pd.date_range(origin + pd.Timedelta(hours=1), periods=max_h, freq="h", tz="UTC")
            future_idx = future_idx[future_idx <= series.index.max()]
            if future_idx.empty:
                break
            future_exog = full_exog.loc[future_idx]

            forecast = current.get_forecast(steps=len(future_idx), exog=future_exog)
            predicted = forecast.predicted_mean

            split = "val" if origin <= val_end else "test"
            for h in horizons:
                target_ts = origin + pd.Timedelta(hours=h)
                if target_ts not in predicted.index:
                    continue
                rows.append({
                    "forecast_timestamp": origin,
                    "target_timestamp": target_ts,
                    "horizon_hours": h,
                    "predicted_demand_mw": float(predicted.loc[target_ts]),
                    "actual_demand_mw": float(series.loc[target_ts]) if target_ts in series.index else None,
                    SPLIT_COL: split,
                })
            n_origins += 1
            if n_origins % 200 == 0:
                logger.info("    ...%d forecast origins processed (%s)", n_origins, origin)

            # Advance the STATE (not the parameters) through the next cadence
            # block so the next iteration's forecast starts from what actually
            # happened, not from what was predicted.
            #
            # `.extend()`, not `.append()`: append() re-filters the ENTIRE
            # series-so-far on every call (O(n) each time -> O(n^2) over ~1,500
            # cadence steps - this is what made the first version of this loop
            # time out). extend() re-initializes a fresh state-space model from
            # the current filtered state and only filters the NEW block, so
            # each step costs O(cadence) instead of O(elapsed_so_far).
            block_idx = holdout.index[(holdout.index > cursor) & (holdout.index <= origin + pd.Timedelta(hours=cadence))]
            if len(block_idx) == 0:
                block_idx = holdout.index[(holdout.index > cursor) & (holdout.index <= origin)]
            if len(block_idx) > 0:
                current = current.extend(series.loc[block_idx], exog=full_exog.loc[block_idx])
                cursor = block_idx.max()

    if not rows:
        run = _failed_run_contract(cfg, region_code, horizons, RuntimeError("no forecast rows produced"))
        return RegionResult(region_code=region_code, run=run, status="FAILED",
                            failure_reason="no forecast rows produced")

    all_preds = pd.DataFrame(rows)
    smape_eps = float(cfg.get("evaluation.smape_epsilon", 1.0))

    horizon_results: Dict[int, HorizonSeries] = {}
    for h in horizons:
        part = all_preds[all_preds["horizon_hours"] == h].copy()
        metrics: Dict[str, Dict[str, float]] = {}
        for split_name in ("val", "test"):
            split_part = part[part[SPLIT_COL] == split_name]
            split_part = split_part[split_part["actual_demand_mw"].notna()]
            if split_part.empty:
                continue
            scored = split_part.rename(columns={"actual_demand_mw": "y"})
            metrics[split_name] = evaluate_split(scored, smape_epsilon=smape_eps, horizon_hours=h)
        horizon_results[h] = HorizonSeries(horizon_hours=h, predictions=part, metrics=metrics)
        test_m = metrics.get("test", {})
        logger.info("  h=%-3d n_origins_in_test=%-5d test MAE=%8.1f WAPE=%5.2f%% sMAPE=%5.2f%%",
                    h, int((part[SPLIT_COL] == "test").sum()),
                    test_m.get("mae", float("nan")), test_m.get("wape", float("nan")),
                    test_m.get("smape", float("nan")))

    run = ModelRunContract(
        model_name=cfg["project.model_name"],
        model_type=cfg["project.model_type"],
        model_version=cfg["project.model_version"],
        region_code=region_code,
        training_start=train_y.index.min(),
        training_end=train_y.index.max(),
        horizons=horizons,
        feature_version=cfg["dhr.feature_version"],
        code_version=cfg["project.code_version"],
        status="SUCCESS",
        n_features=int(full_exog.shape[1]),
        n_training_rows=int(len(train_y)),
        metadata={
            "strategy": "single_fit_rolling_origin_state_update",
            "arima_order": list(order),
            "aic": float(results.aic),
            "origin_cadence_hours": cadence,
            "n_forecast_origins": n_origins,
            "supported_horizons": horizons,
            "trained_horizons": horizons,
            "failed_horizons": [],
            "data_available_through": region_df["timestamp_utc"].max().isoformat(),
            "note": "forecast rows are cadence-spaced origins, not one row per hour "
                    "(see forecast.origin_cadence_hours) - fewer rows than LightGBM by design",
        },
    )

    return RegionResult(
        region_code=region_code, run=run, horizons=horizon_results,
        n_origins=n_origins, fit_seconds=fit_timer.seconds, status="SUCCESS",
    )


def _failed_run_contract(cfg: Config, region_code: str, horizons: List[int], exc: Exception) -> ModelRunContract:
    now = pd.Timestamp.now(tz="UTC")
    return ModelRunContract(
        model_name=cfg["project.model_name"],
        model_type=cfg["project.model_type"],
        model_version=cfg["project.model_version"],
        region_code=region_code,
        training_start=now,
        training_end=now,
        horizons=horizons,
        feature_version=cfg["dhr.feature_version"],
        code_version=cfg["project.code_version"],
        status="FAILED",
        failure_reason=f"{type(exc).__name__}: {exc}",
        metadata={"supported_horizons": horizons, "trained_horizons": [], "failed_horizons": horizons},
    )
