"""
Forecast accuracy metrics.

These four are what model_evaluations stores, and what cross-model comparison
runs on. Definitions are pinned here so DHR+ARIMA, LightGBM and PatchTST/TFT are
never scored by three subtly different formulas - which is the classic way a
model comparison quietly becomes meaningless.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error = sum|e| / sum|y|, as a percentage.

    Preferred over MAPE for this problem: MAPE divides each error by its own
    actual, so a single low-demand hour can dominate the score. WAPE's single
    aggregate denominator is stable, scale-free across regions of very different
    size (DAYTON peaks ~3.7 GW, PJME ~62 GW), and directly interpretable.
    """
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """Symmetric MAPE, percentage. Denominators below `epsilon` are dropped
    rather than allowed to explode."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    valid = denom > epsilon
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[valid] - y_pred[valid]) / denom[valid]) * 100.0)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error. Positive = model under-forecasts on average."""
    return float(np.mean(y_true - y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float("nan") if ss_tot == 0 else float(1.0 - ss_res / ss_tot)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, smape_epsilon: float = 1.0
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    n_dropped = int((~finite).sum())
    if n_dropped:
        logger.debug("Metrics: dropped %d non-finite pairs", n_dropped)
    y_true, y_pred = y_true[finite], y_pred[finite]

    if y_true.size == 0:
        return {k: float("nan") for k in
                ("mae", "rmse", "wape", "smape", "bias", "r2")} | {"n_observations": 0}

    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "smape": smape(y_true, y_pred, smape_epsilon),
        "bias": bias(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "n_observations": int(y_true.size),
    }


def naive_seasonal_baseline(
    frame: pd.DataFrame, horizon_hours: int, target_col: str = "y"
) -> Optional[np.ndarray]:
    """Seasonal-naive reference: prediction for t+h is demand at the same hour
    one week earlier. A model that cannot beat this is not earning its keep, so
    every run reports the comparison.

    For h <= 168 the required value is lag_(168-h) relative to origin t, which
    the feature block already contains and which is strictly in the past.
    """
    needed_lag = 168 - horizon_hours
    col = f"lag_{needed_lag}"
    if needed_lag < 0 or col not in frame.columns:
        return None
    return frame[col].to_numpy(dtype=float)


def evaluate_split(
    frame: pd.DataFrame,
    pred_col: str = "predicted_demand_mw",
    target_col: str = "y",
    smape_epsilon: float = 1.0,
    horizon_hours: Optional[int] = None,
) -> Dict[str, float]:
    metrics = compute_metrics(
        frame[target_col].to_numpy(dtype=float),
        frame[pred_col].to_numpy(dtype=float),
        smape_epsilon,
    )
    if horizon_hours is not None:
        baseline = naive_seasonal_baseline(frame, horizon_hours, target_col)
        if baseline is not None:
            base = compute_metrics(frame[target_col].to_numpy(dtype=float), baseline, smape_epsilon)
            metrics["baseline_mae"] = base["mae"]
            metrics["baseline_wape"] = base["wape"]
            if base["mae"] and np.isfinite(base["mae"]) and base["mae"] > 0:
                # Skill score > 0 means we beat seasonal-naive.
                metrics["skill_vs_naive"] = float(1.0 - metrics["mae"] / base["mae"])
    return metrics
