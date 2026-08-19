"""
Residual-based anomaly detection on LightGBM forecasts.

SCOPE BOUNDARY
--------------
The project spec is explicit that anomaly detection must run against ONE clearly
defined forecast and must not mix predictions from different models. This module
therefore scores LightGBM's own residuals at a SINGLE configured horizon
(default 1h) and stamps every row with `detection_method` and the owning
`run_uid`.

Whether LightGBM's forecast is the one the production anomaly view should use is
decided later, by the integration layer, from model_evaluations. What we produce
here are LightGBM-scoped anomaly candidates - correct and complete for this
model, and directly comparable against the other tracks' equivalents.

METHOD
------
Robust z-score on the forecast residual:

    residual  = actual - predicted
    z         = |residual - median(train residuals)| / (1.4826 * MAD(train residuals))

Median/MAD rather than mean/std because the statistics are being estimated from
a sample that itself contains the outliers we are hunting. A handful of large
excursions inflates a standard deviation enough to hide the next one; MAD has a
50% breakdown point and barely moves.

Residuals are additionally conditioned on hour-of-day. A 400 MW miss at the 07:00
ramp is ordinary; the same miss at 03:00 is not. Pooling all hours into one
threshold systematically over-flags peak hours and under-flags overnight ones.

WHICH SPLIT FITS THE THRESHOLD
------------------------------
Validation, not training. This matters more than it looks. A boosted tree's
in-sample residuals are far tighter than its out-of-sample residuals, so a scale
estimated on the training split is badly optimistic - in a first run on AEP it
flagged 20.5% of test hours, which is a broken threshold rather than a finding.
Validation residuals are genuinely out-of-sample yet sit strictly before the test
period, so they are both realistic and leakage-free.

Scoring the test period against thresholds derived from that same period would
let the anomalies define their own normal, so test is never in `fit_on_splits`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .contracts import ANOMALY_CONTRACT_COLUMNS
from .feature_engineering import TARGET_TS_COL
from .splits import SPLIT_COL

logger = logging.getLogger(__name__)

MAD_TO_SIGMA = 1.4826
_MIN_HOUR_SAMPLES = 50


@dataclass
class ResidualThresholds:
    """Fitted per-hour location/scale of the residual distribution."""

    global_center: float
    global_scale: float
    per_hour: Dict[int, Dict[str, float]] = field(default_factory=dict)
    fitted_on_splits: List[str] = field(default_factory=list)
    n_fit_rows: int = 0
    robust: bool = True

    def center_scale(self, hour: int) -> tuple[float, float]:
        stats = self.per_hour.get(int(hour))
        if stats and stats["n"] >= _MIN_HOUR_SAMPLES and stats["scale"] > 0:
            return stats["center"], stats["scale"]
        return self.global_center, self.global_scale

    def to_dict(self) -> Dict[str, object]:
        return {
            "global_center": self.global_center,
            "global_scale": self.global_scale,
            "fitted_on_splits": self.fitted_on_splits,
            "n_fit_rows": self.n_fit_rows,
            "robust": self.robust,
            "per_hour": {str(k): v for k, v in sorted(self.per_hour.items())},
        }


def _center_scale(values: np.ndarray, robust: bool) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0
    if robust:
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        scale = mad * MAD_TO_SIGMA
        if scale <= 0:  # degenerate (>50% identical residuals) - fall back
            scale = float(np.std(values))
        return center, scale
    return float(np.mean(values)), float(np.std(values))


def fit_thresholds(predictions: pd.DataFrame, cfg: Config) -> ResidualThresholds:
    """Estimate residual location/scale from the configured fitting splits."""
    fit_splits = list(cfg.get("anomaly.fit_on_splits", ["train", "val"]))
    robust = bool(cfg.get("anomaly.robust", True))

    fit_df = predictions[predictions[SPLIT_COL].isin(fit_splits)].copy()
    fit_df = fit_df[np.isfinite(fit_df["actual_demand_mw"]) & np.isfinite(fit_df["predicted_demand_mw"])]
    if fit_df.empty:
        raise ValueError(f"No rows in splits {fit_splits} to fit anomaly thresholds")

    fit_df["residual_mw"] = fit_df["actual_demand_mw"] - fit_df["predicted_demand_mw"]
    fit_df["hour"] = pd.DatetimeIndex(fit_df[TARGET_TS_COL]).hour

    g_center, g_scale = _center_scale(fit_df["residual_mw"].to_numpy(float), robust)

    per_hour: Dict[int, Dict[str, float]] = {}
    for hour, grp in fit_df.groupby("hour"):
        c, s = _center_scale(grp["residual_mw"].to_numpy(float), robust)
        per_hour[int(hour)] = {"center": c, "scale": s, "n": int(len(grp))}

    thresholds = ResidualThresholds(
        global_center=g_center,
        global_scale=g_scale,
        per_hour=per_hour,
        fitted_on_splits=fit_splits,
        n_fit_rows=int(len(fit_df)),
        robust=robust,
    )
    logger.info(
        "  Residual thresholds fitted on %s (%d rows): center=%.2f MW scale=%.2f MW (%s)",
        fit_splits, len(fit_df), g_center, g_scale, "robust MAD" if robust else "mean/std",
    )
    return thresholds


def _classify_severity(score: float, bands: Dict[str, float]) -> str:
    if not np.isfinite(score):
        return "NONE"
    if score >= bands.get("critical", 7.0):
        return "CRITICAL"
    if score >= bands.get("high", 5.0):
        return "HIGH"
    if score >= bands.get("medium", 3.5):
        return "MEDIUM"
    if score >= bands.get("low", 2.5):
        return "LOW"
    return "NONE"


def _build_reason(row: pd.Series) -> str:
    """Human-readable explanation. This is the seed of the wider explanation
    layer - deliberately factual and derived only from stored quantities, so the
    API never has to reverse-engineer why a row was flagged."""
    if not row["is_anomaly"]:
        return "Within expected forecast error band."
    direction = "exceeded" if row["residual_mw"] > 0 else "fell short of"
    return (
        f"Actual demand {direction} the LightGBM forecast by "
        f"{abs(row['residual_mw']):.0f} MW ({abs(row['deviation_percent']):.1f}%), "
        f"a {row['anomaly_score']:.1f}-sigma deviation against the hour-{int(row['hour']):02d} "
        f"residual distribution. Severity {row['severity']}."
    )


def detect_anomalies(
    predictions: pd.DataFrame,
    cfg: Config,
    region_code: str,
    horizon_hours: int,
    run_uid: str,
    thresholds: Optional[ResidualThresholds] = None,
) -> tuple[pd.DataFrame, ResidualThresholds]:
    """Score the configured splits and return contract-shaped anomaly rows."""
    thresholds = thresholds or fit_thresholds(predictions, cfg)
    score_splits = list(cfg.get("anomaly.score_on_splits", ["test"]))

    df = predictions[predictions[SPLIT_COL].isin(score_splits)].copy()
    df = df[np.isfinite(df["actual_demand_mw"]) & np.isfinite(df["predicted_demand_mw"])]
    if df.empty:
        logger.warning("  No scorable rows for anomaly detection (%s h=%d)", region_code, horizon_hours)
        return pd.DataFrame(columns=ANOMALY_CONTRACT_COLUMNS), thresholds

    df["residual_mw"] = df["actual_demand_mw"] - df["predicted_demand_mw"]
    df["hour"] = pd.DatetimeIndex(df[TARGET_TS_COL]).hour

    centers, scales = zip(*(thresholds.center_scale(h) for h in df["hour"]))
    centers = np.asarray(centers, dtype=float)
    scales = np.asarray(scales, dtype=float)
    safe_scales = np.where(scales > 0, scales, np.nan)

    df["anomaly_score"] = np.abs(df["residual_mw"].to_numpy(float) - centers) / safe_scales
    df["anomaly_score"] = df["anomaly_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        df["deviation_percent"] = np.where(
            np.abs(df["actual_demand_mw"]) > 0,
            df["residual_mw"] / df["actual_demand_mw"] * 100.0,
            0.0,
        )

    bands = dict(cfg.get("anomaly.severity_thresholds", {}))
    df["severity"] = [_classify_severity(s, bands) for s in df["anomaly_score"]]

    min_score = float(cfg.get("anomaly.is_anomaly_min_score", 2.5))
    min_abs = float(cfg.get("anomaly.min_abs_residual_mw", 0.0))
    min_dev = float(cfg.get("anomaly.min_deviation_percent", 0.0))
    df["is_anomaly"] = (
        (df["anomaly_score"] >= min_score)
        & (df["residual_mw"].abs() >= min_abs)
        & (df["deviation_percent"].abs() >= min_dev)
    )

    df["anomaly_direction"] = np.where(
        ~df["is_anomaly"], "NONE", np.where(df["residual_mw"] > 0, "OVER", "UNDER")
    )
    df.loc[~df["is_anomaly"], "severity"] = "NONE"
    df["reason"] = df.apply(_build_reason, axis=1)

    df["run_uid"] = run_uid
    df["region_code"] = region_code
    df["horizon_hours"] = int(horizon_hours)
    df["timestamp_utc"] = df[TARGET_TS_COL]
    df["detection_method"] = cfg.get("anomaly.detection_method", "lgbm_residual_robust_z_v1")

    out = df[ANOMALY_CONTRACT_COLUMNS].reset_index(drop=True)
    n_flagged = int(out["is_anomaly"].sum())
    logger.info(
        "  Anomalies %-9s h=%-3d scored=%d flagged=%d (%.2f%%) | %s",
        region_code, horizon_hours, len(out), n_flagged,
        100.0 * n_flagged / max(len(out), 1),
        out[out["is_anomaly"]]["severity"].value_counts().to_dict() or "{}",
    )
    return out, thresholds
