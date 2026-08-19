"""
Scored (region, horizon) rows -> ANOMALY_CONTRACT_COLUMNS.

The important design point: each row already carries its OWN model_run_id and
forecast_id, pulled straight from v_selected_forecast. Because that view can
legitimately hand back different winning models for different (region,
horizon) cells - that's the entire point of automatic model selection - a
single anomaly-scoring pass produces rows anchored to several different
model_run_ids at once. `run_uid` in the contract is therefore filled in
per-row from whichever model actually produced that forecast, not from a
single "the anomaly detector's own run" identity the way LightGBM/DHR-ARIMA
do it for their own (single-model) outputs.
"""
from __future__ import annotations

import logging

import pandas as pd

from .contracts import ANOMALY_CONTRACT_COLUMNS, validate_anomaly_frame

logger = logging.getLogger(__name__)


def _build_reason(row: pd.Series) -> str:
    if not row["is_anomaly"]:
        return "Within expected forecast error band."
    direction = "exceeded" if row["residual_mw"] > 0 else "fell short of"
    evidence = (
        "both statistical and Isolation Forest evidence agree" if row["strong_anomaly"]
        else "statistical residual evidence only" if row["statistical_only_candidate"]
        else "Isolation Forest calendar-pattern evidence only"
    )
    event_note = f" Part of event {row['event_id']}." if row.get("event_id") else ""
    return (
        f"Actual demand ({row['model_name']} forecast) {direction} prediction by "
        f"{abs(row['residual_mw']):.0f} MW ({abs(row['residual_pct']):.1f}%), "
        f"composite score {row['statistical_anomaly_score']:.2f} "
        f"(IF score {row['if_anomaly_score']:.2f}) - {evidence}. Severity {row['severity']}.{event_note}"
    )


def to_anomaly_contract(scored: pd.DataFrame, cfg) -> pd.DataFrame:
    df = scored.copy()
    df["reason"] = df.apply(_build_reason, axis=1)

    out = pd.DataFrame({
        "run_uid": df["model_run_id"].astype(str),   # per-row winning model's run
        "region_code": df["region_code"],
        "timestamp_utc": df["target_timestamp"],
        "horizon_hours": df["horizon_hours"].astype(int),
        "actual_demand_mw": df["actual_demand_mw"].astype(float),
        "predicted_demand_mw": df["predicted_demand_mw"].astype(float),
        "residual_mw": df["residual_mw"].astype(float),
        "deviation_percent": df["residual_pct"].astype(float),
        "anomaly_score": df["statistical_anomaly_score"].astype(float),
        "severity": df["severity"],
        "is_anomaly": df["is_anomaly"].astype(bool),
        "anomaly_direction": df["anomaly_direction"],
        "reason": df["reason"],
        "detection_method": cfg.get("project.detection_method", "ensight_hybrid_stat_if_v1"),
    })
    # Keep the DB-facing forecast_id and model_run_id (native int, needed by the
    # writer to attach anomalies to their exact parent forecast row) alongside
    # the contract columns rather than discarding them here.
    out["forecast_id"] = df["forecast_id"]
    out["model_run_id"] = df["model_run_id"]

    validated = validate_anomaly_frame(out[ANOMALY_CONTRACT_COLUMNS])
    out = pd.concat([validated, out[["forecast_id", "model_run_id"]]], axis=1)
    logger.info("Adapted %d anomaly rows (%d flagged) to contract shape",
                len(out), int(out["is_anomaly"].sum()))
    return out
