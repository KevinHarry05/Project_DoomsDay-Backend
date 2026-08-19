"""
patchtst_adapter - native RegionResult output -> standardized cross-model contract.

Same shape as the DHR+ARIMA and LightGBM adapters on purpose: the integration
layer (db_writer.py) never needs to know a model is a pooled transformer
instead of a statistical model or a gradient-boosted tree ensemble.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from .contracts import (
    ANOMALY_CONTRACT_COLUMNS,
    EVALUATION_CONTRACT_COLUMNS,
    FORECAST_CONTRACT_COLUMNS,
    validate_evaluation_frame,
    validate_forecast_frame,
)
from .model import RegionResult

logger = logging.getLogger(__name__)

ADAPTER_NAME = "patchtst_adapter"


def to_forecast_contract(result: RegionResult, cfg, splits: Optional[List[str]] = None) -> pd.DataFrame:
    splits = splits or list(cfg.get("database.write_forecast_splits", ["test"]))
    model_name = cfg["project.model_name"]
    run_uid = result.run.run_uid

    rows: List[pd.DataFrame] = []
    for horizon, hseries in sorted(result.horizons.items()):
        part = hseries.predictions[hseries.predictions["split_name"].isin(splits)].copy()
        if part.empty:
            continue
        out = pd.DataFrame({
            "model_name": model_name,
            "run_uid": run_uid,
            "region_code": result.region_code,
            "forecast_timestamp": part["forecast_timestamp"],
            "target_timestamp": part["target_timestamp"],
            "horizon_hours": int(horizon),
            "predicted_demand_mw": part["predicted_demand_mw"].astype(float),
            "actual_demand_mw": part["actual_demand_mw"],
            "split_name": part["split_name"],
        })
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=FORECAST_CONTRACT_COLUMNS)
    frame = pd.concat(rows, ignore_index=True).sort_values(["horizon_hours", "target_timestamp"])
    return validate_forecast_frame(frame.reset_index(drop=True))


def to_evaluation_contract(result: RegionResult, cfg) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    run_uid = result.run.run_uid
    for horizon, hseries in sorted(result.horizons.items()):
        for split_name, metrics in hseries.metrics.items():
            part = hseries.predictions[hseries.predictions["split_name"] == split_name]
            part = part[part["actual_demand_mw"].notna()]
            if part.empty:
                continue
            rows.append({
                "run_uid": run_uid,
                "region_code": result.region_code,
                "horizon_hours": int(horizon),
                "split_name": split_name,
                "evaluation_start": part["target_timestamp"].min(),
                "evaluation_end": part["target_timestamp"].max(),
                "n_observations": int(metrics.get("n_observations", len(part))),
                "mae": _f(metrics.get("mae")),
                "rmse": _f(metrics.get("rmse")),
                "wape": _f(metrics.get("wape")),
                "smape": _f(metrics.get("smape")),
                "training_time_seconds": _f(result.fit_seconds) if split_name == "test" else None,
                "inference_time_ms": None,
                "bias": _f(metrics.get("bias")),
                "r2": _f(metrics.get("r2")),
                "baseline_mae": _f(metrics.get("baseline_mae")),
                "baseline_wape": _f(metrics.get("baseline_wape")),
                "skill_vs_naive": _f(metrics.get("skill_vs_naive")),
            })
    if not rows:
        return pd.DataFrame(columns=EVALUATION_CONTRACT_COLUMNS)
    frame = pd.DataFrame(rows)
    validate_evaluation_frame(frame)
    return frame


def _f(value):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(out) else out


def to_anomaly_contract(result: RegionResult, cfg):
    # Anomaly detection is a separate, deliberately-later stage - same
    # convention as the other tracks. Empty by design until enabled.
    return pd.DataFrame(columns=ANOMALY_CONTRACT_COLUMNS), {}


def adapt_region(result: RegionResult, cfg) -> Dict[str, object]:
    forecasts = to_forecast_contract(result, cfg)
    evaluations = to_evaluation_contract(result, cfg)
    anomalies, threshold_meta = to_anomaly_contract(result, cfg)
    payload = {
        "adapter": ADAPTER_NAME,
        "model_run": result.run,
        "forecasts": forecasts,
        "evaluations": evaluations,
        "anomalies": anomalies,
        "anomaly_thresholds": threshold_meta,
    }
    logger.info("Adapted %-9s -> run_uid=%s forecasts=%d evaluations=%d",
                result.region_code, result.run.run_uid[:12], len(forecasts), len(evaluations))
    return payload


def describe_capabilities(cfg) -> Dict[str, object]:
    return {
        "model_name": cfg["project.model_name"],
        "model_type": cfg["project.model_type"],
        "model_version": cfg["project.model_version"],
        "adapter": ADAPTER_NAME,
        "supported_horizons": cfg.horizons,
        "forecast_generation_method": "ONE shared PatchTST transformer trained pooled across "
                                      "all regions; direct multi-horizon output per origin "
                                      f"(cadence {cfg.get('sequence.origin_cadence_hours', 24)}h), "
                                      "not autoregressive.",
        "input_file_pattern": "<REGION>_clean.csv",
        "input_columns": ["Load_Area", "Datetime_UTC", "Datetime_EPT", "Demand_MW", "Missing_Flag"],
        "output_format": "contracts.FORECAST_CONTRACT_COLUMNS",
        "region_handling": "single pooled/global model; one model_runs row per region for "
                            "schema consistency (metadata.pooled_global_model=true)",
        "architecture": "patch embedding -> transformer encoder -> linear projection head",
        "scale_down_from_notebook": {
            "d_model": "128 -> %s" % cfg.get("model.d_model"),
            "n_layers": "3 -> %s" % cfg.get("model.n_layers"),
            "n_heads": "8 -> %s" % cfg.get("model.n_heads"),
        },
        "metrics_produced": ["mae", "rmse", "wape", "smape"],
        "notes": (
            "CPU-feasible rebuild, not the team's GPU notebook - see README.md "
            "'Scope and limitations' for the full rationale and honest accuracy "
            "expectations relative to a full-scale run."
        ),
    }
