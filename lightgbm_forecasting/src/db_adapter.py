"""
lightgbm_adapter - native LightGBM output -> standardized cross-model contract.

This is the seam the integration spec asks for. Everything model-specific stops
here; everything downstream (db_writer, comparison, anomaly consumption, API)
sees only the shapes defined in contracts.py.

The sibling adapters - statistical_adapter, patchtst_adapter, tft_adapter - must
expose the same three functions with the same return shapes. If a fourth model
appears, it writes an adapter and nothing else in the stack changes.

Mapping performed here (LightGBM native -> contract):
    forecast_timestamp   <- origin t of the supervised row
    target_timestamp     <- t + h
    horizon_hours        <- h (model-per-horizon, so never inferred)
    predicted_demand_mw  <- booster.predict(...)
    actual_demand_mw     <- label y (NULL for genuine future forecasts)
    region_code          <- region of the booster; resolved to region_id at write
    run_uid              <- deterministic hash; resolved to model_run_id at write

Note on model_run_id: it is a database-assigned surrogate key, so it cannot
exist before the write. The adapter carries `run_uid` instead and db_writer
swaps it for the real model_run_id after inserting the parent model_runs row.
This is what keeps the write order (model_runs -> forecasts) honest without the
adapter inventing an identifier.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .anomaly import detect_anomalies
from .config import Config
from .contracts import (
    ANOMALY_CONTRACT_COLUMNS,
    EVALUATION_CONTRACT_COLUMNS,
    FORECAST_CONTRACT_COLUMNS,
    ModelRunContract,
    validate_anomaly_frame,
    validate_evaluation_frame,
    validate_forecast_frame,
)
from .feature_engineering import ORIGIN_TS_COL, TARGET_TS_COL
from .model import RegionResult
from .splits import SPLIT_COL

logger = logging.getLogger(__name__)

ADAPTER_NAME = "lightgbm_adapter"


# ---------------------------------------------------------------------------
def to_forecast_contract(
    result: RegionResult, cfg: Config, splits: Optional[List[str]] = None
) -> pd.DataFrame:
    """Flatten every horizon's predictions into the standardized forecast frame."""
    splits = splits or list(cfg.get("database.write_forecast_splits", ["test"]))
    model_name = cfg["project.model_name"]
    run_uid = result.run.run_uid

    rows: List[pd.DataFrame] = []
    for horizon, hres in sorted(result.horizons.items()):
        if hres.status != "SUCCESS" or hres.predictions.empty:
            continue
        part = hres.predictions[hres.predictions[SPLIT_COL].isin(splits)].copy()
        if part.empty:
            continue
        out = pd.DataFrame({
            "model_name": model_name,
            "run_uid": run_uid,
            "region_code": result.region_code,
            "forecast_timestamp": part[ORIGIN_TS_COL],
            "target_timestamp": part[TARGET_TS_COL],
            "horizon_hours": int(horizon),
            "predicted_demand_mw": part["predicted_demand_mw"].astype(float),
            "actual_demand_mw": part["actual_demand_mw"].astype(float),
            "split_name": part[SPLIT_COL],
        })
        rows.append(out)

    if not rows:
        logger.warning("Region %s produced no forecast rows for splits %s",
                       result.region_code, splits)
        return pd.DataFrame(columns=FORECAST_CONTRACT_COLUMNS)

    frame = pd.concat(rows, ignore_index=True)
    frame = frame.sort_values(["horizon_hours", "target_timestamp"]).reset_index(drop=True)
    return validate_forecast_frame(frame)


# ---------------------------------------------------------------------------
def to_evaluation_contract(result: RegionResult, cfg: Config) -> pd.DataFrame:
    """One row per (run, region, horizon, split) -> model_evaluations.

    All three splits are emitted, but comparison must use `test` only. Training
    metrics are kept for diagnosing over/under-fitting, not for ranking models.
    """
    rows: List[Dict[str, object]] = []
    run_uid = result.run.run_uid

    for horizon, hres in sorted(result.horizons.items()):
        if hres.status != "SUCCESS":
            continue
        for split_name, metrics in hres.metrics.items():
            part = hres.predictions[hres.predictions[SPLIT_COL] == split_name]
            if part.empty:
                continue
            rows.append({
                "run_uid": run_uid,
                "region_code": result.region_code,
                "horizon_hours": int(horizon),
                "split_name": split_name,
                "evaluation_start": part[TARGET_TS_COL].min(),
                "evaluation_end": part[TARGET_TS_COL].max(),
                "n_observations": int(metrics.get("n_observations", len(part))),
                "mae": _f(metrics.get("mae")),
                "rmse": _f(metrics.get("rmse")),
                "wape": _f(metrics.get("wape")),
                "smape": _f(metrics.get("smape")),
                # Training cost is attributed to the run, so it is recorded once
                # per horizon (on the test row) rather than duplicated per split.
                "training_time_seconds": _f(hres.training_time_seconds) if split_name == "test" else None,
                "inference_time_ms": _f(hres.inference_time_ms) if split_name == "test" else None,
                # Extra diagnostics kept out of the strict contract columns.
                "bias": _f(metrics.get("bias")),
                "r2": _f(metrics.get("r2")),
                "baseline_mae": _f(metrics.get("baseline_mae")),
                "baseline_wape": _f(metrics.get("baseline_wape")),
                "skill_vs_naive": _f(metrics.get("skill_vs_naive")),
                "best_iteration": hres.best_iteration,
            })

    if not rows:
        return pd.DataFrame(columns=EVALUATION_CONTRACT_COLUMNS)

    frame = pd.DataFrame(rows)
    validate_evaluation_frame(frame)  # raises if the contract subset is incomplete
    return frame


def _f(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if pd.isna(out) else out


# ---------------------------------------------------------------------------
def to_anomaly_contract(result: RegionResult, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Run anomaly detection at the single configured horizon."""
    if not cfg.get("anomaly.enabled", True):
        return pd.DataFrame(columns=ANOMALY_CONTRACT_COLUMNS), {}

    horizon = int(cfg.get("anomaly.detection_horizon_hours", 1))
    hres = result.horizons.get(horizon)
    if hres is None or hres.status != "SUCCESS" or hres.predictions.empty:
        logger.warning(
            "Region %s: anomaly horizon %dh unavailable (status=%s) - skipping detection",
            result.region_code, horizon, getattr(hres, "status", "MISSING"),
        )
        return pd.DataFrame(columns=ANOMALY_CONTRACT_COLUMNS), {}

    frame, thresholds = detect_anomalies(
        hres.predictions, cfg, result.region_code, horizon, result.run.run_uid
    )
    return validate_anomaly_frame(frame), thresholds.to_dict() if thresholds else {}


# ---------------------------------------------------------------------------
def adapt_region(
    result: RegionResult, cfg: Config
) -> Dict[str, object]:
    """Single entry point mirroring what every sibling adapter must expose."""
    forecasts = to_forecast_contract(result, cfg)
    evaluations = to_evaluation_contract(result, cfg)
    anomalies, threshold_meta = to_anomaly_contract(result, cfg)

    payload: Dict[str, object] = {
        "adapter": ADAPTER_NAME,
        "model_run": result.run,
        "forecasts": forecasts,
        "evaluations": evaluations,
        "anomalies": anomalies,
        "anomaly_thresholds": threshold_meta,
    }
    logger.info(
        "Adapted %-9s -> run_uid=%s forecasts=%d evaluations=%d anomalies=%d(flagged=%d)",
        result.region_code, result.run.run_uid[:12], len(forecasts), len(evaluations),
        len(anomalies), int(anomalies["is_anomaly"].sum()) if not anomalies.empty else 0,
    )
    return payload


def describe_capabilities(cfg: Config) -> Dict[str, object]:
    """Section-7 model capability declaration. The integration layer records this
    so an unsupported horizon is visibly absent rather than silently fabricated."""
    return {
        "model_name": cfg["project.model_name"],
        "model_type": cfg["project.model_type"],
        "model_version": cfg["project.model_version"],
        "adapter": ADAPTER_NAME,
        "supported_horizons": cfg.horizons,
        "forecast_generation_method": "direct multi-horizon (one booster per horizon)",
        "input_window": {
            "max_lag_hours": max(cfg["features.lags"]),
            "max_rolling_window_hours": max(cfg["features.rolling_windows"]),
            "effective_history_required_hours": max(
                max(cfg["features.lags"]), max(cfg["features.rolling_windows"])
            ),
        },
        "input_file_pattern": "<REGION>_clean.csv",
        "input_columns": ["Load_Area", "Datetime_UTC", "Datetime_EPT", "Demand_MW", "Missing_Flag"],
        "output_format": "contracts.FORECAST_CONTRACT_COLUMNS",
        "region_handling": "one independent model_run per region_code",
        "exogenous_features": "none (no weather in source data)",
        "scaling": "none required (tree ensemble)",
        "metrics_produced": ["mae", "rmse", "wape", "smape"],
        "supports_future_forecast_without_actuals": True,
        "notes": (
            "actual_demand_mw is populated for backtest rows and NULL for genuine "
            "future forecasts until the observation arrives."
        ),
    }
