"""
Standardized cross-model contracts.

These dataclasses are the ONLY structures the integration layer consumes.
Every forecasting track (DHR+ARIMA, LightGBM, PatchTST/TFT) must emit these
exact shapes, so no model-specific logic ever reaches the database layer.

Field names mirror the PostgreSQL column names one-to-one on purpose.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# Column order for the standardized forecast frame. The integration layer and
# the DB writer both rely on this exact ordering.
FORECAST_CONTRACT_COLUMNS: List[str] = [
    "model_name",
    "run_uid",
    "region_code",
    "forecast_timestamp",
    "target_timestamp",
    "horizon_hours",
    "predicted_demand_mw",
    "actual_demand_mw",
    "split_name",
]

EVALUATION_CONTRACT_COLUMNS: List[str] = [
    "run_uid",
    "region_code",
    "horizon_hours",
    "split_name",
    "evaluation_start",
    "evaluation_end",
    "n_observations",
    "mae",
    "rmse",
    "wape",
    "smape",
    "training_time_seconds",
    "inference_time_ms",
]

ANOMALY_CONTRACT_COLUMNS: List[str] = [
    "run_uid",
    "region_code",
    "timestamp_utc",
    "horizon_hours",
    "actual_demand_mw",
    "predicted_demand_mw",
    "residual_mw",
    "deviation_percent",
    "anomaly_score",
    "severity",
    "is_anomaly",
    "anomaly_direction",
    "reason",
    "detection_method",
]

VALID_SEVERITIES = ("NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL")
VALID_DIRECTIONS = ("OVER", "UNDER", "NONE")


@dataclass
class ModelRunContract:
    """One actual execution of one model for one region -> model_runs row.

    run_uid is a deterministic hash of everything that defines the run. Re-running
    identical code on identical data produces an identical run_uid, which is what
    makes the database write idempotent.
    """

    model_name: str
    model_type: str
    model_version: str
    region_code: str
    training_start: datetime
    training_end: datetime
    horizons: List[int]
    feature_version: str
    code_version: str
    status: str = "SUCCESS"          # SUCCESS | FAILED | PARTIAL
    n_features: Optional[int] = None
    n_training_rows: Optional[int] = None
    failure_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None

    @property
    def run_uid(self) -> str:
        payload = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "code_version": self.code_version,
            "feature_version": self.feature_version,
            "region_code": self.region_code,
            "training_start": _iso(self.training_start),
            "training_end": _iso(self.training_end),
            "horizons": sorted(int(h) for h in self.horizons),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["run_uid"] = self.run_uid
        out["training_start"] = _iso(self.training_start)
        out["training_end"] = _iso(self.training_end)
        out["created_at"] = _iso(self.created_at) if self.created_at else None
        return out


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def validate_forecast_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly if a model emits something the DB layer cannot accept."""
    missing = [c for c in FORECAST_CONTRACT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Forecast contract violation - missing columns: {missing}")

    if df.empty:
        return df[FORECAST_CONTRACT_COLUMNS].copy()

    out = df[FORECAST_CONTRACT_COLUMNS].copy()

    for col in ("forecast_timestamp", "target_timestamp"):
        ts = pd.to_datetime(out[col], utc=True, errors="coerce")
        if ts.isna().any():
            raise ValueError(f"Forecast contract violation - unparseable {col}")
        out[col] = ts

    if (out["horizon_hours"] <= 0).any():
        raise ValueError("Forecast contract violation - horizon_hours must be > 0")

    if out["predicted_demand_mw"].isna().any():
        raise ValueError("Forecast contract violation - predicted_demand_mw contains NULL")

    # horizon_hours must actually equal the origin->target clock distance.
    delta = (out["target_timestamp"] - out["forecast_timestamp"]).dt.total_seconds() / 3600.0
    if not (delta.round(6) == out["horizon_hours"].astype(float)).all():
        raise ValueError(
            "Forecast contract violation - horizon_hours does not match "
            "(target_timestamp - forecast_timestamp)"
        )

    key = ["run_uid", "region_code", "target_timestamp", "horizon_hours"]
    dupes = int(out.duplicated(subset=key).sum())
    if dupes:
        raise ValueError(
            f"Forecast contract violation - {dupes} duplicate rows on {key}; "
            "this would break the forecasts uniqueness constraint"
        )

    out["horizon_hours"] = out["horizon_hours"].astype(int)
    return out


def validate_evaluation_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in EVALUATION_CONTRACT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Evaluation contract violation - missing columns: {missing}")
    return df[EVALUATION_CONTRACT_COLUMNS].copy()


def validate_anomaly_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ANOMALY_CONTRACT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Anomaly contract violation - missing columns: {missing}")
    out = df[ANOMALY_CONTRACT_COLUMNS].copy()
    if out.empty:
        return out
    bad_sev = set(out["severity"].unique()) - set(VALID_SEVERITIES)
    if bad_sev:
        raise ValueError(f"Anomaly contract violation - invalid severity values: {bad_sev}")
    bad_dir = set(out["anomaly_direction"].unique()) - set(VALID_DIRECTIONS)
    if bad_dir:
        raise ValueError(f"Anomaly contract violation - invalid direction values: {bad_dir}")
    return out
