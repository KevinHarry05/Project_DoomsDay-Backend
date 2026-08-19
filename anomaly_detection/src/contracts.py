"""Same ANOMALY_CONTRACT_COLUMNS shape used by both forecasting tracks, copied
so this package has no import dependency on either sibling package."""
from __future__ import annotations

from typing import List

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


def validate_anomaly_frame(df):
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
