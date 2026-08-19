"""
Port of the EnerSight notebook's cells 27-34: Isolation Forest cross-check,
hybrid statistical+ML flag, and temporal event grouping.

WHY A SECOND, INDEPENDENT DETECTOR AT ALL
------------------------------------------
The statistical score (statistical_features.py) is a residual-magnitude
detector: it is blind to *when* a large residual happens, only *how* large it
is relative to its group's own history. Isolation Forest is given only
calendar-cyclical features (hour/day-of-week/month, as sin/cos pairs) - not
residual magnitude, not demand level - so it isolates points that sit in an
unusual REGION OF TIME (e.g. a residual that recurs at an unusual hour pattern
combination) independently of how big that residual is. Two detectors that
fail in different ways and agreeing is much stronger evidence than either
alone; that is the entire reasoning behind the hybrid flag.

Per the notebook: IF_Prediction (the raw -1/1 output) is NOT used directly as
the final flag. The continuous decision-function score is used instead so
severity can be graded rather than binary.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import Config
from .statistical_features import GROUP_COLS

logger = logging.getLogger(__name__)


def _cyclical_features(ts: pd.Series) -> pd.DataFrame:
    idx = pd.DatetimeIndex(ts)
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    month = idx.month.to_numpy()
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        "month_sin": np.sin(2 * np.pi * month / 12.0),
        "month_cos": np.cos(2 * np.pi * month / 12.0),
    }, index=ts.index)


def add_isolation_forest_scores(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Fit one Isolation Forest per (region, horizon) group - consistent with
    every other baseline in this module being group-scoped, and avoids one
    region's calendar rhythm dominating another's."""
    out = df.copy()
    n_estimators = int(cfg.get("isolation_forest.n_estimators", 300))
    contamination = float(cfg.get("isolation_forest.contamination", 0.01))
    random_state = int(cfg.get("isolation_forest.random_state", 42))

    feats = _cyclical_features(out["target_timestamp"])
    out = pd.concat([out, feats], axis=1)
    feature_cols = list(cfg.get(
        "isolation_forest.features",
        ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"],
    ))

    if_score = pd.Series(0.0, index=out.index)
    if_candidate = pd.Series(False, index=out.index)

    for key, idx in out.groupby(list(GROUP_COLS)).groups.items():
        sub = out.loc[idx, feature_cols]
        if len(sub) < max(20, n_estimators // 10):
            logger.warning("  Skipping Isolation Forest for group %s (only %d rows)", key, len(sub))
            continue
        model = IsolationForest(
            n_estimators=n_estimators, contamination=contamination,
            random_state=random_state, n_jobs=-1,
        )
        model.fit(sub)
        # decision_function: LOWER (more negative) = more anomalous. Flip sign
        # so higher IF_Anomaly_Score = more anomalous, matching the statistical
        # score's convention and making the two directly comparable downstream.
        raw = model.decision_function(sub)
        if_score.loc[idx] = -raw
        if_candidate.loc[idx] = model.predict(sub) == -1

    out["if_anomaly_score"] = if_score
    out["if_candidate_anomaly"] = if_candidate
    out = out.drop(columns=feature_cols)
    logger.info("Isolation Forest scored %d rows (%d candidates flagged, contamination=%.3f)",
                len(out), int(if_candidate.sum()), contamination)
    return out


def add_hybrid_flags(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    stat_threshold = float(cfg.get("statistical.statistical_threshold", 3.0))
    out["statistical_candidate_anomaly"] = out["statistical_anomaly_score"] >= stat_threshold

    out["strong_anomaly"] = out["statistical_candidate_anomaly"] & out["if_candidate_anomaly"]
    out["statistical_only_candidate"] = out["statistical_candidate_anomaly"] & ~out["if_candidate_anomaly"]
    out["if_only_candidate"] = out["if_candidate_anomaly"] & ~out["statistical_candidate_anomaly"]

    out["is_anomaly"] = out["strong_anomaly"] | out["statistical_only_candidate"] | out["if_only_candidate"]

    bands = dict(cfg.get("severity_bands", {}))
    out["severity"] = "NONE"
    out.loc[out["if_only_candidate"], "severity"] = "LOW"
    out.loc[out["statistical_only_candidate"], "severity"] = "MEDIUM"
    out.loc[out["strong_anomaly"], "severity"] = "HIGH"
    out.loc[out["strong_anomaly"] & (out["statistical_anomaly_score"] >= bands.get("critical", 7.0)),
            "severity"] = "CRITICAL"

    out["anomaly_direction"] = np.where(
        ~out["is_anomaly"], "NONE", np.where(out["residual_mw"] > 0, "OVER", "UNDER")
    )

    logger.info(
        "Hybrid flags: strong=%d statistical_only=%d if_only=%d total_flagged=%d / %d scored (%.2f%%)",
        int(out["strong_anomaly"].sum()), int(out["statistical_only_candidate"].sum()),
        int(out["if_only_candidate"].sum()), int(out["is_anomaly"].sum()), len(out),
        100.0 * out["is_anomaly"].sum() / max(len(out), 1),
    )
    return out


_SEVERITY_RANK = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}


def _highest_severity(severities: pd.Series) -> str:
    return max(severities, key=lambda s: _SEVERITY_RANK.get(s, 0))


def add_event_grouping(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Group temporally-consecutive flagged hours (per region+horizon) into one
    Event_Candidate_ID, mirroring the notebook's persistence analysis. A lone
    flagged hour is still a valid event of Observation_Count=1 - this is
    reporting/aggregation, not a second filter."""
    out = df.sort_values(list(GROUP_COLS) + ["target_timestamp"]).copy()
    max_gap = pd.Timedelta(hours=float(cfg.get("event_grouping.max_gap_hours", 1)))

    out["event_id"] = None
    event_counter = 0
    event_rows = []

    for key, grp in out[out["is_anomaly"]].groupby(list(GROUP_COLS)):
        grp = grp.sort_values("target_timestamp")
        gap = grp["target_timestamp"].diff()
        new_event = (gap.isna()) | (gap > max_gap)
        local_event_num = new_event.cumsum()
        for local_id, ev in grp.groupby(local_event_num):
            event_counter += 1
            eid = f"EVT-{event_counter:06d}"
            out.loc[ev.index, "event_id"] = eid
            event_rows.append({
                "event_id": eid,
                "region_code": key[0] if isinstance(key, tuple) else key,
                "horizon_hours": key[1] if isinstance(key, tuple) else None,
                "start_time": ev["target_timestamp"].min(),
                "end_time": ev["target_timestamp"].max(),
                "duration_hours": float((ev["target_timestamp"].max() - ev["target_timestamp"].min())
                                        .total_seconds() / 3600.0) + 1.0,
                "observation_count": int(len(ev)),
                "max_statistical_score": float(ev["statistical_anomaly_score"].max()),
                "mean_statistical_score": float(ev["statistical_anomaly_score"].mean()),
                "max_if_score": float(ev["if_anomaly_score"].max()),
                "max_severity": _highest_severity(ev["severity"]),
            })

    events = pd.DataFrame(event_rows)
    logger.info("Grouped anomalies into %d temporal events", len(events))
    return out, events
