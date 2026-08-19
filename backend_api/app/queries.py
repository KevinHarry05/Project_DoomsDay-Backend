"""
Every function here is a Python-callable version of a SQL shape we already
hand-verified in psql during the demo prep: forecast lookup with the
common-origin fix, historical actuals, model comparison, and anomalies. The
routers and the /ask NLU layer both call these - there is exactly one
implementation of each query shape, never duplicated between "the API" and
"the natural-language layer".
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from .db import fetch_dicts

# The horizon set every forecasting track actually trains (see
# horizons.supported in each model's config.yaml - LightGBM and DHR-ARIMA
# were both deliberately built to match). Forecasts only exist at these
# exact horizons, not every integer hour, so any "give me the next N hours"
# request must be translated into "the supported horizons <= N", not a raw
# range(1, N+1) - the latter silently asks for hours that were never trained
# and always come back in missing_horizons.
SUPPORTED_HORIZONS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]


def horizons_up_to(hours: int) -> list:
    """Map a plain-English 'next N hours' into the actual trained horizon set."""
    return [h for h in SUPPORTED_HORIZONS if h <= hours]


def resolve_region_code(region: str) -> str:
    """Accept case-insensitive region input from an API caller or NLU layer;
    the DB's region_code column is upper-case by convention (AEP, COMED, ...)."""
    return region.strip().upper().replace(" ", "_")


# ---------------------------------------------------------------------------
def get_forecast(region: str, horizons: Sequence[int]) -> dict:
    """The 'next N hours' shape. Picks the latest origin common to every
    requested horizon (see the manual psql fix from demo prep: a naive
    MAX(forecast_timestamp) across all horizons is WRONG because each
    horizon's own last origin differs - longer horizons run out of usable
    origins earlier than short ones, since target = origin + horizon must
    stay inside the data)."""
    region_code = resolve_region_code(region)
    horizons = sorted(set(int(h) for h in horizons))
    if not horizons:
        raise ValueError("At least one horizon is required")

    sql = """
        WITH per_horizon_max AS (
            SELECT horizon_hours, MAX(forecast_timestamp) AS max_ts
            FROM v_selected_forecast
            WHERE region_code = %s AND horizon_hours = ANY(%s)
            GROUP BY horizon_hours
        ),
        common_origin AS (
            SELECT MIN(max_ts) AS forecast_timestamp FROM per_horizon_max
        )
        SELECT vsf.region_code, vsf.model_name, vsf.forecast_timestamp,
               vsf.horizon_hours, vsf.target_timestamp, vsf.predicted_demand_mw,
               vsf.actual_demand_mw
        FROM v_selected_forecast vsf, common_origin co
        WHERE vsf.region_code = %s
          AND vsf.horizon_hours = ANY(%s)
          AND vsf.forecast_timestamp = co.forecast_timestamp
        ORDER BY vsf.horizon_hours
    """
    rows = fetch_dicts(sql, (region_code, horizons, region_code, horizons))
    missing = sorted(set(horizons) - {r["horizon_hours"] for r in rows})
    return {
        "region_code": region_code,
        "requested_horizons": horizons,
        "forecast_timestamp": rows[0]["forecast_timestamp"] if rows else None,
        "points": rows,
        "missing_horizons": missing,   # honest about gaps rather than silently dropping them
    }


# ---------------------------------------------------------------------------
def get_historical(region: str, timestamp_utc: str, exact: bool = True) -> dict:
    """Actual demand lookup - 'what was demand at/around time T'. `exact=False`
    returns the single closest reading instead of requiring a grid-exact hit,
    for callers (or an NLU layer) that can't guarantee alignment to the hour."""
    region_code = resolve_region_code(region)
    if exact:
        sql = """
            SELECT r.region_code, d.timestamp_utc, d.demand_mw
            FROM demand_data d JOIN regions r ON r.region_id = d.region_id
            WHERE r.region_code = %s AND d.timestamp_utc = %s
        """
        rows = fetch_dicts(sql, (region_code, timestamp_utc))
        if rows:
            return {"region_code": region_code, "match": "exact", "reading": rows[0]}
    sql = """
        SELECT r.region_code, d.timestamp_utc, d.demand_mw
        FROM demand_data d JOIN regions r ON r.region_id = d.region_id
        WHERE r.region_code = %s
        ORDER BY ABS(EXTRACT(EPOCH FROM (d.timestamp_utc - %s::timestamptz)))
        LIMIT 1
    """
    rows = fetch_dicts(sql, (region_code, timestamp_utc))
    if not rows:
        return {"region_code": region_code, "match": "none", "reading": None}
    return {"region_code": region_code, "match": "nearest", "reading": rows[0]}


# ---------------------------------------------------------------------------
def get_model_comparison(region: Optional[str], horizon_hours: Optional[int]) -> List[dict]:
    """'Which model is winning' - straight read of v_model_ranking, the view
    that already enforces test-split-only, minimum-sample-size, RANK() OVER
    PARTITION BY (region, horizon). No ranking logic is re-implemented here."""
    where, params = [], []
    if region:
        where.append("region_code = %s")
        params.append(resolve_region_code(region))
    if horizon_hours is not None:
        where.append("horizon_hours = %s")
        params.append(int(horizon_hours))
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    sql = f"""
        SELECT region_code, horizon_hours, model_name, model_rank, wape, mae, rmse,
               smape, skill_vs_naive, models_compared
        FROM v_model_ranking
        {clause}
        ORDER BY region_code, horizon_hours, model_rank
    """
    return fetch_dicts(sql, tuple(params))


# ---------------------------------------------------------------------------
def get_anomalies(region: Optional[str], severity: Optional[str], limit: int = 50) -> List[dict]:
    where, params = ["a.is_anomaly = TRUE"], []
    if region:
        # anomalies table only has region_id - resolve via regions. Aliased
        # a.region_id explicitly: both `anomalies` (a) and `regions` (r) have
        # a region_id column, so the bare name is ambiguous to Postgres even
        # though only one of them is semantically correct here.
        where.append("a.region_id = (SELECT region_id FROM regions WHERE region_code = %s)")
        params.append(resolve_region_code(region))
    if severity:
        where.append("a.severity = %s")
        params.append(severity.upper())
    clause = " AND ".join(where)
    # Severity ranks detector AGREEMENT (how many methods fired), not raw
    # magnitude - a statistical-only MEDIUM can have a bigger anomaly_score
    # than an IF-agreeing HIGH. Sorting by score alone therefore surfaces a
    # page of MEDIUMs before any CRITICAL/HIGH ever appears. Sort by severity
    # rank first, score as the tiebreaker within a severity band.
    sql = f"""
        SELECT r.region_code, a.timestamp_utc, a.actual_demand_mw, a.predicted_demand_mw,
               a.anomaly_score, a.severity, a.anomaly_direction, a.reason, a.detection_method
        FROM anomalies a JOIN regions r ON r.region_id = a.region_id
        WHERE {clause}
        ORDER BY
            CASE a.severity
                WHEN 'CRITICAL' THEN 4 WHEN 'HIGH' THEN 3
                WHEN 'MEDIUM' THEN 2 WHEN 'LOW' THEN 1 ELSE 0
            END DESC,
            a.anomaly_score DESC
        LIMIT %s
    """
    params.append(limit)
    return fetch_dicts(sql, tuple(params))
