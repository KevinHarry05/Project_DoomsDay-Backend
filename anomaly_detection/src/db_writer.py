"""
Writes scored anomalies into the shared `anomalies` table (and derives
`alerts` from them), reusing the exact idempotency key the two forecasting
packages already established: (forecast_id, detection_method).

Unlike the forecasting packages' db_writer.upsert_anomalies (which has to
INNER JOIN back onto `forecasts` by timestamp to recover forecast_id, because
a single model's own pipeline never knew its rows' surrogate keys), this
writer already has forecast_id on every row - it came straight out of
v_selected_forecast in data_source.py. So the join step that exists in the
sibling packages is unnecessary here and is deliberately not repeated.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import pandas as pd

try:
    from psycopg2.extras import execute_values
except ImportError as exc:  # pragma: no cover
    raise ImportError("psycopg2-binary is required") from exc

logger = logging.getLogger(__name__)

ANOMALY_WRITE_COLUMNS = (
    "forecast_id", "region_id", "timestamp_utc", "actual_demand_mw",
    "predicted_demand_mw", "residual_mw", "anomaly_score", "severity",
    "is_anomaly", "detection_method", "deviation_percent", "anomaly_direction",
    "reason",
)


def inspect_schema(conn, schema: str = "public") -> Dict[str, Dict[str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns WHERE table_schema = %s
            ORDER BY table_name, ordinal_position
            """,
            (schema,),
        )
        out: Dict[str, Dict[str, str]] = {}
        for table, column, dtype in cur.fetchall():
            out.setdefault(table, {})[column] = dtype
    return out


def upsert_anomalies(conn, contract_df: pd.DataFrame, live: Dict[str, Dict[str, str]],
                      batch_size: int = 10000) -> int:
    if contract_df.empty:
        return 0
    cols = [c for c in ANOMALY_WRITE_COLUMNS if c in live.get("anomalies", {})]
    missing = set(ANOMALY_WRITE_COLUMNS) - set(cols) - {"region_id"}
    if missing:
        logger.warning("anomalies table missing recommended columns %s - those values dropped", missing)

    df = contract_df.copy()
    if "region_id" not in df.columns:
        raise RuntimeError("contract_df must carry region_id before writing (see run_anomaly_detection.py)")
    for col in cols:
        if col not in df.columns:
            df[col] = None
    df = df.where(pd.notna(df), None)

    records = [tuple(row) for row in df[cols].itertuples(index=False, name=None)]
    conflict = "(forecast_id, detection_method)" if "detection_method" in cols else "(forecast_id)"
    updatable = [c for c in cols if c not in ("forecast_id", "detection_method")]
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in updatable)

    sql = f"""
        INSERT INTO anomalies ({", ".join(cols)}) VALUES %s
        ON CONFLICT {conflict} DO UPDATE SET {set_clause}
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, records, page_size=batch_size)
    n_flagged = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
    logger.info("anomalies -> %d rows upserted (%d flagged)", len(records), n_flagged)
    return len(records)


def upsert_alerts(conn, live: Dict[str, Dict[str, str]],
                   region_ids: Sequence[int],
                   severities: Sequence[str] = ("HIGH", "CRITICAL")) -> int:
    if "anomalies" not in live or "alerts" not in live or not region_ids:
        return 0
    has_message = "message" in live.get("alerts", {})
    message_expr = (
        "a.severity || ' demand anomaly: actual ' || ROUND(a.actual_demand_mw::numeric, 0) "
        "|| ' MW vs forecast ' || ROUND(a.predicted_demand_mw::numeric, 0) || ' MW'"
    ) if has_message else "NULL"

    cols = ["anomaly_id", "region_id", "alert_type", "severity", "status"]
    selects = ["a.anomaly_id", "a.region_id", "'ENERGY_ANOMALY'", "a.severity", "'OPEN'"]
    if has_message:
        cols.insert(4, "message")
        selects.insert(4, message_expr)

    sql = f"""
        INSERT INTO alerts ({", ".join(cols)})
        SELECT {", ".join(selects)}
        FROM anomalies a
        WHERE a.region_id = ANY(%s)
          AND a.is_anomaly = TRUE AND a.severity = ANY(%s)
          AND NOT EXISTS (SELECT 1 FROM alerts x WHERE x.anomaly_id = a.anomaly_id)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (list(region_ids), list(severities)))
        inserted = cur.rowcount
    logger.info("alerts -> %d new rows", max(inserted, 0))
    return max(inserted, 0)


def resolve_region_ids_by_code(conn, region_codes: Sequence[str]) -> Dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT region_code, region_id FROM regions WHERE region_code = ANY(%s)",
                    (list(region_codes),))
        return {code: rid for code, rid in cur.fetchall()}
