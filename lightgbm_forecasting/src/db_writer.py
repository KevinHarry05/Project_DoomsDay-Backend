"""
PostgreSQL integration layer.

PRINCIPLES ENFORCED HERE
------------------------
1. Credentials come only from the DATABASE_URL environment variable. Nothing is
   read from a file and nothing is logged.
2. No column is invented. The writer introspects information_schema at startup,
   writes only columns that actually exist, and prints exactly which recommended
   columns are missing along with the migration that adds them.
3. Write order follows the foreign keys:
       regions -> model_runs -> forecasts -> model_evaluations -> anomalies -> alerts
   A child row is never attempted before its parent exists.
4. Every write is idempotent. Re-running the same model output updates in place
   instead of appending duplicates, keyed on the natural business keys:
       model_runs        (run_uid)
       forecasts         (model_run_id, region_id, target_timestamp, horizon_hours)
       model_evaluations (model_run_id, horizon_hours, split_name)
       anomalies         (forecast_id, detection_method)
5. regions is read, never written. The 12 rows are reference data that already
   exist; inserting from here is how you end up with duplicate region codes.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError as exc:  # pragma: no cover
    raise ImportError("psycopg2-binary is required for database integration") from exc

from .config import Config
from .contracts import ModelRunContract

logger = logging.getLogger(__name__)

CORE_TABLES = (
    "regions", "demand_data", "model_runs", "forecasts",
    "model_evaluations", "anomalies", "alerts",
)

# Columns this writer will populate when present. Anything absent is skipped and
# reported - never silently created, never silently dropped without a warning.
DESIRED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "model_runs": (
        "model_name", "model_type", "region_id", "training_start", "training_end",
        "model_version", "run_uid", "status", "n_features", "n_training_rows",
        "failure_reason", "metadata",
    ),
    "forecasts": (
        "model_run_id", "region_id", "forecast_timestamp", "target_timestamp",
        "horizon_hours", "predicted_demand_mw", "actual_demand_mw",
    ),
    "model_evaluations": (
        "model_run_id", "evaluation_start", "evaluation_end", "horizon_hours",
        "mae", "rmse", "wape", "smape", "training_time_seconds", "inference_time_ms",
        "split_name", "n_observations", "bias", "r2", "skill_vs_naive",
    ),
    "anomalies": (
        "forecast_id", "region_id", "timestamp_utc", "actual_demand_mw",
        "predicted_demand_mw", "residual_mw", "anomaly_score", "severity",
        "is_anomaly", "detection_method", "deviation_percent", "anomaly_direction",
        "reason",
    ),
    "alerts": (
        "anomaly_id", "region_id", "alert_type", "severity", "message", "status",
    ),
}

# Columns that must exist or the write cannot proceed at all.
REQUIRED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "regions": ("region_id", "region_code"),
    "model_runs": ("model_run_id", "model_name", "model_type", "region_id"),
    "forecasts": (
        "forecast_id", "model_run_id", "region_id", "forecast_timestamp",
        "target_timestamp", "horizon_hours", "predicted_demand_mw",
    ),
    "model_evaluations": ("model_run_id", "horizon_hours"),
    "anomalies": ("forecast_id", "region_id", "timestamp_utc"),
    "alerts": ("anomaly_id", "region_id"),
}


class SchemaMismatch(RuntimeError):
    """Raised when the live schema cannot accept a write."""


# ---------------------------------------------------------------------------
@contextmanager
def connect(cfg: Config) -> Iterator["psycopg2.extensions.connection"]:
    env_var = cfg.get("database.env_var", "DATABASE_URL")
    dsn = os.environ.get(env_var)
    if not dsn:
        raise RuntimeError(
            f"{env_var} is not set. Export the connection string in the shell that runs "
            "this job; never place it in config.yaml or in source control."
        )
    conn = psycopg2.connect(dsn)
    try:
        timeout = int(cfg.get("database.statement_timeout_ms", 300000))
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = %s", (timeout,))
        # Log the destination host without ever echoing the password.
        params = conn.get_dsn_parameters()
        logger.info(
            "Connected to PostgreSQL %s@%s:%s/%s",
            params.get("user"), params.get("host"), params.get("port"), params.get("dbname"),
        )
        yield conn
    finally:
        conn.close()


def inspect_schema(conn, schema: str = "public") -> Dict[str, Dict[str, str]]:
    query = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = %s
        ORDER BY table_name, ordinal_position
    """
    out: Dict[str, Dict[str, str]] = {}
    with conn.cursor() as cur:
        cur.execute(query, (schema,))
        for table, column, dtype in cur.fetchall():
            out.setdefault(table, {})[column] = dtype
    return out


def preflight(conn, cfg: Config) -> Dict[str, Any]:
    """Verify the live schema before writing anything. Returns a report."""
    schema_name = cfg.get("database.schema", "public")
    live = inspect_schema(conn, schema_name)

    missing_tables = [t for t in CORE_TABLES if t not in live]
    if missing_tables:
        raise SchemaMismatch(
            f"Missing tables in schema '{schema_name}': {missing_tables}. "
            "Run sql/00_recommended_migrations.sql (or the base DDL) first."
        )

    hard_failures: List[str] = []
    for table, required in REQUIRED_COLUMNS.items():
        absent = [c for c in required if c not in live.get(table, {})]
        if absent:
            hard_failures.append(f"{table}: missing required columns {absent}")
    if hard_failures:
        raise SchemaMismatch("; ".join(hard_failures))

    optional_missing: Dict[str, List[str]] = {}
    for table, desired in DESIRED_COLUMNS.items():
        absent = [c for c in desired if c not in live.get(table, {})]
        if absent:
            optional_missing[table] = absent

    if optional_missing:
        logger.warning(
            "Recommended columns absent - these values will NOT be persisted: %s. "
            "Apply sql/00_recommended_migrations.sql to capture them.",
            optional_missing,
        )
        if "run_uid" in optional_missing.get("model_runs", []):
            logger.warning(
                "model_runs.run_uid is absent: idempotency will fall back to the "
                "natural key (model_name, model_version, region_id, training_start, "
                "training_end). Adding run_uid is strongly recommended."
            )
    else:
        logger.info("Preflight: schema accepts every recommended column.")

    return {"schema": schema_name, "live_tables": sorted(live), "optional_missing": optional_missing}


# ---------------------------------------------------------------------------
def resolve_region_ids(conn, region_codes: Sequence[str]) -> Dict[str, int]:
    """Read-only lookup. regions is reference data, so a missing code is an
    error to surface, not a row to invent."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT region_code, region_id FROM regions WHERE region_code = ANY(%s)",
            (list(region_codes),),
        )
        mapping = {code: rid for code, rid in cur.fetchall()}
    missing = sorted(set(region_codes) - set(mapping))
    if missing:
        raise SchemaMismatch(
            f"region_code(s) {missing} not present in regions. "
            "Insert the reference rows before running integration."
        )
    return mapping


def get_latest_training_end(
    conn, model_name: str, region_id: int
) -> Optional["pd.Timestamp"]:
    """Most recent successful/partial training_end for this model+region.

    This is what scheduled_retrain.py compares against the newest local data
    point to decide whether a region actually needs retraining, rather than
    blindly retraining everything on every scheduled fire.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT MAX(training_end) FROM model_runs
            WHERE model_name = %s AND region_id = %s
              AND status IN ('SUCCESS', 'PARTIAL')
            """,
            (model_name, region_id),
        )
        row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return pd.Timestamp(row[0]).tz_convert("UTC") if pd.Timestamp(row[0]).tzinfo \
        else pd.Timestamp(row[0]).tz_localize("UTC")


def get_latest_data_available_through(
    conn, model_name: str, region_id: int
) -> Optional["pd.Timestamp"]:
    """Full data extent (train+val+test) the most recent run for this
    model+region actually saw, read from model_runs.metadata.

    This is the correct comparison point for "has new data arrived", unlike
    training_end (which is only the train-split cutoff and always sits ~30%
    behind the data a run saw, since val/test are held out by design).
    Falls back to training_end if metadata/the field is absent (e.g. schema
    predates this column, or a run was written before this field existed).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT metadata ->> 'data_available_through', training_end
            FROM model_runs
            WHERE model_name = %s AND region_id = %s
              AND status IN ('SUCCESS', 'PARTIAL')
            ORDER BY COALESCE(
                (metadata ->> 'data_available_through')::timestamptz,
                training_end
            ) DESC NULLS LAST
            LIMIT 1
            """,
            (model_name, region_id),
        )
        row = cur.fetchone()
    if not row:
        return None
    raw = row[0] or row[1]
    if raw is None:
        return None
    ts = pd.Timestamp(raw)
    return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")


def _present(live: Dict[str, Dict[str, str]], table: str, columns: Iterable[str]) -> List[str]:
    available = live.get(table, {})
    return [c for c in columns if c in available]


# ---------------------------------------------------------------------------
def upsert_model_run(
    conn, run: ModelRunContract, region_id: int, live: Dict[str, Dict[str, str]]
) -> int:
    """Insert or update the parent model_runs row and return its model_run_id."""
    import json

    values: Dict[str, Any] = {
        "model_name": run.model_name,
        "model_type": run.model_type,
        "region_id": region_id,
        "training_start": run.training_start,
        "training_end": run.training_end,
        "model_version": run.model_version,
        "run_uid": run.run_uid,
        "status": run.status,
        "n_features": run.n_features,
        "n_training_rows": run.n_training_rows,
        "failure_reason": run.failure_reason,
        "metadata": json.dumps(run.metadata),
    }
    cols = _present(live, "model_runs", DESIRED_COLUMNS["model_runs"])
    payload = [values[c] for c in cols]
    placeholders = ", ".join(["%s"] * len(cols))
    col_list = ", ".join(cols)

    has_run_uid = "run_uid" in cols
    with conn.cursor() as cur:
        if has_run_uid:
            updatable = [c for c in cols if c != "run_uid"]
            set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in updatable)
            cur.execute(
                f"""
                INSERT INTO model_runs ({col_list}) VALUES ({placeholders})
                ON CONFLICT (run_uid) DO UPDATE SET {set_clause}
                RETURNING model_run_id
                """,
                payload,
            )
            model_run_id = cur.fetchone()[0]
        else:
            # Natural-key fallback keeps the run idempotent without run_uid.
            cur.execute(
                """
                SELECT model_run_id FROM model_runs
                WHERE model_name = %s AND region_id = %s
                  AND training_start = %s AND training_end = %s
                ORDER BY model_run_id DESC LIMIT 1
                """,
                (run.model_name, region_id, run.training_start, run.training_end),
            )
            row = cur.fetchone()
            if row:
                model_run_id = row[0]
                logger.info("  Reusing existing model_run_id=%s (natural key match)", model_run_id)
            else:
                cur.execute(
                    f"INSERT INTO model_runs ({col_list}) VALUES ({placeholders}) "
                    "RETURNING model_run_id",
                    payload,
                )
                model_run_id = cur.fetchone()[0]

    logger.info("  model_runs -> model_run_id=%s (%s, region_id=%s, status=%s)",
                model_run_id, run.model_name, region_id, run.status)
    return int(model_run_id)


def upsert_forecasts(
    conn, forecasts: pd.DataFrame, model_run_id: int, region_id: int,
    live: Dict[str, Dict[str, str]], batch_size: int = 10000,
) -> int:
    if forecasts.empty:
        return 0
    cols = _present(live, "forecasts", DESIRED_COLUMNS["forecasts"])
    df = forecasts.copy()
    df["model_run_id"] = model_run_id
    df["region_id"] = region_id
    df = df.where(pd.notna(df), None)

    records = [tuple(row) for row in df[cols].itertuples(index=False, name=None)]
    updatable = [c for c in cols if c not in
                 ("model_run_id", "region_id", "target_timestamp", "horizon_hours")]
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in updatable)

    sql = f"""
        INSERT INTO forecasts ({", ".join(cols)}) VALUES %s
        ON CONFLICT (model_run_id, region_id, target_timestamp, horizon_hours)
        DO UPDATE SET {set_clause}
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, records, page_size=batch_size)
    logger.info("  forecasts -> %d rows upserted", len(records))
    return len(records)


def fetch_forecast_ids(
    conn, model_run_id: int, region_id: int, horizon_hours: int
) -> pd.DataFrame:
    """Retrieve surrogate forecast_ids so anomalies can reference their parent."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT forecast_id, target_timestamp
            FROM forecasts
            WHERE model_run_id = %s AND region_id = %s AND horizon_hours = %s
            """,
            (model_run_id, region_id, horizon_hours),
        )
        rows = cur.fetchall()
    out = pd.DataFrame(rows, columns=["forecast_id", "target_timestamp"])
    if not out.empty:
        out["target_timestamp"] = pd.to_datetime(out["target_timestamp"], utc=True)
    return out


def upsert_evaluations(
    conn, evaluations: pd.DataFrame, model_run_id: int, live: Dict[str, Dict[str, str]]
) -> int:
    if evaluations.empty:
        return 0
    cols = _present(live, "model_evaluations", DESIRED_COLUMNS["model_evaluations"])
    df = evaluations.copy()
    df["model_run_id"] = model_run_id
    for col in cols:
        if col not in df.columns:
            df[col] = None
    df = df.where(pd.notna(df), None)

    records = [tuple(row) for row in df[cols].itertuples(index=False, name=None)]
    has_split = "split_name" in cols
    conflict_cols = "(model_run_id, horizon_hours, split_name)" if has_split \
        else "(model_run_id, horizon_hours)"
    updatable = [c for c in cols if c not in ("model_run_id", "horizon_hours", "split_name")]
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in updatable)

    sql = f"""
        INSERT INTO model_evaluations ({", ".join(cols)}) VALUES %s
        ON CONFLICT {conflict_cols} DO UPDATE SET {set_clause}
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, records)
    logger.info("  model_evaluations -> %d rows upserted", len(records))
    return len(records)


def upsert_anomalies(
    conn, anomalies: pd.DataFrame, model_run_id: int, region_id: int,
    live: Dict[str, Dict[str, str]], batch_size: int = 10000,
) -> int:
    """Join anomalies onto their forecast_id, then upsert. Rows whose parent
    forecast was not persisted are dropped with a warning rather than orphaned."""
    if anomalies.empty:
        return 0
    horizon = int(anomalies["horizon_hours"].iloc[0])
    parents = fetch_forecast_ids(conn, model_run_id, region_id, horizon)
    if parents.empty:
        logger.warning("  No parent forecasts found for h=%dh - skipping anomalies", horizon)
        return 0

    df = anomalies.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    before = len(df)
    df = df.merge(parents, left_on="timestamp_utc", right_on="target_timestamp", how="inner")
    if len(df) < before:
        logger.warning("  Dropped %d anomaly rows with no persisted parent forecast",
                       before - len(df))
    if df.empty:
        return 0

    df["region_id"] = region_id
    cols = _present(live, "anomalies", DESIRED_COLUMNS["anomalies"])
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
    logger.info("  anomalies -> %d rows upserted (%d flagged)",
                len(records), int(df["is_anomaly"].sum()) if "is_anomaly" in df else 0)
    return len(records)


def upsert_alerts(
    conn, model_run_id: int, region_id: int, live: Dict[str, Dict[str, str]],
    severities: Sequence[str] = ("HIGH", "CRITICAL"),
) -> int:
    """Generate alerts from the anomalies just written.

    Done in SQL rather than Python so the alert set is always derived from what
    the database actually holds, and so a re-run cannot duplicate alerts.
    """
    if "anomalies" not in live or "alerts" not in live:
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
        JOIN forecasts f ON f.forecast_id = a.forecast_id
        WHERE f.model_run_id = %s AND a.region_id = %s
          AND a.is_anomaly = TRUE AND a.severity = ANY(%s)
          AND NOT EXISTS (SELECT 1 FROM alerts x WHERE x.anomaly_id = a.anomaly_id)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (model_run_id, region_id, list(severities)))
        inserted = cur.rowcount
    logger.info("  alerts -> %d new rows", max(inserted, 0))
    return max(inserted, 0)


# ---------------------------------------------------------------------------
def write_region_payload(conn, payload: Dict[str, Any], cfg: Config,
                         live: Dict[str, Dict[str, str]]) -> Dict[str, int]:
    """Write one region's adapted output in strict foreign-key order."""
    run: ModelRunContract = payload["model_run"]
    region_id = resolve_region_ids(conn, [run.region_code])[run.region_code]
    batch = int(cfg.get("database.batch_size", 10000))

    model_run_id = upsert_model_run(conn, run, region_id, live)
    n_fc = upsert_forecasts(conn, payload["forecasts"], model_run_id, region_id, live, batch)
    n_ev = upsert_evaluations(conn, payload["evaluations"], model_run_id, live)
    n_an = upsert_anomalies(conn, payload["anomalies"], model_run_id, region_id, live, batch)
    n_al = upsert_alerts(conn, model_run_id, region_id, live)

    return {
        "model_run_id": model_run_id,
        "region_id": region_id,
        "forecasts": n_fc,
        "evaluations": n_ev,
        "anomalies": n_an,
        "alerts": n_al,
    }
