"""
Reads the ONE input the anomaly stage is allowed to depend on: v_selected_forecast.

This is the operationalization of the EnerSight notebook's manual step. In the
notebook, a human trained a forecasting model in one Colab session, downloaded a
"final_forecast_outputs" zip, and uploaded it into a second Colab session to run
anomaly detection. There is no equivalent upload step here: v_selected_forecast
already resolves, per (region, horizon), whichever model - LightGBM, DHR_ARIMA,
or later PatchTST/TFT - currently has the lowest test-set WAPE (see
v_model_ranking / v_best_model in 10_model_comparison.sql). Querying that view
IS "give me the final forecast output", done automatically and kept current
every time a new model run changes the standings.

Because forecasts are only ever persisted for the `test` split (see
database.write_forecast_splits in both forecasting configs), every row this
module reads is already out-of-sample - exactly the population anomaly
detection should be scored against.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator, List, Optional, Sequence

import pandas as pd

try:
    import psycopg2
except ImportError as exc:  # pragma: no cover
    raise ImportError("psycopg2-binary is required for the anomaly detection stage") from exc

from .config import Config

logger = logging.getLogger(__name__)


@contextmanager
def connect(cfg: Config) -> Iterator["psycopg2.extensions.connection"]:
    import os

    env_var = cfg.get("database.env_var", "DATABASE_URL")
    dsn = os.environ.get(env_var)
    if not dsn:
        raise RuntimeError(f"{env_var} is not set - export the connection string first.")
    conn = psycopg2.connect(dsn)
    try:
        timeout = int(cfg.get("database.statement_timeout_ms", 300000))
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = %s", (timeout,))
        params = conn.get_dsn_parameters()
        logger.info("Connected to PostgreSQL %s@%s:%s/%s",
                    params.get("user"), params.get("host"), params.get("port"), params.get("dbname"))
        yield conn
    finally:
        conn.close()


def view_exists(conn, view_name: str = "v_selected_forecast") -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.views WHERE table_schema='public' AND table_name=%s",
            (view_name,),
        )
        return cur.fetchone() is not None


def fetch_selected_forecasts(
    conn,
    region_codes: Optional[Sequence[str]] = None,
    horizons: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Pull the current winning-model forecast series, ready to be scored.

    Columns returned: forecast_id, region_id, region_code, model_run_id,
    model_name, forecast_timestamp, target_timestamp, horizon_hours,
    predicted_demand_mw, actual_demand_mw, residual_mw.
    """
    if not view_exists(conn):
        raise RuntimeError(
            "v_selected_forecast does not exist. Apply sql/10_model_comparison.sql "
            "(from the lightgbm_forecasting package - it is model-agnostic) first."
        )

    where = ["f.actual_demand_mw IS NOT NULL"]
    params: List[object] = []
    if region_codes:
        where.append("b.region_code = ANY(%s)")
        params.append(list(region_codes))
    if horizons:
        where.append("f.horizon_hours = ANY(%s)")
        params.append([int(h) for h in horizons])

    sql = f"""
        SELECT
            f.forecast_id, f.region_id, b.region_code, f.model_run_id,
            b.best_model_name AS model_name,
            f.forecast_timestamp, f.target_timestamp, f.horizon_hours,
            f.predicted_demand_mw, f.actual_demand_mw,
            f.actual_demand_mw - f.predicted_demand_mw AS residual_mw
        FROM forecasts f
        JOIN v_best_model b
          ON b.best_model_run_id = f.model_run_id AND b.horizon_hours = f.horizon_hours
        WHERE {" AND ".join(where)}
        ORDER BY b.region_code, f.horizon_hours, f.target_timestamp
    """
    df = pd.read_sql(sql, conn, params=params)
    for col in ("forecast_timestamp", "target_timestamp"):
        df[col] = pd.to_datetime(df[col], utc=True)
    logger.info("Fetched %d selected-forecast rows spanning %d region(s), %d horizon(s)",
                len(df), df["region_code"].nunique(), df["horizon_hours"].nunique())
    return df
