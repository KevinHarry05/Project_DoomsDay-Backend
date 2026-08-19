"""
Single connection-pool module every router shares. DATABASE_URL is the only
credential source (same rule enforced across all three model packages) - it
is never read from a file, never logged, never hardcoded.
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from typing import Iterator

import psycopg2
import psycopg2.extras
from psycopg2 import pool

_POOL: "psycopg2.pool.SimpleConnectionPool | None" = None


def init_pool(minconn: int = 1, maxconn: int = 10) -> None:
    global _POOL
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "DATABASE_URL is not set. Export it before starting the API, e.g.\n"
            '  $env:DATABASE_URL = "postgresql://postgres:<pw>@localhost:5432/energy_forecasting"'
        )
    _POOL = psycopg2.pool.SimpleConnectionPool(minconn, maxconn, dsn)


def close_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.closeall()
        _POOL = None


@contextmanager
def get_conn() -> Iterator["psycopg2.extensions.connection"]:
    if _POOL is None:
        raise RuntimeError("Connection pool not initialized - call init_pool() at startup")
    conn = _POOL.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _POOL.putconn(conn)


def _json_safe(value):
    """NaN / +Inf / -Inf are legal Postgres double precision values (an older
    LightGBM/DHR-ARIMA write, or a metric computed on a zero-sample split,
    can leave one in model_evaluations) but FastAPI's JSONResponse renders
    with allow_nan=False - so any of them reaching json.dumps() is a 500,
    not a slow query. Every row from every endpoint funnels through
    fetch_dicts(), so scrubbing here fixes it once for the whole API instead
    of trusting every writer to have scrubbed on the way in."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fetch_dicts(sql: str, params: tuple = ()) -> list[dict]:
    """Run a read-only query, return rows as plain dicts (JSON-serializable
    after datetime/Decimal handling is done by the caller / pydantic model)."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [
                {k: _json_safe(v) for k, v in dict(row).items()}
                for row in cur.fetchall()
            ]
