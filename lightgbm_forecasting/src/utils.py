"""Shared helpers: timing, JSON-safe serialization, output directory layout."""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Timer:
    """Context manager capturing wall-clock duration in seconds and milliseconds."""

    def __init__(self) -> None:
        self.seconds: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.seconds = time.perf_counter() - self._start

    @property
    def milliseconds(self) -> float:
        return self.seconds * 1000.0


@contextmanager
def log_stage(name: str) -> Iterator[None]:
    logger.info("START  %s", name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info("DONE   %s (%.2fs)", name, time.perf_counter() - t0)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_output_dirs(root: Path) -> Dict[str, Path]:
    """Create the standard output tree and return the resolved sub-paths."""
    layout = {
        "root": root,
        "models": root / "models",
        "forecasts": root / "forecasts",
        "evaluations": root / "evaluations",
        "anomalies": root / "anomalies",
        "runs": root / "runs",
        "reports": root / "reports",
        "logs": root / "logs",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


class _JSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, cls=_JSONEncoder)
    logger.debug("Wrote %s", path)


def write_table(df: pd.DataFrame, path_no_ext: Path, also_csv: bool = True) -> Dict[str, str]:
    """Persist a frame as Parquet (canonical) plus optional CSV (human review)."""
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    parquet_path = path_no_ext.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        written["parquet"] = str(parquet_path)
    except Exception as exc:  # pragma: no cover - pyarrow/fastparquet absent
        logger.warning("Parquet write failed (%s); CSV only", exc)
    if also_csv:
        csv_path = path_no_ext.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        written["csv"] = str(csv_path)
    return written


def to_utc_series(values: Any) -> pd.Series:
    return pd.to_datetime(values, utc=True)
