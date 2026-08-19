"""
Load the cleaned regional demand series.

Input contract (confirmed against the actual files in clean_dataset/):
    Load_Area, Datetime_UTC, Datetime_EPT, Demand_MW, Missing_Flag

Datetime_UTC is the canonical instant and is parsed as tz-aware UTC. Datetime_EPT
is deliberately ignored for modelling - it is a local-time convenience column and
using it would silently reintroduce the timezone ambiguity the cleaning step
already resolved.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


class RegionDataError(RuntimeError):
    """Raised when a region's file is absent or structurally unusable."""


def region_file_path(cfg: Config, region_code: str) -> Path:
    return cfg.raw_data_dir / f"{region_code}_clean.csv"


def available_regions(cfg: Config) -> List[str]:
    found = []
    for region in cfg.all_regions:
        if region_file_path(cfg, region).exists():
            found.append(region)
    return found


def load_region(cfg: Config, region_code: str) -> pd.DataFrame:
    """Return a clean, gap-filled, hourly-indexed frame for one region.

    Output columns:
        timestamp_utc (tz-aware UTC, hourly, strictly increasing, no duplicates)
        region_code
        demand_mw     (float, may contain NaN where the source had gaps)
        missing_flag  (int 0/1, 1 where the value is absent or source-flagged)
        is_imputed    (bool, True for rows inserted by grid reindexing)
    """
    path = region_file_path(cfg, region_code)
    if not path.exists():
        raise RegionDataError(f"No cleaned file for region {region_code}: {path}")

    ts_col = cfg["data.timestamp_col"]
    tgt_col = cfg["data.target_col"]
    reg_col = cfg["data.region_col"]
    flag_col = cfg["data.missing_flag_col"]

    df = pd.read_csv(path, usecols=[reg_col, ts_col, tgt_col, flag_col])
    if df.empty:
        raise RegionDataError(f"Region {region_code} file is empty: {path}")

    df = df.rename(
        columns={
            reg_col: "region_code",
            ts_col: "timestamp_utc",
            tgt_col: "demand_mw",
            flag_col: "missing_flag",
        }
    )
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    if df["timestamp_utc"].isna().any():
        n_bad = int(df["timestamp_utc"].isna().sum())
        raise RegionDataError(f"Region {region_code}: {n_bad} unparseable timestamps")

    df["demand_mw"] = pd.to_numeric(df["demand_mw"], errors="coerce")
    df["missing_flag"] = pd.to_numeric(df["missing_flag"], errors="coerce").fillna(0).astype(int)
    df["region_code"] = region_code

    # Duplicate timestamps would corrupt every lag feature. The upstream cleaning
    # reported zero duplicates; we re-assert rather than trust.
    n_dupes = int(df["timestamp_utc"].duplicated().sum())
    if n_dupes:
        raise RegionDataError(
            f"Region {region_code}: {n_dupes} duplicate timestamps - refusing to build "
            "lag features on an ambiguous index"
        )

    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    if cfg.get("data.reindex_full_hourly_grid", True):
        df = _reindex_hourly(df, region_code, freq=cfg.get("data.freq", "h"))
    else:
        df["is_imputed"] = False

    df.loc[df["demand_mw"].isna(), "missing_flag"] = 1

    logger.info(
        "Loaded %-9s rows=%d  span=%s -> %s  gaps_filled=%d  nan_demand=%d",
        region_code,
        len(df),
        df["timestamp_utc"].min(),
        df["timestamp_utc"].max(),
        int(df["is_imputed"].sum()),
        int(df["demand_mw"].isna().sum()),
    )
    return df


def _reindex_hourly(df: pd.DataFrame, region_code: str, freq: str = "h") -> pd.DataFrame:
    """Place the series on a complete hourly grid.

    Why this matters: pandas shift/rolling operate positionally. If a region is
    missing hour 03:00, then shift(24) no longer means "24 clock hours ago". A
    complete grid makes every lag a true clock offset. Inserted rows keep
    demand_mw = NaN - LightGBM handles NaN natively, so we do NOT impute a value
    and pretend it was observed.
    """
    grid = pd.date_range(
        df["timestamp_utc"].min(), df["timestamp_utc"].max(), freq=freq, tz="UTC"
    )
    original = set(df["timestamp_utc"])
    out = (
        df.set_index("timestamp_utc")
        .reindex(grid)
        .rename_axis("timestamp_utc")
        .reset_index()
    )
    out["region_code"] = region_code
    out["missing_flag"] = out["missing_flag"].fillna(1).astype(int)
    out["is_imputed"] = ~out["timestamp_utc"].isin(original)
    n_added = int(out["is_imputed"].sum())
    if n_added:
        logger.debug("Region %s: inserted %d grid rows for missing hours", region_code, n_added)
    return out


def load_regions(cfg: Config, regions: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Load several regions, skipping (with a warning) any that fail."""
    targets = regions or cfg.default_regions
    loaded: Dict[str, pd.DataFrame] = {}
    for region in targets:
        try:
            loaded[region] = load_region(cfg, region)
        except RegionDataError as exc:
            logger.error("Skipping region %s: %s", region, exc)
    if not loaded:
        raise RegionDataError("No regions could be loaded - check paths.raw_data_dir")
    return loaded


def summarize_region(df: pd.DataFrame) -> Dict[str, object]:
    """Small profile used in the run manifest for traceability."""
    demand = df["demand_mw"]
    return {
        "region_code": df["region_code"].iloc[0],
        "n_rows": int(len(df)),
        "start_utc": df["timestamp_utc"].min().isoformat(),
        "end_utc": df["timestamp_utc"].max().isoformat(),
        "n_missing_demand": int(demand.isna().sum()),
        "n_grid_inserted": int(df["is_imputed"].sum()),
        "demand_min": float(demand.min()),
        "demand_max": float(demand.max()),
        "demand_mean": float(demand.mean()),
    }
