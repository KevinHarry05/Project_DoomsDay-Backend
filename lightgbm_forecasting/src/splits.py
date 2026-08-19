"""
Chronological train / validation / test splitting.

Random or K-fold splitting is invalid here: it lets the model train on hours
that sit between test hours, which in a series this autocorrelated is close to
handing it the answer. Everything below cuts strictly on time.

EMBARGO
-------
A row's label lives h hours after its origin. Without a buffer, the last train
rows carry labels drawn from the first validation hours. We therefore drop
`embargo_hours` (>= max horizon) of origins at each boundary. The cost is a day
of data; the benefit is that reported validation and test error are honest.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from .config import Config
from .feature_engineering import ORIGIN_TS_COL

logger = logging.getLogger(__name__)

SPLIT_COL = "split_name"
SPLIT_ORDER = ("train", "val", "test")


@dataclass
class SplitBoundaries:
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    embargo_hours: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "train_end_utc": self.train_end.isoformat(),
            "val_end_utc": self.val_end.isoformat(),
            "embargo_hours": self.embargo_hours,
        }


def _as_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def compute_boundaries(frame: pd.DataFrame, cfg: Config) -> SplitBoundaries:
    origins = pd.DatetimeIndex(frame[ORIGIN_TS_COL]).sort_values()
    if len(origins) < 100:
        raise ValueError(f"Too few rows ({len(origins)}) to split chronologically")

    embargo = int(cfg.get("split.embargo_hours", 24))
    method = str(cfg.get("split.method", "fraction")).lower()

    if method == "date":
        train_end = _as_utc(cfg["split.train_end"])
        val_end = _as_utc(cfg["split.val_end"])
        # A fixed calendar date can fall outside a short region's span (NI and
        # PJM_Load both end years before the others), so fall back gracefully.
        if not (origins.min() < train_end < val_end < origins.max()):
            logger.warning(
                "Configured split dates fall outside this region's span (%s -> %s); "
                "falling back to fraction split",
                origins.min(), origins.max(),
            )
            return _fraction_boundaries(origins, cfg, embargo)
        return SplitBoundaries(train_end, val_end, embargo)

    return _fraction_boundaries(origins, cfg, embargo)


def _fraction_boundaries(origins: pd.DatetimeIndex, cfg: Config, embargo: int) -> SplitBoundaries:
    train_frac = float(cfg.get("split.train_frac", 0.70))
    val_frac = float(cfg.get("split.val_frac", 0.15))
    n = len(origins)
    train_idx = max(1, int(n * train_frac) - 1)
    val_idx = max(train_idx + 1, int(n * (train_frac + val_frac)) - 1)
    val_idx = min(val_idx, n - 2)
    return SplitBoundaries(origins[train_idx], origins[val_idx], embargo)


def assign_splits(
    frame: pd.DataFrame, cfg: Config, boundaries: Optional[SplitBoundaries] = None
) -> pd.DataFrame:
    """Attach a `split_name` column. Embargoed rows are labelled `embargo` and
    are excluded from training, evaluation and persistence alike."""
    bounds = boundaries or compute_boundaries(frame, cfg)
    out = frame.copy()
    origin = pd.DatetimeIndex(out[ORIGIN_TS_COL])
    gap = pd.Timedelta(hours=bounds.embargo_hours)

    out[SPLIT_COL] = "embargo"
    out.loc[origin <= bounds.train_end - gap, SPLIT_COL] = "train"
    out.loc[(origin > bounds.train_end) & (origin <= bounds.val_end - gap), SPLIT_COL] = "val"
    out.loc[origin > bounds.val_end, SPLIT_COL] = "test"

    counts = out[SPLIT_COL].value_counts().to_dict()
    logger.info(
        "Split h=%s -> train=%d val=%d test=%d embargo=%d | train_end=%s val_end=%s",
        out["horizon_hours"].iloc[0] if "horizon_hours" in out else "?",
        counts.get("train", 0), counts.get("val", 0),
        counts.get("test", 0), counts.get("embargo", 0),
        bounds.train_end, bounds.val_end,
    )
    for name in SPLIT_ORDER:
        if counts.get(name, 0) == 0:
            raise ValueError(f"Split '{name}' is empty - adjust split config for this region")
    return out


def split_frames(frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        name: frame[frame[SPLIT_COL] == name].reset_index(drop=True)
        for name in SPLIT_ORDER
    }


def assert_chronological_integrity(frames: Dict[str, pd.DataFrame]) -> None:
    """Hard guarantee that no split overlaps or precedes its predecessor."""
    prev_name = None
    prev_max = None
    for name in SPLIT_ORDER:
        part = frames.get(name)
        if part is None or part.empty:
            continue
        lo = part[ORIGIN_TS_COL].min()
        hi = part[ORIGIN_TS_COL].max()
        if prev_max is not None and lo <= prev_max:
            raise AssertionError(
                f"Chronological violation: {name} starts {lo} but {prev_name} ends {prev_max}"
            )
        prev_name, prev_max = name, hi
