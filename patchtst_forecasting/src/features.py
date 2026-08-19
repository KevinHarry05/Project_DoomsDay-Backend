"""
Sequence construction for PatchTST.

Builds fixed-length lookback windows (multivariate: scaled demand + cyclical
calendar features) and the multi-horizon targets that go with each window
("origin"). Mirrors the DHR+ARIMA / LightGBM tracks' data contract (Load_Area,
Datetime_UTC, Demand_MW, Missing_Flag) - this module is the only place that
knows how a raw region frame becomes model-ready tensors.

Design choices, and why:
  - Per-region z-score scaling, fit on the TRAIN portion only. Demand scale
    varies wildly across regions (DAYTON ~3.7GW peak vs PJME ~62GW); a shared
    global model needs every region on a comparable numeric scale to learn
    cross-region patterns at all, and fitting the scaler only on train avoids
    leaking val/test statistics into the model.
  - Origins spaced `origin_cadence_hours` apart (default 24h) rather than
    every hour - this is what keeps the sequence count small enough to train
    on a 2-thread CPU sandbox; see config.yaml for the full rationale.
  - Chronological train/val/test split by origin position (not random) -
    consistent with the rest of the project and required for time series
    (a randomly-shuffled split would let the model "see the future").
  - `max_origins_per_region`: if the cadence-spaced origin count still
    exceeds this cap, evenly-spaced indices are kept across the FULL time
    range (not just the head) so the subsample still spans the whole series
    before the chronological split is applied.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = out["timestamp_utc"]
    hour = ts.dt.hour.to_numpy()
    dow = ts.dt.dayofweek.to_numpy()
    month = ts.dt.month.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    out["is_weekend"] = (dow >= 5).astype(float)
    return out


@dataclass
class RegionScaler:
    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


@dataclass
class Origin:
    region_code: str
    origin_idx: int
    forecast_timestamp: pd.Timestamp
    split_name: str


@dataclass
class RegionSequenceSet:
    region_code: str
    scaler: RegionScaler
    feature_matrix: np.ndarray          # [n_rows, n_features], NaN-filled where demand missing
    demand_raw: np.ndarray              # [n_rows] actual MW (NaN where missing), for target lookup
    timestamps: pd.Series                # [n_rows]
    origins: List[Origin]


def _fit_scaler(demand_raw: np.ndarray, train_end_idx: int) -> RegionScaler:
    train_vals = demand_raw[:train_end_idx]
    train_vals = train_vals[np.isfinite(train_vals)]
    if train_vals.size == 0:
        raise ValueError("No finite training-window demand values to fit scaler on")
    mean = float(train_vals.mean())
    std = float(train_vals.std())
    if std < 1e-6:
        std = 1.0
    return RegionScaler(mean=mean, std=std)


def build_region_sequences(
    df: pd.DataFrame, cfg: Config, horizons: List[int]
) -> Optional[RegionSequenceSet]:
    region_code = df["region_code"].iloc[0]
    feat_df = add_calendar_features(df)
    feature_cols = cfg.get("sequence.feature_columns")
    lookback = int(cfg.get("sequence.lookback_hours", 168))
    cadence = int(cfg.get("sequence.origin_cadence_hours", 24))
    max_origins = int(cfg.get("sequence.max_origins_per_region", 700))
    max_h = max(horizons)

    n = len(feat_df)
    first_valid = lookback
    last_valid = n - max_h - 1
    if last_valid <= first_valid:
        logger.warning("Region %s: not enough rows (%d) for lookback=%d + max_horizon=%d",
                        region_code, n, lookback, max_h)
        return None

    demand_raw = feat_df["demand_mw"].to_numpy(dtype=float)

    # Fit the scaler on demand up to the chronological 70% point of the USABLE
    # origin range, matching the train/val/test fractions used downstream.
    train_frac = 1.0 - float(cfg.get("training.val_frac", 0.15)) - float(cfg.get("training.test_frac", 0.15))
    approx_train_end = first_valid + int((last_valid - first_valid) * train_frac)
    scaler = _fit_scaler(demand_raw, approx_train_end)

    demand_scaled = np.where(np.isfinite(demand_raw), scaler.transform(demand_raw), np.nan)
    feat_df = feat_df.assign(demand_scaled=demand_scaled)

    feature_matrix = feat_df[feature_cols].to_numpy(dtype=float)

    candidate_idx = np.arange(first_valid, last_valid + 1, cadence)
    if candidate_idx.size > max_origins:
        keep = np.linspace(0, candidate_idx.size - 1, max_origins).round().astype(int)
        candidate_idx = candidate_idx[np.unique(keep)]
        logger.info("Region %s: subsampled %d -> %d origins (max_origins_per_region cap)",
                    region_code, np.arange(first_valid, last_valid + 1, cadence).size, candidate_idx.size)

    val_frac = float(cfg.get("training.val_frac", 0.15))
    test_frac = float(cfg.get("training.test_frac", 0.15))
    n_c = candidate_idx.size
    n_train = int(n_c * (1.0 - val_frac - test_frac))
    n_val = int(n_c * val_frac)

    origins: List[Origin] = []
    for pos, idx in enumerate(candidate_idx):
        split_name = "train" if pos < n_train else ("val" if pos < n_train + n_val else "test")
        origins.append(Origin(
            region_code=region_code,
            origin_idx=int(idx),
            forecast_timestamp=feat_df["timestamp_utc"].iloc[idx],
            split_name=split_name,
        ))

    logger.info("Region %-9s origins=%d (train=%d val=%d test=%d) scaler(mean=%.1f std=%.1f)",
                region_code, n_c, n_train, n_val, n_c - n_train - n_val, scaler.mean, scaler.std)

    return RegionSequenceSet(
        region_code=region_code,
        scaler=scaler,
        feature_matrix=feature_matrix,
        demand_raw=demand_raw,
        timestamps=feat_df["timestamp_utc"],
        origins=origins,
    )


def window_and_targets(seq: RegionSequenceSet, origin: Origin, cfg: Config, horizons: List[int]):
    """Return (input_window [lookback, n_features], targets dict h -> (target_ts, actual_mw_or_nan))."""
    lookback = int(cfg.get("sequence.lookback_hours", 168))
    idx = origin.origin_idx
    window = seq.feature_matrix[idx - lookback: idx, :]
    # Any NaN in the window (grid-inserted missing hours) is filled with 0 in
    # SCALED space, i.e. "at the series mean" - a neutral placeholder, not a
    # spurious signal, and consistent with how the scaler treats gaps.
    window = np.nan_to_num(window, nan=0.0)
    targets = {}
    for h in horizons:
        t_idx = idx + h
        target_ts = seq.timestamps.iloc[t_idx]
        actual = seq.demand_raw[t_idx]
        targets[h] = (target_ts, actual)
    return window.astype(np.float32), targets
