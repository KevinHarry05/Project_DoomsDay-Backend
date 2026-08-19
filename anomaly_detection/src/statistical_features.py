"""
Port of the EnerSight notebook's statistical anomaly features (its cells 20-26),
generalized from "one column of pre-uploaded forecasts" to "N (region, horizon)
groups pulled live from v_selected_forecast".

Every baseline below is fit PER GROUP (region_code, horizon_hours), not
globally. The notebook computed a per-region baseline because AEP's normal
residual scale is nothing like EKPC's; the same argument applies across
horizons within one region - a 24h-ahead residual is routinely 5-10x the
scale of a 1h-ahead residual for the same series, so pooling horizons into one
baseline would flag every long-horizon forecast as anomalous. This is the one
deliberate generalization made versus the original notebook.

FEATURES (mirrors the notebook 1:1 within a group)
----------------------------------------------------
1. Historical baseline   : group-level mean/std/median/MAD of the residual.
2. Robust_Z              : 0.6745 * (residual - median) / MAD   (MAD_EPSILON guard)
3. Rolling baseline       : rolling median/MAD over the last ROLLING_WINDOW hours
                            (min_periods so the first few hours of a group don't
                            silently produce nonsense from a near-empty window),
                            used to build a second, LOCAL robust z-score that
                            reacts to regime shifts the global baseline can't see.
4. Capping                : |Z| capped at Z_CAP, residual% capped at PCT_CAP -
                            stops one monstrous outlier from dominating the
                            composite score's scale for every other row.
5. Ramp feature           : |change in residual vs the previous hour|, itself
                            z-scored against a group-specific median/MAD of that
                            change series. The very first observation in a group
                            has no "previous hour" - flagged Ramp_Available=False
                            and given a neutral (zero) ramp contribution rather
                            than a fabricated one.
6. Statistical_Anomaly_Score : weighted sum of the four capped/normalized
                            components (weights sum to 1.0, validated below).
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)

GROUP_COLS = ("region_code", "horizon_hours")


def _robust_z(x: pd.Series, center: pd.Series, mad: pd.Series, k: float, eps: float) -> pd.Series:
    safe_mad = mad.where(mad > eps, eps)
    return k * (x - center) / safe_mad


def add_statistical_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.sort_values(list(GROUP_COLS) + ["target_timestamp"]).copy()

    mad_eps = float(cfg.get("statistical.mad_epsilon", 1e-6))
    k = float(cfg.get("statistical.mad_to_sigma", 0.6745))
    z_cap = float(cfg.get("statistical.z_cap", 10.0))
    pct_cap = float(cfg.get("statistical.pct_cap", 50.0))
    window = int(cfg.get("statistical.rolling_window_hours", 24))
    min_periods = int(cfg.get("statistical.rolling_min_periods", 12))

    w_hist = float(cfg.get("statistical.weight_historical", 0.30))
    w_roll = float(cfg.get("statistical.weight_rolling", 0.30))
    w_pct = float(cfg.get("statistical.weight_percentage", 0.20))
    w_ramp = float(cfg.get("statistical.weight_ramp", 0.20))
    weight_sum = w_hist + w_roll + w_pct + w_ramp
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"Anomaly score weights must sum to 1.0, got {weight_sum}")

    grp = out.groupby(list(GROUP_COLS))["residual_mw"]

    # ---- 1+2: group-level historical robust baseline -----------------------
    out["hist_median"] = grp.transform("median")
    out["hist_mad"] = grp.transform(lambda s: (s - s.median()).abs().median())
    out["robust_z_historical"] = _robust_z(out["residual_mw"], out["hist_median"], out["hist_mad"], k, mad_eps)

    # ---- 3: rolling local baseline (per group, chronological) --------------
    roll_median = out.groupby(list(GROUP_COLS))["residual_mw"].transform(
        lambda s: s.rolling(window=window, min_periods=min_periods).median()
    )
    roll_mad = out.groupby(list(GROUP_COLS))["residual_mw"].transform(
        lambda s: (s - s.rolling(window=window, min_periods=min_periods).median())
        .abs().rolling(window=window, min_periods=min_periods).median()
    )
    out["rolling_median"] = roll_median
    out["rolling_mad"] = roll_mad
    out["robust_z_rolling"] = _robust_z(out["residual_mw"], roll_median.fillna(out["hist_median"]),
                                         roll_mad.fillna(out["hist_mad"]), k, mad_eps)
    # Rows before min_periods observations exist in their group fall back to
    # the historical baseline rather than producing NaN - documented, not hidden.
    out["rolling_baseline_available"] = roll_median.notna() & roll_mad.notna()

    # ---- 4: capped z-scores + capped residual percentage --------------------
    out["robust_z_historical_capped"] = out["robust_z_historical"].clip(-z_cap, z_cap)
    out["robust_z_rolling_capped"] = out["robust_z_rolling"].clip(-z_cap, z_cap)
    with np.errstate(divide="ignore", invalid="ignore"):
        residual_pct = np.where(
            out["actual_demand_mw"].abs() > 0,
            out["residual_mw"] / out["actual_demand_mw"] * 100.0,
            0.0,
        )
    out["residual_pct"] = residual_pct
    out["residual_pct_capped"] = np.clip(residual_pct, -pct_cap, pct_cap)

    # ---- 5: ramp feature (change vs previous hour, per group) ---------------
    prev_residual = out.groupby(list(GROUP_COLS))["residual_mw"].shift(1)
    ramp_raw = (out["residual_mw"] - prev_residual).abs()
    out["ramp_available"] = prev_residual.notna()
    out["ramp_raw"] = ramp_raw.fillna(0.0)

    # Only rows with a genuine previous-hour observation contribute to the
    # group's ramp baseline - the first row of each group is excluded rather
    # than silently pulling the baseline toward its fabricated zero.
    valid_ramp = out.loc[out["ramp_available"], list(GROUP_COLS) + ["ramp_raw"]]
    ramp_stats = valid_ramp.groupby(list(GROUP_COLS))["ramp_raw"].agg(
        ramp_median="median",
        ramp_mad=lambda s: (s - s.median()).abs().median(),
    ).reset_index()
    out = out.merge(ramp_stats, on=list(GROUP_COLS), how="left")
    out["ramp_median"] = out["ramp_median"].fillna(0.0)
    out["ramp_mad"] = out["ramp_mad"].fillna(0.0)
    ramp_z = k * (out["ramp_raw"] - out["ramp_median"]) / out["ramp_mad"].where(out["ramp_mad"] > mad_eps, mad_eps)
    out["ramp_z_capped"] = np.where(out["ramp_available"], ramp_z.clip(-z_cap, z_cap), 0.0)

    # ---- 6: weighted composite score ----------------------------------------
    # Each component normalized to a common 0-z_cap scale before weighting, so
    # no single component (e.g. a 50%-capped percentage vs a 10-capped z-score)
    # silently dominates just because of its native numeric range.
    hist_component = out["robust_z_historical_capped"].abs()
    roll_component = out["robust_z_rolling_capped"].abs()
    pct_component = (out["residual_pct_capped"].abs() / pct_cap) * z_cap
    ramp_component = out["ramp_z_capped"].abs()

    out["statistical_anomaly_score"] = (
        w_hist * hist_component + w_roll * roll_component
        + w_pct * pct_component + w_ramp * ramp_component
    )

    logger.info(
        "Statistical features built for %d rows across %d (region, horizon) groups "
        "(weights hist=%.2f roll=%.2f pct=%.2f ramp=%.2f)",
        len(out), out.groupby(list(GROUP_COLS)).ngroups, w_hist, w_roll, w_pct, w_ramp,
    )
    return out
