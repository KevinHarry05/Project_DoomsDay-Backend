"""
LightGBM training / prediction for one region across all supported horizons.

STRATEGY: DIRECT MULTI-HORIZON
------------------------------
One independent booster per (region, horizon). Model h=24 is trained to map
features known at t directly onto demand at t+24.

The alternative - train a single 1-step model and feed its own output back 24
times - is rejected because prediction error compounds multiplicatively and the
recursive input distribution drifts away from anything the model saw in
training. Direct costs us 4x training time and buys honest 24-hour error.

MODEL SCOPE
-----------
One booster per region, not one global model with region as a categorical.
Reasons: (1) demand scale spans two orders of magnitude across regions
(DAYTON ~2 GW mean, PJME ~32 GW), (2) regional coverage windows differ by
years - NI ends in 2011 while COMED begins there - so a global chronological
split would be incoherent, and (3) `model_runs.region_id` already models a run
as region-scoped, so per-region boosters map onto the database design without
contortion.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import Config
from .contracts import ModelRunContract
from .evaluation import evaluate_split
from .feature_engineering import (
    ORIGIN_TS_COL,
    TARGET_COL,
    TARGET_TS_COL,
    build_all_horizons,
    feature_columns,
)
from .splits import (
    SPLIT_COL,
    assert_chronological_integrity,
    assign_splits,
    compute_boundaries,
    split_frames,
)
from .utils import Timer

logger = logging.getLogger(__name__)


@dataclass
class HorizonResult:
    """Everything produced by one (region, horizon) booster."""

    region_code: str
    horizon_hours: int
    booster: Optional[lgb.Booster]
    best_iteration: Optional[int]
    feature_names: List[str]
    metrics: Dict[str, Dict[str, float]]          # split -> metric dict
    predictions: pd.DataFrame                      # origin/target/actual/pred/split
    training_time_seconds: float
    inference_time_ms: float
    feature_importance: pd.DataFrame
    status: str = "SUCCESS"
    failure_reason: Optional[str] = None


@dataclass
class RegionResult:
    region_code: str
    run: ModelRunContract
    horizons: Dict[int, HorizonResult] = field(default_factory=dict)
    split_boundaries: Dict[str, object] = field(default_factory=dict)
    diagnostics: Dict[str, object] = field(default_factory=dict)

    @property
    def succeeded_horizons(self) -> List[int]:
        return sorted(h for h, r in self.horizons.items() if r.status == "SUCCESS")

    @property
    def failed_horizons(self) -> List[int]:
        return sorted(h for h, r in self.horizons.items() if r.status != "SUCCESS")


# ---------------------------------------------------------------------------
def train_horizon(
    frame: pd.DataFrame, cfg: Config, region_code: str, horizon_hours: int
) -> HorizonResult:
    """Train and score one booster. Never raises - a failed horizon is recorded
    so the surviving horizons and other models stay usable."""
    feat_cols = feature_columns(frame.drop(columns=[SPLIT_COL], errors="ignore"))
    parts = split_frames(frame)
    assert_chronological_integrity(parts)

    empty = pd.DataFrame(columns=[ORIGIN_TS_COL, TARGET_TS_COL, "actual_demand_mw",
                                  "predicted_demand_mw", SPLIT_COL])
    try:
        params = dict(cfg["model.params"])
        train_set = lgb.Dataset(
            parts["train"][feat_cols], label=parts["train"][TARGET_COL],
            feature_name=feat_cols, free_raw_data=False,
        )
        val_set = lgb.Dataset(
            parts["val"][feat_cols], label=parts["val"][TARGET_COL],
            reference=train_set, feature_name=feat_cols, free_raw_data=False,
        )

        with Timer() as train_timer:
            booster = lgb.train(
                params,
                train_set,
                num_boost_round=int(cfg["model.num_boost_round"]),
                valid_sets=[train_set, val_set],
                valid_names=["train", "val"],
                callbacks=[
                    lgb.early_stopping(int(cfg["model.early_stopping_rounds"]), verbose=False),
                    lgb.log_evaluation(int(cfg.get("model.log_evaluation_period", 200))),
                ],
            )

        best_iter = booster.best_iteration or booster.current_iteration()

        # Inference timing is measured on the test split only, so the number
        # written to model_evaluations.inference_time_ms is comparable across
        # models rather than being an artefact of split size.
        preds: Dict[str, np.ndarray] = {}
        inference_ms = 0.0
        for name, part in parts.items():
            if part.empty:
                preds[name] = np.array([])
                continue
            with Timer() as infer_timer:
                preds[name] = booster.predict(part[feat_cols], num_iteration=best_iter)
            if name == "test":
                inference_ms = infer_timer.milliseconds

        smape_eps = float(cfg.get("evaluation.smape_epsilon", 1.0))
        metrics: Dict[str, Dict[str, float]] = {}
        pred_rows = []
        for name, part in parts.items():
            if part.empty:
                continue
            scored = part.copy()
            scored["predicted_demand_mw"] = preds[name]
            metrics[name] = evaluate_split(
                scored, smape_epsilon=smape_eps, horizon_hours=horizon_hours
            )
            pred_rows.append(
                scored[[ORIGIN_TS_COL, TARGET_TS_COL, TARGET_COL,
                        "predicted_demand_mw", SPLIT_COL]]
                .rename(columns={TARGET_COL: "actual_demand_mw"})
            )

        predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else empty
        predictions["horizon_hours"] = horizon_hours
        predictions["region_code"] = region_code

        importance = pd.DataFrame({
            "feature": booster.feature_name(),
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }).sort_values("gain", ascending=False).reset_index(drop=True)
        importance.insert(0, "horizon_hours", horizon_hours)
        importance.insert(0, "region_code", region_code)

        test_m = metrics.get("test", {})
        logger.info(
            "  %-9s h=%-3d iters=%-5s  test MAE=%8.1f  RMSE=%8.1f  WAPE=%5.2f%%  "
            "sMAPE=%5.2f%%  skill_vs_naive=%s",
            region_code, horizon_hours, best_iter,
            test_m.get("mae", float("nan")), test_m.get("rmse", float("nan")),
            test_m.get("wape", float("nan")), test_m.get("smape", float("nan")),
            f"{test_m['skill_vs_naive']:.3f}" if "skill_vs_naive" in test_m else "n/a",
        )

        return HorizonResult(
            region_code=region_code,
            horizon_hours=horizon_hours,
            booster=booster,
            best_iteration=int(best_iter),
            feature_names=feat_cols,
            metrics=metrics,
            predictions=predictions,
            training_time_seconds=train_timer.seconds,
            inference_time_ms=inference_ms,
            feature_importance=importance,
        )

    except Exception as exc:  # noqa: BLE001 - deliberate: isolate horizon failures
        logger.exception("Horizon %dh FAILED for region %s", horizon_hours, region_code)
        return HorizonResult(
            region_code=region_code,
            horizon_hours=horizon_hours,
            booster=None,
            best_iteration=None,
            feature_names=feat_cols,
            metrics={},
            predictions=empty,
            training_time_seconds=0.0,
            inference_time_ms=0.0,
            feature_importance=pd.DataFrame(columns=["region_code", "horizon_hours",
                                                     "feature", "gain", "split"]),
            status="FAILED",
            failure_reason=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
def train_region(
    region_df: pd.DataFrame, cfg: Config, horizons: Optional[List[int]] = None
) -> RegionResult:
    """Full training pass for one region across every requested horizon."""
    region_code = region_df["region_code"].iloc[0]
    horizons = horizons or cfg.horizons
    logger.info("=" * 78)
    logger.info("REGION %s | horizons=%s", region_code, horizons)
    logger.info("=" * 78)

    frames, _ = build_all_horizons(region_df, cfg, horizons)

    # Boundaries are computed once on the shortest-horizon frame and reused, so
    # every horizon is evaluated over the same wall-clock test window. Without
    # this, h=1 and h=24 would be scored on slightly different periods and their
    # metrics would not be comparable.
    reference = frames[min(horizons)]
    bounds = compute_boundaries(reference, cfg)

    results: Dict[int, HorizonResult] = {}
    for h in horizons:
        labelled = assign_splits(frames[h], cfg, boundaries=bounds)
        results[h] = train_horizon(labelled, cfg, region_code, h)

    ok = [h for h, r in results.items() if r.status == "SUCCESS"]
    train_part = assign_splits(reference, cfg, boundaries=bounds)
    train_only = train_part[train_part[SPLIT_COL] == "train"]

    run = ModelRunContract(
        model_name=cfg["project.model_name"],
        model_type=cfg["project.model_type"],
        model_version=cfg["project.model_version"],
        region_code=region_code,
        training_start=train_only[ORIGIN_TS_COL].min(),
        training_end=train_only[ORIGIN_TS_COL].max(),
        horizons=sorted(horizons),
        feature_version=cfg["features.feature_version"],
        code_version=cfg["project.code_version"],
        status="SUCCESS" if len(ok) == len(horizons) else ("PARTIAL" if ok else "FAILED"),
        n_features=len(feature_columns(reference)),
        n_training_rows=int(len(train_only)),
        failure_reason=None if len(ok) == len(horizons) else
        "; ".join(f"h={h}: {results[h].failure_reason}" for h in results if results[h].status != "SUCCESS"),
        metadata={
            "supported_horizons": sorted(horizons),
            "trained_horizons": ok,
            "failed_horizons": sorted(set(horizons) - set(ok)),
            "strategy": "direct_multi_horizon",
            "objective": cfg["model.params"]["objective"],
            "split_method": cfg.get("split.method", "fraction"),
            "scaling": "none (gradient-boosted trees are scale-invariant)",
            # Full extent of the local series this run actually saw - INCLUDING
            # the held-out val/test tail, not just the train split. This is
            # deliberately separate from training_start/training_end (which
            # stay the true train-split window for that column's own meaning).
            # scheduled_retrain.py compares new local data against THIS value,
            # not against training_end - otherwise the val/test tail (held out
            # by design, ~30% of every run) would look like "new data" on every
            # single check and defeat the point of skip-if-nothing-changed.
            "data_available_through": region_df["timestamp_utc"].max().isoformat(),
        },
    )

    return RegionResult(
        region_code=region_code,
        run=run,
        horizons=results,
        split_boundaries=bounds.as_dict(),
        diagnostics={"n_feature_columns": len(feature_columns(reference))},
    )
