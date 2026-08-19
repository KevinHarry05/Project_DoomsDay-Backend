#!/usr/bin/env python3
"""
Integrate a single already-finished region straight from its per-region
artefacts, without waiting for the full multi-region run_pipeline.py run (and
its combined model_runs.json manifest) to complete.

Used when a background training run has finished region N but is still
training region N+1 onward - the artefacts for the finished region are already
on disk and safe to load.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, setup_logging
from src.contracts import ModelRunContract
from src.db_writer import connect, inspect_schema, preflight, resolve_region_ids, write_region_payload
from src.utils import ensure_output_dirs, utcnow


def main(region: str) -> int:
    cfg = load_config()
    setup_logging(cfg)
    dirs = ensure_output_dirs(cfg.output_dir)

    fc = pd.read_parquet(dirs["forecasts"] / f"{region}_forecasts.parquet")
    ev = pd.read_parquet(dirs["evaluations"] / f"{region}_evaluations.parquet")
    for col in ("forecast_timestamp", "target_timestamp"):
        fc[col] = pd.to_datetime(fc[col], utc=True)
    for col in ("evaluation_start", "evaluation_end"):
        ev[col] = pd.to_datetime(ev[col], utc=True)

    horizons = sorted(fc["horizon_hours"].unique().tolist())
    train_window = fc["forecast_timestamp"]
    # test-split forecast_timestamps only, to describe the actual training window
    test_fc = fc[fc["split_name"] == "test"] if "split_name" in fc.columns else fc

    run = ModelRunContract(
        model_name=cfg["project.model_name"],
        model_type=cfg["project.model_type"],
        model_version=cfg["project.model_version"],
        region_code=region,
        training_start=fc["forecast_timestamp"].min(),
        training_end=fc["forecast_timestamp"].max(),
        horizons=horizons,
        feature_version=cfg["features.feature_version"],
        code_version=cfg["project.code_version"],
        status="SUCCESS",
        n_features=None,
        n_training_rows=None,
        metadata={"strategy": "direct_multi_horizon", "trained_horizons": horizons,
                  "integrated_early": True},
    )

    payload = {
        "model_run": run,
        "forecasts": fc,
        "evaluations": ev,
        "anomalies": pd.DataFrame(columns=["run_uid", "region_code", "timestamp_utc",
                                            "horizon_hours", "actual_demand_mw",
                                            "predicted_demand_mw", "residual_mw",
                                            "deviation_percent", "anomaly_score", "severity",
                                            "is_anomaly", "anomaly_direction", "reason",
                                            "detection_method"]),
    }

    with connect(cfg) as conn:
        live = inspect_schema(conn, cfg.get("database.schema", "public"))
        preflight(conn, cfg)
        resolve_region_ids(conn, [region])
        stats = write_region_payload(conn, payload, cfg, live)
        conn.commit()
        print(f"Committed {region}: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "AEP"))
