#!/usr/bin/env python3
"""
Anomaly detection - third stage of the pipeline, run AFTER at least one
forecasting track has loaded results into Postgres.

Unlike run_pipeline.py in the two forecasting packages, this script talks to
the database on BOTH ends: it reads its only input (v_selected_forecast) live
from Postgres, and writes results straight back - there is no intermediate
"train once, load later" split, because there is nothing to train. This
mirrors exactly how the reference notebook's logic is meant to run in
production: score whatever the current best forecast is, whenever asked.

Usage:
    python scripts/run_anomaly_detection.py --all-regions
    python scripts/run_anomaly_detection.py --regions AEP EKPC --horizons 1 24
    python scripts/run_anomaly_detection.py --all-regions --no-write-db   (dry run, writes parquet only)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adapter import to_anomaly_contract                       # noqa: E402
from src.config import load_config, setup_logging                 # noqa: E402
from src.data_source import connect, fetch_selected_forecasts     # noqa: E402
from src.db_writer import (                                       # noqa: E402
    inspect_schema, resolve_region_ids_by_code, upsert_alerts, upsert_anomalies,
)
from src.hybrid_scoring import add_event_grouping, add_hybrid_flags, add_isolation_forest_scores  # noqa: E402
from src.statistical_features import add_statistical_features     # noqa: E402

logger = logging.getLogger("run_anomaly_detection")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid statistical + Isolation Forest anomaly detection")
    p.add_argument("--config", default=None)
    p.add_argument("--regions", nargs="+", default=None)
    p.add_argument("--all-regions", action="store_true")
    p.add_argument("--horizons", nargs="+", type=int, default=None)
    p.add_argument("--no-write-db", action="store_true",
                   help="Score and write parquet/CSV only; skip the anomalies/alerts write.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("#" * 78)
    logger.info("# Hybrid Anomaly Detection - reading from v_selected_forecast")
    logger.info("#" * 78)

    with connect(cfg) as conn:
        selected = fetch_selected_forecasts(
            conn,
            region_codes=None if args.all_regions else args.regions,
            horizons=args.horizons,
        )
        if selected.empty:
            logger.error("v_selected_forecast returned no rows for the requested scope. "
                         "Has at least one forecasting track's output been loaded yet?")
            return 1

        scored = add_statistical_features(selected, cfg)
        scored = add_isolation_forest_scores(scored, cfg)
        scored = add_hybrid_flags(scored, cfg)
        scored, events = add_event_grouping(scored, cfg)

        contract_df = to_anomaly_contract(scored, cfg)

        # Persist artefacts regardless of --no-write-db, same convention as
        # the forecasting packages (outputs always land on disk; the DB write
        # is a separate, skippable step).
        contract_df.drop(columns=["forecast_id", "model_run_id"]).to_parquet(
            out_dir / "anomalies.parquet", index=False)
        contract_df.drop(columns=["forecast_id", "model_run_id"]).to_csv(
            out_dir / "anomalies.csv", index=False)
        events.to_parquet(out_dir / "anomaly_events.parquet", index=False)
        events.to_csv(out_dir / "anomaly_events.csv", index=False)
        logger.info("Wrote %d anomaly rows and %d events to %s", len(contract_df), len(events), out_dir)

        n_written = 0
        if not args.no_write_db:
            live = inspect_schema(conn)
            region_map = resolve_region_ids_by_code(conn, contract_df["region_code"].unique().tolist())
            contract_df["region_id"] = contract_df["region_code"].map(region_map)
            missing_region = contract_df["region_id"].isna()
            if missing_region.any():
                logger.warning("Dropping %d rows with unresolved region_code", int(missing_region.sum()))
                contract_df = contract_df[~missing_region]

            batch = int(cfg.get("database.batch_size", 10000))
            n_written = upsert_anomalies(conn, contract_df, live, batch)
            upsert_alerts(conn, live, list(region_map.values()))
            conn.commit()

    _print_summary(contract_df, events)
    return 0


def _print_summary(contract_df: pd.DataFrame, events: pd.DataFrame) -> None:
    logger.info("")
    logger.info("=" * 78)
    logger.info("ANOMALY DETECTION SUMMARY")
    logger.info("=" * 78)
    n = len(contract_df)
    n_flag = int(contract_df["is_anomaly"].sum())
    logger.info("Scored %d rows | flagged %d (%.2f%%) | events %d",
                n, n_flag, 100.0 * n_flag / max(n, 1), len(events))
    if n_flag:
        by_sev = contract_df[contract_df["is_anomaly"]]["severity"].value_counts()
        logger.info("Severity breakdown: %s", by_sev.to_dict())
        by_region = contract_df[contract_df["is_anomaly"]].groupby("region_code").size()
        logger.info("By region: %s", by_region.to_dict())


if __name__ == "__main__":
    raise SystemExit(main())
