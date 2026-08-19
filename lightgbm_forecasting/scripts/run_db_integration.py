#!/usr/bin/env python3
"""
Stage 2 - Load stage-1 artefacts into PostgreSQL.

Split from training on purpose: a database outage, a bad credential or a schema
gap should never cost a 40-minute training run. This stage reads what stage 1
wrote to disk and is safe to retry as many times as needed.

Credentials come exclusively from the DATABASE_URL environment variable.

    # local
    export DATABASE_URL="postgresql://postgres@localhost:5432/energy_forecasting"
    # supabase (session pooler)
    export DATABASE_URL="postgresql://postgres.<ref>:<pw>@aws-0-ap-south-1.pooler.supabase.com:5432/postgres?sslmode=require"

Usage:
    python scripts/run_db_integration.py                # write everything found
    python scripts/run_db_integration.py --dry-run      # preflight + counts only
    python scripts/run_db_integration.py --regions AEP
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, setup_logging   # noqa: E402
from src.contracts import ModelRunContract          # noqa: E402
from src.db_writer import (                          # noqa: E402
    connect, inspect_schema, preflight, resolve_region_ids, write_region_payload,
)
from src.utils import ensure_output_dirs, utcnow, write_json  # noqa: E402

logger = logging.getLogger("run_db_integration")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load LightGBM outputs into PostgreSQL")
    p.add_argument("--config", default=None)
    p.add_argument("--output-dir", default=None, help="Stage-1 output directory")
    p.add_argument("--regions", nargs="+", default=None, help="Subset of regions to load")
    p.add_argument("--dry-run", action="store_true",
                   help="Run preflight and report what would be written, then roll back")
    return p.parse_args()


def _read_table(base: Path) -> pd.DataFrame:
    """Prefer Parquet (exact dtypes); fall back to CSV."""
    parquet, csv = base.with_suffix(".parquet"), base.with_suffix(".csv")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()


def load_stage1_payloads(dirs: Dict[str, Path], regions: Optional[List[str]]) -> List[Dict]:
    runs_path = dirs["runs"] / "model_runs.json"
    if not runs_path.exists():
        raise FileNotFoundError(
            f"{runs_path} not found - run scripts/run_pipeline.py first"
        )
    import json
    with open(runs_path, "r", encoding="utf-8") as fh:
        run_records = json.load(fh)

    payloads: List[Dict] = []
    for record in run_records:
        region = record["region_code"]
        if regions and region not in regions:
            continue
        if record.get("status") == "FAILED":
            logger.warning("Skipping region %s - stage 1 recorded status=FAILED", region)
            continue

        run = ModelRunContract(
            model_name=record["model_name"],
            model_type=record["model_type"],
            model_version=record["model_version"],
            region_code=region,
            training_start=pd.Timestamp(record["training_start"]),
            training_end=pd.Timestamp(record["training_end"]),
            horizons=record["horizons"],
            feature_version=record["feature_version"],
            code_version=record["code_version"],
            status=record.get("status", "SUCCESS"),
            n_features=record.get("n_features"),
            n_training_rows=record.get("n_training_rows"),
            failure_reason=record.get("failure_reason"),
            metadata=record.get("metadata", {}),
        )
        if run.run_uid != record["run_uid"]:
            logger.warning(
                "run_uid drift for %s (recomputed %s vs stored %s) - config changed "
                "since stage 1; using recomputed value",
                region, run.run_uid[:12], record["run_uid"][:12],
            )

        forecasts = _read_table(dirs["forecasts"] / f"{region}_forecasts")
        evaluations = _read_table(dirs["evaluations"] / f"{region}_evaluations")
        anomalies = _read_table(dirs["anomalies"] / f"{region}_anomalies")

        for frame, cols in ((forecasts, ("forecast_timestamp", "target_timestamp")),
                            (evaluations, ("evaluation_start", "evaluation_end")),
                            (anomalies, ("timestamp_utc",))):
            for col in cols:
                if col in frame.columns:
                    frame[col] = pd.to_datetime(frame[col], utc=True)

        payloads.append({
            "model_run": run,
            "forecasts": forecasts,
            "evaluations": evaluations,
            "anomalies": anomalies,
        })
    return payloads


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)

    out_root = Path(args.output_dir) if args.output_dir else cfg.output_dir
    dirs = ensure_output_dirs(out_root)

    payloads = load_stage1_payloads(dirs, args.regions)
    if not payloads:
        logger.error("No stage-1 payloads found under %s", out_root)
        return 1

    logger.info("Loaded %d region payload(s) from %s", len(payloads), out_root)

    report: Dict[str, object] = {
        "started_at": utcnow().isoformat(),
        "dry_run": args.dry_run,
        "regions": {},
    }

    with connect(cfg) as conn:
        schema_report = preflight(conn, cfg)
        report["preflight"] = schema_report
        live = inspect_schema(conn, cfg.get("database.schema", "public"))

        # Fail fast on unknown regions before writing a single row.
        resolve_region_ids(conn, [p["model_run"].region_code for p in payloads])

        try:
            for payload in payloads:
                region = payload["model_run"].region_code
                logger.info("-" * 70)
                logger.info("Writing region %s", region)
                if args.dry_run:
                    report["regions"][region] = {
                        "would_write_forecasts": int(len(payload["forecasts"])),
                        "would_write_evaluations": int(len(payload["evaluations"])),
                        "would_write_anomalies": int(len(payload["anomalies"])),
                    }
                    continue
                stats = write_region_payload(conn, payload, cfg, live)
                conn.commit()   # commit per region so a later failure keeps earlier work
                report["regions"][region] = stats
                logger.info("Committed region %s: %s", region, stats)

            if args.dry_run:
                conn.rollback()
                logger.info("Dry run complete - nothing was written.")

        except Exception:
            conn.rollback()
            logger.exception("Integration failed - current region rolled back")
            report["error"] = "see log"
            write_json(dirs["reports"] / "db_integration_report.json", report)
            return 1

    report["finished_at"] = utcnow().isoformat()
    write_json(dirs["reports"] / "db_integration_report.json", report)
    logger.info("=" * 70)
    logger.info("Integration report: %s", report["regions"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
