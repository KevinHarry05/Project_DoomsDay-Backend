#!/usr/bin/env python3
"""
Scheduled retrain wrapper for LightGBM.

Implements the loop:

    new historical demand arrives -> retrain -> generate new forecasts -> update DB

Meant to be registered with a scheduler (Windows Task Scheduler locally, cron on
Linux) and run unattended - it makes its own decision about which regions
actually need retraining, retrains only those, writes straight to PostgreSQL,
and never duplicates a row thanks to the same idempotent upsert keys the manual
scripts already use (run_uid, and the forecast/evaluation conflict keys).

WHY NOT JUST RETRAIN EVERY TIME IT FIRES
-----------------------------------------
Retraining a region that has no new data since its last run wastes real time
(minutes per region) for a result that will be byte-for-byte identical - same
run_uid, same rows, upserted over themselves. Before touching a region this
script checks two things:

  1. How much NEW historical data exists locally beyond what LightGBM has
     already seen for that region (its last model_runs.training_end).
  2. Whether that amount clears `retrain.min_new_hours` (default 168h = one
     week) - a fresh half day of readings is not worth a multi-minute retrain.

`--force` bypasses both checks and retrains every requested region regardless.

WHY THIS STAYS SAFE TO RUN UNATTENDED
--------------------------------------
- One region's failure is logged and skipped; it never stops the others
  (same isolation pattern as run_pipeline.py / run_db_integration.py).
- Every write is the same upsert used everywhere else, so a retry after a
  crash mid-run cannot double-insert anything.
- Nothing is deleted. An old model_run is superseded (the comparison views
  pick the newest SUCCESS/PARTIAL run per model+region+horizon), never
  removed - so history of past runs is preserved for audit.

USAGE
-----
    # one-off manual run, all regions, only retrains what actually needs it
    python scripts/scheduled_retrain.py --all-regions

    # what would happen, without training or writing anything
    python scripts/scheduled_retrain.py --all-regions --dry-run

    # force a full retrain regardless of how much new data exists
    python scripts/scheduled_retrain.py --all-regions --force

REGISTERING ON WINDOWS TASK SCHEDULER (weekly, Sunday 02:00)
--------------------------------------------------------------
    schtasks /Create /SC WEEKLY /D SUN /ST 02:00 ^
        /TN "LightGBM_Weekly_Retrain" ^
        /TR "cmd /c cd /d C:\\Users\\kani2\\OneDrive\\Documents\\BACKEND\\lightgbm_forecasting && set DATABASE_URL=postgresql://... && python scripts\\scheduled_retrain.py --all-regions >> logs\\retrain.log 2>&1"

The DATABASE_URL is embedded in the /TR command line here only because
Task Scheduler has no simple per-task secret store; if that is a concern,
wrap the call in a .bat file that reads the URL from a local, non-synced
file instead of typing it into the scheduled task definition directly.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, setup_logging                      # noqa: E402
from src.data_loader import load_region, region_file_path               # noqa: E402
from src.db_adapter import adapt_region                                 # noqa: E402
from src.db_writer import (                                             # noqa: E402
    connect, get_latest_data_available_through, inspect_schema, preflight,
    resolve_region_ids, write_region_payload,
)
from src.model import train_region                                      # noqa: E402
from src.utils import ensure_output_dirs, utcnow, write_json            # noqa: E402

logger = logging.getLogger("scheduled_retrain")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scheduled LightGBM retrain + DB update")
    p.add_argument("--config", default=None)
    p.add_argument("--regions", nargs="+", default=None)
    p.add_argument("--all-regions", action="store_true")
    p.add_argument("--horizons", nargs="+", type=int, default=None)
    p.add_argument("--force", action="store_true",
                   help="Retrain every requested region regardless of how much new data exists")
    p.add_argument("--min-new-hours", type=int, default=None,
                   help="Override retrain.min_new_hours from config")
    p.add_argument("--enable-anomaly", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Decide what would retrain, touch nothing")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def latest_local_timestamp(cfg, region: str) -> Optional[pd.Timestamp]:
    """Peek at the newest timestamp in a region's file without loading the
    whole series - this runs on every scheduled fire, so it should be cheap."""
    path = region_file_path(cfg, region)
    if not path.exists():
        return None
    ts_col = cfg["data.timestamp_col"]
    col = pd.read_csv(path, usecols=[ts_col])[ts_col]
    return pd.to_datetime(col, utc=True, errors="coerce").max()


def decide_regions_to_retrain(
    conn, cfg, regions: List[str], model_name: str, min_new_hours: int, force: bool
) -> Dict[str, Dict[str, object]]:
    """Return {region: {"action": "retrain"|"skip", "reason": str, ...}}."""
    region_ids = resolve_region_ids(conn, regions)
    decisions: Dict[str, Dict[str, object]] = {}

    for region in regions:
        latest_local = latest_local_timestamp(cfg, region)
        if latest_local is None or pd.isna(latest_local):
            decisions[region] = {"action": "skip", "reason": "no local data file found"}
            continue

        if force:
            decisions[region] = {
                "action": "retrain", "reason": "--force",
                "latest_local": latest_local.isoformat(),
            }
            continue

        last_seen = get_latest_data_available_through(conn, model_name, region_ids[region])
        if last_seen is None:
            decisions[region] = {
                "action": "retrain", "reason": "no prior successful run for this model+region",
                "latest_local": latest_local.isoformat(),
            }
            continue

        new_hours = (latest_local - last_seen).total_seconds() / 3600.0
        if new_hours >= min_new_hours:
            decisions[region] = {
                "action": "retrain",
                "reason": f"{new_hours:.0f}h of new data beyond what the last run saw "
                          f"(threshold {min_new_hours}h)",
                "latest_local": latest_local.isoformat(),
                "last_run_data_through": last_seen.isoformat(),
            }
        else:
            decisions[region] = {
                "action": "skip",
                "reason": f"only {new_hours:.0f}h of new data beyond what the last run saw "
                          f"(threshold {min_new_hours}h) - not worth a retrain yet",
                "latest_local": latest_local.isoformat(),
                "last_run_data_through": last_seen.isoformat(),
            }
    return decisions


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)
    if args.enable_anomaly:
        cfg._data.setdefault("anomaly", {})["enabled"] = True

    regions = cfg.all_regions if args.all_regions else (args.regions or cfg.default_regions)
    horizons = sorted(args.horizons) if args.horizons else cfg.horizons
    min_new_hours = args.min_new_hours if args.min_new_hours is not None \
        else int(cfg.get("retrain.min_new_hours", 168))
    model_name = cfg["project.model_name"]

    out_root = Path(args.output_dir) if args.output_dir else cfg.output_dir
    dirs = ensure_output_dirs(out_root)

    report: Dict[str, object] = {
        "started_at": utcnow().isoformat(),
        "model_name": model_name,
        "requested_regions": regions,
        "horizons": horizons,
        "min_new_hours": min_new_hours,
        "force": args.force,
        "dry_run": args.dry_run,
        "decisions": {},
        "results": {},
    }

    logger.info("=" * 78)
    logger.info("SCHEDULED RETRAIN - %s", model_name)
    logger.info("regions=%s  min_new_hours=%d  force=%s  dry_run=%s",
                regions, min_new_hours, args.force, args.dry_run)
    logger.info("=" * 78)

    with connect(cfg) as conn:
        live = inspect_schema(conn, cfg.get("database.schema", "public"))
        preflight(conn, cfg)

        decisions = decide_regions_to_retrain(conn, cfg, regions, model_name, min_new_hours, args.force)
        report["decisions"] = decisions
        for region, decision in decisions.items():
            logger.info("  %-9s -> %-7s (%s)", region, decision["action"], decision["reason"])

        to_retrain = [r for r, d in decisions.items() if d["action"] == "retrain"]
        if not to_retrain:
            logger.info("Nothing needs retraining right now.")
            report["finished_at"] = utcnow().isoformat()
            write_json(dirs["reports"] / "retrain_report.json", report)
            return 0

        if args.dry_run:
            logger.info("Dry run - would retrain: %s. Stopping without training or writing.", to_retrain)
            conn.rollback()
            report["finished_at"] = utcnow().isoformat()
            write_json(dirs["reports"] / "retrain_report.json", report)
            return 0

        for region in to_retrain:
            logger.info("-" * 70)
            logger.info("RETRAINING %s", region)
            try:
                region_df = load_region(cfg, region)
                result = train_region(region_df, cfg, horizons)
                payload = adapt_region(result, cfg)

                stats = write_region_payload(conn, payload, cfg, live)
                conn.commit()

                report["results"][region] = {
                    "status": "SUCCESS",
                    "run_uid": result.run.run_uid,
                    "trained_horizons": result.succeeded_horizons,
                    "failed_horizons": result.failed_horizons,
                    **stats,
                }
                logger.info("  Committed %s: %s", region, stats)

            except Exception as exc:  # noqa: BLE001 - one region must not sink the run
                conn.rollback()
                logger.exception("  RETRAIN FAILED for %s - skipping, other regions continue", region)
                report["results"][region] = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}

    report["finished_at"] = utcnow().isoformat()
    write_json(dirs["reports"] / "retrain_report.json", report)

    n_ok = sum(1 for r in report["results"].values() if r.get("status") == "SUCCESS")
    n_fail = sum(1 for r in report["results"].values() if r.get("status") == "FAILED")
    logger.info("=" * 78)
    logger.info("Retrain complete: %d succeeded, %d failed, %d skipped (no new data)",
                n_ok, n_fail, len(regions) - len(to_retrain))
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
