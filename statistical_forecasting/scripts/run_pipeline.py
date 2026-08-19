#!/usr/bin/env python3
"""
Stage 1 - Fit DHR+ARIMA, roll it through the held-out period, write artefacts.

Same two-stage shape as the LightGBM track: this never touches PostgreSQL.
Run scripts/run_db_integration.py afterward to load the results.

Usage:
    python scripts/run_pipeline.py --regions AEP
    python scripts/run_pipeline.py --all-regions
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, setup_logging        # noqa: E402
from src.data_loader import load_region, summarize_region # noqa: E402
from src.db_adapter import adapt_region, describe_capabilities  # noqa: E402
from src.model import train_region                        # noqa: E402
from src.utils import ensure_output_dirs, utcnow, write_json, write_table  # noqa: E402

logger = logging.getLogger("run_pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DHR+ARIMA energy demand forecasting pipeline")
    p.add_argument("--config", default=None)
    p.add_argument("--regions", nargs="+", default=None)
    p.add_argument("--all-regions", action="store_true")
    p.add_argument("--horizons", nargs="+", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)

    out_root = Path(args.output_dir) if args.output_dir else cfg.output_dir
    dirs = ensure_output_dirs(out_root)

    regions: List[str] = cfg.all_regions if args.all_regions else (args.regions or cfg.default_regions)
    horizons = sorted(args.horizons) if args.horizons else cfg.horizons

    logger.info("#" * 78)
    logger.info("# DHR+ARIMA Energy Demand Forecasting - %s v%s",
                cfg["project.model_name"], cfg["project.model_version"])
    logger.info("# regions=%s horizons=%s", regions, horizons)
    logger.info("#" * 78)

    manifest: Dict[str, object] = {
        "started_at": utcnow().isoformat(),
        "model_name": cfg["project.model_name"],
        "model_version": cfg["project.model_version"],
        "code_version": cfg["project.code_version"],
        "capabilities": describe_capabilities(cfg),
        "requested_regions": regions,
        "horizons": horizons,
        "regions": {},
        "failures": {},
    }

    all_forecasts, all_evaluations, run_records = [], [], []

    for region in regions:
        try:
            region_df = load_region(cfg, region)
            profile = summarize_region(region_df)

            result = train_region(region_df, cfg, horizons)
            if result.status != "SUCCESS":
                raise RuntimeError(result.failure_reason or "unknown failure")

            payload = adapt_region(result, cfg)
            write_table(payload["forecasts"], dirs["forecasts"] / f"{region}_forecasts")
            write_table(payload["evaluations"], dirs["evaluations"] / f"{region}_evaluations")

            all_forecasts.append(payload["forecasts"])
            all_evaluations.append(payload["evaluations"])
            run_records.append(result.run.to_dict())

            manifest["regions"][region] = {
                "run_uid": result.run.run_uid,
                "status": result.run.status,
                "data_profile": profile,
                "n_origins": result.n_origins,
                "fit_seconds": result.fit_seconds,
                "n_forecast_rows": int(len(payload["forecasts"])),
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("REGION %s FAILED", region)
            manifest["failures"][region] = f"{type(exc).__name__}: {exc}"

    if not run_records:
        logger.error("No region completed successfully.")
        write_json(dirs["reports"] / "run_manifest.json", manifest)
        return 1

    combined_fc = pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()
    combined_ev = pd.concat(all_evaluations, ignore_index=True) if all_evaluations else pd.DataFrame()
    write_table(combined_fc, dirs["forecasts"] / "ALL_forecasts", also_csv=False)
    write_table(combined_ev, dirs["evaluations"] / "ALL_evaluations")
    write_json(dirs["runs"] / "model_runs.json", run_records)

    manifest["finished_at"] = utcnow().isoformat()
    manifest["totals"] = {
        "regions_succeeded": len(run_records),
        "regions_failed": len(manifest["failures"]),
        "forecast_rows": int(len(combined_fc)),
        "evaluation_rows": int(len(combined_ev)),
    }
    write_json(dirs["reports"] / "run_manifest.json", manifest)

    _print_summary(combined_ev, cfg, manifest)
    return 0 if not manifest["failures"] else 2


def _print_summary(evaluations: pd.DataFrame, cfg, manifest) -> None:
    logger.info("")
    logger.info("=" * 78)
    logger.info("TEST-SET PERFORMANCE")
    logger.info("=" * 78)
    if evaluations.empty:
        logger.info("(no evaluations)")
        return
    test = evaluations[evaluations["split_name"] == "test"].copy()
    cols = [c for c in ["region_code", "horizon_hours", "n_observations", "mae", "rmse",
                        "wape", "smape", "skill_vs_naive"] if c in test.columns]
    view = test[cols].sort_values(["region_code", "horizon_hours"])
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:,.3f}"):
        logger.info("\n%s", view.to_string(index=False))
    logger.info("Totals: %s", manifest["totals"])
    if manifest["failures"]:
        logger.warning("Failed regions: %s", manifest["failures"])
    logger.info("Artefacts written under: %s", cfg.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
