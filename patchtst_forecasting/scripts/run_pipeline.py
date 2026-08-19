#!/usr/bin/env python3
"""
Stage 1 - Train ONE pooled PatchTST model across regions, evaluate per region,
write artefacts. Never touches PostgreSQL - run scripts/run_db_integration.py
afterward to load the results (identical two-stage shape as the other tracks).

Usage:
    python scripts/run_pipeline.py --regions AEP COMED
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

from src.config import load_config, setup_logging          # noqa: E402
from src.data_loader import load_region, summarize_region  # noqa: E402
from src.db_adapter import adapt_region, describe_capabilities  # noqa: E402
from src.model import evaluate_region, train_global         # noqa: E402
from src.utils import ensure_output_dirs, utcnow, write_json, write_table  # noqa: E402

logger = logging.getLogger("run_pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PatchTST energy demand forecasting pipeline")
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
    logger.info("# PatchTST Energy Demand Forecasting - %s v%s",
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

    logger.info("Loading %d regions for pooled training...", len(regions))
    regions_data: Dict[str, pd.DataFrame] = {}
    profiles: Dict[str, dict] = {}
    for region in regions:
        try:
            df = load_region(cfg, region)
            regions_data[region] = df
            profiles[region] = summarize_region(df)
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not load region %s: %s", region, exc)
            manifest["failures"][region] = f"load error: {exc}"

    if not regions_data:
        logger.error("No region data loaded.")
        write_json(dirs["reports"] / "run_manifest.json", manifest)
        return 1

    logger.info("Training pooled global PatchTST across %d regions...", len(regions_data))
    global_result = train_global(regions_data, cfg, horizons)
    manifest["training"] = {
        "n_train_samples": global_result.n_train_samples,
        "n_val_samples": global_result.n_val_samples,
        "n_params": global_result.n_params,
        "epochs_run": global_result.epochs_run,
        "stopped_reason": global_result.stopped_reason,
        "best_val_loss": global_result.best_val_loss,
        "fit_seconds": global_result.fit_seconds,
    }
    logger.info("Training complete: %s", manifest["training"])

    all_forecasts, all_evaluations, run_records = [], [], []

    for region in regions_data:
        try:
            result = evaluate_region(global_result, region, cfg)
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
                "data_profile": profiles.get(region, {}),
                "n_origins": result.n_origins,
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
