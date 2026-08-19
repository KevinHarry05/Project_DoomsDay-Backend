#!/usr/bin/env python3
"""
Stage 1 - Train LightGBM, evaluate, detect anomalies, write standardized outputs.

This stage never touches PostgreSQL. It produces contract-shaped artefacts on
disk so the model can be re-run, inspected and reviewed without a database, and
so stage 2 (run_db_integration.py) can be retried independently if a write fails.

Usage:
    python scripts/run_pipeline.py                          # default region subset
    python scripts/run_pipeline.py --regions AEP COMED DOM
    python scripts/run_pipeline.py --all-regions
    python scripts/run_pipeline.py --horizons 1 24 --skip-leakage-check
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
from src.data_loader import load_region, summarize_region   # noqa: E402
from src.db_adapter import adapt_region, describe_capabilities  # noqa: E402
from src.feature_engineering import assert_no_future_leakage    # noqa: E402
from src.model import train_region                         # noqa: E402
from src.utils import ensure_output_dirs, utcnow, write_json, write_table  # noqa: E402

logger = logging.getLogger("run_pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LightGBM energy demand forecasting pipeline")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--regions", nargs="+", default=None, help="Region codes to run")
    p.add_argument("--all-regions", action="store_true", help="Run every configured region")
    p.add_argument("--horizons", nargs="+", type=int, default=None, help="Override horizons")
    p.add_argument("--skip-leakage-check", action="store_true",
                   help="Skip the perturbation leakage test (it is slow but worth running)")
    p.add_argument("--enable-anomaly", action="store_true",
                   help="Also run anomaly detection (off by default - forecasting is "
                        "verified as its own stage first)")
    p.add_argument("--output-dir", default=None, help="Override output directory")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)
    if args.enable_anomaly:
        cfg._data.setdefault("anomaly", {})["enabled"] = True

    out_root = Path(args.output_dir) if args.output_dir else cfg.output_dir
    dirs = ensure_output_dirs(out_root)

    regions: List[str] = (
        cfg.all_regions if args.all_regions else (args.regions or cfg.default_regions)
    )
    horizons = sorted(args.horizons) if args.horizons else cfg.horizons

    logger.info("#" * 78)
    logger.info("# LightGBM Energy Demand Forecasting - %s v%s",
                cfg["project.model_name"], cfg["project.model_version"])
    logger.info("# regions=%s", regions)
    logger.info("# horizons=%s hours", horizons)
    logger.info("# data=%s", cfg.raw_data_dir)
    logger.info("#" * 78)

    manifest: Dict[str, object] = {
        "started_at": utcnow().isoformat(),
        "model_name": cfg["project.model_name"],
        "model_version": cfg["project.model_version"],
        "code_version": cfg["project.code_version"],
        "feature_version": cfg["features.feature_version"],
        "capabilities": describe_capabilities(cfg),
        "config_path": str(cfg.path),
        "requested_regions": regions,
        "horizons": horizons,
        "regions": {},
        "failures": {},
    }

    all_forecasts: List[pd.DataFrame] = []
    all_evaluations: List[pd.DataFrame] = []
    all_anomalies: List[pd.DataFrame] = []
    all_importance: List[pd.DataFrame] = []
    run_records: List[Dict[str, object]] = []

    for region in regions:
        try:
            region_df = load_region(cfg, region)
            profile = summarize_region(region_df)

            leakage_report = None
            if not args.skip_leakage_check:
                # Run the perturbation test on the longest horizon - it exercises
                # the widest window and is therefore the strictest check.
                leakage_report = assert_no_future_leakage(region_df, cfg, max(horizons), n_probes=8)
                if not leakage_report["passed"]:
                    raise RuntimeError(
                        f"Leakage detected in features: {leakage_report['leaking_features']}"
                    )

            result = train_region(region_df, cfg, horizons)
            payload = adapt_region(result, cfg)

            # Persist per-region artefacts.
            write_table(payload["forecasts"], dirs["forecasts"] / f"{region}_forecasts")
            write_table(payload["evaluations"], dirs["evaluations"] / f"{region}_evaluations")
            if not payload["anomalies"].empty:
                write_table(payload["anomalies"], dirs["anomalies"] / f"{region}_anomalies")

            importance = pd.concat(
                [h.feature_importance for h in result.horizons.values()
                 if not h.feature_importance.empty],
                ignore_index=True,
            ) if result.horizons else pd.DataFrame()
            if not importance.empty:
                write_table(importance, dirs["reports"] / f"{region}_feature_importance",
                            also_csv=True)
                all_importance.append(importance)

            for horizon, hres in result.horizons.items():
                if hres.booster is not None:
                    hres.booster.save_model(
                        str(dirs["models"] / f"{region}_h{horizon}.txt"),
                        num_iteration=hres.best_iteration,
                    )

            all_forecasts.append(payload["forecasts"])
            all_evaluations.append(payload["evaluations"])
            if not payload["anomalies"].empty:
                all_anomalies.append(payload["anomalies"])
            run_records.append(result.run.to_dict())

            manifest["regions"][region] = {
                "run_uid": result.run.run_uid,
                "status": result.run.status,
                "data_profile": profile,
                "split_boundaries": result.split_boundaries,
                "trained_horizons": result.succeeded_horizons,
                "failed_horizons": result.failed_horizons,
                "n_features": result.run.n_features,
                "n_training_rows": result.run.n_training_rows,
                "leakage_check": leakage_report,
                "anomaly_thresholds": payload["anomaly_thresholds"],
                "n_forecast_rows": int(len(payload["forecasts"])),
                "n_anomaly_rows": int(len(payload["anomalies"])),
                "n_anomalies_flagged": int(payload["anomalies"]["is_anomaly"].sum())
                if not payload["anomalies"].empty else 0,
            }

        except Exception as exc:  # noqa: BLE001 - one region must not sink the run
            logger.exception("REGION %s FAILED", region)
            manifest["failures"][region] = f"{type(exc).__name__}: {exc}"

    if not run_records:
        logger.error("No region completed successfully.")
        write_json(dirs["reports"] / "run_manifest.json", manifest)
        return 1

    # Combined artefacts - what stage 2 and the comparison layer consume.
    combined_fc = pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()
    combined_ev = pd.concat(all_evaluations, ignore_index=True) if all_evaluations else pd.DataFrame()
    combined_an = pd.concat(all_anomalies, ignore_index=True) if all_anomalies else pd.DataFrame()

    write_table(combined_fc, dirs["forecasts"] / "ALL_forecasts", also_csv=False)
    write_table(combined_ev, dirs["evaluations"] / "ALL_evaluations")
    if not combined_an.empty:
        write_table(combined_an, dirs["anomalies"] / "ALL_anomalies", also_csv=False)
    if all_importance:
        write_table(pd.concat(all_importance, ignore_index=True),
                    dirs["reports"] / "ALL_feature_importance", also_csv=False)
    write_json(dirs["runs"] / "model_runs.json", run_records)

    manifest["finished_at"] = utcnow().isoformat()
    manifest["totals"] = {
        "regions_succeeded": len(run_records),
        "regions_failed": len(manifest["failures"]),
        "forecast_rows": int(len(combined_fc)),
        "evaluation_rows": int(len(combined_ev)),
        "anomaly_rows": int(len(combined_an)),
        "anomalies_flagged": int(combined_an["is_anomaly"].sum()) if not combined_an.empty else 0,
    }
    write_json(dirs["reports"] / "run_manifest.json", manifest)

    _print_summary(combined_ev, cfg, manifest)
    return 0 if not manifest["failures"] else 2


def _print_summary(evaluations: pd.DataFrame, cfg, manifest: Dict[str, object]) -> None:
    logger.info("")
    logger.info("=" * 78)
    logger.info("TEST-SET PERFORMANCE (the only split used for model comparison)")
    logger.info("=" * 78)
    if evaluations.empty:
        logger.info("(no evaluations)")
        return
    test = evaluations[evaluations["split_name"] == "test"].copy()
    cols = ["region_code", "horizon_hours", "n_observations", "mae", "rmse",
            "wape", "smape", "skill_vs_naive"]
    cols = [c for c in cols if c in test.columns]
    view = test[cols].sort_values(["region_code", "horizon_hours"])
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:,.3f}"):
        logger.info("\n%s", view.to_string(index=False))
    logger.info("")
    logger.info("Totals: %s", manifest["totals"])
    if manifest["failures"]:
        logger.warning("Failed regions: %s", manifest["failures"])
    logger.info("Artefacts written under: %s", cfg.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
