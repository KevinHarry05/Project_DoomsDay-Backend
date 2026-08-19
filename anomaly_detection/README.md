# Anomaly Detection (Stage 3)

Hybrid statistical + Isolation Forest anomaly detector, ported from the
`Copy_of_EnerSight_Anomaly_Detection_2.ipynb` reference notebook. This is the
stage that runs AFTER LightGBM and DHR-ARIMA (and later PatchTST/TFT) have
both loaded their forecasts into Postgres.

## What changed vs. the notebook

The notebook's flow was: train a model in one Colab session -> export a
"final_forecast_outputs" zip -> manually upload that zip into a second Colab
session -> run detection on the uploaded file. There is no upload step here.

This package reads its only input directly from `v_selected_forecast`, the
SQL view already built in `lightgbm_forecasting/sql/10_model_comparison.sql`.
That view resolves, per `(region, horizon)`, whichever model currently has the
lowest test-set WAPE, so:

- Anomaly detection is never hardcoded to "always score LightGBM" or "always
  score DHR-ARIMA" - it automatically follows the winner.
- If DHR-ARIMA overtakes LightGBM for AEP at h=24 next week, this script's
  next run scores AEP/h=24 against DHR-ARIMA's residuals with zero code
  changes.
- Every anomaly row already carries the exact `forecast_id` and
  `model_run_id` of the forecast that produced it, so results are always
  traceable back to one specific model run - never a blend of models.

## Algorithm (per `(region_code, horizon_hours)` group)

1. **Statistical score** (`src/statistical_features.py`): robust (median/MAD)
   historical baseline, a 24h rolling local baseline, a capped z-score on
   each, a capped residual-percentage term, and a ramp (hour-over-hour
   change) term - combined into `statistical_anomaly_score` with weights
   0.30 / 0.30 / 0.20 / 0.20 (historical / rolling / percentage / ramp).
2. **Isolation Forest** (`src/hybrid_scoring.py`): one forest per group,
   fit on cyclical calendar features only (hour/day-of-week/month as
   sin/cos) - never on demand magnitude, so differently-sized regions
   aren't biased against each other.
3. **Hybrid flag**: `Strong_Anomaly` when both detectors agree,
   `Statistical_Only_Candidate` / `IF_Only_Candidate` otherwise. Severity is
   derived from which combination fired plus the composite score.
4. **Event grouping**: consecutive flagged hours (gap <= 1h) within one
   `(region, horizon)` are merged into one `Event_Candidate_ID` with
   start/end/duration/observation count/max & mean scores - written to
   `outputs/anomaly_events.{parquet,csv}` (not a DB table; a jury-facing
   summary artefact).

One deliberate generalization from the notebook: every baseline here is
fit per `(region_code, horizon_hours)`, not just per region, because a 24h
DHR-ARIMA residual and a 1h LightGBM residual live on completely different
scales even for the same region.

## Running it (against your LOCAL Postgres, per the storage-cap decision)

```powershell
cd C:\Users\kani2\OneDrive\Documents\BACKEND\anomaly_detection
pip install -r requirements.txt
$env:DATABASE_URL = "postgresql://<local_user>:<local_pass>@localhost:5432/energy_forecasting"
python scripts\run_anomaly_detection.py --all-regions
```

Requires `sql/10_model_comparison.sql` to already be applied (it is, on your
local DB) and at least one model's forecasts to already be loaded (both
LightGBM and DHR-ARIMA are, once you run DHR-ARIMA's `run_db_integration.py`
locally).

Flags:
- `--regions AEP EKPC` / `--horizons 1 24` to scope a run.
- `--no-write-db` to score and inspect `outputs/anomalies.csv` without
  touching the database (safe dry run).

## Output

- `outputs/anomalies.{parquet,csv}` - every scored row, contract-shaped
  (`ANOMALY_CONTRACT_COLUMNS`), whether flagged or not.
- `outputs/anomaly_events.{parquet,csv}` - grouped events, for a jury-facing
  "here are the 8 distinct incidents we found" view rather than a wall of
  individual hourly rows.
- `anomalies` and `alerts` tables in Postgres, upserted idempotently on
  `(forecast_id, detection_method)` - safe to re-run after new forecasts
  land.
