# Smart Building Energy Forecasting & Anomaly Detection

Hourly electricity demand forecasting and anomaly detection across 12 PJM
grid regions, built around one shared PostgreSQL schema that multiple
competing forecasting models write into. A SQL comparison layer picks the
best model per (region, horizon) automatically, an anomaly detector scores
whichever model wins, and a backend API + UI sit on top so the whole system
is queryable in plain English.

## The big idea

Every model package is independent (different code, different training
approach, even different languages of thought - statistical vs. gradient
boosting vs. transformer) but all of them write into the **same** tables
using the **same** contract (`contracts.py` in each package - identical
column shapes, identical validation rules). That's what makes it possible
for `v_model_ranking` / `v_best_model` to compare them fairly and pick a
winner per cell without any package knowing the others exist.

```
 LightGBM ----\
 DHR+ARIMA -----> forecasts / model_evaluations (shared schema) --> v_best_model --> API --> UI
 PatchTST -----/                                                        |
                                                                   anomaly_detection
```

## Folder map

```
backend_api/            FastAPI backend + single-page UI ("ask a question, get a table")
statistical_forecasting/  DHR+ARIMA - 1h-7d, Fourier/trend/ARIMA state, cheap, fast
lightgbm_forecasting/     LightGBM - 1h-24h, lag/rolling/calendar features, very strong short-range
patchtst_forecasting/     PatchTST - 24h-7d, scaled-down transformer, wins long-range
anomaly_detection/        Statistical + Isolation Forest hybrid, scores whichever model won
clean_dataset/            Raw per-region hourly demand CSVs, the common input to every model
```

## The three models, and why each exists

| Model | Trained range | Why it's there |
|---|---|---|
| LightGBM | 1-24h | Very strong with lag/rolling/calendar features - wins almost every short-horizon cell |
| DHR+ARIMA | 1-168h (7 days) | Statistical baseline, Fourier+trend+ARIMA state, fast and cheap, competes at long range |
| PatchTST | 24-168h (7 days) | Patch-based transformer, direct multi-horizon output, wins long-range because it doesn't compound error the way ARIMA's rolling state does |

A fourth model (TFT) was deliberately **not** built. A 4-month DHR-ARIMA
extension was tested first to see if an existing model could reach that far
without a new architecture - it couldn't (WAPE climbs past 90% and one
region exceeds 100%). That negative result is documented in
`statistical_forecasting/LONG_HORIZON_EXPERIMENT.md` and is the reason the
model roster stopped at three.

## Running everything, in order

Each model package is a two-stage pipeline: **Stage 1** (`run_pipeline.py`)
trains and writes Parquet/CSV outputs to disk, no database needed. **Stage
2** (`run_db_integration.py`) loads those outputs into PostgreSQL. Splitting
these means a database outage never costs a training run.

```powershell
# 1. Each model package (repeat for lightgbm_forecasting, statistical_forecasting, patchtst_forecasting)
cd <package>
pip install -r requirements.txt
python scripts\run_pipeline.py --all-regions
$env:DATABASE_URL = "postgresql://postgres@localhost:5432/energy_forecasting"
python scripts\run_db_integration.py

# 2. Anomaly detection (after at least one model is loaded)
cd anomaly_detection
pip install -r requirements.txt
$env:DATABASE_URL = "postgresql://postgres@localhost:5432/energy_forecasting"
python scripts\run_anomaly_detection.py --all-regions

# 3. Backend API
cd backend_api
pip install -r requirements.txt
$env:DATABASE_URL = "postgresql://postgres@localhost:5432/energy_forecasting"
uvicorn app.main:app --reload --port 8000

# 4. UI - just open backend_api/ui/index.html in a browser
#    (API base URL defaults to http://127.0.0.1:8000)
```

## Verifying it's wired correctly

Each package ships its own `sql/20_validation_checks.sql` (duplicates,
orphaned foreign keys, null/invalid values - all should read 0). The
cross-model comparison lives in `sql/10_model_comparison.sql`
(`v_model_performance`, `v_model_ranking`, `v_best_model`,
`v_model_coverage`, `v_selected_forecast`) and is identical across every
package, since the schema is shared.

Two queries worth running to see the whole system prove itself:

```sql
-- Who wins each long-horizon cell, and by how much?
SELECT model_name, horizon_hours, ROUND(wape::numeric,2) AS wape_pct, mae, n_observations
FROM v_model_ranking
WHERE region_code = 'AEP' AND horizon_hours IN (24, 72, 168)
ORDER BY horizon_hours, model_rank;

-- The automatic handoff: LightGBM at 1-24h, PatchTST/DHR-ARIMA beyond
SELECT horizon_hours, best_model_name, COUNT(*) AS n_regions_won
FROM v_best_model
GROUP BY horizon_hours, best_model_name
ORDER BY horizon_hours, best_model_name;
```

Or just ask the UI: *"which model is best for AEP"* pulls the same ranking,
live, as a table.

## Talking to the system in plain English

The backend's `/ask` endpoint (and the UI built on it) uses a small
rule-based parser (`backend_api/app/nlu.py`) - no LLM, no API key required.
It recognizes forecast requests ("next 3 hours for AEP"), historical lookups
("previous day same time for AEP"), model comparisons ("which model is best
for PJME"), and anomaly queries ("any critical anomalies in DOM"). The
parsing layer is deliberately isolated from the database layer so a real LLM
can be swapped in later without touching anything downstream - see the
README inside `backend_api/` for exactly where that seam is.
