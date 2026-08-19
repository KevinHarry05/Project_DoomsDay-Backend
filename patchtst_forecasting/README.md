# PatchTST Energy Demand Forecasting

Deep-learning track of the Smart Building Energy Forecasting project. A
patch-based transformer trained as ONE shared/global model pooled across all
12 PJM regions, targeting the long-horizon band (24h - 7 days) called out for
PatchTST specifically in the project's model-selection table.

```
DHR + ARIMA   1-24 hours and long-term trend forecasting   Strong on regular seasonality/trend
LightGBM      1-24 hours                                    Very strong with lag/rolling/calendar features
PatchTST      24 hours -> several days/weeks                Designed for long sequence, multi-step forecasting  <- this package
TFT           24 hours -> weeks/months                       Multi-horizon, historical + known-future features
```

## Scope and limitations (read this first)

A teammate had already built a PatchTST/TFT notebook with a well-designed
architecture (`PatchTSTLightweight`: `d_model=128, n_heads=8, n_layers=3`,
1.37M params, 30-epoch plan on a GPU). That run was checked before any of
this package was written, and **it never finished**: training stopped mid
batch 5,500/6,774 of epoch 1 of 30, no checkpoint was ever saved, and it
depends on a pre-built "Feature_Engineered_Dataset" file that does not exist
in this environment (only the raw `clean_dataset` CSVs do). There was nothing
reusable to load - only the architectural ideas.

This package is a **from-scratch, CPU-feasible rebuild** of those ideas,
sized to actually finish training in this sandbox (2 CPU threads, no GPU),
per the explicit instruction to go with whatever is quickest to build. That
trade-off is real and worth being upfront about in front of a jury:

- **Model size**: `d_model=32, n_heads=2, n_layers=2` (~26K params) vs. the
  notebook's `d_model=128, n_heads=8, n_layers=3` (~1.37M params) - roughly
  50x smaller.
- **Training data per region**: origins are spaced 24h apart (not every hour)
  and capped at 700 per region (~8,400 pooled samples across 12 regions) vs.
  the notebook's 867K training sequences - a deliberate, documented
  subsample, not silent truncation (see `sequence.max_origins_per_region` in
  `config.yaml`, and `training.n_origins` in the run manifest).
- **Epochs**: 12 max epochs, wall-clock capped at 25 minutes, vs. the
  notebook's 30-epoch plan. The actual run finished all 12 epochs in ~13
  seconds - well inside budget - because the model and data are both small.
- **Result**: test-set WAPE in the 4-15% range across regions for horizons
  24h-168h. That is materially worse than a properly-sized, fully-trained
  GPU model would likely achieve, but it is a genuine, working, trained
  model producing non-degenerate predictions (verified: predictions track
  actual demand with real variance, not a flatlined mean - see the
  verification step below), not a placeholder.

If a full-scale, GPU-trained version becomes available later (e.g. the
teammate finishes their notebook), swapping it in only requires a new
`db_adapter.py`/`model.py` pair that emits the same `contracts.py` shapes -
nothing downstream (schema, comparison views, anomaly detection, API, UI)
needs to change. That is the whole point of the shared contract.

## Architecture

```
[lookback=168h, 8 features] input window
    -> overlapping patches (patch_length=16, stride=8 -> 20 patches)
    -> linear patch embedding -> d_model=32
    -> Transformer encoder (2 layers, 2 heads, dim_feedforward=64)
    -> flatten -> linear projection head
    -> [7] direct multi-horizon output (24, 48, 72, 96, 120, 144, 168h)
```

Direct multi-step forecasting (one forward pass predicts all 7 horizons at
once), not autoregressive - matches the notebook's own design and is the
reason PatchTST-style models are used for long horizons where autoregressive
error accumulation would otherwise dominate.

**One shared model, not 12.** All 12 regions' training sequences are pooled
into a single training set; one set of weights is learned across all of
them, then evaluated separately per region. This is both far cheaper than 12
independent transformers and closer to how PatchTST is normally deployed
(learn cross-series structure, not per-series noise). Every `model_runs` row
this produces carries `metadata.pooled_global_model = true` and
`metadata.n_regions_pooled` so this is never mistaken for 12 independently
fit models when someone is reading raw database rows later.

**Features per timestep** (all available at inference time, no future
leakage): `demand_scaled` (per-region z-score, scaler fit on the train split
only) plus cyclical calendar encodings (`hour_sin/cos`, `dow_sin/cos`,
`month_sin/cos`, `is_weekend`).

**Horizons**: `[24, 48, 72, 96, 120, 144, 168]` - deliberately the "several
days/weeks" band from the model-selection table, non-overlapping with
LightGBM's 1-24h range, so `v_model_ranking`/`v_best_model` reflect each
model competing in the range it was actually designed for rather than three
models fighting over hour 1.

## Files

```
config/config.yaml       All tunables - sequence/model/training sizing is
                          fully documented inline with the CPU-budget rationale.
src/
  config.py               Dotted-access YAML config loader (shared convention).
  contracts.py             Cross-model contract dataclasses + validators (identical
                            to DHR+ARIMA / LightGBM - the whole point is one shape).
  data_loader.py            Loads <REGION>_clean.csv, reindexes to a full hourly grid.
  features.py                Calendar features, per-region scaler, sequence/origin
                              construction, chronological train/val/test split.
  model.py                    PatchTST nn.Module, pooled training loop
                               (train_global), per-region evaluation (evaluate_region).
  evaluation.py                 mae/rmse/wape/smape - identical formulas to every
                                 other track.
  db_adapter.py                  RegionResult -> contracts.py frames.
  db_writer.py                    PostgreSQL writer - byte-identical to the other
                                   tracks (schema-introspecting, idempotent, FK-ordered).
scripts/
  run_pipeline.py            Stage 1: train + evaluate, writes outputs/ (no DB).
  run_db_integration.py      Stage 2: load outputs/ into PostgreSQL.
sql/                        Same migration / comparison / validation SQL as the
                             other tracks (schema is shared, so the SQL is too).
```

## Running it

```bash
pip install -r requirements.txt

# Stage 1: train + evaluate (no database needed)
python scripts/run_pipeline.py --all-regions

# Stage 2: load into PostgreSQL (local recommended; DATABASE_URL only)
export DATABASE_URL="postgresql://postgres@localhost:5432/energy_forecasting"
python scripts/run_db_integration.py
```

Stage 1 writes `outputs/forecasts/<REGION>_forecasts.parquet`,
`outputs/evaluations/<REGION>_evaluations.parquet`,
`outputs/runs/model_runs.json`, and `outputs/reports/run_manifest.json`
(includes the training curve summary and, if any region failed to load or
evaluate, exactly why - no silent skips).

## Verification performed before shipping

- Training loss decreased monotonically-ish across all 12 epochs
  (0.29 -> 0.19 train loss, 0.22 -> 0.19 val loss) - not flatlined, not
  diverging.
- All 12 regions succeeded, 0 failures, 8,904 forecast rows / 252 evaluation
  rows written.
- Test-set WAPE ranges ~4-15% across regions/horizons - in the same
  ballpark as DHR+ARIMA's long-horizon numbers, which is a reasonable sanity
  bound (neither suspiciously perfect nor degenerate).
- Spot-checked AEP h=168 predictions: mean 14,109 MW vs actual mean 13,445
  MW, std 1,407 MW (not a flatlined constant), min/max span 12,106-18,609 MW
  tracking the actual 10,746-20,513 MW range - genuine variance, not a
  mean-predictor collapsing under load.
