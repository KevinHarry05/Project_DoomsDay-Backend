# LightGBM Energy Demand Forecasting & Anomaly Detection

Machine-learning track of the **Smart Building Energy Forecasting & Anomaly
Detection** project. Forecasts hourly electricity demand for 12 PJM regions at
multiple horizons, flags abnormal consumption, and writes everything into the
shared PostgreSQL schema through a standardized contract.

This package is one of three competing forecasting tracks. It does **not** feed
DHR+ARIMA or PatchTST/TFT and is not fed by them — all three run independently
on the same input and converge only at the database.

```
                    cleaned regional demand (clean_dataset/)
                                    |
                    +---------------+---------------+
                    |               |               |
               DHR + ARIMA      LightGBM      PatchTST / TFT
             (statistical)   (this package)   (deep learning)
                    |               |               |
              statistical_    lightgbm_       patchtst_ /
                adapter        adapter        tft_adapter
                    |               |               |
                    +---------------+---------------+
                                    |
                    standardized forecast contract
                                    |
                              PostgreSQL
                    model_runs -> forecasts -> model_evaluations
                                    |
                       model comparison (per region + horizon)
                                    |
                          selected forecast -> residual
                                    |
                       anomalies -> alerts -> backend API
```

---

## 1. Quick start

```bash
pip install -r requirements.txt

# Stage 1 - train, evaluate, detect anomalies, write artefacts to disk
python scripts/run_pipeline.py --regions AEP COMED DAYTON

# Stage 2 - load those artefacts into PostgreSQL
export DATABASE_URL="postgresql://postgres@localhost:5432/energy_forecasting"
python scripts/run_db_integration.py --dry-run     # preflight first
python scripts/run_db_integration.py
```

Before the first stage-2 run, apply the additive migration:

```bash
psql -U postgres -d energy_forecasting -f sql/00_recommended_migrations.sql
psql -U postgres -d energy_forecasting -f sql/10_model_comparison.sql
```

The two stages are separate on purpose. A database outage, a wrong credential or
a missing column should never cost a long training run — stage 2 reads what
stage 1 left on disk and is safe to retry.

---

## 2. Layout

```
lightgbm_forecasting/
├── config/config.yaml              every tunable, one place
├── src/
│   ├── config.py                   config loader
│   ├── contracts.py                cross-model contracts + validators
│   ├── data_loader.py              read clean_dataset, hourly grid, integrity
│   ├── feature_engineering.py      leakage-safe features + self-check
│   ├── splits.py                   chronological split with embargo
│   ├── model.py                    LightGBM training, direct multi-horizon
│   ├── evaluation.py               MAE / RMSE / WAPE / sMAPE + naive baseline
│   ├── anomaly.py                  robust residual anomaly detection
│   ├── db_adapter.py               lightgbm_adapter -> contract
│   └── db_writer.py                idempotent PostgreSQL writes
├── scripts/
│   ├── run_pipeline.py             stage 1
│   └── run_db_integration.py       stage 2
├── sql/
│   ├── 00_recommended_migrations.sql   additive, idempotent
│   ├── 01_baseline_schema_reference.sql  reference only, do NOT run on prod
│   ├── 02_inspect_live_schema.sql      verify before changing anything
│   ├── 10_model_comparison.sql         best-model selection views
│   └── 20_validation_checks.sql        post-integration checks
└── outputs/                        stage-1 artefacts
```

---

## 3. Input

Confirmed against the actual files, not assumed:

| Property | Value |
|---|---|
| Location | `clean_dataset/<REGION>_clean.csv` |
| Columns | `Load_Area, Datetime_UTC, Datetime_EPT, Demand_MW, Missing_Flag` |
| Rows | 1,090,176 across 12 regions |
| Coverage | 1998-04-01 → 2018-08-03 (varies by region) |
| Duplicates | none |

**This data is cleaned, not feature-engineered.** The five columns match
`demand_staging` exactly. All modelling features are therefore built inside this
package, in `feature_engineering.py`, versioned as `feature_version` and stamped
onto every `model_run`. When a shared feature-engineered dataset becomes
available, this module is the thing to replace — nothing else changes.

Two coverage facts that affect splitting:

- `NI` ends 2011-01-01, exactly where `COMED` begins.
- `PJM_Load` ends 2002-01-01, exactly where `PJME` begins.

These are renamed/absorbed zones, not gaps. A fixed calendar split date would
produce an empty test set for both, so the default split is fractional per
region and the date-based mode falls back automatically.

`Datetime_EPT` is deliberately unused. `Datetime_UTC` is the canonical instant;
touching the local-time column would reintroduce the ambiguity the cleaning step
already resolved.

---

## 4. Method

### Direct multi-horizon
One independent booster per `(region, horizon)`, horizons 1 / 6 / 12 / 24 h.
Model `h=24` maps features known at `t` directly onto demand at `t+24`.

The recursive alternative — a 1-step model fed its own output 24 times — is
rejected because error compounds and the model's inputs drift into a
distribution it never trained on. Direct costs 4× training time and buys honest
24-hour error.

### Per-region models
Not one global model with region as a categorical. Demand scale spans two orders
of magnitude (DAYTON ~2 GW mean, PJME ~32 GW), coverage windows differ by years,
and `model_runs.region_id` already makes a run region-scoped — so per-region
boosters map onto the schema without contortion.

### Features (67 demand-derived + calendar)
- Lags of demand at the origin: 0–504 h
- Rolling mean / std / min / max over 3–720 h windows ending at `t`
- EWM levels, differences and percent changes at 1 / 24 / 168 h
- Position within recent range, volatility and level ratios
- Calendar of the **target** hour: hour, day-of-week, month, interactions
- Fourier terms: daily, weekly, yearly
- US federal holidays plus day-before / day-after flags

No scaling. Gradient-boosted trees are scale-invariant, so a scaler would add a
fit/transform leakage risk for no benefit.

No weather. The source has no temperature column, and temperature is the
dominant exogenous driver of electricity demand. This is the single
highest-value future improvement, and inventing it now would be fabricating data
the source does not contain.

### Leakage control
The rule: a forecast made at `t` for `t+h` may use demand observations at or
before `t`, and calendar attributes of `t+h`. Nothing else.

| Risk | Control |
|---|---|
| Future demand as a feature | Features indexed at `t`; only the label is shifted (`shift(-h)`) |
| Rolling/lag windows reaching forward | All windows end at `t`; enforced by grid reindexing |
| Scaler fitted on all data | No scaler used |
| Random splits | Chronological only |
| Train/val bleed via the label | 24-hour embargo at each boundary |
| Anomaly thresholds seeing test | Thresholds fitted on validation only |

`assert_no_future_leakage()` proves it rather than asserting it: it corrupts the
series after a cut instant, rebuilds features, and confirms no row at or before
the cut changed. It runs by default in stage 1 and currently passes across all
67 demand-derived features.

### Splits
Chronological 70 / 15 / 15 with a 24-hour embargo between splits. Boundaries are
computed once per region on the shortest-horizon frame and reused across all
horizons, so every horizon is scored over the same wall-clock window and the
numbers are actually comparable.

### Metrics
MAE, RMSE, WAPE, sMAPE — pinned in `evaluation.py` so all three tracks are
scored by identical formulas. WAPE is primary: MAPE divides each error by its
own actual, so one low-demand hour can dominate; WAPE's single aggregate
denominator is stable and comparable across regions of very different size.

Every run also reports a **seasonal-naive baseline** (same hour last week) and
`skill_vs_naive`. A model that can't beat that isn't earning its keep.

### Anomaly detection
Robust z-score on the forecast residual:

```
residual = actual - predicted
z        = |residual - median(val residuals)| / (1.4826 * MAD(val residuals))
```

Median/MAD rather than mean/std, because the statistics are estimated from a
sample that itself contains the outliers being hunted — a few large excursions
inflate a standard deviation enough to hide the next one.

Thresholds are conditioned on **hour of day**. A 400 MW miss at the 07:00 ramp
is ordinary; the same miss at 03:00 is not. Pooling all hours over-flags peaks
and under-flags overnight.

Fitted on **validation**, not training. A boosted tree's in-sample residuals are
far tighter than its out-of-sample ones; the first run fitted on train and
flagged 20.5% of test hours, which is a broken threshold rather than a finding.
Validation is out-of-sample and strictly before test, so it is realistic and
leakage-free. The rate landed at 2.6%.

Detection runs at **one** configured horizon (default 1 h) and every row carries
its `detection_method` and owning `run_uid`, so predictions from different
models can never be mixed into one residual series.

---

## 5. Contracts

`src/contracts.py` defines what the sibling adapters must also produce. Field
names mirror the PostgreSQL columns one-to-one.

**Forecast** — `model_name, run_uid, region_code, forecast_timestamp,
target_timestamp, horizon_hours, predicted_demand_mw, actual_demand_mw,
split_name`

Validated on every emit: timestamps parse as UTC, `horizon_hours > 0`,
predictions non-null, no duplicates on the uniqueness key, and
`horizon_hours == target_timestamp - forecast_timestamp`. That last check has
teeth — an adapter that writes the target into both timestamp columns looks
completely fine until something compares them.

**Evaluation** — `run_uid, region_code, horizon_hours, split_name,
evaluation_start, evaluation_end, n_observations, mae, rmse, wape, smape,
training_time_seconds, inference_time_ms`

**Anomaly** — `run_uid, region_code, timestamp_utc, horizon_hours,
actual_demand_mw, predicted_demand_mw, residual_mw, deviation_percent,
anomaly_score, severity, is_anomaly, anomaly_direction, reason,
detection_method`

`deviation_percent`, `anomaly_direction` and `reason` exist because the DHR+ARIMA
pipeline already produces them. Rather than discard them to fit the basic table,
LightGBM emits the same three, so the anomaly view is uniform across models.

### On `model_run_id`
It's a database-assigned surrogate key, so it cannot exist before the write. The
adapter carries `run_uid` — a deterministic SHA-256 over model name, version,
code version, feature version, region, training window and horizon set — and
`db_writer` exchanges it for the real `model_run_id` after inserting the parent
row. That keeps the write order honest without any adapter inventing an
identifier.

### Model capability declaration
`describe_capabilities()` publishes supported horizons, required input window,
output format and region handling into the run manifest. An unsupported horizon
is recorded as absent rather than filled with a fabricated number.

| | |
|---|---|
| Supported horizons | 1, 6, 12, 24 h |
| Input window | 720 h of history required |
| Generation | direct multi-horizon, one booster per horizon |
| Region handling | one `model_run` per `region_code` |
| Exogenous | none |
| Future forecasts without actuals | supported (`actual_demand_mw` NULL) |

---

## 6. Database integration

Write order follows the foreign keys:

```
regions (read only) -> model_runs -> forecasts -> model_evaluations -> anomalies -> alerts
```

`regions` is reference data. This package reads it and errors on an unknown
code; it never inserts, because that is how duplicate region rows happen.

### Idempotency
Every write is an upsert on a natural business key:

| Table | Conflict key |
|---|---|
| `model_runs` | `run_uid` |
| `forecasts` | `(model_run_id, region_id, target_timestamp, horizon_hours)` |
| `model_evaluations` | `(model_run_id, horizon_hours, split_name)` |
| `anomalies` | `(forecast_id, detection_method)` |
| `alerts` | `anomaly_id` |

Verified: running the integration twice produces identical row counts, the same
`model_run_id`, and zero duplicate alerts.

### No invented columns
`db_writer` introspects `information_schema` at startup, writes only columns
that actually exist, and names precisely which recommended ones are missing
along with the migration that adds them. Required columns missing → hard stop
before a single row is written.

### Error isolation
Each region commits independently, so a failure in region 7 keeps regions 1–6.
A failed horizon is recorded as `status='PARTIAL'` with a `failure_reason`
rather than silently vanishing — which is what makes "PatchTST failed, the other
three are fine" visible to the comparison layer instead of looking like a model
nobody ran.

### Backend connection
Credentials come only from `DATABASE_URL`, read from the environment, never
logged, never in `config.yaml`, never in git. The connection log line prints
user/host/port/database and no password.

```bash
# local
export DATABASE_URL="postgresql://postgres@localhost:5432/energy_forecasting"

# supabase - session pooler, TLS required
export DATABASE_URL="postgresql://postgres.<project-ref>:<password>@aws-0-ap-south-1.pooler.supabase.com:5432/postgres?sslmode=require"
```

The backend holds this server-side and exposes only its own API. The frontend
never receives a database credential.

---

## 7. Model comparison

`sql/10_model_comparison.sql` holds the selection rule in one place so no
service can quietly hardcode a favourite. No model is named anywhere in it.

| | |
|---|---|
| Scope | per `(region_id, horizon_hours)` — the winner may differ per cell |
| Eligible | `split_name = 'test'` only, `n_observations >= 100` |
| Primary | WAPE ascending |
| Tie-break | MAE, then RMSE, then most recent run |
| Eligibility | `model_runs.status IN ('SUCCESS','PARTIAL')` |

Views: `v_model_performance`, `v_model_ranking`, `v_best_model`,
`v_model_coverage`, `v_selected_forecast`.

`v_selected_forecast` joins `forecasts` to `v_best_model`, which is what makes
it structurally impossible to mix models within one residual series.

`v_model_coverage` answers "which cells lost a competitor?", so an absent model
reads as absent rather than as one that happened not to win.

Verified against a two-model scratch database — the winner correctly differed by
horizon, chosen from the metrics rather than from a hardcoded name.

---

## 8. Results

AEP, chronological test set (18,191 hours, 2016-07 → 2018-08):

| Horizon | MAE (MW) | RMSE (MW) | WAPE | sMAPE | Skill vs naive |
|---|---|---|---|---|---|
| 1 h | 104.3 | 138.0 | 0.70% | 0.71% | 0.93 |
| 24 h | 685.6 | 908.8 | 4.62% | 4.55% | 0.53 |

Anomalies at h=1: 470 of 18,191 hours flagged (2.58%) — 383 LOW, 71 MEDIUM,
9 HIGH, 7 CRITICAL.

The 1 h → 24 h degradation is expected and honest: without weather, a day-ahead
forecast is extrapolating load shape from history alone.

---

## 9. Verification performed

| Check | Result |
|---|---|
| Feature leakage perturbation test | PASS — 67 features, 2 horizons |
| Contract validation on emit | PASS |
| Migration SQL applies cleanly | PASS |
| Comparison views build and rank | PASS |
| Full DB write path | PASS — run, forecasts, evals, anomalies, alerts |
| Idempotency (run twice) | PASS — identical counts |
| `sql/20_validation_checks.sql` | PASS — all 13 checks |

Two real bugs were found and fixed by this process: the anomaly threshold split
(20.5% → 2.6% flag rate) and a partial unique index that `ON CONFLICT` could not
match.

---

## 10. Known limitations

1. **No weather.** The largest accuracy gap, especially at 24 h. Adding a
   temperature join is the highest-value next step.
2. **h=1 does not early-stop within 3,000 rounds** — it is still improving at
   the cap, so `num_boost_round` has headroom.
3. **No hyperparameter search.** Parameters are sensible defaults, identical
   across regions. A per-region time-series-CV search would likely help.
4. **No prediction intervals.** LightGBM quantile regression would give the
   bounds that the long-horizon table anticipates.
5. **Backtest, not live.** Every forecast currently carries an actual. Genuine
   forward forecasting with `actual_demand_mw` NULL is supported by the contract
   but not yet exercised.

---

## 11. Integration checklist for the other tracks

Each remaining model needs an adapter exposing the same three functions:

```python
to_forecast_contract(result, cfg)    -> DataFrame[FORECAST_CONTRACT_COLUMNS]
to_evaluation_contract(result, cfg)  -> DataFrame[EVALUATION_CONTRACT_COLUMNS]
to_anomaly_contract(result, cfg)     -> (DataFrame[ANOMALY_CONTRACT_COLUMNS], dict)
describe_capabilities(cfg)           -> dict
```

For DHR+ARIMA, whose native output is `Datetime_UTC, actual_demand, forecast,
residual`, the mapping is:

| Native | Contract |
|---|---|
| `Datetime_UTC` | `target_timestamp` |
| `actual_demand` | `actual_demand_mw` |
| `forecast` | `predicted_demand_mw` |
| `region` | `region_code` |
| `"DHR_ARIMA"` | `model_name` |

`forecast_timestamp` and `horizon_hours` must come from how the run actually
produced each value, not be back-filled with a constant. Once the adapter
returns the contract shape, `db_writer` and every comparison view work
unchanged — no other code moves.
