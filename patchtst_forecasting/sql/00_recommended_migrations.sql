-- ============================================================================
-- Recommended migrations for multi-model integration
--
-- RUN THIS ON THE LOCAL DATABASE FIRST, VERIFY, THEN ON SUPABASE.
--
-- Every statement is additive and idempotent. Nothing here drops a column,
-- drops a table, or rewrites existing data. The local database stays a valid
-- backup throughout.
--
-- Scope note: these columns are what the LightGBM track needs in order to write
-- without discarding information it actually produces. They are equally needed
-- by the DHR+ARIMA and PatchTST/TFT tracks, since all three share the contract.
-- ============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. model_runs : run identity, versioning and failure recording
-- ---------------------------------------------------------------------------
-- run_uid is the idempotency key. It is a deterministic hash of
-- (model_name, model_version, code_version, feature_version, region_code,
--  training window, horizon set). Re-running identical code on identical data
-- produces the same hash, so the write updates in place instead of appending a
-- duplicate run. Without this column the integration cannot be safely re-run.
ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS run_uid        TEXT;
ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS model_version  TEXT;

-- status lets a failed or partial run be recorded rather than silently absent,
-- which is what makes "PatchTST failed, the other three are fine" visible to
-- the comparison layer instead of looking like a model that was never run.
ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS status          TEXT DEFAULT 'SUCCESS';
ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS failure_reason  TEXT;
ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS n_features      INTEGER;
ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS n_training_rows BIGINT;

-- Free-form per-model detail (hyperparameters, supported vs trained horizons,
-- strategy). JSONB keeps model-specific fields out of the shared columns.
ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS metadata        JSONB;

-- Deliberately NOT a partial index. `ON CONFLICT (run_uid)` cannot match a
-- partial unique index unless the statement repeats the predicate, which is
-- easy to forget and fails at write time rather than migration time. A plain
-- unique index is safe here because PostgreSQL treats NULLs as distinct, so any
-- pre-existing model_runs rows with run_uid IS NULL coexist without conflict.
CREATE UNIQUE INDEX IF NOT EXISTS uq_model_runs_run_uid
    ON model_runs (run_uid);

ALTER TABLE model_runs DROP CONSTRAINT IF EXISTS chk_model_runs_status;
ALTER TABLE model_runs ADD CONSTRAINT chk_model_runs_status
    CHECK (status IS NULL OR status IN ('SUCCESS', 'PARTIAL', 'FAILED', 'RUNNING'));


-- ---------------------------------------------------------------------------
-- 2. forecasts : the uniqueness constraint the design already calls for
-- ---------------------------------------------------------------------------
-- This is what makes ON CONFLICT possible and what stops a re-run from doubling
-- the table. Create it CONCURRENTLY on a large populated table if needed.
CREATE UNIQUE INDEX IF NOT EXISTS uq_forecasts_run_region_target_horizon
    ON forecasts (model_run_id, region_id, target_timestamp, horizon_hours);


-- ---------------------------------------------------------------------------
-- 3. model_evaluations : split identity and sample size
-- ---------------------------------------------------------------------------
-- split_name is the important one. Without it, a train-set MAE and a test-set
-- MAE are indistinguishable rows, and a model comparison that accidentally
-- ranks on training metrics will look completely normal while being wrong.
ALTER TABLE model_evaluations ADD COLUMN IF NOT EXISTS split_name     TEXT;
ALTER TABLE model_evaluations ADD COLUMN IF NOT EXISTS n_observations BIGINT;

-- Supporting diagnostics: signed bias exposes systematic over/under-forecasting
-- that MAE hides; skill_vs_naive says whether the model beats seasonal-naive.
ALTER TABLE model_evaluations ADD COLUMN IF NOT EXISTS bias           DOUBLE PRECISION;
ALTER TABLE model_evaluations ADD COLUMN IF NOT EXISTS r2             DOUBLE PRECISION;
ALTER TABLE model_evaluations ADD COLUMN IF NOT EXISTS skill_vs_naive DOUBLE PRECISION;

UPDATE model_evaluations SET split_name = 'test' WHERE split_name IS NULL;

ALTER TABLE model_evaluations DROP CONSTRAINT IF EXISTS chk_model_evaluations_split;
ALTER TABLE model_evaluations ADD CONSTRAINT chk_model_evaluations_split
    CHECK (split_name IS NULL OR split_name IN ('train', 'val', 'test'));

CREATE UNIQUE INDEX IF NOT EXISTS uq_model_evaluations_run_horizon_split
    ON model_evaluations (model_run_id, horizon_hours, split_name);


-- ---------------------------------------------------------------------------
-- 4. anomalies : preserve the richer fields the statistical pipeline produces
-- ---------------------------------------------------------------------------
-- The DHR+ARIMA anomaly pipeline already emits deviation_percent,
-- anomaly_direction and reason. Rather than discard them to fit the basic
-- table, add them - LightGBM populates the same three, so the anomaly view is
-- uniform across models.
ALTER TABLE anomalies ADD COLUMN IF NOT EXISTS deviation_percent DOUBLE PRECISION;
ALTER TABLE anomalies ADD COLUMN IF NOT EXISTS anomaly_direction TEXT;
ALTER TABLE anomalies ADD COLUMN IF NOT EXISTS reason            TEXT;

ALTER TABLE anomalies DROP CONSTRAINT IF EXISTS chk_anomalies_direction;
ALTER TABLE anomalies ADD CONSTRAINT chk_anomalies_direction
    CHECK (anomaly_direction IS NULL OR anomaly_direction IN ('OVER', 'UNDER', 'NONE'));

ALTER TABLE anomalies DROP CONSTRAINT IF EXISTS chk_anomalies_severity;
ALTER TABLE anomalies ADD CONSTRAINT chk_anomalies_severity
    CHECK (severity IS NULL OR severity IN ('NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'));

-- One anomaly row per (forecast, detection method). Keying on forecast_id alone
-- would stop two detectors from ever scoring the same forecast, which is
-- exactly what we want to allow when comparing detection approaches.
CREATE UNIQUE INDEX IF NOT EXISTS uq_anomalies_forecast_method
    ON anomalies (forecast_id, detection_method);


-- ---------------------------------------------------------------------------
-- 5. alerts : stop duplicate alerts for the same anomaly
-- ---------------------------------------------------------------------------
CREATE UNIQUE INDEX IF NOT EXISTS uq_alerts_anomaly
    ON alerts (anomaly_id);


-- ---------------------------------------------------------------------------
-- 6. Indexes matching the documented access patterns
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_demand_data_region_time
    ON demand_data (region_id, timestamp_utc);

CREATE INDEX IF NOT EXISTS idx_forecasts_region_target
    ON forecasts (region_id, target_timestamp);

-- Serves "give me AEP's next 24h from the selected run" without scanning.
CREATE INDEX IF NOT EXISTS idx_forecasts_run_horizon_target
    ON forecasts (model_run_id, horizon_hours, target_timestamp);

CREATE INDEX IF NOT EXISTS idx_model_runs_region
    ON model_runs (region_id);

CREATE INDEX IF NOT EXISTS idx_model_runs_name_region
    ON model_runs (model_name, region_id);

CREATE INDEX IF NOT EXISTS idx_model_evaluations_run
    ON model_evaluations (model_run_id);

CREATE INDEX IF NOT EXISTS idx_anomalies_region_time
    ON anomalies (region_id, timestamp_utc);

-- Serves the "show me only real anomalies" API path; partial so it stays small.
CREATE INDEX IF NOT EXISTS idx_anomalies_flagged
    ON anomalies (region_id, timestamp_utc)
    WHERE is_anomaly = TRUE;

CREATE INDEX IF NOT EXISTS idx_alerts_region_status
    ON alerts (region_id, status);

COMMIT;

-- ---------------------------------------------------------------------------
-- Verification (run after COMMIT)
-- ---------------------------------------------------------------------------
-- \d+ model_runs
-- \d+ forecasts
-- SELECT indexname FROM pg_indexes WHERE schemaname='public' ORDER BY tablename, indexname;
