-- ============================================================================
-- Model comparison and best-model selection
--
-- The selection rule is DATA-DRIVEN and lives here, in one place, so no service
-- can quietly hardcode a favourite. Nothing in this file names LightGBM,
-- DHR_ARIMA, PatchTST or TFT.
--
-- Selection rule
--   scope           : (region_id, horizon_hours)  -- the winner may differ per cell
--   eligible rows   : split_name = 'test' only    -- never rank on training metrics
--   primary metric  : WAPE ascending              -- scale-free across regions
--   tie-breakers    : MAE, then RMSE, then most recent run
--   eligibility     : model_runs.status IN ('SUCCESS','PARTIAL')
--                     and the evaluation has a non-null WAPE over enough samples
-- ============================================================================

-- ---------------------------------------------------------------------------
-- v_model_performance : one tidy row per (region, horizon, model run)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_model_performance AS
SELECT
    r.region_id,
    r.region_code,
    mr.model_run_id,
    mr.model_name,
    mr.model_type,
    mr.model_version,
    mr.status                AS run_status,
    me.horizon_hours,
    me.split_name,
    me.n_observations,
    me.mae,
    me.rmse,
    me.wape,
    me.smape,
    me.bias,
    me.skill_vs_naive,
    me.training_time_seconds,
    me.inference_time_ms,
    me.evaluation_start,
    me.evaluation_end,
    mr.training_start,
    mr.training_end,
    mr.created_at            AS run_created_at
FROM model_evaluations me
JOIN model_runs mr ON mr.model_run_id = me.model_run_id
JOIN regions    r  ON r.region_id     = mr.region_id;


-- ---------------------------------------------------------------------------
-- v_model_ranking : competing models ranked within each (region, horizon)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_model_ranking AS
WITH eligible AS (
    SELECT *
    FROM v_model_performance
    WHERE split_name = 'test'
      AND run_status IN ('SUCCESS', 'PARTIAL')
      AND wape IS NOT NULL
      AND COALESCE(n_observations, 0) >= 100   -- refuse to rank on a tiny sample
),
-- If the same model has several runs for a cell, keep only its newest.
latest_per_model AS (
    SELECT DISTINCT ON (region_id, horizon_hours, model_name) *
    FROM eligible
    ORDER BY region_id, horizon_hours, model_name, run_created_at DESC, model_run_id DESC
)
SELECT
    *,
    RANK() OVER (
        PARTITION BY region_id, horizon_hours
        ORDER BY wape ASC, mae ASC, rmse ASC, run_created_at DESC
    ) AS model_rank,
    COUNT(*) OVER (PARTITION BY region_id, horizon_hours) AS models_compared
FROM latest_per_model;


-- ---------------------------------------------------------------------------
-- v_best_model : the single winner per (region, horizon)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_best_model AS
SELECT
    region_id,
    region_code,
    horizon_hours,
    model_run_id       AS best_model_run_id,
    model_name         AS best_model_name,
    model_type         AS best_model_type,
    model_version      AS best_model_version,
    wape               AS best_wape,
    mae                AS best_mae,
    rmse               AS best_rmse,
    smape              AS best_smape,
    skill_vs_naive,
    n_observations,
    models_compared,
    evaluation_start,
    evaluation_end
FROM v_model_ranking
WHERE model_rank = 1;


-- ---------------------------------------------------------------------------
-- v_model_coverage : which models are missing from which cells
-- ---------------------------------------------------------------------------
-- Answers "PatchTST failed - which region/horizon combinations lost a
-- competitor?" so an absent model reads as absent rather than as a model that
-- happened not to win.
CREATE OR REPLACE VIEW v_model_coverage AS
WITH cells AS (
    SELECT DISTINCT region_id, region_code, horizon_hours FROM v_model_performance
),
models AS (
    SELECT DISTINCT model_name FROM model_runs
)
SELECT
    c.region_id,
    c.region_code,
    c.horizon_hours,
    m.model_name,
    EXISTS (
        SELECT 1 FROM v_model_ranking vr
        WHERE vr.region_id = c.region_id
          AND vr.horizon_hours = c.horizon_hours
          AND vr.model_name = m.model_name
    ) AS has_test_evaluation
FROM cells c
CROSS JOIN models m;


-- ---------------------------------------------------------------------------
-- v_selected_forecast : the forecast series the API and anomaly layer should use
-- ---------------------------------------------------------------------------
-- Joining forecasts to v_best_model is what guarantees a single consistent model
-- per (region, horizon) and makes it structurally impossible to mix predictions
-- from different models into one residual series.
CREATE OR REPLACE VIEW v_selected_forecast AS
SELECT
    f.forecast_id,
    b.region_code,
    f.region_id,
    f.model_run_id,
    b.best_model_name AS model_name,
    f.forecast_timestamp,
    f.target_timestamp,
    f.horizon_hours,
    f.predicted_demand_mw,
    f.actual_demand_mw,
    f.actual_demand_mw - f.predicted_demand_mw AS residual_mw
FROM forecasts f
JOIN v_best_model b
  ON b.best_model_run_id = f.model_run_id
 AND b.horizon_hours     = f.horizon_hours;


-- ============================================================================
-- Example queries
-- ============================================================================

-- "Which model is most accurate for AEP at 24 hours?"
-- SELECT * FROM v_best_model
-- WHERE region_code = 'AEP' AND horizon_hours = 24;

-- "Compare all models for AEP, 24-hour horizon."
-- SELECT model_name, model_rank, mae, rmse, wape, smape, skill_vs_naive
-- FROM v_model_ranking
-- WHERE region_code = 'AEP' AND horizon_hours = 24
-- ORDER BY model_rank;

-- "Does the best model differ by region or horizon?"  (the whole point)
-- SELECT horizon_hours, best_model_name, COUNT(*) AS n_regions
-- FROM v_best_model
-- GROUP BY horizon_hours, best_model_name
-- ORDER BY horizon_hours, n_regions DESC;

-- "Give AEP's next 24 hours from the selected model."
-- SELECT target_timestamp, horizon_hours, predicted_demand_mw, model_name
-- FROM v_selected_forecast
-- WHERE region_code = 'AEP' AND horizon_hours <= 24
-- ORDER BY target_timestamp;

-- "Which models are missing for any cell?"
-- SELECT * FROM v_model_coverage WHERE NOT has_test_evaluation ORDER BY region_code, horizon_hours;
