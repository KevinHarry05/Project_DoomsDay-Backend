-- ============================================================================
-- Post-integration validation
--
-- Run after every integration, and again after the Supabase restore. Every
-- check is read-only. The intended result for each is stated inline; anything
-- else needs investigation before the run is declared good.
-- ============================================================================

\echo '=== 1. Row counts by table ==='
SELECT 'regions'           AS table_name, COUNT(*) FROM regions
UNION ALL SELECT 'demand_data',       COUNT(*) FROM demand_data
UNION ALL SELECT 'model_runs',        COUNT(*) FROM model_runs
UNION ALL SELECT 'forecasts',         COUNT(*) FROM forecasts
UNION ALL SELECT 'model_evaluations', COUNT(*) FROM model_evaluations
UNION ALL SELECT 'anomalies',         COUNT(*) FROM anomalies
UNION ALL SELECT 'alerts',            COUNT(*) FROM alerts
ORDER BY table_name;


\echo '=== 2. Duplicate forecasts (EXPECT 0 rows) ==='
-- The uniqueness constraint should make this impossible; the check verifies the
-- constraint is actually present and enforced rather than assumed.
SELECT model_run_id, region_id, target_timestamp, horizon_hours, COUNT(*) AS n
FROM forecasts
GROUP BY 1, 2, 3, 4
HAVING COUNT(*) > 1
ORDER BY n DESC
LIMIT 20;


\echo '=== 3. Orphaned foreign keys (EXPECT 0 for every row) ==='
SELECT 'forecasts.model_run_id -> model_runs' AS relationship, COUNT(*) AS orphans
FROM forecasts f LEFT JOIN model_runs mr ON mr.model_run_id = f.model_run_id
WHERE mr.model_run_id IS NULL
UNION ALL
SELECT 'forecasts.region_id -> regions', COUNT(*)
FROM forecasts f LEFT JOIN regions r ON r.region_id = f.region_id
WHERE r.region_id IS NULL
UNION ALL
SELECT 'model_runs.region_id -> regions', COUNT(*)
FROM model_runs mr LEFT JOIN regions r ON r.region_id = mr.region_id
WHERE r.region_id IS NULL
UNION ALL
SELECT 'model_evaluations.model_run_id -> model_runs', COUNT(*)
FROM model_evaluations me LEFT JOIN model_runs mr ON mr.model_run_id = me.model_run_id
WHERE mr.model_run_id IS NULL
UNION ALL
SELECT 'anomalies.forecast_id -> forecasts', COUNT(*)
FROM anomalies a LEFT JOIN forecasts f ON f.forecast_id = a.forecast_id
WHERE f.forecast_id IS NULL
UNION ALL
SELECT 'anomalies.region_id -> regions', COUNT(*)
FROM anomalies a LEFT JOIN regions r ON r.region_id = a.region_id
WHERE r.region_id IS NULL
UNION ALL
SELECT 'alerts.anomaly_id -> anomalies', COUNT(*)
FROM alerts al LEFT JOIN anomalies a ON a.anomaly_id = al.anomaly_id
WHERE a.anomaly_id IS NULL;


\echo '=== 4. NULL / invalid forecast values (EXPECT 0 for every row) ==='
SELECT 'null predicted_demand_mw' AS check_name, COUNT(*) AS n
FROM forecasts WHERE predicted_demand_mw IS NULL
UNION ALL
SELECT 'non-positive horizon_hours', COUNT(*) FROM forecasts WHERE horizon_hours <= 0
UNION ALL
SELECT 'negative predicted demand', COUNT(*) FROM forecasts WHERE predicted_demand_mw < 0
UNION ALL
SELECT 'null target_timestamp', COUNT(*) FROM forecasts WHERE target_timestamp IS NULL
UNION ALL
SELECT 'null forecast_timestamp', COUNT(*) FROM forecasts WHERE forecast_timestamp IS NULL;


\echo '=== 5. Timestamp ordering: horizon must equal target - origin (EXPECT 0) ==='
-- Catches an adapter that mapped a timestamp column wrongly. Cheap, and it has
-- caught real bugs: an adapter that writes target_timestamp into both columns
-- looks completely fine until this check runs.
SELECT COUNT(*) AS mismatched_horizon
FROM forecasts
WHERE target_timestamp <= forecast_timestamp
   OR EXTRACT(EPOCH FROM (target_timestamp - forecast_timestamp)) / 3600.0
      <> horizon_hours::double precision;


\echo '=== 6. Model and region coverage ==='
SELECT
    r.region_code,
    mr.model_name,
    f.horizon_hours,
    COUNT(*)                    AS n_forecasts,
    MIN(f.target_timestamp)     AS first_target,
    MAX(f.target_timestamp)     AS last_target,
    COUNT(f.actual_demand_mw)   AS n_with_actuals
FROM forecasts f
JOIN model_runs mr ON mr.model_run_id = f.model_run_id
JOIN regions    r  ON r.region_id     = f.region_id
GROUP BY 1, 2, 3
ORDER BY r.region_code, mr.model_name, f.horizon_hours;


\echo '=== 7. Evaluation sanity ==='
SELECT 'null wape on test split'      AS check_name, COUNT(*) AS n
FROM model_evaluations WHERE split_name = 'test' AND wape IS NULL
UNION ALL
SELECT 'negative error metric', COUNT(*)
FROM model_evaluations WHERE mae < 0 OR rmse < 0 OR wape < 0 OR smape < 0
UNION ALL
SELECT 'rmse < mae (mathematically impossible)', COUNT(*)
FROM model_evaluations WHERE rmse IS NOT NULL AND mae IS NOT NULL AND rmse < mae
UNION ALL
SELECT 'test evaluation missing for a run', COUNT(*)
FROM model_runs mr
WHERE mr.status IN ('SUCCESS', 'PARTIAL')
  AND NOT EXISTS (
      SELECT 1 FROM model_evaluations me
      WHERE me.model_run_id = mr.model_run_id AND me.split_name = 'test'
  );


\echo '=== 8. Anomaly sanity ==='
SELECT 'residual != actual - predicted' AS check_name, COUNT(*) AS n
FROM anomalies
WHERE actual_demand_mw IS NOT NULL AND predicted_demand_mw IS NOT NULL
  AND ABS(residual_mw - (actual_demand_mw - predicted_demand_mw)) > 0.01
UNION ALL
SELECT 'flagged but severity NONE', COUNT(*)
FROM anomalies WHERE is_anomaly = TRUE AND severity = 'NONE'
UNION ALL
SELECT 'not flagged but severity set', COUNT(*)
FROM anomalies WHERE is_anomaly = FALSE AND severity <> 'NONE'
UNION ALL
SELECT 'negative anomaly_score', COUNT(*)
FROM anomalies WHERE anomaly_score < 0;


\echo '=== 9. Anomaly rate per region/method (sanity, not a hard failure) ==='
-- A rate above ~10% almost always means the threshold was fitted on the wrong
-- split rather than that the grid genuinely misbehaved for a tenth of the year.
SELECT
    r.region_code,
    a.detection_method,
    COUNT(*)                                              AS n_scored,
    COUNT(*) FILTER (WHERE a.is_anomaly)                  AS n_flagged,
    ROUND(100.0 * COUNT(*) FILTER (WHERE a.is_anomaly) / NULLIF(COUNT(*), 0), 2) AS pct_flagged,
    COUNT(*) FILTER (WHERE a.severity = 'CRITICAL')       AS n_critical,
    COUNT(*) FILTER (WHERE a.severity = 'HIGH')           AS n_high
FROM anomalies a
JOIN regions r ON r.region_id = a.region_id
GROUP BY 1, 2
ORDER BY pct_flagged DESC;


\echo '=== 10. Forecast vs demand_data consistency (EXPECT small) ==='
-- Where a forecast carries an actual, it should match the historical series.
-- A systematic mismatch means a timezone was reapplied somewhere in the chain.
SELECT
    r.region_code,
    COUNT(*)                                                 AS n_compared,
    COUNT(*) FILTER (WHERE ABS(f.actual_demand_mw - d.demand_mw) > 0.01) AS n_mismatched,
    MAX(ABS(f.actual_demand_mw - d.demand_mw))               AS max_abs_diff
FROM forecasts f
JOIN demand_data d
  ON d.region_id = f.region_id AND d.timestamp_utc = f.target_timestamp
JOIN regions r ON r.region_id = f.region_id
WHERE f.actual_demand_mw IS NOT NULL
GROUP BY 1
ORDER BY n_mismatched DESC;


\echo '=== 11. Indexes present ==='
SELECT tablename, indexname
FROM pg_indexes
WHERE schemaname = 'public'
  AND tablename IN ('regions','demand_data','model_runs','forecasts',
                    'model_evaluations','anomalies','alerts')
ORDER BY tablename, indexname;


\echo '=== 12. Declared foreign keys ==='
SELECT
    tc.table_name,
    kcu.column_name,
    ccu.table_name  AS references_table,
    ccu.column_name AS references_column,
    tc.constraint_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON kcu.constraint_name = tc.constraint_name AND kcu.table_schema = tc.table_schema
JOIN information_schema.constraint_column_usage ccu
  ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public'
ORDER BY tc.table_name, kcu.column_name;


\echo '=== 13. Idempotency probe ==='
-- Run the integration twice, then compare. Identical counts across both runs is
-- the proof that ON CONFLICT is doing its job.
SELECT
    mr.model_name,
    COUNT(DISTINCT mr.model_run_id) AS n_runs,
    COUNT(f.forecast_id)            AS n_forecasts
FROM model_runs mr
LEFT JOIN forecasts f ON f.model_run_id = mr.model_run_id
GROUP BY 1
ORDER BY 1;
