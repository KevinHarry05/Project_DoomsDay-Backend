-- ============================================================================
-- Removes DHR_ARIMA model_runs superseded by a newer run for the same region
-- (the extended 1h-168h retrain), plus their child forecasts /
-- model_evaluations / anomalies / alerts rows, in FK-safe order.
--
-- Purely cosmetic cleanup - v_model_ranking already picks the newest run per
-- (region, horizon, model) automatically, so leaving the old rows in place
-- does not break anything. This just keeps model_runs tidy for anyone
-- eyeballing raw row counts. Safe to run multiple times; a no-op once
-- there is nothing left to clean up.
--
-- Only touches DHR_ARIMA rows that are NOT each region's most recent DHR_ARIMA
-- run. LightGBM rows and the new (post-retrain) DHR_ARIMA rows are untouched.
-- ============================================================================

BEGIN;

CREATE TEMP TABLE _old_dhr_runs AS
SELECT mr.model_run_id
FROM model_runs mr
WHERE mr.model_name = 'DHR_ARIMA'
  AND mr.created_at < (
    SELECT MAX(mr2.created_at)
    FROM model_runs mr2
    WHERE mr2.region_id = mr.region_id AND mr2.model_name = 'DHR_ARIMA'
  );

-- alerts reference anomalies, which reference forecasts, which reference model_runs
DELETE FROM alerts
WHERE anomaly_id IN (
  SELECT a.anomaly_id FROM anomalies a
  JOIN forecasts f ON f.forecast_id = a.forecast_id
  WHERE f.model_run_id IN (SELECT model_run_id FROM _old_dhr_runs)
);

DELETE FROM anomalies
WHERE forecast_id IN (
  SELECT forecast_id FROM forecasts WHERE model_run_id IN (SELECT model_run_id FROM _old_dhr_runs)
);

DELETE FROM model_evaluations
WHERE model_run_id IN (SELECT model_run_id FROM _old_dhr_runs);

DELETE FROM forecasts
WHERE model_run_id IN (SELECT model_run_id FROM _old_dhr_runs);

DELETE FROM model_runs
WHERE model_run_id IN (SELECT model_run_id FROM _old_dhr_runs);

COMMIT;

-- Sanity check: should show exactly 12 rows (one current DHR_ARIMA run per region)
SELECT region_id, COUNT(*) AS dhr_arima_runs_remaining
FROM model_runs WHERE model_name = 'DHR_ARIMA'
GROUP BY region_id ORDER BY region_id;
