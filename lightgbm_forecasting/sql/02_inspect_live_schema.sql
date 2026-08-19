-- ============================================================================
-- INSPECT THE LIVE SCHEMA
--
-- Read-only. Run this FIRST, on the real energy_forecasting database, before
-- applying any migration. Its purpose is to replace assumptions about the
-- current schema with facts, so nothing is changed on the basis of a guess.
--
--   psql -U postgres -d energy_forecasting -f sql/02_inspect_live_schema.sql
-- ============================================================================

\echo '=== 1. Tables and sizes ==='
SELECT
    c.relname AS table_name,
    pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size,
    c.reltuples::bigint AS approx_rows
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind = 'r'
ORDER BY pg_total_relation_size(c.oid) DESC;


\echo '=== 2. Exact columns of every production table ==='
-- Compare this against sql/01_baseline_schema_reference.sql. Any difference is
-- schema drift that the integration layer needs to know about.
SELECT
    table_name,
    ordinal_position AS pos,
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name IN ('regions','demand_data','model_runs','forecasts',
                     'model_evaluations','anomalies','alerts')
ORDER BY table_name, ordinal_position;


\echo '=== 3. Are the timestamp columns actually TIMESTAMPTZ? ==='
-- TIMESTAMP WITHOUT TIME ZONE anywhere here is a correctness problem, not a
-- cosmetic one: it silently drops the offset and makes UTC alignment across
-- models unverifiable.
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND (column_name LIKE '%timestamp%' OR column_name LIKE '%_start' OR column_name LIKE '%_end')
ORDER BY table_name, column_name;


\echo '=== 4. Primary keys ==='
SELECT tc.table_name, tc.constraint_name, string_agg(kcu.column_name, ', ' ORDER BY kcu.ordinal_position) AS columns
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON kcu.constraint_name = tc.constraint_name AND kcu.table_schema = tc.table_schema
WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_schema = 'public'
GROUP BY 1, 2
ORDER BY 1;


\echo '=== 5. Foreign keys ==='
SELECT
    tc.table_name,
    kcu.column_name,
    ccu.table_name  AS references_table,
    ccu.column_name AS references_column,
    rc.delete_rule,
    rc.update_rule
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON kcu.constraint_name = tc.constraint_name AND kcu.table_schema = tc.table_schema
JOIN information_schema.constraint_column_usage ccu
  ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
JOIN information_schema.referential_constraints rc
  ON rc.constraint_name = tc.constraint_name AND rc.constraint_schema = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public'
ORDER BY tc.table_name, kcu.column_name;


\echo '=== 6. Unique and check constraints ==='
SELECT
    rel.relname AS table_name,
    con.conname AS constraint_name,
    CASE con.contype WHEN 'u' THEN 'UNIQUE' WHEN 'c' THEN 'CHECK'
                     WHEN 'p' THEN 'PRIMARY KEY' ELSE con.contype::text END AS kind,
    pg_get_constraintdef(con.oid) AS definition
FROM pg_constraint con
JOIN pg_class rel ON rel.oid = con.conrelid
JOIN pg_namespace n ON n.oid = rel.relnamespace
WHERE n.nspname = 'public' AND con.contype IN ('u','c')
ORDER BY rel.relname, con.conname;


\echo '=== 7. Indexes ==='
SELECT tablename, indexname, indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;


\echo '=== 8. regions reference data (EXPECT the 12 codes) ==='
SELECT region_id, region_code, region_name FROM regions ORDER BY region_id;


\echo '=== 9. demand_data coverage per region ==='
SELECT
    r.region_code,
    COUNT(*)                        AS n_rows,
    MIN(d.timestamp_utc)            AS first_utc,
    MAX(d.timestamp_utc)            AS last_utc,
    COUNT(*) FILTER (WHERE d.missing_flag) AS n_missing_flagged
FROM demand_data d
JOIN regions r ON r.region_id = d.region_id
GROUP BY 1
ORDER BY 1;


\echo '=== 10. demand_data row count reconciliation ==='
-- Documented: staging holds 1,090,176 rows and the insert reported
-- INSERT 0 1090176, yet 11 source rows carry NULL Demand_MW against a NOT NULL
-- column. Those two facts cannot both be true, so this reconciles them before
-- the numbers get carried into the cloud migration.
SELECT
    (SELECT COUNT(*) FROM demand_data)   AS demand_data_rows,
    (SELECT COUNT(*) FROM demand_staging) AS staging_rows,
    (SELECT COUNT(*) FROM demand_staging WHERE demand_mw IS NULL) AS staging_null_demand,
    (SELECT COUNT(*) FROM demand_staging) - (SELECT COUNT(*) FROM demand_data) AS difference;


\echo '=== 11. Duplicate timestamps in demand_data (EXPECT 0 rows) ==='
SELECT region_id, timestamp_utc, COUNT(*) AS n
FROM demand_data
GROUP BY 1, 2
HAVING COUNT(*) > 1
LIMIT 20;


\echo '=== 12. Session timezone (display only, not storage) ==='
-- Confirms that a TIMESTAMPTZ rendered as +05:30 in pgAdmin is a display
-- artefact of the session timezone, not the stored value.
SHOW timezone;
SELECT
    timestamp_utc                        AS as_displayed,
    timestamp_utc AT TIME ZONE 'UTC'     AS as_utc
FROM demand_data
ORDER BY timestamp_utc
LIMIT 3;
