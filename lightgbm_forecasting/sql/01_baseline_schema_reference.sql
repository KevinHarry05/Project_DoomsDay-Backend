-- ============================================================================
-- BASELINE SCHEMA - REFERENCE ONLY
--
-- This reproduces the eight-table design as documented, and is what the
-- integration layer was developed and tested against in a scratch database.
--
-- DO NOT RUN THIS ON energy_forecasting. That database already exists and is
-- populated; running this would be a redesign, not a migration. Its purpose is
-- (a) to make the assumed schema explicit so any drift is visible, and (b) to
-- let anyone stand up a throwaway database for testing.
--
-- To check the real schema against these assumptions instead, run:
--     sql/02_inspect_live_schema.sql
-- ============================================================================

CREATE TABLE IF NOT EXISTS regions (
    region_id    SERIAL PRIMARY KEY,
    region_code  TEXT NOT NULL UNIQUE,
    region_name  TEXT
);

CREATE TABLE IF NOT EXISTS demand_data (
    demand_id     BIGSERIAL PRIMARY KEY,
    region_id     INTEGER NOT NULL REFERENCES regions (region_id),
    timestamp_utc TIMESTAMPTZ NOT NULL,
    demand_mw     DOUBLE PRECISION NOT NULL,
    missing_flag  BOOLEAN DEFAULT FALSE,
    UNIQUE (region_id, timestamp_utc)
);

CREATE TABLE IF NOT EXISTS model_runs (
    model_run_id   BIGSERIAL PRIMARY KEY,
    model_name     TEXT NOT NULL,
    model_type     TEXT NOT NULL,
    region_id      INTEGER NOT NULL REFERENCES regions (region_id),
    training_start TIMESTAMPTZ,
    training_end   TIMESTAMPTZ,
    created_at     TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id         BIGSERIAL PRIMARY KEY,
    model_run_id        BIGINT NOT NULL REFERENCES model_runs (model_run_id),
    region_id           INTEGER NOT NULL REFERENCES regions (region_id),
    forecast_timestamp  TIMESTAMPTZ NOT NULL,
    target_timestamp    TIMESTAMPTZ NOT NULL,
    horizon_hours       INTEGER NOT NULL CHECK (horizon_hours > 0),
    predicted_demand_mw DOUBLE PRECISION NOT NULL,
    actual_demand_mw    DOUBLE PRECISION,
    created_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_evaluations (
    evaluation_id         BIGSERIAL PRIMARY KEY,
    model_run_id          BIGINT NOT NULL REFERENCES model_runs (model_run_id),
    evaluation_start      TIMESTAMPTZ,
    evaluation_end        TIMESTAMPTZ,
    horizon_hours         INTEGER,
    mae                   DOUBLE PRECISION,
    rmse                  DOUBLE PRECISION,
    wape                  DOUBLE PRECISION,
    smape                 DOUBLE PRECISION,
    training_time_seconds DOUBLE PRECISION,
    inference_time_ms     DOUBLE PRECISION,
    created_at            TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS anomalies (
    anomaly_id          BIGSERIAL PRIMARY KEY,
    forecast_id         BIGINT NOT NULL REFERENCES forecasts (forecast_id),
    region_id           INTEGER NOT NULL REFERENCES regions (region_id),
    timestamp_utc       TIMESTAMPTZ NOT NULL,
    actual_demand_mw    DOUBLE PRECISION,
    predicted_demand_mw DOUBLE PRECISION,
    residual_mw         DOUBLE PRECISION,
    anomaly_score       DOUBLE PRECISION,
    severity            TEXT,
    is_anomaly          BOOLEAN DEFAULT FALSE,
    detection_method    TEXT,
    created_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id        BIGSERIAL PRIMARY KEY,
    anomaly_id      BIGINT NOT NULL REFERENCES anomalies (anomaly_id),
    region_id       INTEGER NOT NULL REFERENCES regions (region_id),
    alert_type      TEXT,
    severity        TEXT,
    message         TEXT,
    status          TEXT DEFAULT 'OPEN',
    created_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMPTZ
);

-- Reference data: the 12 PJM region codes.
INSERT INTO regions (region_code, region_name) VALUES
    ('AEP',      'American Electric Power'),
    ('COMED',    'Commonwealth Edison'),
    ('DAYTON',   'Dayton Power and Light'),
    ('DEOK',     'Duke Energy Ohio/Kentucky'),
    ('DOM',      'Dominion Virginia Power'),
    ('DUQ',      'Duquesne Light'),
    ('EKPC',     'East Kentucky Power Cooperative'),
    ('FE',       'FirstEnergy'),
    ('NI',       'Northern Illinois Hub'),
    ('PJM_Load', 'PJM Total Load'),
    ('PJME',     'PJM East'),
    ('PJMW',     'PJM West')
ON CONFLICT (region_code) DO NOTHING;
