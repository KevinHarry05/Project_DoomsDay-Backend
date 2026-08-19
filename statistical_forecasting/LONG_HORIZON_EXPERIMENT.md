# Experiment: extending DHR+ARIMA to a 4-month horizon (2026-08-18)

**Result: reverted. Documented here as a negative result, not deleted from history.**

## Why this was tried

TFT's differentiated pitch (per the project's model-selection table) is
long-horizon forecasting via known-future covariates. Before building a
fourth model, the question was asked: can an existing model already reach a
4-month horizon? DHR+ARIMA was the natural candidate — its forecast is built
from Fourier (seasonal) + linear trend + holiday regressors rather than lag
features or a fixed lookback window, so unlike LightGBM (lags meaningless
past ~1 week) or PatchTST (fixed 168h lookback, no visibility into a season
4 months away), extending its horizon range is a config change, not an
architecture change.

## What was tried

`config.yaml` was changed to add horizons `[336, 504, 672, 1008, 1344, 2016,
2688, 2928]` hours (2 weeks through ~4 months) on top of the existing 1-168h
range, with `forecast.max_horizon_hours: 2928`. No code changes were
needed - `model.py`'s rolling-origin forecast loop is fully horizon-agnostic.
All 12 regions retrained successfully (`regions_succeeded: 12,
regions_failed: 0`, 171,041 forecast rows, 624 evaluation rows) - this was
NOT a crash or a data problem, the pipeline is fully capable of producing
these forecasts.

## What the numbers actually showed

The hypothesis going in was that WAPE would plateau once the ARIMA
state's contribution decayed to near-zero, leaving a pure seasonal+trend
"climatology" forecast. That hypothesis was wrong. Mean WAPE across all 12
regions, test split:

```
h=168  (7 days):   48.72%
h=336  (2 weeks):   69.32%
h=504  (3 weeks):   79.42%
h=672  (4 weeks):   84.55%
h=1008 (6 weeks):   89.17%
h=1344 (8 weeks):   91.14%
h=2016 (12 weeks):  93.85%
h=2688 (16 weeks):  95.59%
h=2928 (~4 months): 95.76%  (min region 16.31%, max region 129.64%)
```

It does eventually slow down past ~2000h, but it slows down at a WAPE level
that is not useful (95.76% mean means the average error is nearly as large
as the demand value itself). The worst region (PJM_Load) exceeded 100% WAPE
at 4 months - worse than a naive prediction would be. PJM_Load also has a
notably shorter/older data span (1998-2002 only, vs. 2004/2005-2018 for most
other regions), which may make it more exposed to this failure mode, but
several other regions were also well past 80% WAPE by 1 month out.

**Likely cause**: the linear trend regressor extrapolates unboundedly
instead of reverting toward a long-run mean. Months out, that can drift
meaningfully away from where demand actually settles. This was not
root-caused further (see "not investigated" below) - it's a plausible
explanation based on the shape of the degradation curve, not a confirmed
diagnosis.

## Decision

Reverted. `config.yaml` is back to `horizons.supported: [..., 168]` and
`forecast.max_horizon_hours: 168`. The project's honest claim for
DHR+ARIMA stays at "1 hour - 7 days", matching what's actually loaded in
the database. The 4-month run's outputs were never loaded into PostgreSQL,
so there is nothing to clean up there.

## Not investigated (possible future work, not started)

- Whether removing or capping the linear trend term (`dhr.add_linear_trend:
  false`, or a damped/mean-reverting trend instead of a pure linear one)
  changes the degradation curve materially.
- Whether a re-estimated (not just state-advanced) ARIMA fit at a much
  longer cadence would behave differently.
- TFT's known-future-covariate design was the original alternative
  considered for genuine long-horizon forecasting - this experiment doesn't
  replace that discussion, it just confirms a config-only extension of
  DHR+ARIMA is not a shortcut around building it.

## Scope decision (2026-08-18)

Per direction: stop the model roster at three - LightGBM, DHR+ARIMA,
PatchTST. TFT is not being built. This file exists so that decision is
backed by an actual tested result, not just a time-budget call.
