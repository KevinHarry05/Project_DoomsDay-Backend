# Backend API

Thin FastAPI layer over the shared Postgres schema. Every endpoint wraps one
of the query shapes already hand-verified live during demo prep (the
common-origin forecast fix, historical actuals, model comparison, anomalies).

## Setup

```powershell
cd C:\Users\kani2\OneDrive\Documents\BACKEND\backend_api
pip install -r requirements.txt
$env:DATABASE_URL = "postgresql://postgres:<your_password>@localhost:5432/energy_forecasting"
uvicorn app.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive Swagger docs — this alone
is a strong jury demo surface, no frontend required.

## Endpoints

- `GET /forecast/{region}?hours=3` or `?horizons=1&horizons=6&horizons=24`
  → the "next N hours" shape, picks the latest origin common to every
  requested horizon (the exact fix worked out live in psql: a naive MAX
  across horizons is wrong because longer horizons run out of usable origins
  sooner than short ones).
- `GET /historical/{region}?days_ago=1` (or `?timestamp=...`)
  → "what was demand at time T" / "same time yesterday" / "last week".
- `GET /compare?region=AEP&horizon_hours=24`
  → wraps `v_model_ranking` directly; omit params for the full board.
- `GET /anomalies?region=AEP&severity=CRITICAL`
  → flagged rows from the `anomalies` table, most severe first.
- `POST /ask` `{"text": "next 3 hour usage for AEP"}`
  → rule-based NLU stub (see `app/nlu.py`) that routes free-form English
  into the same four query shapes above. No LLM, no API key required.

## Swapping in a real LLM for /ask

`app/nlu.py`'s `parse_query()` is a keyword/regex matcher on purpose - it
gets the end-to-end shape working without any external dependency before the
demo. To upgrade it: replace the body of `parse_query()` with a call to the
Anthropic API (needs `ANTHROPIC_API_KEY`), prompted to extract
`{intent, region, horizons, days_ago, severity}` as JSON from the sentence.
Nothing downstream changes - `routers/ask.py` and `queries.py` are already
LLM-agnostic; the model only ever replaces the parsing step.

## Testing without a live database

`app/queries.py`'s `fetch_dicts` is the only DB touchpoint; every function
in that file can be tested by mocking it, e.g.:

```python
from unittest.mock import patch
from app import queries
with patch.object(queries, "fetch_dicts", return_value=[...fixture rows...]):
    ...
```

This is how the endpoints were smoke-tested before ever touching your real
Postgres instance.
