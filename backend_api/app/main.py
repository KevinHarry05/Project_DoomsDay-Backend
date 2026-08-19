"""
Backend API - thin HTTP layer over the shared Postgres schema.

Every endpoint here is a direct wrapper around one of the query shapes in
queries.py, which were hand-verified against the live database during demo
prep (forecast-with-common-origin, historical actuals, model comparison,
anomalies). Nothing in this file talks to the database directly - that
discipline is what keeps /ask (the NLU stub) and the plain REST endpoints
from ever disagreeing about what a given question returns.

Run:
    $env:DATABASE_URL = "postgresql://postgres:<pw>@localhost:5432/energy_forecasting"
    uvicorn app.main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive Swagger docs - useful
for a live jury demo without writing any frontend code.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import close_pool, init_pool
from .routers import anomalies, ask, compare, forecast, historical

app = FastAPI(
    title="Smart Building Energy Forecasting & Anomaly Detection API",
    description="Model-agnostic forecast, historical, comparison, and anomaly endpoints "
                "over a shared PostgreSQL schema written by independent forecasting tracks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # demo-scoped; tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast.router)
app.include_router(historical.router)
app.include_router(compare.router)
app.include_router(anomalies.router)
app.include_router(ask.router)


@app.on_event("startup")
def on_startup() -> None:
    init_pool()


@app.on_event("shutdown")
def on_shutdown() -> None:
    close_pool()


@app.get("/health")
def health():
    return {"status": "ok"}
