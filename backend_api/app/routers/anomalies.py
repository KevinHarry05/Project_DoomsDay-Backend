from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from .. import queries

router = APIRouter(prefix="/anomalies", tags=["anomalies"])


@router.get("")
def anomalies(
    region: Optional[str] = Query(None),
    severity: Optional[str] = Query(None, description="LOW | MEDIUM | HIGH | CRITICAL"),
    limit: int = Query(50, le=500),
):
    return {"anomalies": queries.get_anomalies(region, severity, limit)}
