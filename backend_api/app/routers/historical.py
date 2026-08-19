from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from .. import queries

router = APIRouter(prefix="/historical", tags=["historical"])


@router.get("/{region}")
def historical(
    region: str,
    timestamp: Optional[str] = Query(None, description="ISO timestamp, e.g. 2018-08-02T06:30:00+05:30"),
    relative_to: Optional[str] = Query(
        None, description="ISO timestamp to offset FROM (defaults to now if omitted)"
    ),
    days_ago: Optional[int] = Query(None, description="e.g. days_ago=1 for 'yesterday same time'"),
):
    if timestamp:
        target = timestamp
    elif days_ago is not None:
        base = datetime.fromisoformat(relative_to) if relative_to else datetime.utcnow()
        target = (base - timedelta(days=days_ago)).isoformat()
    else:
        raise HTTPException(400, "Provide ?timestamp=... or ?days_ago=N (optionally with ?relative_to=...)")
    return queries.get_historical(region, target, exact=False)
