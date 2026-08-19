from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from .. import queries

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("/{region}")
def forecast(
    region: str,
    hours: Optional[int] = Query(None, description="Shorthand for horizons=1..hours, e.g. hours=3"),
    horizons: Optional[List[int]] = Query(None, description="Explicit horizon list, e.g. horizons=1&horizons=6&horizons=24"),
):
    if not hours and not horizons:
        raise HTTPException(400, "Provide either ?hours=N or one-or-more ?horizons=")
    horizon_list = horizons or queries.horizons_up_to(hours)
    try:
        return queries.get_forecast(region, horizon_list)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
