from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from .. import queries

router = APIRouter(prefix="/compare", tags=["compare"])


@router.get("")
def compare(
    region: Optional[str] = Query(None),
    horizon_hours: Optional[int] = Query(None),
):
    return {"rankings": queries.get_model_comparison(region, horizon_hours)}
