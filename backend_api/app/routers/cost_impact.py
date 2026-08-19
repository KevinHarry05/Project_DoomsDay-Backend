from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from .. import queries

router = APIRouter(prefix="/cost-impact", tags=["cost-impact"])


@router.get("")
def cost_impact(
    region: Optional[str] = Query(
        None, description="Filter to a single region code, e.g. AEP. Omit for all regions."
    ),
):
    return queries.get_cost_impact(region)
