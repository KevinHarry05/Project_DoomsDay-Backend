from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import queries
from ..nlu import parse_query

router = APIRouter(tags=["ask"])


class AskRequest(BaseModel):
    text: str


@router.post("/ask")
def ask(req: AskRequest):
    parsed = parse_query(req.text)

    if not parsed.region and parsed.intent in ("forecast", "historical"):
        raise HTTPException(400, f"Couldn't identify a region in: {req.text!r}. "
                                  "Try including a region code like AEP, COMED, PJME, etc.")

    if parsed.intent == "forecast":
        result = queries.get_forecast(parsed.region, parsed.horizons)
        return {"intent": parsed.intent, "parsed": parsed.__dict__, "result": result}

    if parsed.intent == "historical":
        target = (datetime.utcnow() - timedelta(days=parsed.days_ago or 1)).isoformat()
        result = queries.get_historical(parsed.region, target, exact=False)
        return {"intent": parsed.intent, "parsed": parsed.__dict__, "result": result}

    if parsed.intent == "compare":
        result = {"rankings": queries.get_model_comparison(parsed.region, None)}
        return {"intent": parsed.intent, "parsed": parsed.__dict__, "result": result}

    if parsed.intent == "anomalies":
        result = {"anomalies": queries.get_anomalies(parsed.region, parsed.severity, 20)}
        return {"intent": parsed.intent, "parsed": parsed.__dict__, "result": result}

    raise HTTPException(422, {
        "message": "Could not classify this question into forecast / historical / compare / anomalies.",
        "parsed": parsed.__dict__,
        "hint": "This is the rule-based NLU stub - see app/nlu.py's docstring for how to swap in a real LLM.",
    })
