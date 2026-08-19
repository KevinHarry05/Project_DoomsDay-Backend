from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter
from pydantic import BaseModel

from .. import queries
from ..llm_nlu import resolve as llm_resolve
from ..nlu import ParsedQuery, parse_query

router = APIRouter(tags=["ask"])


class AskRequest(BaseModel):
    text: str


def _run(parsed: ParsedQuery) -> dict:
    """Same lookup logic for both the rule-based path and the LLM-resolved path."""
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

    # anomalies
    rows = queries.get_anomalies(parsed.region, parsed.severity, 20)
    note = None
    if not rows and parsed.severity:
        # A severity word picked up from casual phrasing (e.g. "critical
        # situation") can filter down to a real severity band that happens to
        # have zero rows for this region right now. Rather than a silent dead
        # end, fall back to showing every severity and say so explicitly -
        # the user should never see "no anomalies found" when anomalies do
        # in fact exist, just not at the exact severity a stray word implied.
        rows = queries.get_anomalies(parsed.region, None, 20)
        if rows:
            note = (
                f"No {parsed.severity}-severity anomalies matched, so showing all "
                "severities instead."
            )
    result = {"anomalies": rows, "note": note}
    return {"intent": parsed.intent, "parsed": parsed.__dict__, "result": result}


@router.post("/ask")
def ask(req: AskRequest):
    parsed = parse_query(req.text)

    needs_region = parsed.intent in ("forecast", "historical")
    resolved_by_rules = parsed.intent != "unknown" and not (needs_region and not parsed.region)

    if resolved_by_rules:
        return _run(parsed)

    # Rule-based parser couldn't fully classify this one - hand it to the LLM
    # fallback (or, if no API key is configured, a friendly static reply).
    # Either way this always returns 200: an unclassifiable question is a
    # normal conversational turn, not a server error.
    fallback = llm_resolve(req.text, hint_region=parsed.region)

    if fallback.mode == "resolve" and fallback.parsed is not None:
        return _run(fallback.parsed)

    return {
        "intent": "chat",
        "parsed": parsed.__dict__,
        "result": {"message": fallback.message},
    }
