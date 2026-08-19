"""
Fallback layer for questions the rule-based parser in nlu.py can't classify.

DESIGN
------
parse_query() in nlu.py stays exactly as-is and remains the FIRST thing every
question goes through - it's free and instant, and it already handles the
common phrasings fine. This module only runs for the leftover cases: intent
came back "unknown", or a region-dependent intent (forecast/historical)
matched but no region could be found in the sentence.

Two tiers, in order:

1. If OPENROUTER_API_KEY is not set, we never fail loudly - we return a
   friendly, locally-generated message that explains what the assistant can
   help with and echoes back anything it *did* understand (e.g. a region it
   found). No network call, no cost, still never shows the user a raw error.

2. If a key IS set, we call an LLM through OpenRouter (openrouter.ai) - an
   OpenAI-compatible endpoint that can route to many providers/models behind
   one API key - and ask it to either:
     a) resolve the sentence into the same {intent, region, horizons,
        days_ago, severity} shape parse_query() would have produced, so it
        can be handed to the exact same queries.py functions the REST
        endpoints use (the LLM never touches the database - it only ever
        replaces the parsing step), or
     b) if the question genuinely isn't answerable by this app (wrong
        domain, missing/unresolvable region, too vague), write a short,
        polite, conversational reply itself - the way a person would say
        "I'm not able to answer that, but here's what I can do" - instead of
        the app returning a technical error box.

We ask for OpenAI-style function/tool calling (`tools` + `tool_choice`)
because that keeps the response deterministic and typed. Free-tier models
routed through OpenRouter are less reliable at honoring `tool_choice`, so if
no tool call comes back we also try to parse a JSON object straight out of
the plain text reply before giving up and falling back to the static
message - belt and suspenders, since this path must never surface a raw
error to the user.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import requests

from .nlu import _REGION_ALIASES, ParsedQuery

logger = logging.getLogger(__name__)

_VALID_REGIONS = sorted(set(_REGION_ALIASES.values()))
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# "openrouter/free" is OpenRouter's auto-router: it always resolves to whichever
# free-tier model is currently live and explicitly filters for tool-calling
# support. Pinning a specific "...:free" model slug instead is fragile -
# OpenRouter periodically retires/renames individual free models (this is a
# common, documented gotcha), which turns into a silent 404 here. Override
# with OPENROUTER_ASK_MODEL if you want to pin a specific paid model instead.
_MODEL = os.environ.get("OPENROUTER_ASK_MODEL", "openrouter/free")
_TIMEOUT_S = float(os.environ.get("OPENROUTER_TIMEOUT_S", "12"))

_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "respond",
        "description": "Resolve the user's question into a structured query, or write a short conversational reply.",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["resolve", "chat"],
                    "description": (
                        "'resolve' if this maps to a forecast/historical/compare/anomalies "
                        "lookup with an identifiable region. 'chat' if it doesn't - wrong "
                        "domain, no identifiable region, too vague, or a general question."
                    ),
                },
                "intent": {
                    "type": "string",
                    "enum": ["forecast", "historical", "compare", "anomalies"],
                    "description": "Only set when mode='resolve'.",
                },
                "region": {
                    "type": "string",
                    "description": f"One of: {', '.join(_VALID_REGIONS)}. Only set when mode='resolve'.",
                },
                "hours": {
                    "type": "integer",
                    "description": "Forecast horizon in hours, only for intent='forecast'.",
                },
                "days_ago": {
                    "type": "integer",
                    "description": "Only for intent='historical'.",
                },
                "severity": {
                    "type": "string",
                    "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                    "description": "Only for intent='anomalies', if mentioned.",
                },
                "message": {
                    "type": "string",
                    "description": (
                        "Only set when mode='chat'. A short (1-3 sentence), warm, "
                        "helpful reply - never a raw error. If the question is off-topic, "
                        "say so politely and mention this app answers questions about grid "
                        "demand forecasts, historical usage, model comparisons, and anomalies "
                        "for regions like AEP, COMED, DOM, PJME, etc."
                    ),
                },
            },
            "required": ["mode"],
        },
    },
}

_SYSTEM_PROMPT = (
    "You are the query parser for GridCast, a grid energy demand forecasting app "
    f"covering these regions: {', '.join(_VALID_REGIONS)}. A rule-based parser already "
    "tried and failed to classify the user's question. Resolve it into a structured "
    "query whenever a reasonable default exists - prefer resolving over asking the user "
    "to clarify. In particular, a region plus a general word like 'demand', 'energy', "
    "'usage', or 'power' with no explicit time frame should resolve to "
    "intent='forecast' with hours=1 (a quick current outlook), not a clarifying chat "
    "reply. Only use mode='chat' when the question is genuinely off-topic (not about "
    "grid energy at all), or no region can be identified anywhere in the question. "
    "You MUST respond by calling the `respond` function - never plain text."
)


@dataclass
class FallbackResult:
    mode: str                      # "resolve" | "chat"
    parsed: Optional[ParsedQuery] = None
    message: str = ""


def _static_fallback(text: str, hint_region: Optional[str]) -> FallbackResult:
    """Zero-cost, zero-dependency fallback used when no API key is configured
    or the LLM call fails/produces nothing usable."""
    if hint_region:
        msg = (
            f"I found the region {hint_region} in your question, but couldn't tell what "
            "you want to know about it. Try asking for a forecast (\"next 6 hours for "
            f"{hint_region}\"), recent history (\"demand 2 days ago for {hint_region}\"), "
            f"the best model (\"which model is best for {hint_region}\"), or anomalies "
            f"(\"any critical anomalies in {hint_region}\")."
        )
    else:
        msg = (
            "I couldn't quite match that to a forecast, historical lookup, model "
            "comparison, or anomaly search. I can answer things like \"next 3 hours for "
            "AEP\", \"demand 2 days ago for DOM\", \"which model is best for PJME\", or "
            "\"any critical anomalies in COMED\" - mind rephrasing with one of those "
            "shapes and a region code?"
        )
    return FallbackResult(mode="chat", message=msg)


def _extract_json_object(text: str) -> Optional[dict]:
    """Best-effort recovery of a JSON object from a free-text model reply,
    for models that ignore tool_choice and just answer in prose. Handles a
    bare object and one wrapped in a ```json fenced block."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None


def _to_result(data: dict, text: str, hint_region: Optional[str]) -> FallbackResult:
    mode = data.get("mode")

    if mode == "resolve":
        intent = data.get("intent")
        region = str(data.get("region") or hint_region or "").upper() or None
        if intent not in ("forecast", "historical", "compare", "anomalies"):
            return _static_fallback(text, hint_region)
        if intent in ("forecast", "historical") and region not in _VALID_REGIONS:
            # Model claimed it could resolve this but didn't give a real region -
            # don't trust it silently, fall back to an honest chat reply instead.
            return _static_fallback(text, hint_region)

        horizons: List[int] = []
        if intent == "forecast":
            from .queries import horizons_up_to
            try:
                hours = int(data.get("hours") or 1)
            except (TypeError, ValueError):
                hours = 1
            horizons = horizons_up_to(hours) or [1]

        days_ago = data.get("days_ago")
        try:
            days_ago = int(days_ago) if days_ago is not None else None
        except (TypeError, ValueError):
            days_ago = None

        parsed = ParsedQuery(
            intent=intent,
            region=region,
            horizons=horizons,
            days_ago=days_ago,
            severity=data.get("severity"),
            raw_text=text,
        )
        return FallbackResult(mode="resolve", parsed=parsed)

    message = str(data.get("message") or "").strip()
    if not message:
        return _static_fallback(text, hint_region)
    return FallbackResult(mode="chat", message=message)


def resolve(text: str, hint_region: Optional[str] = None) -> FallbackResult:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return _static_fallback(text, hint_region)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional but recommended by OpenRouter for attribution/rate-limit visibility.
        "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "https://gridcast.app"),
        "X-Title": "GridCast",
    }
    body = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "tools": [_TOOL_SCHEMA],
        "tool_choice": {"type": "function", "function": {"name": "respond"}},
        "max_tokens": 400,
        "temperature": 0.2,
    }

    try:
        resp = requests.post(_OPENROUTER_URL, headers=headers, json=body, timeout=_TIMEOUT_S)
        if not resp.ok:
            logger.warning("OpenRouter returned %s for model %r: %s", resp.status_code, _MODEL, resp.text[:500])
        resp.raise_for_status()
        payload = resp.json()
        message = (payload.get("choices") or [{}])[0].get("message", {})

        tool_calls = message.get("tool_calls") or []
        data: Optional[dict] = None
        if tool_calls:
            raw_args = tool_calls[0].get("function", {}).get("arguments", "")
            try:
                data = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                data = None

        if data is None:
            # Model ignored tool_choice (common on free-tier models) - try to
            # recover a JSON object from whatever plain text it replied with.
            data = _extract_json_object(message.get("content") or "")

        if data is None:
            logger.warning("OpenRouter reply had no usable structured content; using static reply")
            return _static_fallback(text, hint_region)

        return _to_result(data, text, hint_region)

    except Exception:
        logger.exception("OpenRouter fallback call failed; using static reply")
        return _static_fallback(text, hint_region)
