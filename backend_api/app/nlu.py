"""
Rule-based natural-language router. No API key, no external model call.

This is deliberately the thinnest possible layer: regex/keyword matching over
a small set of phrasings, each mapped to exactly one of the query shapes in
queries.py - the same functions the plain REST endpoints call. It exists to
prove the end-to-end shape ("user types a sentence -> gets a grounded
answer") without taking on an LLM dependency before the demo.

SWAPPING IN A REAL LLM LATER
-----------------------------
Replace `parse_query()`'s body with a call to the Anthropic API (or any LLM),
prompted to extract {intent, region, horizons, severity, days_ago} as JSON
from the user's sentence, then feed that structured object into the exact
same `queries.py` functions used below. Nothing else in this file, or in
routers/ask.py, needs to change - the LLM only ever replaces the parsing
step, never the part that touches the database. That boundary is the reason
this can be swapped without redesigning anything.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .queries import horizons_up_to

_REGION_ALIASES = {
    "aep": "AEP", "comed": "COMED", "com ed": "COMED", "dayton": "DAYTON",
    "deok": "DEOK", "dom": "DOM", "dominion": "DOM", "duq": "DUQ",
    "duquesne": "DUQ", "ekpc": "EKPC", "fe": "FE", "firstenergy": "FE",
    "ni": "NI", "nipsco": "NI", "pjm load": "PJM_Load", "pjm_load": "PJM_Load",
    "pjme": "PJME", "pjmw": "PJMW",
}

_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
    "a day": 24, "day": 24, "week": 168,
}


@dataclass
class ParsedQuery:
    intent: str                    # forecast | historical | compare | anomalies | unknown
    region: Optional[str] = None
    horizons: List[int] = field(default_factory=list)
    days_ago: Optional[int] = None
    severity: Optional[str] = None
    raw_text: str = ""


def _find_region(text: str) -> Optional[str]:
    low = text.lower()
    for alias, code in _REGION_ALIASES.items():
        if alias in low:
            return code
    # bare upper-case region codes typed directly, e.g. "AEP"
    for m in re.finditer(r"\b([A-Z]{2,9}(?:_[A-Z]+)?)\b", text):
        code = m.group(1)
        if code in _REGION_ALIASES.values() or code.lower().replace("_", " ") in _REGION_ALIASES:
            return code
    return None


def _find_hour_count(text: str) -> Optional[int]:
    low = text.lower()
    m = re.search(r"(\d+)\s*hour", low)
    if m:
        return int(m.group(1))
    for word, n in _NUM_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b\s*hour", low):
            return n
    if "next hour" in low or "hourly" in low:
        return 1
    return None


def _find_days_ago(text: str) -> Optional[int]:
    low = text.lower()
    if "yesterday" in low or "previous day" in low:
        return 1
    if "last week" in low or "a week ago" in low:
        return 7
    m = re.search(r"(\d+)\s*day", low)
    if m:
        return int(m.group(1))
    return None


def _find_severity(text: str) -> Optional[str]:
    low = text.lower()
    for sev in ("critical", "high", "medium", "low"):
        if sev in low:
            return sev.upper()
    return None


_COMPARE_WORDS = (
    "which model", "best model", "compare model", "model win", "ranking",
    "top model", "winning model", "most accurate", "accuracy comparison",
    "wape", "model performance", "better model", "model comparison",
    "leaderboard", "rank the model", "score the model", "compare",
)

_ANOMALY_WORDS = (
    "anomaly", "anomalies", "unusual", "spike", "outage", "alert",
    "critical", "situation", "issue", "problem", "concern", "abnormal",
    "irregular", "warning", "risk", "emergency", "flagged", "flag",
    "trouble", "weird", "strange", "odd", "deviation", "fault", "glitch",
    "malfunction", "off-track", "off track", "wrong", "surge", "drop-off",
    "dropoff", "unexpected", "red flag", "incident",
)

_HISTORICAL_WORDS = (
    "was", "were", "usage", "demand", "actual", "history", "historical",
    "recorded", "logged", "recent", "past", "previous", "yesterday", "ago",
    "so far", "up to now", "already happened", "real value", "real usage",
)

_FORECAST_WORDS = (
    "forecast", "predict", "prediction", "predicted", "expected", "next",
    "upcoming", "outlook", "projection", "projected", "future", "tomorrow",
    "today", "now", "current", "estimate", "estimated", "will be", "going to",
    "coming hours", "coming days", "ahead",
)

_DEMAND_WORDS = ("demand", "energy", "usage", "consumption", "load", "power")


def parse_query(text: str) -> ParsedQuery:
    low = text.lower()
    region = _find_region(text)

    if any(k in low for k in _COMPARE_WORDS):
        return ParsedQuery(intent="compare", region=region, raw_text=text)

    if any(k in low for k in _ANOMALY_WORDS):
        return ParsedQuery(intent="anomalies", region=region, severity=_find_severity(text), raw_text=text)

    days_ago = _find_days_ago(text)
    if days_ago is not None and any(k in low for k in _HISTORICAL_WORDS):
        return ParsedQuery(intent="historical", region=region, days_ago=days_ago, raw_text=text)

    hours = _find_hour_count(text)
    if any(k in low for k in _FORECAST_WORDS) or hours:
        horizon_list = horizons_up_to(hours or 1) or [1]
        return ParsedQuery(intent="forecast", region=region, horizons=horizon_list, raw_text=text)

    # A region plus a bare demand/energy word ("what is the energy demand in
    # DOM") doesn't name forecast/historical/compare/anomalies explicitly, but
    # asking the user to disambiguate every time is bad UX for the single most
    # obvious reading: "show me the current outlook". Default to a 1h forecast
    # rather than falling through to "unknown" for this shape specifically.
    if region and any(k in low for k in _DEMAND_WORDS):
        return ParsedQuery(intent="forecast", region=region, horizons=horizons_up_to(1) or [1], raw_text=text)

    return ParsedQuery(intent="unknown", region=region, raw_text=text)
