from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import ValidationError

from .exemplars import select_exemplars
from .prompts import build_messages
from .schema import AssistantResponse

SPECIALISMS_PATH = Path("data/specialisms.json")


def _safe_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normalize_language(text: str, fallback: str = "en") -> str:
    text = (text or "").lower()
    dutch_markers = [
        " ik ", " je ", " jij ", " dit ", " dat ", " een ", " het ", " de ",
        " project ", " graag ", " kunnen ", " wil ",
    ]
    score = sum(1 for marker in dutch_markers if marker in f" {text} ")
    return "nl" if score >= 2 else fallback


def _compute_timeline(current_date: str) -> str:
    base = datetime.strptime(current_date, "%Y-%m-%d").date()
    start = base + timedelta(days=7)
    end = start + timedelta(days=30)
    return f"{start.isoformat()}{end.isoformat()}"


def _default_title(conversation: List[Dict[str, str]], backend_context: Dict[str, Any]) -> str:
    for msg in reversed(conversation):
        if msg.get("role") == "user":
            text = _safe_str(msg.get("content"))
            if text:
                short = text[:80].strip()
                return short[:255]
    user_name = _safe_str(backend_context.get("user_name"))
    return f"{user_name} - SoW draft"[:255] if user_name else "SoW draft"


def _default_display_message(language: str) -> str:
    if language == "nl":
        return "Ik heb een bijgewerkte concept-SoW voor je voorbereid."
    return "I've prepared an updated draft SoW for you."


def _normalize_specialism_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize_specialism_text(text: str) -> Set[str]:
    normalized = _normalize_specialism_text(text)
    return {token for token in normalized.split() if len(token) > 1}


def _load_specialisms() -> List[Dict[str, Any]]:
    if not SPECIALISMS_PATH.exists():
        return []

    with SPECIALISMS_PATH.open(encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        return []

    specialisms: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue

        name = _safe_str(item.get("name"))
        if not name:
            continue

        normalized = _normalize_specialism_text(name)
        tokens = _tokenize_specialism_text(name)

        specialisms.append(
            {
                "id": item.get("id"),
                "field_id": item.get("field_id"),
                "name": name,
                "normalized": normalized,
                "tokens": tokens,
            }
        )

    return specialisms


ALL_SPECIALISMS = _load_specialisms()
SPECIALISM_BY_NORMALIZED = {item["normalized"]: item for item in ALL_SPECIALISMS}


def _infer_specialism(last_user: str, backend_context: Dict[str, Any]) -> str:
    backend_specialism = _safe_str(backend_context.get("specialism"))
    if backend_specialism:
        return backend_specialism

    if not ALL_SPECIALISMS:
        return ""

    text = _safe_str(last_user)
    normalized_text = _normalize_specialism_text(text)
    text_tokens = _tokenize_specialism_text(text)

    if not normalized_text:
        return ""

    exact = SPECIALISM_BY_NORMALIZED.get(normalized_text)
    if exact:
        return exact["name"]

    phrase_matches: List[Tuple[int, int, str]] = []
    for item in ALL_SPECIALISMS:
        candidate = item["normalized"]
        if candidate and candidate in normalized_text:
            phrase_matches.append((len(candidate), len(item["tokens"]), item["name"]))

    if phrase_matches:
        phrase_matches.sort(reverse=True)
        return phrase_matches[0][2]

    scored: List[Tuple[int, int, int, str]] = []
    for item in ALL_SPECIALISMS:
        candidate_tokens = item["tokens"]
        if not candidate_tokens:
            continue

        overlap = text_tokens & candidate_tokens
        if not overlap:
            continue

        overlap_count = len(overlap)
        candidate_size = len(candidate_tokens)
        score = overlap_count * 10

        if overlap_count == candidate_size:
            score += 25

        coverage = overlap_count / candidate_size
        score += int(coverage * 10)

        scored.append((score, overlap_count, candidate_size, item["name"]))

    if scored:
        scored.sort(reverse=True)
        best_score, best_overlap, _, best_name = scored[0]
        if best_score >= 20 and best_overlap >= 2:
            return best_name

    return ""


def _infer_context_profile(
    conversation: List[Dict[str, str]],
    backend_context: Optional[Dict[str, Any]] = None,
    current_sow: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    backend_context = backend_context or {}
    current_sow = current_sow or {}

    last_user = ""
    for msg in reversed(conversation):
        if msg.get("role") == "user":
            last_user = msg.get("content", "")
            break

    language = _normalize_language(last_user, backend_context.get("language_preference", "en"))
    specialism = _infer_specialism(last_user, backend_context)
    project_type = _safe_str(current_sow.get("project_type") or backend_context.get("project_type"))
    mode = _safe_str(backend_context.get("mode", "creator_v2"))
    budget_band = backend_context.get("budget_band")

    return {
        "language": language,
        "specialism": specialism,
        "project_type": project_type,
        "mode": mode,
        "budget_band": budget_band,
    }


def _ensure_minimum_sow(
    sow: Dict[str, Any],
    conversation: List[Dict[str, str]],
    current_date: str,
    backend_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    backend_context = backend_context or {}

    language = sow.get("language")
    if language not in {"nl", "en"}:
        last_user = ""
        for msg in reversed(conversation):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break
        language = _normalize_language(last_user, backend_context.get("language_preference", "en"))

    sow.setdefault("title", _default_title(conversation, backend_context))
    sow.setdefault("purpose", "")
    sow.setdefault("definitionOfDone", "")
    sow.setdefault("boundaries", {})
    sow["boundaries"].setdefault("includedActivities", [])
    sow["boundaries"].setdefault("outOfScope", [])
    sow.setdefault("mustHaveRequirements", [])
    sow.setdefault("niceToHaveRequirements", [])
    sow.setdefault("timeline", _compute_timeline(current_date))
    sow.setdefault("budget", {})
    sow["budget"].setdefault("costestimate", None)
    sow["budget"].setdefault("hourlyrate", None)
    sow["budget"].setdefault("averageweeklyhours", None)
    sow.setdefault("resources", [])
    sow.setdefault("location", {})
    sow["location"].setdefault("workingtype", "hybrid")
    sow["location"].setdefault("worklocation", None)
    sow["language"] = language
    sow.setdefault("type", "timebased")
    sow.setdefault("isFinalized", False)
    sow.setdefault("percentage", 0)

    if sow["location"]["workingtype"] == "remote":
        sow["location"]["worklocation"] = None

    return sow


def _coerce_for_validation(
    payload: Dict[str, Any],
    conversation: List[Dict[str, str]],
    current_date: str,
    backend_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    backend_context = backend_context or {}

    display_message = payload.get("displayMessage")
    if not isinstance(display_message, str) or not display_message.strip():
        lang = backend_context.get("language_preference", "en")
        display_message = _default_display_message(lang)

    sow = payload.get("SoW")
    if not isinstance(sow, dict):
        sow = {}

    sow = _ensure_minimum_sow(sow, conversation, current_date, backend_context)

    if sow.get("type") == "materialbased":
        if sow["budget"].get("costestimate") is None:
            sow["budget"]["costestimate"] = 30000
        sow["budget"]["hourlyrate"] = None
        sow["budget"]["averageweeklyhours"] = None

    if sow.get("type") == "timebased":
        if sow["budget"].get("hourlyrate") is None:
            sow["budget"]["hourlyrate"] = 7500
        if sow["budget"].get("averageweeklyhours") is None:
            sow["budget"]["averageweeklyhours"] = 20

    if sow.get("location", {}).get("workingtype") == "remote":
        sow["location"]["worklocation"] = None

    return {
        "displayMessage": display_message.strip(),
        "SoW": sow,
    }


def parse_assistant_response(
    raw_text: str,
    conversation: List[Dict[str, str]],
    current_date: str,
    backend_context: Optional[Dict[str, Any]] = None,
) -> AssistantResponse:
    backend_context = backend_context or {}

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {exc}") from exc

    payload = _coerce_for_validation(payload, conversation, current_date, backend_context)

    # Small robustness fix: normalize "YYYY-MM-DD YYYY-MM-DD" → "YYYY-MM-DDYYYY-MM-DD"
    sow = payload.get("SoW")
    if isinstance(sow, dict):
        timeline = sow.get("timeline")
        if isinstance(timeline, str) and len(timeline) == 21 and timeline[10] == " ":
            left = timeline[:10]
            right = timeline[11:]
            if left.count("-") == 2 and right.count("-") == 2:
                sow["timeline"] = left + right

    try:
        return AssistantResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Assistant response failed schema validation: {exc}") from exc


def build_generation_request(
    conversation: List[Dict[str, str]],
    current_date: str,
    backend_context: Optional[Dict[str, Any]] = None,
    current_sow: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    backend_context = backend_context or {}
    current_sow = current_sow or {}

    context_profile = _infer_context_profile(
        conversation=conversation,
        backend_context=backend_context,
        current_sow=current_sow,
    )

    exemplars = select_exemplars(context_profile, k=2)

    return build_messages(
        conversation=conversation,
        exemplars=exemplars,
        current_date=current_date,
        backend_context=backend_context,
        current_sow=current_sow,
    )