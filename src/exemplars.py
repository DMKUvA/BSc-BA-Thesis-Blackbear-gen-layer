import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

GOLDEN_PATH = Path("data/golden_sows.json")

with GOLDEN_PATH.open(encoding="utf-8") as f:
    ALL_SOWS: List[Dict[str, Any]] = json.load(f)


def _safe_str(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _get_response_json(exemplar: Dict[str, Any]) -> Dict[str, Any]:
    value = exemplar.get("response_json")
    if not isinstance(value, dict):
        value = exemplar.get("sow_json")  # temporary fallback for older golden files
    if isinstance(value, dict) and "SoW" in value:
        return value
    if isinstance(value, dict):
        return {"displayMessage": exemplar.get("display_message", ""), "SoW": value}
    return {}


def _get_sow(exemplar: Dict[str, Any]) -> Dict[str, Any]:
    response_json = _get_response_json(exemplar)
    sow = response_json.get("SoW")
    return sow if isinstance(sow, dict) else {}


def _has_valid_response_shape(exemplar: Dict[str, Any]) -> bool:
    response_json = _get_response_json(exemplar)
    if not response_json:
        return False
    if set(response_json.keys()) != {"displayMessage", "SoW"}:
        return False
    return isinstance(response_json.get("displayMessage"), str) and isinstance(response_json.get("SoW"), dict)


def _is_good(exemplar: Dict[str, Any]) -> bool:
    return _safe_str(exemplar.get("quality")) == "good"


def _extract_language(exemplar: Dict[str, Any]) -> str:
    sow = _get_sow(exemplar)
    return _safe_str(sow.get("language"))


def _extract_specialism(exemplar: Dict[str, Any]) -> str:
    return _safe_str(exemplar.get("specialism"))


def _extract_project_type(exemplar: Dict[str, Any]) -> str:
    return (
        _safe_str(exemplar.get("project_type"))
        or _safe_str(exemplar.get("projecttype"))
        or _safe_str(exemplar.get("type_label"))
    )


def _extract_budget_band(exemplar: Dict[str, Any]) -> Optional[int]:
    value = exemplar.get("budget_band")
    return value if isinstance(value, int) else None


def _extract_mode(exemplar: Dict[str, Any]) -> str:
    return _safe_str(exemplar.get("mode"))


def _has_non_empty_list_field(sow: Dict[str, Any], key: str) -> bool:
    value = sow.get(key)
    return isinstance(value, list) and any(isinstance(v, str) and v.strip() for v in value)


def _has_valid_boundaries(sow: Dict[str, Any]) -> bool:
    boundaries = sow.get("boundaries")
    if not isinstance(boundaries, dict):
        return False

    included = boundaries.get("includedActivities")
    out_of_scope = boundaries.get("outOfScope")

    return (
        isinstance(included, list)
        and any(isinstance(v, str) and v.strip() for v in included)
        and isinstance(out_of_scope, list)
        and any(isinstance(v, str) and v.strip() for v in out_of_scope)
    )


def _passes_minimum_content_quality(exemplar: Dict[str, Any]) -> bool:
    sow = _get_sow(exemplar)
    if not sow:
        return False

    required_strings = ["title", "purpose", "definitionOfDone", "timeline", "language", "type"]
    for key in required_strings:
        if not isinstance(sow.get(key), str) or not sow.get(key).strip():
            return False

    if not _has_valid_boundaries(sow):
        return False

    if not _has_non_empty_list_field(sow, "mustHaveRequirements"):
        return False

    if not _has_non_empty_list_field(sow, "resources"):
        return False

    return True


def _score_exemplar(exemplar: Dict[str, Any], context_profile: Dict[str, Any]) -> Tuple[int, int, int]:
    score = 0

    target_language = _safe_str(context_profile.get("language", "nl"))
    target_specialism = _safe_str(context_profile.get("specialism"))
    target_project_type = _safe_str(context_profile.get("project_type"))
    target_mode = _safe_str(context_profile.get("mode"))
    target_budget_band = context_profile.get("budget_band")

    ex_language = _extract_language(exemplar)
    ex_specialism = _extract_specialism(exemplar)
    ex_project_type = _extract_project_type(exemplar)
    ex_mode = _extract_mode(exemplar)
    ex_budget_band = _extract_budget_band(exemplar)

    if _is_good(exemplar):
        score += 100

    if _has_valid_response_shape(exemplar):
        score += 60

    if _passes_minimum_content_quality(exemplar):
        score += 40

    if target_language and ex_language == target_language:
        score += 35

    if target_specialism and ex_specialism == target_specialism:
        score += 30

    if target_project_type and ex_project_type == target_project_type:
        score += 20

    if target_mode and ex_mode == target_mode:
        score += 10

    budget_bonus = 0
    if isinstance(target_budget_band, int) and isinstance(ex_budget_band, int):
        budget_bonus = max(0, 15 - abs(ex_budget_band - target_budget_band))
    score += budget_bonus

    tie_break_budget = 0
    if isinstance(target_budget_band, int) and isinstance(ex_budget_band, int):
        tie_break_budget = -abs(ex_budget_band - target_budget_band)

    tie_break_quality = 1 if _passes_minimum_content_quality(exemplar) else 0

    return score, tie_break_quality, tie_break_budget


def _dedupe_exemplars(exemplars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []

    for ex in exemplars:
        response_json = _get_response_json(ex)
        if response_json:
            key = json.dumps(response_json, ensure_ascii=False, sort_keys=True)
        else:
            key = str(ex.get("id") or id(ex))

        if key in seen:
            continue

        seen.add(key)
        deduped.append(ex)

    return deduped


def select_exemplars(context_profile: Dict[str, Any], k: int = 2) -> List[Dict[str, Any]]:
    candidates = [ex for ex in ALL_SOWS if _is_good(ex)]

    if not candidates:
        candidates = list(ALL_SOWS)

    candidates = _dedupe_exemplars(candidates)

    ranked = sorted(
        candidates,
        key=lambda ex: _score_exemplar(ex, context_profile),
        reverse=True,
    )

    selected = ranked[:k]

    if len(selected) < k:
        leftovers = [ex for ex in ALL_SOWS if ex not in selected]
        leftovers = _dedupe_exemplars(leftovers)
        leftovers = sorted(
            leftovers,
            key=lambda ex: _score_exemplar(ex, context_profile),
            reverse=True,
        )
        selected.extend(leftovers[: max(0, k - len(selected))])

    return selected[:k]