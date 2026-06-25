import json
import re
from datetime import datetime, timedelta
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

SOWS_SOURCE = DATA_DIR / "sows.json"
GOLDEN_TARGET = DATA_DIR / "golden_sows.json"


GOOD_IDS = {
    "9f116dcd-34c5-49fa-9b8e-d5ce5b825583",
    "9f314664-0181-4560-9734-ab504e722386",
    "9f314875-5201-4012-a9fa-c7dabbdb24b5",
    "9f4b9c2e-d7be-45e8-bfdf-7e68aa7c473e",
    "9f65b011-0e71-4599-be26-d894d79101df",
    "9f6bb838-a91c-4868-b951-6b5988760d37",
    "9f7132f1-f260-43cc-b41a-430e1045ed00",
    "9fc60208-0f06-4777-b18b-c64319a4fb98",
    "9ff3e467-9a8a-4273-8312-3dc4123f2534",
    "a0288dbf-b287-4db0-a8c4-03c7e9eb96d6",
    "a03e998a-8cd2-4ce4-9cac-d4d4c8bf69c3",
    "a03ed355-b8c7-4360-b9fc-e2b09d2ccddc",
    "a0451d0e-e6a7-4f0b-ab7d-ba5421d1b4bf",
    "a056f8af-35a2-43bc-8d5f-547f8c8fdae7",
    "a05b146f-7782-4b5f-90d3-ae782c00bcce",
    "a0850eea-ac55-4ebb-b12c-3807db60ea04",
    "a0d263ca-c7d7-4d86-a7ec-b967f0905bf4",
    "a0d46618-bbf6-4211-ad0c-3c954dc8405f",
    "a0dfb827-c714-49e6-aeba-38e20d3476c9",
    "a0b57cad-2b1f-479b-ba86-619ae925ee1d",
}

BAD_IDS = {
    "a01bf804-4d12-49f2-97a1-962608ac30bf",
    "a0387e64-b52a-4aa2-95ff-cb25a1d12be7",
    "a0451b2c-03ad-427b-a4ea-a6db607cfd70",
}


def normalize_for_inference(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.lower().strip().split())


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = unescape(text)
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</?(ul|ol|p)>", " ", text, flags=re.I)
    text = re.sub(r"</?li>", ",", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip().strip('"').strip("'")


def find_quality(record_id: str) -> Optional[str]:
    if record_id in GOOD_IDS:
        return "good"
    if record_id in BAD_IDS:
        return "bad"
    return None


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        key = normalize_for_inference(item)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item[:255])
    return result


def parse_list_field(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        cleaned = [clean_text(v) for v in value]
        return _dedupe_keep_order([v for v in cleaned if v])

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []

        for candidate in [raw, raw.strip('"').replace('\\"', '"')]:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    cleaned = [clean_text(v) for v in parsed]
                    return _dedupe_keep_order([v for v in cleaned if v])
            except json.JSONDecodeError:
                pass

        cleaned = unescape(raw)
        cleaned = re.sub(r"</?ul>", " ", cleaned, flags=re.I)
        cleaned = re.sub(r"</?ol>", " ", cleaned, flags=re.I)
        cleaned = re.sub(r"<li>", "|", cleaned, flags=re.I)
        cleaned = re.sub(r"</li>", " ", cleaned, flags=re.I)
        cleaned = cleaned.replace("ulli", "|").replace("liul", " ").replace("lili", "|")
        cleaned = cleaned.replace("\n- ", "|").replace("\n• ", "|").replace("; ", "|")
        cleaned = re.sub(r"\s*\|\s*", "|", cleaned)
        parts = [clean_text(p).strip(" -•,") for p in cleaned.split("|")]
        parts = [p for p in parts if p]
        return _dedupe_keep_order(parts)

    return []


def to_int_or_none(value: Any) -> Optional[int]:
    if value in (None, "", "null"):
        return None
    try:
        if isinstance(value, str):
            value = value.replace("€", "").replace(".", "").replace(",", "").strip()
        return int(value)
    except (TypeError, ValueError):
        return None


def map_type(raw_type: Optional[str]) -> str:
    raw = normalize_for_inference(raw_type or "")
    if raw in {"material_based", "materialbased", "fixed", "fixed price", "project based"}:
        return "materialbased"
    if raw in {"time_based", "timebased", "hourly", "rate", "hours", "day rate"}:
        return "timebased"
    return "timebased"


def infer_language(record: Dict[str, Any]) -> str:
    text = " ".join(
        str(record.get(k, "") or "")
        for k in ["title", "objective", "definition_of_done"]
    ).lower()

    dutch_markers = [
        "het doel", "de opdracht", "wanneer", "opgeleverd",
        "onderzoek", "ondersteuning", "ontwikkeling", "gegevensdeling",
    ]
    for marker in dutch_markers:
        if marker in text:
            return "nl"
    return "en"


def infer_budget_band(cost_estimate: Optional[int], hourly_rate: Optional[int]) -> int:
    if cost_estimate is not None:
        if cost_estimate < 1_000_000:
            return 1
        if cost_estimate < 5_000_000:
            return 2
        return 3

    if hourly_rate is not None:
        if hourly_rate < 7500:
            return 1
        if hourly_rate < 12500:
            return 2
        return 3

    return 1


def infer_context_summary(record: Dict[str, Any]) -> str:
    objective = clean_text(record.get("objective"))
    if objective:
        return objective[:350]
    return clean_text(record.get("title"))[:350]


def infer_is_finalized(record: Dict[str, Any]) -> bool:
    status = normalize_for_inference(record.get("status") or "")
    required_present = all([
        clean_text(record.get("title")),
        clean_text(record.get("objective")),
        clean_text(record.get("definition_of_done")),
    ])
    return status in {"completed", "contracting", "archived"} and required_present


def infer_percentage(record: Dict[str, Any]) -> int:
    status = normalize_for_inference(record.get("status") or "")
    if status == "completed":
        return 100
    if status in {"contracting", "archived", "active", "sourcing"}:
        return 75
    if status == "draft":
        return 30
    return 0


def infer_specialism(title: str, objective: str = "") -> str:
    t = normalize_for_inference(f"{title} {objective}")

    # --- HIGH-SPECIFICITY MATCHES FIRST ---

    # Tender / proposal work
    if any(k in t for k in ["tender", "aanbesteding", "aanbestedingsdocumenten", "offerte", "request for proposal", "rfp"]):
        return "Tender & Proposal Management"

    # RO-advies / permits: process-heavy advisory work
    if any(k in t for k in ["ro-advies", "ro advies", "ruimtelijke ordening", "omgevingsvergunning", "vergunningaanvragen", "bestemmingsplan"]):
        return "Process Management"

    # Records / data-quality / archiving work
    if any(k in t for k in [
        "records management", "recordsmanagement", "archief", "archivering",
        "dossiervorming", "dossierbeheer", "datakwaliteit", "data quality",
        "classificatieschema", "informatiemanagement"
    ]):
        return "Process Management"

    # ESG / policy-heavy projects (explicit "policy" + ESG-ish signals)
    if "policy" in t or "beleid" in t:
        if any(k in t for k in ["esg", "csrd", "csddd", "v sme", "vsme", "workers in the value chain", "circularity", "s2 ", "e5 "]):
            return "Policy Development"

    # QHSE / certification / audits
    if any(k in t for k in ["qhse", "iso 9001", "iso 14001", "iso 45001", "vca", "lead auditor", "audit", "certificering", "certification"]):
        return "Quality & Compliance"

    # Explicit project leadership
    if any(k in t for k in ["projectleider", "project manager", "projectmanagement", "project management"]):
        return "Project Management"

    # PMO / governance
    if any(k in t for k in ["pmo", "programmabureau", "governance board"]):
        return "PMO / Governance"

    # Agile / product roles
    if any(k in t for k in ["agile", "scrum", "scrum master", "product owner", "product ownership"]):
        return "Agile / Scrum"

    # --- STRATEGY / POLICY / CHANGE ---

    # Strategy
    if any(k in t for k in ["strategie", "strategy", "future vision", "toekomstvisie"]):
        return "Business Strategy"

    # Organizational design / reorganisations
    if any(k in t for k in ["organisatieontwerp", "organizational design", "reorganisatie", "restructuring"]):
        return "Organizational Design"

    # Digital transformation
    if any(k in t for k in ["digitale transformatie", "digital transformation"]):
        return "Digital Transformation"

    # Generic policy work (only if we haven't already classified HR / ESG / records / RO above)
    if any(k in t for k in ["beleid", "policy", "richtlijn", "guideline", "framework"]):
        return "Policy Development"

    # --- PROCESS / OPERATIONS ---

    if any(k in t for k in ["process optimization", "procesoptimalisatie", "optimalisatie", "procesverbetering", "process improvement", "lean", "six sigma", "kaizen"]):
        return "Process Optimization"

    if any(k in t for k in ["operational excellence", "operational efficiency"]):
        return "Process Management"

    # --- MARKETING / CONTENT / CREATIVE ---

    # Creative content / animation / video
    if any(k in t for k in ["animatie", "animatiefilm", "animation", "video", "storyboard", "illustratie", "illustration"]):
        return "Content Marketing"

    # Strong content signals
    if any(k in t for k in ["content", "copy", "copywriting", "blog", "blogs", "nieuwsbrief", "newsletter", "social media", "linkedin", "instagram"]):
        return "Content Marketing"

    # Branding should be about brand/positioning, not just “marketing”
    if any(k in t for k in ["rebranding", "brand identity", "merkidentiteit", "positionering", "brand strategy", "merkstrategie", "huisstijl"]):
        return "Branding"

    # Generic marketing (no obvious content cues) falls back to Digital Marketing
    if "marketing" in t:
        return "Digital Marketing"

    # Market research only when we see classic research language
    if any(k in t for k in ["marktonderzoek", "market research", "survey", "enquête", "enquete", "respondenten", "focusgroep", "focus group"]):
        return "Market Research"

    # --- HR / PEOPLE ---

    # Verzuim / casemanagement should *not* be policy
    if any(k in t for k in ["verzuimcoach", "verzuim", "casemanager", "re-integratie", "reintegratie"]):
        return "HR Administration"

    if any(k in t for k in ["recruitment", "werving", "selection", "talent acquisition", "headhunting"]):
        return "Talent Acquisition"

    if any(k in t for k in ["hr analytics", "people analytics"]):
        return "HR Analytics"

    if any(k in t for k in ["learning & development", "learning and development", "l&d", "opleidingsprogramma", "training programma"]):
        return "Learning & Development"

    if any(k in t for k in ["change management", "cultuurverandering", "verandermanagement"]):
        return "Change Management"

    if any(k in t for k in ["coaching", "coach", "facilitatie", "facilitation", "workshop"]):
        return "Coaching & Facilitation"

    if any(k in t for k in ["interne communicatie", "internal communication"]):
        return "Internal Communication"

    # --- DATA / BI / IT ---

    if any(k in t for k in ["bi", "business intelligence", "power bi", "dashboard", "dashboards", "kpi report", "reporting"]):
        return "Reporting & Dashboards"

    if any(k in t for k in ["data analysis", "data-analyse", "data analyse", "analytics", "analyses"]):
        return "Data Analysis"

    if any(k in t for k in ["data engineering", "data engineer", "etl", "pipeline"]):
        return "Data Engineering"

    if any(k in t for k in ["automation", "automatisering", "rpa"]):
        return "Automation"

    if any(k in t for k in ["it support", "helpdesk"]):
        return "IT Support"

    if any(k in t for k in ["system administration", "systeembeheer", "beheerder"]):
        return "System Administration"

    if any(k in t for k in ["cloud", "azure", "aws", "gcp", "infrastructure"]):
        return "Cloud & Infrastructure"

    if any(k in t for k in ["cybersecurity", "security", "informatiebeveiliging", "infosec"]):
        return "Cybersecurity"

    if any(k in t for k in ["software implementation", "implementatie", "erp", "sap", "d365", "lims"]):
        return "Software Implementation"

    # --- DEFAULT ---

    return "Project Management"


def parse_iso_date(text: str) -> Optional[str]:
    text = clean_text(text)
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def build_timeline(start_date_raw: Any, end_date_raw: Any) -> str:
    start = parse_iso_date(start_date_raw)
    end = parse_iso_date(end_date_raw)

    if start and end:
        return f"{start}{end}"

    if start and not end:
        end_date = datetime.strptime(start, "%Y-%m-%d").date() + timedelta(days=30)
        return f"{start}{end_date.isoformat()}"

    if not start and end:
        start_date = datetime.strptime(end, "%Y-%m-%d").date() - timedelta(days=30)
        return f"{start_date.isoformat()}{end}"

    fallback_start = datetime(2026, 1, 1).date()
    fallback_end = fallback_start + timedelta(days=30)
    return f"{fallback_start.isoformat()}{fallback_end.isoformat()}"


def ensure_non_empty_list(items: List[str], fallback: List[str]) -> List[str]:
    cleaned = _dedupe_keep_order([clean_text(x) for x in items if clean_text(x)])
    return cleaned if cleaned else fallback


def normalize_budget_for_type(
    sow_type: str,
    cost_estimate: Optional[int],
    hourly_rate: Optional[int],
    average_weekly_hours: Optional[int],
) -> Dict[str, Optional[int]]:
    if sow_type == "materialbased":
        return {
            "costestimate": max(30000, cost_estimate or 30000),
            "hourlyrate": None,
            "averageweeklyhours": None,
        }

    return {
        "costestimate": None,
        "hourlyrate": max(3000, hourly_rate or 7500),
        "averageweeklyhours": min(40, max(4, average_weekly_hours or 20)),
    }


def build_display_message(language: str, title: str, finalized: bool) -> str:
    if language == "nl":
        if finalized:
            return f"Je concept-SoW voor **{title}** staat klaar en is volledig ingevuld."
        return f"Ik heb een concept-SoW voorbereid voor **{title}**."
    if finalized:
        return f"Your draft SoW for **{title}** is ready and fully filled in."
    return f"I've prepared a draft SoW for **{title}**."


def map_record_to_golden_entry(record: Dict[str, Any], quality: str) -> Dict[str, Any]:
    title = clean_text(record.get("title"))[:255] or "SoW draft"
    objective = clean_text(record.get("objective"))
    language = infer_language(record)
    sow_type = map_type(record.get("type"))

    included_activities = ensure_non_empty_list(
        parse_list_field(record.get("included_activities")),
        ["Core project execution activities aligned with the project objective"],
    )
    out_of_scope = ensure_non_empty_list(
        parse_list_field(record.get("out_of_scope")),
        ["Activities outside the agreed project scope and post-project support"],
    )
    must_have = ensure_non_empty_list(
        parse_list_field(record.get("must_have")),
        ["Relevant professional experience in similar projects"],
    )
    nice_to_have = _dedupe_keep_order(parse_list_field(record.get("nice_to_have")))
    resources = ensure_non_empty_list(
        parse_list_field(record.get("resources")),
        ["Access to key stakeholders, relevant documentation, and required systems"],
    )

    cost_estimate = to_int_or_none(record.get("cost_estimate"))
    hourly_rate = to_int_or_none(record.get("hourly_rate"))
    average_weekly_hours = to_int_or_none(record.get("average_weekly_hours"))
    normalized_budget = normalize_budget_for_type(
        sow_type=sow_type,
        cost_estimate=cost_estimate,
        hourly_rate=hourly_rate,
        average_weekly_hours=average_weekly_hours,
    )

    working_type = normalize_for_inference(record.get("location_type") or "hybrid")
    if working_type not in {"remote", "hybrid", "onsite"}:
        working_type = "hybrid"

    work_location = clean_text(record.get("location"))[:255] or None
    if working_type == "remote":
        work_location = None
    elif working_type in {"hybrid", "onsite"} and not work_location:
        work_location = "tbd"

    is_finalized = infer_is_finalized(record)
    percentage = 100 if is_finalized else infer_percentage(record)

    sow_json = {
        "title": title,
        "purpose": objective or "tbd",
        "definitionOfDone": clean_text(record.get("definition_of_done")) or "tbd",
        "boundaries": {
            "includedActivities": included_activities,
            "outOfScope": out_of_scope,
        },
        "mustHaveRequirements": must_have,
        "niceToHaveRequirements": nice_to_have,
        "timeline": build_timeline(record.get("start_date"), record.get("end_date")),
        "budget": normalized_budget,
        "resources": resources,
        "location": {
            "workingtype": working_type,
            "worklocation": work_location,
        },
        "language": language,
        "type": sow_type,
        "isFinalized": is_finalized,
        "percentage": percentage,
    }

    response_json = {
        "displayMessage": build_display_message(language, title, is_finalized),
        "SoW": sow_json,
    }

    return {
        "id": str(record.get("id") or "").strip(),
        "source_title": title,
        "quality": quality,
        "specialism": infer_specialism(title, objective),
        "budget_band": infer_budget_band(
            cost_estimate=normalized_budget.get("costestimate"),
            hourly_rate=normalized_budget.get("hourlyrate"),
        ),
        "mode": "creator_v2",
        "context_summary": infer_context_summary(record),
        "response_json": response_json,
    }


def main() -> None:
    if not SOWS_SOURCE.exists():
        raise FileNotFoundError(f"Bronbestand niet gevonden: {SOWS_SOURCE}")

    with SOWS_SOURCE.open("r", encoding="utf-8") as f:
        all_sows = json.load(f)

    if not isinstance(all_sows, list):
        raise ValueError("data/sows.json moet een lijst met records bevatten.")

    golden_entries: List[Dict[str, Any]] = []
    matched_ids: List[str] = []
    matched_good = 0
    matched_bad = 0

    for record in all_sows:
        if not isinstance(record, dict):
            continue

        record_id = str(record.get("id") or "").strip()
        quality = find_quality(record_id)
        if quality is None:
            continue

        if quality == "good":
            matched_good += 1
        elif quality == "bad":
            matched_bad += 1

        golden_entries.append(map_record_to_golden_entry(record, quality))
        matched_ids.append(record_id)

    GOLDEN_TARGET.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN_TARGET.open("w", encoding="utf-8") as f:
        json.dump(golden_entries, f, ensure_ascii=False, indent=2)

    print(f"✅ Written {len(golden_entries)} entries to {GOLDEN_TARGET}")
    print(f"Matched good: {matched_good}/{len(GOOD_IDS)}")
    print(f"Matched bad: {matched_bad}/{len(BAD_IDS)}")
    print("\nMatched ids:")
    for rid in matched_ids:
        print(f"- {rid}")

    expected_total = len(GOOD_IDS) + len(BAD_IDS)
    print(f"\nExpected: {expected_total}")
    print(f"Found:    {len(golden_entries)}")

    if len(golden_entries) != expected_total:
        print("\n⚠️ Let op: niet alle ids zijn gevonden. Controleer de matched ids hierboven.")


if __name__ == "__main__":
    main()