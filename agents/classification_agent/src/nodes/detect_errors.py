import os
import json
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from agents.classification_agent.src.utils import RawReview, DetectedError, get_llm_from_config
from agents.classification_agent.src.nodes.category_normalizer import get_normalizer

SYSTEM = """You are an expert at analyzing customer reviews and identifying REAL issues.

Task: Given a customer review, extract ZERO OR MORE concrete PRODUCT/SERVICE PROBLEMS OR FEATURE/ENHANCEMENT REQUESTS that are ACTUALLY MENTIONED in the review text.

**CRITICAL**: ONLY extract issues that are EXPLICITLY stated in the review. DO NOT infer or hallucinate problems. If the review is positive or neutral with no issues, return empty errors array.

**IMPORTANT**: First understand what type of business this review is about, then generate appropriate category names.

Return ONLY valid JSON with this exact shape:
{
  "business_type": "restaurant|software|hotel|retail|service|other",
  "errors": [
    {
      "error_summary": "Brief description of the issue",
      "error_type": ["Category1", "Category2"],
      "severity": "Severity Level",
      "rationale": "Why this categorization?"
    }
  ]
}

Guidelines for category generation:
- **Understand the business context first**
  * Restaurant/Food: Use categories like "Food Quality", "Service", "Cleanliness", "Wait Time", "Pricing", "Ambiance"
  * Software/SaaS: Use categories like "Stability", "Performance", "Authentication", "Billing", "User Interface", "API"
  * Hotel: Use categories like "Room Condition", "Staff", "Amenities", "Cleanliness", "Location", "Check-in"
  * Retail/E-commerce: Use categories like "Product Quality", "Shipping", "Customer Service", "Returns", "Pricing"
  * Booking/Appointment Services: Use categories like "Booking Process", "Wait Time", "Scheduling", "Customer Service"
  * Other businesses: Generate appropriate categories based on industry

- **Category naming - BE CONSISTENT**:
  * Use 1-3 word category names that are GENERIC and REUSABLE
  * Think: "What is the CORE issue?" not specific details
  * Examples of GOOD categories: "Booking Process", "Wait Time", "Service Quality"
  * Examples of BAD categories: "3 hour wait", "slow elongated process", "inefficient scheduling"
  * If multiple reviews mention delays/slowness in the same area, use the SAME category name
  * Avoid overly specific or one-off category names
  * Multiple categories OK if issue spans multiple areas (max 2-3)

- **Severity levels** (use EXACTLY one of these values):
  * "Critical" - for severe issues (crashes, data loss, health/safety, system down)
  * "Major" - for significant problems (broken features, major bugs, poor service)
  * "Minor" - for small issues (UI glitches, minor inconveniences)
  * "Suggestion" - for feature requests and enhancements
  * "None" - if no real issue detected (positive reviews)

- error_summary <= 140 chars, actionable (what/where)
- For feature requests, start with "Feature request:" or "Enhancement:"
- If no problem or request is present, return {"business_type": "...", "errors": []}
- Output must be ONLY the JSON object (no prose)
"""

# few shot to guide the model
FEWSHOT_USER = """Review:
```
Not thrilled about how the mobile app crashes whenever I switch workspaces. Lost my draft twice.
```
Return JSON only:"""

FEWSHOT_ASSISTANT = """{
  "business_type": "software",
  "errors": [{
    "error_summary": "Mobile app crashes when switching workspaces",
    "error_type": ["Stability", "Mobile App"],
    "severity": "Critical",
    "rationale": "User reports reproducible crash and data loss while switching workspaces."
  }]
}"""

USER = """Review:
```
{review_text}
```
Return JSON only:"""

FEWSHOT_USER_2 = """Review:
```
The BBQ tasting palette did not live up to the hype. Food was cold and took 45 minutes.
```
Return JSON only:"""

FEWSHOT_ASSISTANT_2 = """{
  "business_type": "restaurant",
  "errors": [{
    "error_summary": "Disappointing BBQ quality and cold food with long wait",
    "error_type": ["Food Quality", "Food Temperature", "Wait Time"],
    "severity": "Major",
    "rationale": "Multiple food quality issues including temperature and excessive wait time."
  }]
}"""


def make_llm():
    """Create LLM from config.yaml"""
    return get_llm_from_config(temperature=0)

def _json_load(s: str) -> dict:
    if not s:
        return {"errors": []}
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        a, b = s.find("{"), s.rfind("}")
        if a != -1 and b != -1 and b > a:
            try:
                return json.loads(s[a:b+1])
            except Exception:
                return {"errors": []}
        return {"errors": []}

#keyword fallback to guarantees something for obvious cases
FALLBACK_RULES = [
    ("Mobile app crashes when switching workspaces", ["Mobile","Crash"],
     ["crash", "crashes", "crashing", "workspace"]),
    ("Invoice total mismatches usage/billing metrics", ["Billing"],
     ["invoice", "overcharged", "duplicate charge", "billing mismatch", "billing error"]),
    ("Authentication/session expires unexpectedly", ["Auth"],
     ["auth", "authentication", "session expires", "token expires", "login fails", "cannot login", "can't login"]),
    ("API requests time out or are very slow", ["API","Performance"],
     ["timeout", "time out", "very slow", "slow request", "latency", "rate limit"]),
    ("Webhooks deliver duplicates or invalid signatures", ["Webhooks"],
     ["webhook", "duplicate event", "signature fail", "signature verification"]),
    ("Docs are outdated or inconsistent", ["Docs"],
     ["docs outdated", "documentation outdated", "example wrong", "missing example"]),
    ("UI layout shifts or elements off-canvas", ["UI"],
     ["layout shift", "off-canvas", "alignment", "button missing", "overlapping"]),
]



def _fallback_detect(text: str) -> List[DetectedError]:
    s = text.lower()
    out: List[DetectedError] = []
    for summary, types, kws in FALLBACK_RULES:
        if any(k in s for k in kws):
            out.append(DetectedError(
                error_summary=summary[:140],
                error_type=types,
                rationale="Keyword-based heuristic match from review text."
            ))
    return out

#deterministic fallback for SUGGESTIONS
SUGGESTION_TERMS = [
    "feature request", "would be nice", "could you add", "please add",
    "any chance of", "add support for", "add an option to", "option to",
    "ability to", "please allow", "allow us to", "nice to have",
    "would love", "i wish", "it would help if", "consider adding",
    "integration with", "export to csv", "export to excel", "dark mode",
    "offline mode", "bulk edit", "keyboard shortcuts", "advanced filter",
    "saved views", "granular permissions", "custom roles", "api endpoint", "webhook"
]

def _fallback_suggestion(text: str) -> List[DetectedError]:
    t = text.strip()
    t_low = t.lower()
    if not t:
        return []
    for term in SUGGESTION_TERMS:
        if term in t_low:
            # produce a tidy summary for common cases
            if "bulk edit" in t_low:
                summary = "Feature request: bulk edit for tasks"
            elif "offline mode" in t_low:
                summary = "Feature request: offline mode"
            elif "dark mode" in t_low:
                summary = "Feature request: dark mode"
            elif "export to csv" in t_low:
                summary = "Feature request: export to CSV"
            elif "keyboard shortcuts" in t_low:
                summary = "Feature request: keyboard shortcuts"
            elif "granular permissions" in t_low or "custom roles" in t_low:
                summary = "Feature request: granular permissions"
            elif "integration with" in t_low:
                summary = "Feature request: integration with external service"
            else:
                summary = "Feature request: " + t.replace("\n", " ")
            return [DetectedError(
                error_summary=summary[:140],
                error_type=["Other"],
                rationale="Detected enhancement/feature request from user phrasing."
            )]
    return []

def detect_errors_with_ollama(
    review: RawReview,
    ollama_model: str = None,  # deprecated parameter, uses config.yaml now
) -> List[DetectedError]:
    force_fallback = os.getenv("USE_FALLBACK_DETECT", "false").lower() in {"1", "true", "yes"}

    # simpler prompt for claude - no few shots to prevent contamination
    full_prompt = f"""{SYSTEM}

Review to analyze:
```
{review.review[:4000]}
```

Return ONLY the JSON object with business_type and errors array. If no issues found, return {{"business_type": "...", "errors": []}}"""

    raw = ""

    if not force_fallback:
        try:
            llm = make_llm()
            resp = llm.invoke(full_prompt)
            raw = (getattr(resp, "content", "") or "").strip()
        except Exception as exc:
            print(f" LLM call failed ({exc}); using heuristic detection instead.")
            force_fallback = True

    if force_fallback:
        out = _fallback_detect(review.review)
        if not out:
            out = _fallback_suggestion(review.review)
        return out

    data = _json_load(raw)
    items = data.get("errors", [])
    out: List[DetectedError] = []

    # Get category normalizer
    use_normalizer = os.getenv("USE_CATEGORY_NORMALIZATION", "true").lower() in {"1", "true", "yes"}
    normalizer = get_normalizer() if use_normalizer else None

    if isinstance(items, list):
        for e in items:
            if not isinstance(e, dict):
                continue
            summary = (e.get("error_summary") or "").strip()[:140]
            types = e.get("error_type") or []
            if isinstance(types, str):
                types = [types]

            types = [t.strip() for t in types if isinstance(t, str) and t.strip()] or ["Other"]

            # Normalize categories for semantic consistency
            if normalizer:
                types = normalizer.normalize_categories(types)

            severity = (e.get("severity") or "None").strip()  # Must be: Critical, Major, Minor, Suggestion, None
            rationale = (e.get("rationale") or "").strip()
            if summary:
                out.append(DetectedError(
                    error_summary=summary,
                    error_type=types,
                    severity=severity,  # Include severity from LLM
                    rationale=rationale
                ))

    if not out:
        out = _fallback_detect(review.review)
    if not out:
        out = _fallback_suggestion(review.review)

    return out
