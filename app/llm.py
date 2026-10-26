"""Risk assessment via Claude, with a deterministic offline stand-in.

LLM_PROVIDER=anthropic -> real Claude call with strict structured output.
LLM_PROVIDER=mock      -> rule engine below. Default, and what the offline
                          tests use, so no offline test result ever depends
                          on a network call.

The LLM's output is advisory only (`recommendation`): it is never binding.
Deterministic eligibility and the final decision both live outside this
module, in app/graph.py. Grounding enforcement (empty retrieval, malformed
response, citing a policy ref that was not retrieved) also lives in
app/graph.py, which is the single place all fail-closed checks are made.

The mock is not a stub that returns a constant: it encodes real textual
cues from the dispute reason/description, so the two providers are
interchangeable in the graph and the workflow tests exercise real branching.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.config import get_settings
from app.schemas import LLMAssessment, PolicyCitation

MOCK_ASSESSOR = "mock-rules/v1"

SYSTEM_PROMPT = """You are a card-dispute investigation analyst.

You receive one cardholder dispute, the transaction it concerns, and the policy
excerpts retrieved for it. Assess the dispute against those excerpts only -
never invent a policy, and never cite a policy you were not given.

Produce an advisory recommendation only - you never decide a case, and you
never move money. Set recommendation to "approve" (side with the cardholder)
or "reject" (side with the merchant). A human reviewer makes the real decision.

Set risk_score between 0.0 and 1.0, where higher means more concern about the
dispute's validity or complexity. cited_policy_refs must be exact
"policy_id@version" strings copied from the excerpts you were given - never a
policy_id alone, never a version you were not shown."""


class LLMError(RuntimeError):
    """Raised when the configured LLM provider cannot produce a usable assessment."""


# --------------------------------------------------------------------------- #
# Deterministic mock
# --------------------------------------------------------------------------- #

_DELIVERY_CUES = re.compile(r"deliver|return|track|refus|not arriv|never arriv", re.IGNORECASE)
_MERCHANT_CONTACT_CUES = re.compile(r"promis|refund|merchant|contact|processed", re.IGNORECASE)
_DUPLICATE_CUES = re.compile(r"same (amount|merchant|charge)|twice|duplicate", re.IGNORECASE)


def _mock_assess(case: dict[str, Any], citations: list[PolicyCitation]) -> LLMAssessment:
    reason = case["reason"]
    description = case.get("description", "")
    amount_minor = int(case["amount_minor"])
    days_since = int(case["days_since_transaction"])
    settings = get_settings()

    flags: list[str] = []
    score = 0.05

    if reason == "unauthorized":
        score += 0.15
        if amount_minor >= settings.high_value_minor:
            score += 0.30
            flags.append("enhanced_verification_required")
    elif reason in ("product_not_received", "defective_merchandise"):
        if not _DELIVERY_CUES.search(description):
            score += 0.25
            flags.append("vague_description")
    elif reason == "credit_not_processed":
        if not _MERCHANT_CONTACT_CUES.search(description):
            score += 0.15
            flags.append("no_merchant_contact_referenced")
    elif reason == "duplicate_charge":
        score += 0.05
        if not _DUPLICATE_CUES.search(description):
            score += 0.10
            flags.append("duplicate_not_corroborated")
    elif reason == "other":
        score += 0.10
        flags.append("unclassified_reason")

    if days_since > settings.dispute_filing_window_days:
        score += 0.40
        flags.append("filed_after_dispute_window")

    score = round(min(score, 1.0), 2)
    recommendation = "reject" if score >= 0.6 else "approve"

    if flags:
        rationale = (
            f"Concerns noted: {', '.join(flags)}. {reason} dispute on a "
            f"{amount_minor / 100:.2f} {case['currency']} transaction with "
            f"{case['merchant_name']} ({days_since} days after the transaction). "
            f"Advisory recommendation: {recommendation}."
        )
    else:
        rationale = (
            f"No concern flagged. {reason} dispute filed {days_since} days after a "
            f"{amount_minor / 100:.2f} {case['currency']} transaction with "
            f"{case['merchant_name']}, within policy norms."
        )

    cited: list[str] = []
    for citation in citations:
        if citation.ref not in cited:
            cited.append(citation.ref)
    return LLMAssessment(
        risk_score=score,
        recommendation=recommendation,  # type: ignore[arg-type]
        rationale=rationale,
        risk_flags=flags,
        cited_policy_refs=cited[:3],
    )


# --------------------------------------------------------------------------- #
# Claude
# --------------------------------------------------------------------------- #


def _build_user_prompt(case: dict[str, Any], citations: list[PolicyCitation]) -> str:
    excerpts = "\n\n".join(
        f"[{c.ref} clause {c.clause}] {c.title}\n{c.excerpt}" for c in citations
    ) or "(no policy excerpts retrieved)"
    facts = {
        "dispute_reason": case["reason"],
        "dispute_description": case["description"],
        "days_since_transaction": case["days_since_transaction"],
        "amount": f"{case['amount_minor'] / 100:.2f}",
        "currency": case["currency"],
        "merchant_name": case["merchant_name"],
        "merchant_category": case["merchant_category"],
    }
    return (
        "Pending dispute:\n"
        f"{json.dumps(facts, indent=2, default=str)}\n\n"
        "Retrieved policy excerpts:\n"
        f"{excerpts}"
    )


def _anthropic_assess(case: dict[str, Any], citations: list[PolicyCitation]) -> LLMAssessment:
    import anthropic

    settings = get_settings()
    client_kwargs: dict[str, Any] = {}
    if settings.anthropic_api_key:
        client_kwargs["api_key"] = settings.anthropic_api_key
    if settings.anthropic_base_url:
        client_kwargs["base_url"] = settings.anthropic_base_url

    client = anthropic.Anthropic(**client_kwargs)
    try:
        response = client.messages.parse(
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(case, citations)}],
            output_format=LLMAssessment,
        )
    except anthropic.APIError as exc:  # network, auth, rate limit, bad request
        raise LLMError(f"Anthropic request failed: {exc}") from exc

    assessment = response.parsed_output
    if assessment is None:
        raise LLMError("Anthropic returned no parsable assessment")
    return assessment


# --------------------------------------------------------------------------- #


def assess(case: dict[str, Any], citations: list[PolicyCitation]) -> tuple[LLMAssessment, str]:
    """Return (assessment, assessed_by). Raises LLMError on provider failure.

    `case` must have: reason, description, amount_minor, currency,
    merchant_name, merchant_category, days_since_transaction.
    """
    provider = get_settings().llm_provider.lower()
    if provider == "mock":
        return _mock_assess(case, citations), MOCK_ASSESSOR
    if provider == "anthropic":
        model = get_settings().anthropic_model
        return _anthropic_assess(case, citations), f"anthropic:{model}"
    raise LLMError(f"Unknown LLM_PROVIDER {provider!r} (expected 'mock' or 'anthropic')")
