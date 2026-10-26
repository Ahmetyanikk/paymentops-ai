"""Pydantic schemas: LLM structured output, API requests, API responses.

`extra="forbid"` on every request schema + strict field types is what turns
unknown fields / malformed UUIDs / blank or oversized text / invalid reason
codes into a 422 before any endpoint body runs - see app/api.py, which never
touches the database until FastAPI has already validated the request.

CaseResponse is intentionally flat (no nested `assessment`/`approval_request`/
`outcome` wrapper objects) - this is the restored original contract:
`recommendation`, `risk_flags` and `policy_citations` are top-level fields.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal, get_args
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models import DISPUTE_REASONS

DisputeReason = Literal[
    "duplicate_charge",
    "unauthorized",
    "product_not_received",
    "defective_merchandise",
    "credit_not_processed",
    "other",
]
assert set(get_args(DisputeReason)) == set(DISPUTE_REASONS)  # keep schema and model in sync

Decision = Literal["approve", "reject"]
CaseStatus = Literal["RUNNING", "PENDING_APPROVAL", "APPROVED", "REJECTED", "FAILED"]


class PolicyCitation(BaseModel):
    """A policy chunk returned by Qdrant retrieval, with version + effective-date metadata."""

    policy_id: str
    version: str
    effective_date: date
    expires_date: date | None = None
    title: str
    clause: str
    score: float
    excerpt: str

    @property
    def ref(self) -> str:
        """Canonical "id@version" string the LLM must cite exactly."""
        return f"{self.policy_id}@{self.version}"


class LLMAssessment(BaseModel):
    """Exact JSON contract the LLM must return.

    No field constraints here on purpose: the provider's strict JSON-schema mode
    rejects some keywords, so ranges are clamped in app.llm instead.
    """

    model_config = ConfigDict(extra="forbid")

    risk_score: float = Field(description="0.0 (no concern) to 1.0 (high concern).")
    recommendation: Decision
    rationale: str = Field(description="Two or three sentences a dispute analyst can act on.")
    risk_flags: list[str] = Field(description="Short tags for each policy concern found.")
    cited_policy_refs: list[str] = Field(
        description='Exact "policy_id@version" strings taken from the supplied policy excerpts.'
    )


class TransactionOut(BaseModel):
    id: str
    card_last4: str
    merchant_name: str
    merchant_category: str
    amount: Decimal
    amount_minor: int
    currency: str
    occurred_at: datetime


class DisputeOut(BaseModel):
    id: str
    transaction_id: str
    reason: DisputeReason
    description: str
    created_at: datetime


class CreateInvestigationRequest(BaseModel):
    """POST /v1/investigations body. Idempotency-Key is a required header, not here."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: UUID
    dispute_reason: DisputeReason
    description: str = Field(min_length=1, max_length=2000)

    @field_validator("description")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("description must not be blank or whitespace-only")
        return value


class DecisionRequest(BaseModel):
    """POST /v1/investigations/{case_id}/decision body.

    Reviewer identity comes from the required Authorization header, never from
    the body - a caller cannot assert someone else's identity.
    """

    model_config = ConfigDict(extra="forbid")

    decision: Decision
    reason: str = Field(default="", max_length=2000)


class CaseResponse(BaseModel):
    """Flat response for POST /v1/investigations, GET .../{case_id}, and
    POST .../{case_id}/decision. Fields not yet applicable to the case's
    current status are null/empty rather than omitted."""

    case_id: str
    transaction_id: str
    status: CaseStatus
    failure_reason: str | None = None

    # Assessment - advisory, set once analysis completes.
    question: str | None = None
    recommendation: Decision | None = None
    risk_score: float | None = None
    risk_flags: list[str] = Field(default_factory=list)
    policy_citations: list[PolicyCitation] = Field(default_factory=list)
    rationale: str | None = None
    eligible_for_approval: bool | None = None
    eligibility_reasons: list[str] = Field(default_factory=list)

    # Decision - set once a reviewer decides (status APPROVED or REJECTED).
    decision_id: str | None = None
    decision: Decision | None = None
    decided_by: str | None = None
    decision_reason: str | None = None
    decided_at: datetime | None = None
    simulated: bool = True
    funds_moved: Literal[False] = False


class AuditEventOut(BaseModel):
    id: str
    case_id: str
    transaction_id: str | None
    dispute_id: str | None
    event_type: str
    actor: str
    payload: dict[str, Any]
    created_at: datetime


class HealthResponse(BaseModel):
    status: Literal["ok"]
    database: str
    qdrant: str
    llm_provider: str
    policies_indexed: int
