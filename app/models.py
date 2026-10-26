"""PostgreSQL schema for the payment dispute investigation workflow.

Portability note: every column type here is also valid on SQLite, so the same
models back the Docker/PostgreSQL deployment and the dependency-free test run.
Money is stored as integer *minor units* (cents) - never as float.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

# Investigation.status - state machine enforced in app/graph.py and app/api.py.
#   RUNNING -> PENDING_APPROVAL -> DECIDING -> APPROVED | REJECTED
#      \-> FAILED                    \-> PENDING_APPROVAL (reverted on error/reclaim)
INV_RUNNING = "RUNNING"
INV_PENDING_APPROVAL = "PENDING_APPROVAL"
INV_DECIDING = "DECIDING"  # transient: claimed by one decision request, not yet resolved; never surfaced externally
INV_APPROVED = "APPROVED"
INV_REJECTED = "REJECTED"
INV_FAILED = "FAILED"  # terminal: policy grounding, LLM output, or a duplicate-approval race could not be trusted

DISPUTE_REASONS = (
    "duplicate_charge",
    "unauthorized",
    "product_not_received",
    "defective_merchandise",
    "credit_not_processed",
    "other",
)

# Human decision verb, and the LLM's advisory recommendation use the same two
# verbs. "approve" sides with the cardholder (credit recommended); "reject"
# sides with the merchant. Recording only - no credit is ever issued here.
DECISIONS = ("approve", "reject")


def new_id() -> str:
    return uuid4().hex


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    """A synthetic card transaction a cardholder may dispute."""

    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    card_last4: Mapped[str] = mapped_column(String(4), nullable=False)
    merchant_name: Mapped[str] = mapped_column(String(200), nullable=False)
    merchant_category: Mapped[str] = mapped_column(String(50), nullable=False)
    amount_minor: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    disputes: Mapped[list[Dispute]] = relationship(back_populates="transaction")

    @property
    def amount(self) -> Decimal:
        # scaleb keeps the two minor-unit digits: 542350 -> Decimal('5423.50').
        return Decimal(self.amount_minor).scaleb(-2)


class Dispute(Base):
    """One cardholder claim against a transaction. Created once at intake."""

    __tablename__ = "disputes"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    transaction_id: Mapped[str] = mapped_column(ForeignKey("transactions.id"), nullable=False)
    reason: Mapped[str] = mapped_column(String(30), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    transaction: Mapped[Transaction] = relationship(back_populates="disputes")


class Investigation(Base):
    """One workflow run over one dispute. `case_id` is the LangGraph thread id.

    `transaction_id` is denormalised from `dispute.transaction_id` for one
    reason only: it lets `ux_one_approved_dispute_per_transaction` be a plain
    single-table partial unique index, which is what makes "two investigations
    on the same transaction can never both be APPROVED" an atomic, database-
    enforced guarantee rather than an application-level check that a race can
    slip past. See app/graph.py record_outcome and the DuplicateApprovalError
    path.
    """

    __tablename__ = "investigations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    case_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    dispute_id: Mapped[str] = mapped_column(ForeignKey("disputes.id"), nullable=False)
    transaction_id: Mapped[str] = mapped_column(ForeignKey("transactions.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=INV_RUNNING)
    failure_reason: Mapped[str | None] = mapped_column(Text)

    # `claimed_at` is diagnostic only now ("when was this last touched") -
    # it is NOT what gates recovery safety. Real exclusive ownership is the
    # PostgreSQL session-level advisory lock in app.graph.case_ownership,
    # held on a dedicated connection for the full duration of any graph
    # operation; a timestamp can never prove a worker has died (a slow-but-
    # alive worker can exceed any threshold), only "is this connection
    # still alive" can, and only the database itself can answer that.
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Set to the acquiring worker's ownership token for the duration it
    # holds the case lock. Any write that clears a RUNNING/DECIDING claim
    # must match this token (`WHERE lock_token = :token`) before mutating -
    # defence in depth so a worker's own stale cleanup can never clear a
    # *different* worker's claim, even if lock discipline were ever
    # violated by a bug elsewhere. See app.graph.revert_claim.
    lock_token: Mapped[str | None] = mapped_column(String(32))

    # The decision payload a `DECIDING` claim is about to submit, persisted
    # *before* the graph is resumed so reconciling a genuinely abandoned
    # claim (the lock is free - its holder is actually gone) can replay
    # exactly this, not guess at it.
    pending_decision: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    # Assessment (filled by the assess_risk node). Advisory only - never binding.
    risk_score: Mapped[float | None] = mapped_column(Float)
    recommendation: Mapped[str | None] = mapped_column(String(20))
    rationale: Mapped[str | None] = mapped_column(Text)
    risk_flags: Mapped[list[str] | None] = mapped_column(JSON)
    policy_citations: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    assessed_by: Mapped[str | None] = mapped_column(String(80))

    # Deterministic eligibility (computed in code, never by the LLM).
    eligible_for_approval: Mapped[bool | None] = mapped_column(Boolean)
    eligibility_reasons: Mapped[list[str] | None] = mapped_column(JSON)

    # Outcome (filled by the record_outcome node). Simulated - no funds move.
    decision_id: Mapped[str | None] = mapped_column(String(32))  # stable id, set once, never reassigned
    decision: Mapped[str | None] = mapped_column(String(20))  # approve|reject
    decided_by: Mapped[str | None] = mapped_column(String(120))
    decision_reason: Mapped[str | None] = mapped_column(Text)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    dispute: Mapped[Dispute] = relationship()


# Database-enforced invariant: at most one APPROVED investigation per
# transaction, full stop - no application-level check (cached eligibility
# snapshot or a live re-check) can race past this, because it is the same
# single UPDATE statement in app.graph.record_outcome that either succeeds
# alone or is rejected by the index. Both dialects get the same condition.
ONE_APPROVED_PER_TRANSACTION_INDEX = "ux_one_approved_dispute_per_transaction"
Index(
    ONE_APPROVED_PER_TRANSACTION_INDEX,
    Investigation.transaction_id,
    unique=True,
    postgresql_where=(Investigation.decision == "approve"),
    sqlite_where=(Investigation.decision == "approve"),
)


class IdempotencyKey(Base):
    """Maps a client-supplied Idempotency-Key to the case it created.

    `request_fingerprint` is a hash of the canonicalised intake payload so a
    retry with the same key but different content is detected as a conflict
    rather than silently reusing the first case. `key` being the primary key
    is also what makes a concurrent double-insert under the same key a
    detectable IntegrityError rather than a silent duplicate - see
    app.graph.intake's race handling.
    """

    __tablename__ = "idempotency_keys"

    key: Mapped[str] = mapped_column(String(200), primary_key=True)
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    case_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditEvent(Base):
    """Append-only audit trail. Written by workflow nodes and API handlers."""

    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    case_id: Mapped[str] = mapped_column(String(64), nullable=False)
    transaction_id: Mapped[str | None] = mapped_column(String(36))
    dispute_id: Mapped[str | None] = mapped_column(String(32))
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    actor: Mapped[str] = mapped_column(String(120), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


Index("ix_audit_events_case_created", AuditEvent.case_id, AuditEvent.created_at)


def transaction_to_dict(transaction: Transaction) -> dict[str, Any]:
    return {
        "id": transaction.id,
        "card_last4": transaction.card_last4,
        "merchant_name": transaction.merchant_name,
        "merchant_category": transaction.merchant_category,
        "amount_minor": transaction.amount_minor,
        "amount": str(transaction.amount),
        "currency": transaction.currency,
        "occurred_at": transaction.occurred_at.isoformat(),
    }


def dispute_to_dict(dispute: Dispute) -> dict[str, Any]:
    return {
        "id": dispute.id,
        "transaction_id": dispute.transaction_id,
        "reason": dispute.reason,
        "description": dispute.description,
        "created_at": dispute.created_at.isoformat(),
    }


def log_audit(
    session: Session,
    *,
    case_id: str,
    event_type: str,
    actor: str,
    transaction_id: str | None = None,
    dispute_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> AuditEvent:
    event = AuditEvent(
        case_id=case_id,
        transaction_id=transaction_id,
        dispute_id=dispute_id,
        event_type=event_type,
        actor=actor,
        payload=payload or {},
    )
    session.add(event)
    return event
