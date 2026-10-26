"""Dispute policy corpus + synthetic card transactions.

Fixed primary keys so the README example requests are reproducible.
Run with:  python -m app.seed
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import select

from app.db import init_db, session_scope
from app.models import INV_APPROVED, Investigation, Transaction, log_audit, new_id
from app.retrieval import ensure_index

_POLICY_EFFECTIVE = date(2025, 1, 1)
_TODAY = date.today()

POLICY_DOCS: list[dict[str, Any]] = [
    {
        "policy_id": "DP-POL-001",
        "version": "1.0",
        "effective_date": _POLICY_EFFECTIVE,
        "title": "Dispute Filing Window",
        "clause": "1.1",
        "text": (
            "A dispute must be filed within 60 days of the transaction date. A dispute filed "
            "after 60 days is ineligible for approval and must be rejected unless a documented "
            "exception has been separately authorised."
        ),
    },
    {
        "policy_id": "DP-POL-002",
        "version": "1.0",
        "effective_date": _POLICY_EFFECTIVE,
        "title": "Duplicate Charge Disputes",
        "clause": "2.1",
        "text": (
            "A duplicate charge dispute requires a matching merchant and amount posted within "
            "7 days of the disputed transaction. A transaction may receive at most one approved "
            "dispute credit; a second dispute against an already-approved transaction is "
            "ineligible for approval."
        ),
    },
    {
        "policy_id": "DP-POL-003",
        "version": "1.0",
        "effective_date": _POLICY_EFFECTIVE,
        "title": "Unauthorized Transaction Disputes",
        "clause": "3.1",
        "text": (
            "A dispute claiming an unauthorized transaction of 5,000.00 or more requires "
            "enhanced verification - confirmation that the card was not in the cardholder's "
            "possession and a review of recent account activity - before approval."
        ),
    },
    {
        "policy_id": "DP-POL-004",
        "version": "1.0",
        "effective_date": _POLICY_EFFECTIVE,
        "title": "Not Received or Defective Merchandise",
        "clause": "4.1",
        "text": (
            "A product-not-received or defective-merchandise dispute must reference an expected "
            "delivery date, return attempt, or merchant communication in its description. A "
            "description with no such reference is escalated rather than approved outright."
        ),
    },
    {
        "policy_id": "DP-POL-005",
        "version": "1.0",
        "effective_date": _POLICY_EFFECTIVE,
        "title": "Credit Not Processed",
        "clause": "5.1",
        "text": (
            "When a cardholder disputes a credit the merchant has not processed, the merchant "
            "must be given 14 days from the refund promise date before the dispute is escalated "
            "to a chargeback."
        ),
    },
    {
        "policy_id": "DP-POL-006",
        "version": "1.0",
        "effective_date": _POLICY_EFFECTIVE,
        "title": "Reason Code Validation",
        "clause": "6.1",
        "text": (
            "Only a recognised dispute reason code - duplicate charge, unauthorized, product not "
            "received, defective merchandise, credit not processed, or other - may be "
            "investigated. An unrecognised reason code is escalated for manual categorisation."
        ),
    },
    {
        "policy_id": "DP-POL-007",
        "version": "1.0",
        "effective_date": _POLICY_EFFECTIVE,
        "title": "Segregation of Duties and Audit",
        "clause": "7.1",
        "text": (
            "The analyst or system that investigates a case may not also decide it. Every "
            "approval and rejection is recorded with reviewer identity, timestamp and reason in the "
            "immutable audit log."
        ),
    },
    {
        "policy_id": "DP-POL-008",
        "version": "1.0",
        "effective_date": _POLICY_EFFECTIVE,
        "title": "No Automatic Resolution",
        "clause": "8.1",
        "text": (
            "Every investigation requires an explicit human decision before it reaches a "
            "terminal state; this system never resolves a dispute automatically."
        ),
    },
    {
        "policy_id": "DP-POL-008",
        "version": "1.0",
        "effective_date": _POLICY_EFFECTIVE,
        "title": "No Automatic Resolution",
        "clause": "8.2",
        "text": (
            "This system records a decision only. It never issues a provisional credit and "
            "never initiates a transfer; an approved decision must be executed separately in the "
            "core banking platform."
        ),
    },
    {
        # Not yet in effect - exercises the effective-date exclusion in
        # app.retrieval.search. Lexically overlaps with "dispute"/"verify" so
        # it would otherwise rank for ordinary dispute queries.
        "policy_id": "DP-POL-009",
        "version": "1.0",
        "effective_date": _TODAY + timedelta(days=365),
        "expires_date": None,
        "title": "Biometric Dispute Verification",
        "clause": "9.1",
        "text": (
            "A dispute must be verified against the cardholder's biometric signature before "
            "approval. This control is not yet active and must not be applied to any dispute."
        ),
    },
    {
        # Superseded before today - exercises the expiry exclusion. Also
        # overlaps lexically with ordinary dispute/credit vocabulary.
        "policy_id": "DP-POL-010",
        "version": "1.0",
        "effective_date": _TODAY - timedelta(days=730),
        "expires_date": _TODAY - timedelta(days=365),
        "title": "Legacy Paper Dispute Form Requirement",
        "clause": "10.1",
        "text": (
            "A dispute credit requires a signed paper dispute form mailed to the processing "
            "centre. This requirement was retired and must not be applied to any current dispute."
        ),
    },
]

_NOW = datetime.now(UTC)

TRANSACTIONS: list[dict[str, Any]] = [
    {
        "id": "11111111-1111-4111-8111-111111111111",
        "card_last4": "4412",
        "merchant_name": "Cornerstone Coffee Roasters",
        "merchant_category": "food_and_drink",
        "amount_minor": 1850,  # 18.50
        "currency": "USD",
        "occurred_at": _NOW - timedelta(days=5),
    },
    {
        "id": "22222222-2222-4222-8222-222222222222",
        "card_last4": "9087",
        "merchant_name": "Voltline Electronics",
        "merchant_category": "electronics",
        "amount_minor": 134_900,  # 1,349.00
        "currency": "USD",
        "occurred_at": _NOW - timedelta(days=10),
    },
    {
        "id": "33333333-3333-4333-8333-333333333333",
        "card_last4": "2310",
        "merchant_name": "Aurora Streaming Media",
        "merchant_category": "subscriptions",
        "amount_minor": 2499,  # 24.99
        "currency": "USD",
        "occurred_at": _NOW - timedelta(days=90),  # outside the 60-day filing window
    },
    {
        "id": "44444444-4444-4444-8444-444444444444",
        "card_last4": "7781",
        "merchant_name": "Meridian Home Goods",
        "merchant_category": "retail",
        "amount_minor": 32_000,  # 320.00
        "currency": "USD",
        "occurred_at": _NOW - timedelta(days=20),  # already has an approved dispute, see below
    },
    {
        "id": "55555555-5555-4555-8555-555555555555",
        "card_last4": "6654",
        "merchant_name": "Northbridge Travel Agency",
        "merchant_category": "travel",
        "amount_minor": 612_000,  # 6,120.00 - over the enhanced-verification threshold
        "currency": "USD",
        "occurred_at": _NOW - timedelta(days=2),
    },
    {
        "id": "66666666-6666-4666-8666-666666666666",
        "card_last4": "3321",
        "merchant_name": "Pinehollow Appliance Repair",
        "merchant_category": "services",
        "amount_minor": 45_000,  # 450.00
        "currency": "USD",
        "occurred_at": _NOW - timedelta(days=12),
    },
]

# A prior, already-decided investigation against txn 4 (seeded directly - its
# own dispute/decision were not taken through the API). Exercises the
# duplicate-approved-dispute eligibility rule in app/graph.py.
_PRIOR_DISPUTE_ID = "dsp-prior-0004"
_PRIOR_CASE_ID = "case-prior-0004"


def seed_database() -> tuple[int, int]:
    """Insert transactions (and one prior decided case) not already present. Idempotent."""
    init_db()
    from app.models import Dispute  # local import: avoid a cycle with app.graph

    transactions_added = 0
    with session_scope() as session:
        existing = set(session.scalars(select(Transaction.id)))
        for row in TRANSACTIONS:
            if row["id"] not in existing:
                session.add(Transaction(**row))
                transactions_added += 1
        session.flush()

        # NOTE: `_PRIOR_CASE_ID` is the business key (Investigation.case_id),
        # not the primary key (Investigation.id, which is a separate
        # auto-generated id) - session.get() looks up by primary key, so a
        # naive session.get(Investigation, _PRIOR_CASE_ID) would never match
        # and this fixture would try to re-insert on every run against a
        # persistent database (caught by tests/test_integration.py against
        # real PostgreSQL; a fresh-per-test SQLite file never exercised the
        # second run).
        prior_exists = session.query(Investigation).filter_by(case_id=_PRIOR_CASE_ID).first() is not None
        if not prior_exists:
            dispute = Dispute(
                id=_PRIOR_DISPUTE_ID,
                transaction_id="44444444-4444-4444-8444-444444444444",
                reason="product_not_received",
                description="Package tracking shows delivered 2026-01-02 but never arrived.",
                created_at=_NOW - timedelta(days=15),
            )
            session.add(dispute)
            investigation = Investigation(
                case_id=_PRIOR_CASE_ID,
                dispute_id=_PRIOR_DISPUTE_ID,
                transaction_id=dispute.transaction_id,
                status=INV_APPROVED,
                risk_score=0.3,
                recommendation="approve",
                rationale="Seed fixture: prior approved dispute for duplicate-eligibility tests.",
                risk_flags=[],
                policy_citations=[],
                assessed_by="seed-fixture",
                eligible_for_approval=True,
                eligibility_reasons=[],
                decision_id=new_id(),
                decision="approve",
                decided_by="seed-fixture",
                decision_reason="Seed fixture.",
                decided_at=_NOW - timedelta(days=14),
                completed_at=_NOW - timedelta(days=14),
            )
            session.add(investigation)
            session.flush()
            log_audit(
                session,
                case_id=_PRIOR_CASE_ID,
                event_type="decision_recorded",
                actor="seed-fixture",
                transaction_id=dispute.transaction_id,
                dispute_id=dispute.id,
                payload={"decision": "approve", "seed": True},
            )
    return transactions_added, (0 if prior_exists else 1)


def seed_all() -> dict[str, int]:
    transactions_added, cases_added = seed_database()
    policy_chunks = ensure_index(POLICY_DOCS)
    return {
        "transactions_added": transactions_added,
        "prior_cases_added": cases_added,
        "policy_chunks_indexed": policy_chunks,
    }


if __name__ == "__main__":
    print(seed_all())
