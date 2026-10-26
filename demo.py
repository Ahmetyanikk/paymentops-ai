"""End-to-end demonstration against a running PaymentOps AI API.

    uvicorn app.api:app --port 8000     # terminal 1 (after: python -m app.seed)
    python demo.py                      # terminal 2

Walks: a routine, eligible dispute through to a human approval; an
idempotent retry and a conflicting retry of the same intake request; a
dispute that deterministic eligibility rules refuse to let anyone approve;
and the resulting audit trail. Every decision shown here is simulated; no
credit is ever issued and no transfer is ever initiated.

Safe to run more than once against the same persistent database (Docker
Compose volumes included) without deleting anything first: every intake
uses a fresh Idempotency-Key each run, and every response is checked for its
actual status code before any success-only field is read - including the
one place a *second* run is expected to behave differently from the first
(see step 3): once TXN_DUPLICATE has an approved dispute from an earlier
run, this script's own attempt to approve a new dispute on it correctly
gets refused (409) by the one-approval-per-transaction rule it is there to
demonstrate, and the script shows that refusal rather than tripping over it.
"""

from __future__ import annotations

import os
import sys
from uuid import uuid4

import httpx

BASE_URL = os.environ.get("PAYMENTOPS_URL", "http://localhost:8000")
REVIEWER_KEY = os.environ.get("PAYMENTOPS_REVIEWER_KEY", "demo-reviewer-key")

TXN_DUPLICATE = "11111111-1111-4111-8111-111111111111"  # routine, corroborated, eligible
TXN_OUTSIDE_WINDOW = "33333333-3333-4333-8333-333333333333"  # occurred 90 days ago


def show(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def create_investigation(client: httpx.Client, idem_key: str, transaction_id: str, reason: str, description: str):
    return client.post(
        "/v1/investigations",
        headers={"Idempotency-Key": idem_key},
        json={"transaction_id": transaction_id, "dispute_reason": reason, "description": description},
    )


def decide(client: httpx.Client, case_id: str, decision: str, reason: str):
    return client.post(
        f"/v1/investigations/{case_id}/decision",
        headers={"Authorization": f"Bearer {REVIEWER_KEY}"},
        json={"decision": decision, "reason": reason},
    )


def require(response: httpx.Response, *expected_codes: int, context: str) -> dict:
    """Every call site checks the status code before touching the body -
    this is that check, in one place, so a response that doesn't look like
    what was expected prints a clear message instead of a raw KeyError."""
    if response.status_code not in expected_codes:
        print(f"\n  !! Unexpected response during {context}: HTTP {response.status_code}")
        print(f"     {response.text[:500]}")
        raise SystemExit(1)
    return response.json() if response.content else {}


def main() -> int:
    client = httpx.Client(base_url=BASE_URL, timeout=60.0)

    show("1. Health")
    health = client.get("/healthz")
    print(require(health, 200, context="health check"))

    show("2. Transactions available to dispute")
    transactions = require(client.get("/v1/transactions"), 200, context="list transactions")
    for txn in transactions:
        print(f"  {txn['id']}  {txn['amount']:>10} {txn['currency']}  {txn['merchant_name']:<28} {txn['occurred_at']}")

    # ------------------------------------------------------------------ #
    show("3. Open a routine dispute (duplicate charge, corroborated, eligible)")
    idem_key_1 = f"demo-{uuid4().hex[:8]}"  # fresh every run - safe to re-run without wiping the database
    opened = require(
        create_investigation(client, idem_key_1, TXN_DUPLICATE, "duplicate_charge", "Same amount charged twice by this merchant."),
        201,  # always a brand-new case - idem_key_1 is a fresh uuid4 every run
        context="open routine dispute",
    )
    case_id = opened["case_id"]
    print(f"  case_id          : {case_id}")
    print(f"  status           : {opened['status']}")
    print(f"  question         : {opened['question']}")
    print(f"  recommendation   : {opened['recommendation']}  (risk {opened['risk_score']})")
    print(f"  eligible         : {opened['eligible_for_approval']}")
    print("  citations        :")
    for citation in opened["policy_citations"]:
        print(f"      {citation['policy_id']}@{citation['version']} clause {citation['clause']} "
              f"(score {citation['score']}, effective {citation['effective_date']}) - {citation['title']}")

    print("\n  >> Graph is interrupted and durably checkpointed (PostgreSQL under Docker")
    print("     Compose, SQLite otherwise). The API process could restart here and")
    print("     the case would still be waiting - see README for a restart proof.")

    show("4. Idempotent retry of the intake request")
    replay = require(
        create_investigation(client, idem_key_1, TXN_DUPLICATE, "duplicate_charge", "Same amount charged twice by this merchant."),
        200, context="identical intake retry",
    )
    print(f"  identical retry  -> HTTP 200, same case_id: {replay['case_id'] == case_id}")
    conflict = create_investigation(client, idem_key_1, TXN_DUPLICATE, "duplicate_charge", "A completely different story.")
    conflict_body = require(conflict, 409, context="conflicting intake retry")
    print(f"  conflicting retry -> HTTP 409: {conflict_body['detail']}")

    show("5. A reviewer approves the case")
    decided = decide(client, case_id, "approve", "Confirmed duplicate with the cardholder's statement.")
    if decided.status_code == 200:
        body = require(decided, 200, context="approve the case")
        print(f"  status           : {body['status']}")
        print(f"  decision         : {body['decision']} by {body['decided_by']}")
        print(f"  funds moved      : {body['funds_moved']}  (always False)")

        print("\n  Replaying the exact same decision again returns the stored result, not an error:")
        replay_decision = require(decide(client, case_id, "approve", "Confirmed duplicate with the cardholder's statement."), 200, context="replay the decision")
        print(f"  replay           -> HTTP 200, same decision_id: {replay_decision['decision_id'] == body['decision_id']}")
        print("  A *different* decision on the same case is refused:")
        conflicting_decision = decide(client, case_id, "reject", "Changed my mind.")
        conflicting_body = require(conflicting_decision, 409, context="conflicting decision")
        print(f"  conflicting      -> HTTP 409: {conflicting_body['detail']}")
    elif decided.status_code == 409:
        # A previous run of this script already approved a dispute on
        # TXN_DUPLICATE - the one-approval-per-transaction rule (the exact
        # thing step 6 below demonstrates for a *different* reason) now
        # correctly refuses a second one. Expected on repeated runs; not an
        # error. Reject this case instead so the demo keeps moving.
        print(f"  approve          -> HTTP 409 (TXN_DUPLICATE already has an approved dispute from an earlier run)")
        print(f"                      {decided.json()['detail']}")
        fallback = require(decide(client, case_id, "reject", "Already credited via an earlier demo run."), 200, context="fallback rejection")
        print(f"  rejected instead -> HTTP 200, decision: {fallback['decision']}")
    else:
        require(decided, 200, context="approve the case")  # raises with a clear message

    # ------------------------------------------------------------------ #
    show("6. A dispute deterministic rules will not let anyone approve")
    idem_key_2 = f"demo-{uuid4().hex[:8]}"
    blocked_case = require(
        create_investigation(client, idem_key_2, TXN_OUTSIDE_WINDOW, "credit_not_processed", "Merchant promised a refund that never arrived."),
        201, context="open an out-of-window dispute",
    )
    blocked_case_id = blocked_case["case_id"]
    print(f"  eligible_for_approval : {blocked_case['eligible_for_approval']}")
    print(f"  eligibility_reasons   : {blocked_case['eligibility_reasons']}")

    attempt = decide(client, blocked_case_id, "approve", "approve anyway")
    attempt_body = require(attempt, 409, context="attempt to approve an ineligible case")
    print(f"\n  attempting to approve it anyway -> HTTP 409")
    print(f"  {attempt_body['detail']}")

    print("\n  Rejection is always allowed, regardless of eligibility:")
    denied = require(decide(client, blocked_case_id, "reject", "Outside the 60-day filing window."), 200, context="reject the ineligible case")
    print(f"  reject           -> HTTP 200, decision: {denied['decision']}, funds_moved: {denied['funds_moved']}")

    show("7. Audit trail")
    audit = require(client.get(f"/v1/investigations/{case_id}/audit"), 200, context="fetch audit trail")
    for event in audit:
        print(f"  {event['created_at']}  {event['event_type']:<22} {event['actor']}")
        print(f"      {event['payload']}")

    show("8. Unauthenticated decision attempts fail closed")
    no_auth = client.post(f"/v1/investigations/{blocked_case_id}/decision", json={"decision": "approve", "reason": ""})
    require(no_auth, 401, context="missing auth header")
    print(f"  no Authorization header -> HTTP {no_auth.status_code}")
    bad_key = client.post(
        f"/v1/investigations/{blocked_case_id}/decision",
        headers={"Authorization": "Bearer not-a-real-key"},
        json={"decision": "approve", "reason": ""},
    )
    require(bad_key, 401, context="unrecognised key")
    print(f"  unrecognised key         -> HTTP {bad_key.status_code}")

    print("\nDone. No credit was issued and no transfer was initiated - decisions are recorded only.")
    print("Re-run this script again any time - it does not need the database reset first.\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except httpx.HTTPError as exc:
        print(f"Could not reach the API at {BASE_URL}: {exc}", file=sys.stderr)
        print("Start it with: uvicorn app.api:app --port 8000", file=sys.stderr)
        sys.exit(1)
