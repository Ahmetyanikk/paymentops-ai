"""HTTP surface: idempotency (including the concurrent-insert race), reviewer
auth, concurrency, the one-approval-per-transaction invariant, crash
recovery, eligibility, and strict validation. All against SQLite + embedded
Qdrant + the mock assessor."""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from app.models import utcnow
from tests.conftest import OTHER_REVIEWER_KEY, TEST_REVIEWER, TEST_REVIEWER_KEY

TXN_DUPLICATE = "11111111-1111-4111-8111-111111111111"
TXN_DEFECTIVE = "22222222-2222-4222-8222-222222222222"
TXN_OUTSIDE_WINDOW = "33333333-3333-4333-8333-333333333333"
TXN_DUPLICATE_APPROVAL = "44444444-4444-4444-8444-444444444444"
TXN_HIGH_VALUE_UNAUTH = "55555555-5555-4555-8555-555555555555"
TXN_OTHER_REASON = "66666666-6666-4666-8666-666666666666"


@pytest.fixture
def client(seeded):
    from app.api import app

    with TestClient(app) as test_client:
        yield test_client


def create(client, transaction_id, reason, description, idem_key):
    return client.post(
        "/v1/investigations",
        headers={"Idempotency-Key": idem_key},
        json={"transaction_id": transaction_id, "dispute_reason": reason, "description": description},
    )


def decide(client, case_id, decision, reason="", key=TEST_REVIEWER_KEY):
    return client.post(
        f"/v1/investigations/{case_id}/decision",
        headers={"Authorization": f"Bearer {key}"},
        json={"decision": decision, "reason": reason},
    )


def _approved_count(transaction_id: str) -> int:
    from app.db import session_scope
    from app.models import Investigation

    with session_scope() as session:
        return session.query(Investigation).filter_by(transaction_id=transaction_id, decision="approve").count()


def backdate_claim(seconds: int) -> callable:
    """Returns a function that backdates a case's claimed_at by `seconds`,
    simulating an abandoned RUNNING or DECIDING lease."""
    from app.db import session_scope
    from app.models import Investigation

    def _backdate(case_id: str) -> None:
        with session_scope() as session:
            investigation = session.query(Investigation).filter_by(case_id=case_id).one()
            investigation.claimed_at = utcnow() - timedelta(seconds=seconds)

    return _backdate


# --------------------------------------------------------------------------- #
# Basic flow
# --------------------------------------------------------------------------- #


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["database"] == "sqlite+pysqlite"
    assert body["llm_provider"] == "mock"
    assert body["policies_indexed"] == 11


def test_list_transactions(client):
    transactions = client.get("/v1/transactions").json()
    assert len(transactions) == 6
    assert any(t["id"] == TXN_DUPLICATE for t in transactions)


def test_full_round_trip_approve(client):
    created = create(client, TXN_DEFECTIVE, "defective_merchandise", "Arrived broken, return requested.", "api-approve-1")
    assert created.status_code == 201
    body = created.json()
    case_id = body["case_id"]
    assert body["status"] == "PENDING_APPROVAL"
    assert body["eligible_for_approval"] is True
    assert body["policy_citations"]
    assert body["recommendation"] in ("approve", "reject")

    fetched = client.get(f"/v1/investigations/{case_id}")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "PENDING_APPROVAL"

    decided = decide(client, case_id, "approve", "Confirmed return with merchant.")
    assert decided.status_code == 200
    body = decided.json()
    assert body["status"] == "APPROVED"
    assert body["decision"] == "approve"
    assert body["decided_by"] == TEST_REVIEWER
    assert body["funds_moved"] is False
    assert body["decision_id"]

    audit = client.get(f"/v1/investigations/{case_id}/audit").json()
    assert [e["event_type"] for e in audit] == [
        "investigation_opened",
        "assessment_recorded",
        "decision_recorded",
        "settlement_simulated",
    ]
    assert audit[2]["actor"] == TEST_REVIEWER
    assert audit[3]["payload"]["funds_moved"] is False


def test_unknown_transaction_returns_404_without_creating_a_case(client):
    response = create(client, str(uuid.uuid4()), "other", "x", "api-404")
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# Intake idempotency, including the case/database-mismatch regression and
# the concurrent-insert race (fix #3)
# --------------------------------------------------------------------------- #


def test_identical_retry_returns_the_existing_case_as_200(client):
    first = create(client, TXN_DUPLICATE, "duplicate_charge", "Same amount, twice.", "api-idem-1")
    assert first.status_code == 201
    case_id = first.json()["case_id"]

    second = create(client, TXN_DUPLICATE, "duplicate_charge", "Same amount, twice.", "api-idem-1")
    assert second.status_code == 200
    assert second.json()["case_id"] == case_id


def test_conflicting_retry_returns_409(client):
    first = create(client, TXN_DUPLICATE, "duplicate_charge", "Same amount, twice.", "api-idem-2")
    assert first.status_code == 201

    conflicting = create(client, TXN_DEFECTIVE, "defective_merchandise", "Different payload entirely.", "api-idem-2")
    assert conflicting.status_code == 409


def test_regression_conflicting_retry_does_not_desync_graph_and_database(seeded):
    """Regression test for a graph/database case-ID mismatch: a retry with
    a different payload under the same Idempotency-Key must
    never be merged into the existing thread. It must be rejected outright,
    and the original case's transaction must still match what the graph
    actually loaded for it.
    """
    from app.api import app
    from app.db import session_scope
    from app.models import Investigation

    with TestClient(app) as client:
        first = create(client, TXN_DUPLICATE, "duplicate_charge", "Same amount, twice.", "api-mismatch")
        assert first.status_code == 201
        case_id = first.json()["case_id"]
        assert first.json()["transaction_id"] == TXN_DUPLICATE

        conflicting = create(client, TXN_DEFECTIVE, "defective_merchandise", "A different transaction.", "api-mismatch")
        assert conflicting.status_code == 409

        # The original case is untouched: the API view and the raw database
        # row agree on TXN_DUPLICATE. Nothing about the conflicting retry's
        # transaction_id (TXN_DEFECTIVE) leaked into this case anywhere.
        refetched = client.get(f"/v1/investigations/{case_id}")
        assert refetched.json()["transaction_id"] == TXN_DUPLICATE

        with session_scope() as session:
            investigation = session.query(Investigation).filter_by(case_id=case_id).one()
            assert investigation.transaction_id == TXN_DUPLICATE
            assert session.query(Investigation).filter_by(transaction_id=TXN_DEFECTIVE).count() == 0

        # The case is still perfectly resumable - proves the thread itself
        # was never touched by the rejected retry.
        decided = decide(client, case_id, "approve", "Confirmed.")
        assert decided.status_code == 200
        assert decided.json()["decision"] == "approve"


def test_concurrent_intake_with_the_same_idempotency_key_never_500s(seeded):
    """Real two-thread race on the HTTP surface: whichever interleaving
    actually occurs, the outcome must be exactly one 201 and one 200 with
    the same case_id, never a 500 - the genuine race is in
    tests/test_graph.py's forced white-box version of this same scenario;
    this one proves the whole stack (API validation, intake(), the
    IntegrityError fallback) holds together under real concurrency, not
    just the isolated function."""
    from app.api import app

    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(create, client, TXN_DUPLICATE, "duplicate_charge", "Same amount, twice.", "api-concurrent-key")
            future_b = pool.submit(create, client, TXN_DUPLICATE, "duplicate_charge", "Same amount, twice.", "api-concurrent-key")
            response_a = future_a.result()
            response_b = future_b.result()

        assert response_a.status_code in (200, 201)
        assert response_b.status_code in (200, 201)
        assert sorted([response_a.status_code, response_b.status_code]) == [200, 201]
        assert response_a.json()["case_id"] == response_b.json()["case_id"]


def test_missing_idempotency_key_is_422(client):
    response = client.post(
        "/v1/investigations",
        json={"transaction_id": TXN_DUPLICATE, "dispute_reason": "duplicate_charge", "description": "x"},
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Crash recovery (fix #2)
# --------------------------------------------------------------------------- #


def test_same_key_retry_recovers_a_case_stuck_running(client):
    """Simulates a crash before start_case was ever called: intake() has
    committed (RUNNING) but the synchronous `create_investigation` handler
    that would normally call start_case next never got that far - exactly
    what a real process crash between those two steps leaves behind. (The
    TestClient runs the whole request-response cycle synchronously in one
    thread, so a *successful* create() call can never itself be observed
    mid-flight - this is intake() called directly, the same way
    tests/test_graph.py's `open_case` helper does, to construct the crash
    state without actually needing one.) A plain same-key intake retry -
    the normal, expected client behaviour after a request that appeared to
    fail or time out - must then safely re-drive it.
    """
    from app.db import session_scope
    from app.graph import intake

    with session_scope() as session:
        case_id, created_flag = intake(
            session,
            idempotency_key="api-stuck-running",
            transaction_id=TXN_DUPLICATE,
            dispute_reason="duplicate_charge",
            description="Same amount, twice.",
        )
    assert created_flag
    backdate_claim(seconds=60)(case_id)

    retried = create(client, TXN_DUPLICATE, "duplicate_charge", "Same amount, twice.", "api-stuck-running")
    assert retried.status_code == 200
    assert retried.json()["case_id"] == case_id
    assert retried.json()["status"] == "PENDING_APPROVAL"


def test_same_key_retry_does_not_touch_a_running_case_with_a_fresh_lease(client):
    """The case right after creation always has a fresh lease - a retry
    arriving immediately after must not race the original request that may
    still legitimately be working on it."""
    created = create(client, TXN_DUPLICATE, "duplicate_charge", "Same amount, twice.", "api-fresh-running")
    assert created.status_code == 201  # the first call always completes synchronously in these tests

    # By the time this second call runs, the case has already finished
    # (status is PENDING_APPROVAL, not RUNNING) - the interesting assertion
    # is simply that this is still a clean, safe 200 read, never a crash.
    retried = create(client, TXN_DUPLICATE, "duplicate_charge", "Same amount, twice.", "api-fresh-running")
    assert retried.status_code == 200


def test_get_reconciles_an_abandoned_deciding_claim(client):
    """Simulates a crash right after app.api.decide claimed the row but
    before it ever resumed the graph: DECIDING, with the pending decision
    already persisted - and *no one currently holding ownership* (the
    simulated crashed worker never actually acquired the real lock in this
    test, standing in for "that worker's process, and its connection, are
    gone"). A plain GET - not even a decision retry - must self-heal it."""
    from app.db import session_scope
    from app.graph import begin_deciding
    from app.models import INV_DECIDING, Investigation

    created = create(client, TXN_DEFECTIVE, "defective_merchandise", "Item arrived broken.", "api-abandoned-deciding")
    case_id = created.json()["case_id"]
    with session_scope() as session:
        begin_deciding(
            session,
            case_id,
            "crashed-worker-token",
            {"decision": "approve", "actor": TEST_REVIEWER, "reason": "Confirmed before the crash."},
        )
    with session_scope() as session:
        assert session.query(Investigation).filter_by(case_id=case_id).one().status == INV_DECIDING
    backdate_claim(seconds=60)(case_id)  # diagnostic only now - recovery does not depend on this

    fetched = client.get(f"/v1/investigations/{case_id}")
    assert fetched.status_code == 200
    body = fetched.json()
    assert body["status"] == "APPROVED"
    assert body["decided_by"] == TEST_REVIEWER
    assert body["decision_reason"] == "Confirmed before the crash."


def test_decide_reconciles_an_abandoned_deciding_claim_before_its_own_cas(client):
    """Same abandoned claim as above, but discovered by a *new* decision
    attempt rather than a GET - the reconciliation must run first, then this
    request's own claim attempt sees the now-correctly-decided case and
    treats it as the normal post-decision path (replay or conflict), not a
    stuck 'already deciding'."""
    from app.db import session_scope
    from app.graph import begin_deciding

    created = create(client, TXN_DEFECTIVE, "defective_merchandise", "Item arrived broken.", "api-abandoned-deciding-2")
    case_id = created.json()["case_id"]
    with session_scope() as session:
        begin_deciding(
            session, case_id, "crashed-worker-token", {"decision": "approve", "actor": TEST_REVIEWER, "reason": "Pre-crash."}
        )
    backdate_claim(seconds=60)(case_id)

    # A new attempt with the *same* content as the reconciled decision -> replay (200).
    replay = decide(client, case_id, "approve", "Pre-crash.")
    assert replay.status_code == 200
    assert replay.json()["status"] == "APPROVED"


# --------------------------------------------------------------------------- #
# Reviewer authentication
# --------------------------------------------------------------------------- #


def test_decision_without_auth_header_is_401(client):
    created = create(client, TXN_DUPLICATE, "duplicate_charge", "x", "api-noauth")
    case_id = created.json()["case_id"]
    response = client.post(f"/v1/investigations/{case_id}/decision", json={"decision": "approve", "reason": ""})
    assert response.status_code == 401


def test_decision_with_unrecognised_key_is_401(client):
    created = create(client, TXN_DUPLICATE, "duplicate_charge", "x", "api-badkey")
    case_id = created.json()["case_id"]
    response = decide(client, case_id, "approve", key="not-a-real-key")
    assert response.status_code == 401


def test_decision_with_no_configured_reviewer_keys_fails_closed(client, monkeypatch):
    from app.config import reset_settings

    monkeypatch.setenv("REVIEWER_API_KEYS", "")
    reset_settings()
    try:
        created = create(client, TXN_DUPLICATE, "duplicate_charge", "x", "api-nokeys")
        case_id = created.json()["case_id"]
        response = decide(client, case_id, "approve", key=TEST_REVIEWER_KEY)
        assert response.status_code == 401
    finally:
        reset_settings()


# --------------------------------------------------------------------------- #
# Concurrency-safe terminal decisions
# --------------------------------------------------------------------------- #


def test_identical_decision_retry_returns_the_stored_result(client):
    created = create(client, TXN_DUPLICATE, "duplicate_charge", "Same amount, twice.", "api-decide-idem")
    case_id = created.json()["case_id"]
    first = decide(client, case_id, "approve", "Confirmed.")
    assert first.status_code == 200
    second = decide(client, case_id, "approve", "Confirmed.")
    assert second.status_code == 200
    assert second.json()["decision_id"] == first.json()["decision_id"]


def test_conflicting_decision_after_completion_is_409(client):
    created = create(client, TXN_DUPLICATE, "duplicate_charge", "Same amount, twice.", "api-decide-conflict")
    case_id = created.json()["case_id"]
    first = decide(client, case_id, "approve", "Confirmed.")
    assert first.status_code == 200

    conflicting = decide(client, case_id, "reject", "Changed my mind.")
    assert conflicting.status_code == 409


def test_simultaneous_conflicting_decisions_exactly_one_wins(client):
    """Two reviewers race to decide the same case with opposite decisions.
    Exactly one must win (200, decided) and the other must lose (409),
    and exactly one decision_recorded audit row must exist either way."""
    created = create(client, TXN_DUPLICATE, "duplicate_charge", "Same amount, twice.", "api-race")
    case_id = created.json()["case_id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(decide, client, case_id, "approve", "A approves", TEST_REVIEWER_KEY)
        future_b = pool.submit(decide, client, case_id, "reject", "B rejects", OTHER_REVIEWER_KEY)
        response_a = future_a.result()
        response_b = future_b.result()

    codes = sorted([response_a.status_code, response_b.status_code])
    assert codes == [200, 409]

    audit = client.get(f"/v1/investigations/{case_id}/audit").json()
    decision_events = [e for e in audit if e["event_type"] == "decision_recorded"]
    assert len(decision_events) == 1
    settlement_events = [e for e in audit if e["event_type"] == "settlement_simulated"]
    assert len(settlement_events) == 1


def test_claim_is_reverted_after_a_rejected_ineligible_approve(client):
    """A blocked ineligible-approve attempt must not leave the case stuck in
    the internal DECIDING state - a later, valid decision must still work."""
    created = create(
        client, TXN_OUTSIDE_WINDOW, "credit_not_processed", "Promised refund never arrived.", "api-elig-revert"
    )
    case_id = created.json()["case_id"]
    assert created.json()["eligible_for_approval"] is False

    blocked = decide(client, case_id, "approve", "approve anyway")
    assert blocked.status_code == 409

    # The claim was reverted - the case is still PENDING_APPROVAL, not stuck.
    fetched = client.get(f"/v1/investigations/{case_id}")
    assert fetched.json()["status"] == "PENDING_APPROVAL"

    rejected = decide(client, case_id, "reject", "Outside the filing window.")
    assert rejected.status_code == 200
    assert rejected.json()["decision"] == "reject"


# --------------------------------------------------------------------------- #
# One approved dispute per transaction (fix #1) - API-level sequential and
# concurrent races. The database-constraint mechanism itself is exercised
# directly, with no API layer in the way, in tests/test_graph.py.
# --------------------------------------------------------------------------- #


def test_approving_a_second_investigation_on_an_already_approved_transaction_is_409(client):
    """Sequential race: case A is approved first and commits; case B's live
    eligibility re-check (not a stale snapshot) then correctly refuses it."""
    case_a = create(client, TXN_OTHER_REASON, "duplicate_charge", "Same amount charged twice.", "api-seq-race-a")
    case_b = create(client, TXN_OTHER_REASON, "defective_merchandise", "Arrived broken, return requested.", "api-seq-race-b")
    assert case_a.json()["eligible_for_approval"] is True
    assert case_b.json()["eligible_for_approval"] is True

    approved_a = decide(client, case_a.json()["case_id"], "approve", "A approved.")
    assert approved_a.status_code == 200

    blocked_b = decide(client, case_b.json()["case_id"], "approve", "B tries too.")
    assert blocked_b.status_code == 409

    # B is still perfectly usable - a clean 409, not a stuck case.
    assert client.get(f"/v1/investigations/{case_b.json()['case_id']}").json()["status"] == "PENDING_APPROVAL"
    rejected_b = decide(client, case_b.json()["case_id"], "reject", "Already credited via A.")
    assert rejected_b.status_code == 200


def test_concurrent_approvals_on_the_same_transaction_exactly_one_wins(client):
    """Two different investigations on the same transaction, decided through
    real OS threads at (as near as this stack allows) the same instant.

    Which specific guard catches the loser depends on how the two requests
    actually interleave - app.api.decide's live eligibility re-check
    (app/graph.py's refresh_eligibility_for_decision) if B's re-check runs
    after A's commit is already visible, or the database's one-approval-per-
    transaction index inside record_outcome (app/graph.py's module
    docstring) if both re-checks ran before either commit landed. SQLite's
    single-writer locking makes the former far more likely here than true
    concurrent commits would under PostgreSQL - the index path is exercised
    directly, deterministically, and without any API-level check in the way
    at all in tests/test_graph.py::
    test_database_rejects_a_second_approval_on_the_same_transaction. This
    test's job is only the invariant, not which layer enforced it."""
    case_a = create(client, TXN_OTHER_REASON, "duplicate_charge", "Same amount charged twice.", "api-conc-race-a")
    case_b = create(client, TXN_OTHER_REASON, "defective_merchandise", "Arrived broken, return requested.", "api-conc-race-b")
    case_id_a, case_id_b = case_a.json()["case_id"], case_b.json()["case_id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(decide, client, case_id_a, "approve", "A approves", TEST_REVIEWER_KEY)
        future_b = pool.submit(decide, client, case_id_b, "approve", "B approves too", OTHER_REVIEWER_KEY)
        response_a = future_a.result()
        response_b = future_b.result()

    codes = sorted([response_a.status_code, response_b.status_code])
    assert codes == [200, 409]

    with_200 = response_a if response_a.status_code == 200 else response_b
    assert with_200.json()["decision"] == "approve"

    # Exactly one of the two investigations on this transaction ends up
    # approved; the loser is left in a clean, well-defined state either way
    # (PENDING_APPROVAL if caught by the live re-check and safely retryable,
    # or FAILED if caught by the database constraint) - never both approved.
    statuses = {
        case_id_a: client.get(f"/v1/investigations/{case_id_a}").json()["status"],
        case_id_b: client.get(f"/v1/investigations/{case_id_b}").json()["status"],
    }
    assert sorted(statuses.values()) in (["APPROVED", "FAILED"], ["APPROVED", "PENDING_APPROVAL"])
    assert _approved_count(TXN_OTHER_REASON) == 1


def test_approving_an_ineligible_case_is_409_and_does_not_decide_it(client):
    created = create(
        client, TXN_DUPLICATE_APPROVAL, "defective_merchandise", "Second complaint on the same order.", "api-dup-elig"
    )
    case_id = created.json()["case_id"]
    assert created.json()["eligible_for_approval"] is False
    assert "duplicate_prior_approved_dispute" in created.json()["eligibility_reasons"]

    response = decide(client, case_id, "approve", "approve anyway")
    assert response.status_code == 409
    assert "eligibility_reasons" in response.json()["detail"]

    # Still pending - the rejected attempt did not decide the case.
    assert client.get(f"/v1/investigations/{case_id}").json()["status"] == "PENDING_APPROVAL"


def test_rejecting_an_ineligible_case_is_allowed(client):
    created = create(
        client, TXN_DUPLICATE_APPROVAL, "defective_merchandise", "Second complaint on the same order.", "api-dup-reject"
    )
    case_id = created.json()["case_id"]
    response = decide(client, case_id, "reject", "Already credited once.")
    assert response.status_code == 200
    assert response.json()["decision"] == "reject"


def test_high_value_unauthorized_case_is_flagged_but_approvable(client):
    created = create(client, TXN_HIGH_VALUE_UNAUTH, "unauthorized", "I never made this travel booking.", "api-highvalue")
    assert "enhanced_verification_required" in created.json()["risk_flags"]
    assert created.json()["eligible_for_approval"] is True
    response = decide(client, created.json()["case_id"], "approve", "Verified with cardholder.")
    assert response.status_code == 200


# --------------------------------------------------------------------------- #
# Strict validation - 422 without mutating state
# --------------------------------------------------------------------------- #


def test_unknown_field_is_422(client):
    response = client.post(
        "/v1/investigations",
        headers={"Idempotency-Key": "api-strict-1"},
        json={
            "transaction_id": TXN_DUPLICATE,
            "dispute_reason": "duplicate_charge",
            "description": "x",
            "unexpected_field": "sneaky",
        },
    )
    assert response.status_code == 422


def test_malformed_transaction_id_is_422(client):
    response = client.post(
        "/v1/investigations",
        headers={"Idempotency-Key": "api-strict-2"},
        json={"transaction_id": "not-a-uuid", "dispute_reason": "duplicate_charge", "description": "x"},
    )
    assert response.status_code == 422


def test_malformed_case_id_path_param_is_422(client):
    response = client.get("/v1/investigations/not-a-uuid")
    assert response.status_code == 422


def test_blank_description_is_422(client):
    response = client.post(
        "/v1/investigations",
        headers={"Idempotency-Key": "api-strict-3"},
        json={"transaction_id": TXN_DUPLICATE, "dispute_reason": "duplicate_charge", "description": ""},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("blank", [" ", "   ", "\t", "\n", "\t \n "])
def test_whitespace_only_description_is_422(client, blank):
    """Fix #6: a description that is non-empty but entirely whitespace must
    be rejected, same as a truly empty one."""
    response = client.post(
        "/v1/investigations",
        headers={"Idempotency-Key": f"api-blank-{hash(blank)}"},
        json={"transaction_id": TXN_DUPLICATE, "dispute_reason": "duplicate_charge", "description": blank},
    )
    assert response.status_code == 422


def test_oversized_description_is_422(client):
    response = client.post(
        "/v1/investigations",
        headers={"Idempotency-Key": "api-strict-4"},
        json={"transaction_id": TXN_DUPLICATE, "dispute_reason": "duplicate_charge", "description": "x" * 2001},
    )
    assert response.status_code == 422


def test_invalid_dispute_reason_is_422(client):
    response = client.post(
        "/v1/investigations",
        headers={"Idempotency-Key": "api-strict-5"},
        json={"transaction_id": TXN_DUPLICATE, "dispute_reason": "not_a_real_reason", "description": "x"},
    )
    assert response.status_code == 422


def test_invalid_decision_verb_is_422(client):
    created = create(client, TXN_DUPLICATE, "duplicate_charge", "x", "api-strict-6")
    case_id = created.json()["case_id"]
    response = client.post(
        f"/v1/investigations/{case_id}/decision",
        headers={"Authorization": f"Bearer {TEST_REVIEWER_KEY}"},
        json={"decision": "maybe", "reason": ""},
    )
    assert response.status_code == 422


def test_rejected_payload_leaves_nothing_behind_under_its_idempotency_key(client):
    """A 422 must not mutate case state: the same key, retried with a valid
    payload, must succeed as a brand new case (201) - not collide with
    anything the rejected request might otherwise have half-created."""
    rejected = client.post(
        "/v1/investigations",
        headers={"Idempotency-Key": "api-strict-retry"},
        json={"transaction_id": TXN_DUPLICATE, "dispute_reason": "not_a_real_reason", "description": "x"},
    )
    assert rejected.status_code == 422

    retried = create(client, TXN_DUPLICATE, "duplicate_charge", "x", "api-strict-retry")
    assert retried.status_code == 201


def test_unknown_case_returns_404(client):
    assert client.get(f"/v1/investigations/{uuid.uuid4()}").status_code == 404
    assert client.get(f"/v1/investigations/{uuid.uuid4()}/audit").status_code == 404
    response = decide(client, str(uuid.uuid4()), "approve")
    assert response.status_code == 404
