"""Workflow behaviour: always-interrupt, fail-closed grounding, eligibility,
durability, idempotency, node re-execution safety, duplicate-approval
enforcement, and crash recovery."""

from __future__ import annotations

import threading
from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from app.db import session_scope
from app.graph import (
    AssessmentFailedError,
    DuplicateApprovalError,
    IdempotencyConflict,
    IneligibleDecisionError,
    TransactionNotFound,
    begin_deciding,
    build_checkpointer,
    build_graph,
    case_ownership,
    case_view,
    fingerprint,
    intake,
    resolve_stuck_state,
    revert_claim,
    start_case,
    submit_decision,
    submit_decision_reliably,
)
from app.models import AuditEvent, Dispute, Investigation, utcnow
from app.schemas import LLMAssessment

TXN_DUPLICATE = "11111111-1111-4111-8111-111111111111"  # routine, eligible
TXN_DEFECTIVE = "22222222-2222-4222-8222-222222222222"  # routine, eligible, higher amount
TXN_OUTSIDE_WINDOW = "33333333-3333-4333-8333-333333333333"  # occurred 90 days ago
TXN_DUPLICATE_APPROVAL = "44444444-4444-4444-8444-444444444444"  # has a prior approved dispute
TXN_HIGH_VALUE_UNAUTH = "55555555-5555-4555-8555-555555555555"  # over the enhanced-verification threshold
TXN_OTHER_REASON = "66666666-6666-4666-8666-666666666666"


def audit_types(case_id: str) -> list[str]:
    with session_scope() as session:
        return list(
            session.scalars(
                select(AuditEvent.event_type).where(AuditEvent.case_id == case_id).order_by(AuditEvent.created_at, AuditEvent.id)
            )
        )


def open_case(transaction_id: str, reason: str, description: str, idem_key: str) -> str:
    with session_scope() as session:
        case_id, created = intake(
            session, idempotency_key=idem_key, transaction_id=transaction_id, dispute_reason=reason, description=description
        )
        assert created
    return case_id


def backdate_claim(case_id: str, seconds: int) -> None:
    """Simulate an abandoned lease by setting claimed_at into the past."""
    with session_scope() as session:
        investigation = session.query(Investigation).filter_by(case_id=case_id).one()
        investigation.claimed_at = utcnow() - timedelta(seconds=seconds)


def status_of(case_id: str) -> str:
    with session_scope() as session:
        return session.query(Investigation).filter_by(case_id=case_id).one().status


# --------------------------------------------------------------------------- #
# Every analyzed case reaches a real interrupt
# --------------------------------------------------------------------------- #


def test_routine_dispute_still_pauses_for_a_human(graph):
    case_id = open_case(TXN_DUPLICATE, "duplicate_charge", "Same amount charged twice by this merchant.", "k-routine")
    start_case(graph, case_id)

    view = case_view(case_id)
    assert view["status"] == "PENDING_APPROVAL"
    assert view["decision"] is None
    assert view["question"] is not None
    assert "no funds move" in view["question"].lower()
    assert view["eligible_for_approval"] is True
    assert view["policy_citations"]
    assert audit_types(case_id) == ["investigation_opened", "assessment_recorded"]


def test_resume_with_approval_records_the_decision(graph):
    case_id = open_case(TXN_DEFECTIVE, "defective_merchandise", "Item arrived broken, requested a return label.", "k-approve")
    start_case(graph, case_id)

    submit_decision(graph, case_id, {"decision": "approve", "actor": "reviewer-1", "reason": "Return confirmed with merchant."})
    view = case_view(case_id)
    assert view["status"] == "APPROVED"
    assert view["decision"] == "approve"
    assert view["decided_by"] == "reviewer-1"
    assert view["decision_id"]
    assert view["funds_moved"] is False
    assert audit_types(case_id) == [
        "investigation_opened",
        "assessment_recorded",
        "decision_recorded",
        "settlement_simulated",
    ]


def test_resume_with_rejection_records_the_decision(graph):
    case_id = open_case(TXN_OTHER_REASON, "other", "Charge I don't recognise.", "k-reject")
    start_case(graph, case_id)

    submit_decision(graph, case_id, {"decision": "reject", "actor": "reviewer-1", "reason": "Confirmed with cardholder, charge is valid."})
    view = case_view(case_id)
    assert view["status"] == "REJECTED"
    assert view["decision"] == "reject"


# --------------------------------------------------------------------------- #
# Fail-closed policy grounding
# --------------------------------------------------------------------------- #


def test_no_retrieved_policy_fails_closed(graph, monkeypatch):
    import app.graph as graph_module

    monkeypatch.setattr(graph_module, "search", lambda query, as_of=None: [])
    case_id = open_case(TXN_DUPLICATE, "duplicate_charge", "Same amount charged twice.", "k-nopolicy")

    with pytest.raises(AssessmentFailedError) as exc_info:
        start_case(graph, case_id)
    assert "no_applicable_policy_retrieved" in str(exc_info.value)

    view = case_view(case_id)
    assert view["status"] == "FAILED"
    assert view["failure_reason"] == "no_applicable_policy_retrieved"
    assert view["question"] is None
    assert audit_types(case_id) == ["investigation_opened", "analysis_failed"]


def test_llm_error_fails_closed(graph, monkeypatch):
    import app.graph as graph_module
    from app.llm import LLMError

    def boom(case_facts, citations):
        raise LLMError("provider returned malformed JSON")

    monkeypatch.setattr(graph_module, "llm_assess", boom)
    case_id = open_case(TXN_DUPLICATE, "duplicate_charge", "Same amount charged twice.", "k-llmerror")

    with pytest.raises(AssessmentFailedError) as exc_info:
        start_case(graph, case_id)
    assert "llm_error" in str(exc_info.value)
    assert case_view(case_id)["status"] == "FAILED"


def test_llm_citing_an_unretrieved_policy_ref_fails_closed(graph, monkeypatch):
    import app.graph as graph_module

    def hallucinate(case_facts, citations):
        return (
            LLMAssessment(
                risk_score=0.1,
                recommendation="approve",
                rationale="Looks fine.",
                risk_flags=[],
                cited_policy_refs=["DP-POL-999@9.9"],  # never retrieved - must be rejected
            ),
            "mock-hallucinating",
        )

    monkeypatch.setattr(graph_module, "llm_assess", hallucinate)
    case_id = open_case(TXN_DUPLICATE, "duplicate_charge", "Same amount charged twice.", "k-hallucinate")

    with pytest.raises(AssessmentFailedError) as exc_info:
        start_case(graph, case_id)
    assert "llm_cited_unretrieved_policy_refs" in str(exc_info.value)
    assert case_view(case_id)["status"] == "FAILED"


def test_llm_citing_a_retrieved_ref_succeeds(graph, monkeypatch):
    """Sanity check for the two tests above: a valid ref must not be rejected."""
    import app.graph as graph_module

    captured: dict = {}

    def capturing(case_facts, citations):
        ref = citations[0].ref
        captured["ref"] = ref
        return (
            LLMAssessment(
                risk_score=0.1,
                recommendation="approve",
                rationale="Looks fine.",
                risk_flags=[],
                cited_policy_refs=[ref],
            ),
            "mock-capturing",
        )

    monkeypatch.setattr(graph_module, "llm_assess", capturing)
    case_id = open_case(TXN_DUPLICATE, "duplicate_charge", "Same amount charged twice.", "k-validref")
    start_case(graph, case_id)
    view = case_view(case_id)
    assert view["status"] == "PENDING_APPROVAL"
    stored = view["policy_citations"][0]
    assert f"{stored['policy_id']}@{stored['version']}" == captured["ref"]


# --------------------------------------------------------------------------- #
# Policy effective dates (fix #4) - wiring check; see test_retrieval.py for
# the filtering mechanism itself.
# --------------------------------------------------------------------------- #


def test_retrieve_policies_passes_the_dispute_filing_date_as_of(graph, monkeypatch):
    import app.graph as graph_module

    captured: dict = {}
    real_search = graph_module.search

    def capturing_search(query, k=None, as_of=None):
        captured["as_of"] = as_of
        return real_search(query, k=k, as_of=as_of)

    monkeypatch.setattr(graph_module, "search", capturing_search)
    case_id = open_case(TXN_DUPLICATE, "duplicate_charge", "Same amount charged twice.", "k-asof")
    start_case(graph, case_id)

    with session_scope() as session:
        investigation = session.query(Investigation).filter_by(case_id=case_id).one()
        dispute = session.get(Dispute, investigation.dispute_id)
        expected = dispute.created_at.date()
    assert captured["as_of"] == expected


# --------------------------------------------------------------------------- #
# Deterministic eligibility (never the LLM's call)
# --------------------------------------------------------------------------- #


def test_dispute_filed_after_window_is_ineligible_for_approval(graph):
    case_id = open_case(
        TXN_OUTSIDE_WINDOW, "credit_not_processed", "Merchant promised a refund that never arrived.", "k-window"
    )
    start_case(graph, case_id)
    view = case_view(case_id)
    assert view["eligible_for_approval"] is False
    assert "filed_after_dispute_window" in view["eligibility_reasons"]

    # Rejection is always allowed, regardless of eligibility.
    submit_decision(graph, case_id, {"decision": "reject", "actor": "reviewer-1", "reason": "Outside the filing window."})
    assert case_view(case_id)["status"] == "REJECTED"


def test_graph_level_defence_in_depth_blocks_a_direct_ineligible_approve(graph):
    """app.api.decide checks eligibility before ever calling submit_decision, so
    this path is only reached by a caller that bypasses the API entirely.

    Unlike the duplicate-approval race (which can only be discovered inside
    record_outcome, since nothing earlier can see a concurrent commit), a
    stale-eligibility approve is caught by record_outcome's own stored
    snapshot. Either way the result is the same: a clean terminal `FAILED`
    case, not a thread silently stuck re-trying the same bad decision
    forever (LangGraph's interrupt is one-shot - see app/graph.py's module
    docstring).
    """
    case_id = open_case(
        TXN_OUTSIDE_WINDOW, "credit_not_processed", "Merchant promised a refund that never arrived.", "k-window-bypass"
    )
    start_case(graph, case_id)

    with pytest.raises(IneligibleDecisionError):
        submit_decision(graph, case_id, {"decision": "approve", "actor": "reviewer-1", "reason": "approve anyway"})

    view = case_view(case_id)
    assert view["status"] == "FAILED"
    assert view["failure_reason"] == "ineligible_approval_attempted"
    assert audit_types(case_id)[-1] == "decision_conflict"

    # A later touch does not get stuck retrying - replaying the same bad
    # resume against an already-FAILED thread raises the same clean error.
    with pytest.raises(IneligibleDecisionError):
        submit_decision(graph, case_id, {"decision": "approve", "actor": "reviewer-1", "reason": "approve anyway"})


def test_duplicate_prior_approved_dispute_is_ineligible(graph):
    case_id = open_case(
        TXN_DUPLICATE_APPROVAL, "defective_merchandise", "A second, unrelated complaint about the same order.", "k-dup"
    )
    start_case(graph, case_id)
    view = case_view(case_id)
    assert view["eligible_for_approval"] is False
    assert "duplicate_prior_approved_dispute" in view["eligibility_reasons"]


def test_high_value_unauthorized_dispute_is_flagged_but_still_eligible(graph):
    case_id = open_case(
        TXN_HIGH_VALUE_UNAUTH, "unauthorized", "I never made this travel booking.", "k-highvalue"
    )
    start_case(graph, case_id)
    view = case_view(case_id)
    assert "enhanced_verification_required" in view["risk_flags"]
    assert view["eligible_for_approval"] is True


# --------------------------------------------------------------------------- #
# One approved dispute per transaction, enforced in the database (fix #1)
# --------------------------------------------------------------------------- #


def test_database_rejects_a_second_approval_on_the_same_transaction(graph):
    """Both investigations are assessed *before* either is decided, so both
    legitimately see eligible_for_approval=True at assessment time - this is
    exactly the sequential race app.api.decide's live re-check exists to
    catch, exercised here one level down: directly at the graph/database
    boundary, with no API-layer check in the way at all."""
    case_a = open_case(TXN_OTHER_REASON, "duplicate_charge", "Same amount charged twice.", "k-race-a")
    case_b = open_case(TXN_OTHER_REASON, "defective_merchandise", "Arrived broken, return requested.", "k-race-b")
    start_case(graph, case_a)
    start_case(graph, case_b)
    assert case_view(case_a)["eligible_for_approval"] is True
    assert case_view(case_b)["eligible_for_approval"] is True

    submit_decision(graph, case_a, {"decision": "approve", "actor": "reviewer-1", "reason": "A approved first."})
    assert case_view(case_a)["status"] == "APPROVED"

    with pytest.raises(DuplicateApprovalError):
        submit_decision(graph, case_b, {"decision": "approve", "actor": "reviewer-1", "reason": "B tries too."})

    view_b = case_view(case_b)
    assert view_b["status"] == "FAILED"
    assert view_b["failure_reason"] == "duplicate_approval_conflict"
    assert audit_types(case_b)[-1] == "decision_conflict"

    # The transaction still has exactly one approval - A's - never two.
    with session_scope() as session:
        approved_count = (
            session.query(Investigation)
            .filter_by(transaction_id=TXN_OTHER_REASON, decision="approve")
            .count()
        )
        assert approved_count == 1


def test_duplicate_approval_conflict_leaves_no_misleading_success_state(graph):
    """Checkpoint consistency: for a duplicate-approval conflict, (a) the
    PostgreSQL business row - the sole source of
    truth `case_view` ever reads - is `FAILED`, with no decision_id and no
    terminal approval recorded; (b) the raw LangGraph checkpoint for the
    losing thread was never given a chance to say otherwise, because
    `record_outcome` raises *before* returning an `outcome` state update at
    all, so there is no `outcome` value in the checkpoint for the losing
    thread to be misread from in the first place."""
    import app.graph as graph_module

    case_a = open_case(TXN_OTHER_REASON, "duplicate_charge", "Same amount charged twice.", "k-consistency-a")
    case_b = open_case(TXN_OTHER_REASON, "defective_merchandise", "Arrived broken, return requested.", "k-consistency-b")
    start_case(graph, case_a)
    start_case(graph, case_b)
    submit_decision(graph, case_a, {"decision": "approve", "actor": "reviewer-1", "reason": "A approved first."})

    with pytest.raises(DuplicateApprovalError):
        submit_decision(graph, case_b, {"decision": "approve", "actor": "reviewer-1", "reason": "B tries too."})

    # (a) PostgreSQL business state - checked directly, not through any view.
    with session_scope() as session:
        investigation_b = session.query(Investigation).filter_by(case_id=case_b).one()
        assert investigation_b.status == "FAILED"
        assert investigation_b.decision_id is None
        assert investigation_b.decision is None
        assert investigation_b.decided_at is None
        assert investigation_b.failure_reason == "duplicate_approval_conflict"

    # The public API-shaped view (what a client actually sees) agrees.
    view_b = case_view(case_b)
    assert view_b["status"] == "FAILED"
    assert view_b["decision_id"] is None
    assert view_b["decision"] is None
    assert view_b["funds_moved"] is False

    # (b) The raw checkpoint itself: record_outcome's early `raise` means
    # the superstep that would have carried an `outcome` value never
    # committed one - there is nothing success-shaped here to misread.
    snapshot = graph.get_state(graph_module.thread_config(case_b))
    assert snapshot.values.get("outcome") is None


def test_rejecting_the_loser_of_a_duplicate_approval_race_is_unaffected(graph):
    """Rejection never touches the unique index (it only guards decision='approve'),
    so a case can still be rejected after a sibling investigation on the same
    transaction was approved - the index only ever blocks a *second approval*."""
    case_a = open_case(TXN_OTHER_REASON, "duplicate_charge", "Same amount charged twice.", "k-race-reject-a")
    case_b = open_case(TXN_OTHER_REASON, "defective_merchandise", "Arrived broken, return requested.", "k-race-reject-b")
    start_case(graph, case_a)
    start_case(graph, case_b)

    submit_decision(graph, case_a, {"decision": "approve", "actor": "reviewer-1", "reason": "A approved."})
    submit_decision(graph, case_b, {"decision": "reject", "actor": "reviewer-1", "reason": "Already credited via A."})
    assert case_view(case_b)["status"] == "REJECTED"


# --------------------------------------------------------------------------- #
# Durability
# --------------------------------------------------------------------------- #


def test_paused_case_survives_a_rebuilt_graph_and_checkpointer(graph, seeded):
    """Simulates a process restart: same store, brand new graph object."""
    case_id = open_case(TXN_DEFECTIVE, "defective_merchandise", "Item arrived broken.", "k-durable")
    start_case(graph, case_id)
    graph.checkpointer.conn.close()
    del graph

    revived = build_graph(build_checkpointer())
    try:
        view = case_view(case_id)
        assert view["status"] == "PENDING_APPROVAL"

        submit_decision(revived, case_id, {"decision": "approve", "actor": "reviewer-1", "reason": "Resumed."})
        view = case_view(case_id)
        assert view["status"] == "APPROVED"
        assert view["decided_by"] == "reviewer-1"
    finally:
        revived.checkpointer.conn.close()


# --------------------------------------------------------------------------- #
# Node re-execution safety
# --------------------------------------------------------------------------- #


def test_interrupt_does_not_re_run_the_llm_or_duplicate_audit_rows(graph, monkeypatch):
    """The expensive call sits before the interrupt and must happen exactly once."""
    import app.graph as graph_module

    calls: list[str] = []
    original = graph_module.llm_assess

    def counting_assess(case_facts, citations):
        calls.append(case_facts["reason"])
        return original(case_facts, citations)

    monkeypatch.setattr(graph_module, "llm_assess", counting_assess)

    case_id = open_case(TXN_DEFECTIVE, "defective_merchandise", "Item arrived broken.", "k-once")
    start_case(graph, case_id)
    assert calls == ["defective_merchandise"]

    submit_decision(graph, case_id, {"decision": "approve", "actor": "reviewer-1", "reason": "ok"})
    assert calls == ["defective_merchandise"], "resume must not replay the assessment node"

    events = audit_types(case_id)
    assert events.count("investigation_opened") == 1
    assert events.count("assessment_recorded") == 1
    assert events.count("decision_recorded") == 1


def test_recording_an_outcome_twice_is_a_no_op(graph):
    """Guards the crash-replay path through record_outcome."""
    case_id = open_case(TXN_DEFECTIVE, "defective_merchandise", "Item arrived broken.", "k-twice")
    start_case(graph, case_id)
    submit_decision(graph, case_id, {"decision": "approve", "actor": "reviewer-1", "reason": "first"})

    from app.graph import record_outcome

    replayed = record_outcome(
        {"case_id": case_id, "decision": {"decision": "reject", "actor": "someone-else", "reason": "conflicting"}}
    )
    assert replayed["outcome"]["decided_by"] == "reviewer-1"
    assert audit_types(case_id).count("decision_recorded") == 1


# --------------------------------------------------------------------------- #
# Crash recovery (fix #2)
# --------------------------------------------------------------------------- #


def test_resolve_stuck_state_redrives_a_stuck_running_case(graph):
    """Simulates a crash before assess_risk ever committed anything: the
    case is RUNNING. Correctness here does not depend on how stale
    claimed_at looks any more - only on whether ownership can be acquired -
    so this deliberately does NOT backdate claimed_at at all."""
    case_id = open_case(TXN_DUPLICATE, "duplicate_charge", "Same amount charged twice.", "k-stuck-running")

    with case_ownership(case_id) as token:
        assert token is not None
        resolve_stuck_state(graph, case_id)

    view = case_view(case_id)
    assert view["status"] == "PENDING_APPROVAL"
    assert audit_types(case_id).count("investigation_opened") == 1
    assert audit_types(case_id).count("assessment_recorded") == 1


def test_resolve_stuck_state_on_an_already_settled_case_is_a_noop(graph):
    case_id = open_case(TXN_DUPLICATE, "duplicate_charge", "Same amount charged twice.", "k-already-pending")
    start_case(graph, case_id)
    assert status_of(case_id) == "PENDING_APPROVAL"

    with case_ownership(case_id) as token:
        assert token is not None
        resolve_stuck_state(graph, case_id)  # nothing to do - must not re-run assess_risk

    assert status_of(case_id) == "PENDING_APPROVAL"
    assert audit_types(case_id).count("assessment_recorded") == 1


def test_resolve_stuck_state_replays_an_abandoned_deciding_claim(graph):
    """Simulates a crash after a decision request claimed the row but
    before it ever resumed the graph: DECIDING, with the pending decision
    already persisted at claim time - that persistence is exactly what
    makes this reconcilable."""
    case_id = open_case(TXN_DEFECTIVE, "defective_merchandise", "Item arrived broken.", "k-abandoned-deciding")
    start_case(graph, case_id)
    pending = {"decision": "approve", "actor": "reviewer-1", "reason": "Confirmed before the crash."}
    with session_scope() as session:
        begin_deciding(session, case_id, "crashed-worker-token", pending)

    with case_ownership(case_id) as token:
        assert token is not None
        resolve_stuck_state(graph, case_id)

    view = case_view(case_id)
    assert view["status"] == "APPROVED"
    assert view["decided_by"] == "reviewer-1"
    assert view["decision_reason"] == "Confirmed before the crash."


# --------------------------------------------------------------------------- #
# Regression suite for the exclusive-ownership mechanism itself.
#
# These use real threading.Event synchronisation barriers, not timing
# assumptions - a worker is "slow but alive" because the test controls
# exactly when it is allowed to proceed, not because it merely sleeps.
# --------------------------------------------------------------------------- #


def test_regression_a_slow_but_alive_running_worker_cannot_be_reclaimed(graph, monkeypatch):
    """Test A. The core race this mechanism exists to prevent: an active
    RUNNING worker exceeds the old (now-unused) staleness timeout while
    genuinely still working.
    A competing recovery attempt must get no ownership at all, and must
    cause no overlapping execution of assess_risk."""
    import app.graph as graph_module

    case_id = open_case(TXN_DUPLICATE, "duplicate_charge", "Same amount charged twice.", "k-slow-running")
    backdate_claim(case_id, seconds=10_000)  # looks ancient by the old heuristic - must not matter

    worker_inside = threading.Event()
    release_worker = threading.Event()
    calls = {"n": 0}
    original_assess = graph_module.llm_assess

    def slow_assess(case_facts, citations):
        calls["n"] += 1
        worker_inside.set()
        assert release_worker.wait(timeout=5), "test harness did not release the worker in time"
        return original_assess(case_facts, citations)

    monkeypatch.setattr(graph_module, "llm_assess", slow_assess)

    def worker_one():
        with graph_module.case_ownership(case_id) as token:
            assert token is not None
            graph_module.start_case(graph, case_id)

    t1 = threading.Thread(target=worker_one)
    t1.start()
    try:
        assert worker_inside.wait(timeout=5), "worker 1 must have entered the slow call"

        # A second worker attempts to take over WHILE worker 1 is still
        # alive and holding the lock, mid-operation.
        with graph_module.case_ownership(case_id) as token2:
            assert token2 is None, "a second worker must NOT acquire ownership while worker 1 is still alive"
    finally:
        release_worker.set()
        t1.join(timeout=5)

    assert calls["n"] == 1, "no overlapping graph execution - the slow worker's single call is the only one"
    assert case_view(case_id)["status"] == "PENDING_APPROVAL"


def test_regression_b_slow_but_alive_deciding_worker_retains_exclusive_ownership(seeded, monkeypatch):
    """Test B. Same reproduction for the DECIDING claim: an active decision
    submission exceeds the old staleness timeout while genuinely still
    resuming the graph. A competing recovery attempt must get no ownership,
    and the original worker's claim/pending_decision must be untouched."""
    import app.graph as graph_module

    case_id = open_case(TXN_DEFECTIVE, "defective_merchandise", "Item arrived broken.", "k-slow-deciding")

    worker_inside = threading.Event()
    release_worker = threading.Event()
    calls = {"n": 0}
    original_record_outcome = graph_module.record_outcome

    def slow_record_outcome(state):
        calls["n"] += 1
        worker_inside.set()
        assert release_worker.wait(timeout=5), "test harness did not release the worker in time"
        return original_record_outcome(state)

    # Patched *before* building this test's own graph object, since
    # build_graph resolves `record_outcome` as a module-global at the time
    # it runs - the `graph` fixture's own graph was already built earlier
    # with the unpatched node, so a fresh one is needed here.
    monkeypatch.setattr(graph_module, "record_outcome", slow_record_outcome)
    checkpointer = graph_module.build_checkpointer()
    graph = graph_module.build_graph(checkpointer)
    try:
        start_case(graph, case_id)
        pending = {"decision": "approve", "actor": "reviewer-1", "reason": "Confirmed before the crash."}

        def worker_one():
            with graph_module.case_ownership(case_id) as token:
                assert token is not None
                with session_scope() as session:
                    begin_deciding(session, case_id, token, pending)
                graph_module.submit_decision_reliably(graph, case_id, pending)

        t1 = threading.Thread(target=worker_one)
        t1.start()
        try:
            assert worker_inside.wait(timeout=5), "worker 1 must have entered the slow call"
            backdate_claim(case_id, seconds=10_000)  # looks ancient - must not matter

            with graph_module.case_ownership(case_id) as token2:
                assert token2 is None, "must not acquire ownership while the DECIDING worker is still alive"

            # The attempt above touched nothing - the claim is exactly as
            # worker 1 left it.
            with session_scope() as session:
                investigation = session.query(Investigation).filter_by(case_id=case_id).one()
                assert investigation.status == "DECIDING"
                assert investigation.pending_decision == pending
        finally:
            release_worker.set()
            t1.join(timeout=5)

        assert calls["n"] == 1
        assert case_view(case_id)["status"] == "APPROVED"
    finally:
        checkpointer.conn.close()


def test_regression_c_a_stale_exception_handler_cannot_clobber_a_successors_claim(graph):
    """Test C. Direct test of the token-fencing defence in depth: a
    worker's own cleanup, using its *own* (now-stale) token, must be a
    no-op against a claim that a different token currently owns - even
    though the lock already makes the end-to-end scenario (two workers
    truly both believing they own the same case) structurally impossible.
    This is what makes that guarantee hold even if lock discipline were
    ever violated by a bug elsewhere."""
    case_id = open_case(TXN_DEFECTIVE, "defective_merchandise", "Item arrived broken.", "k-fenced-revert")
    start_case(graph, case_id)

    stale_token = "stale-worker-a-token"
    current_token = "current-worker-b-token"
    pending_b = {"decision": "approve", "actor": "worker-b", "reason": "B's decision"}

    with session_scope() as session:
        begin_deciding(session, case_id, current_token, pending_b)

    # Worker A's (stale) failure cleanup, fenced on its OWN old token.
    revert_claim(case_id, stale_token)

    with session_scope() as session:
        investigation = session.query(Investigation).filter_by(case_id=case_id).one()
        assert investigation.status == "DECIDING", "a differently-tokened revert must not have touched this claim"
        assert investigation.lock_token == current_token
        assert investigation.pending_decision == pending_b

    # The *correct* token still works normally.
    revert_claim(case_id, current_token)
    with session_scope() as session:
        investigation = session.query(Investigation).filter_by(case_id=case_id).one()
        assert investigation.status == "PENDING_APPROVAL"
        assert investigation.pending_decision is None


def test_regression_d_a_genuinely_abandoned_case_can_still_be_recovered(graph):
    """Test D. The flip side of A/B: once an owner is *actually* gone (here:
    its `with case_ownership(...)` block has exited - standing in for the
    process terminating, which is what actually drops a PostgreSQL
    session-level advisory lock), a later caller must be able to acquire
    ownership and finish the job. The fix is "no live owner can be
    overridden", not "nothing can ever be recovered"."""
    import app.graph as graph_module

    case_id = open_case(TXN_DUPLICATE, "duplicate_charge", "Same amount charged twice.", "k-truly-abandoned")
    backdate_claim(case_id, seconds=10_000)

    # Simulate a worker that acquired ownership and then crashed outright -
    # standing in for process termination, which releases the real
    # PostgreSQL lock the instant the connection drops.
    with graph_module.case_ownership(case_id):
        pass

    with graph_module.case_ownership(case_id) as token:
        assert token is not None, "a truly abandoned case must still be recoverable"
        resolve_stuck_state(graph, case_id)

    assert case_view(case_id)["status"] == "PENDING_APPROVAL"


def test_submit_decision_reliably_catches_up_a_lagging_checkpoint(graph, monkeypatch):
    """Simulates the crash window between assess_risk's database commit and
    its own LangGraph checkpoint commit: force the *first* resume to land on
    a fresh interrupt instead of consuming the decision (that is exactly
    what a plain Command(resume=...) does on a thread whose checkpoint is
    still waiting on assess_risk - see the module docstring's LangGraph
    resume fact 2), and confirm submit_decision_reliably's bounded one-shot
    catch-up still finishes the job without looping unboundedly."""
    case_id = open_case(TXN_DEFECTIVE, "defective_merchandise", "Item arrived broken.", "k-catchup")
    start_case(graph, case_id)

    import app.graph as graph_module

    calls = {"n": 0}
    real_submit = graph_module.submit_decision

    def flaky_submit(g, cid, decision):
        calls["n"] += 1
        if calls["n"] == 1:
            # Simulate the checkpoint-lag scenario directly: re-run
            # load_case/retrieve_policies/assess_risk (all safely no-ops
            # here, already committed) without ever delivering `decision`.
            g.invoke({"case_id": cid}, config=graph_module.thread_config(cid))
            return
        real_submit(g, cid, decision)

    monkeypatch.setattr(graph_module, "submit_decision", flaky_submit)
    submit_decision_reliably(graph, case_id, {"decision": "approve", "actor": "reviewer-1", "reason": "ok"})

    assert calls["n"] == 2  # exactly one bounded catch-up retry, not a loop
    assert case_view(case_id)["status"] == "APPROVED"


# --------------------------------------------------------------------------- #
# Intake idempotency (unit level - see tests/test_api.py for the HTTP-level
# regression test reproducing the graph/database case mismatch, and for the
# end-to-end concurrent-insert-race test against the real API surface)
# --------------------------------------------------------------------------- #


def test_identical_retry_returns_the_same_case_without_creating_a_second_one(seeded):
    with session_scope() as session:
        first_id, first_created = intake(
            session, idempotency_key="idem-1", transaction_id=TXN_DUPLICATE, dispute_reason="duplicate_charge", description="x"
        )
    with session_scope() as session:
        second_id, second_created = intake(
            session, idempotency_key="idem-1", transaction_id=TXN_DUPLICATE, dispute_reason="duplicate_charge", description="x"
        )
    assert first_created is True
    assert second_created is False
    assert first_id == second_id

    with session_scope() as session:
        rows = session.scalars(select(Investigation).where(Investigation.case_id == first_id)).all()
        assert len(rows) == 1


def test_conflicting_retry_raises_without_touching_the_database(seeded):
    with session_scope() as session:
        case_id, _ = intake(
            session, idempotency_key="idem-2", transaction_id=TXN_DUPLICATE, dispute_reason="duplicate_charge", description="x"
        )

    with session_scope() as session:
        with pytest.raises(IdempotencyConflict) as exc_info:
            intake(
                session,
                idempotency_key="idem-2",
                transaction_id=TXN_DEFECTIVE,  # different transaction - a conflicting payload
                dispute_reason="defective_merchandise",
                description="different",
            )
        assert exc_info.value.existing_case_id == case_id

    # No second investigation was created for the conflicting payload, and the
    # original case still points at the original transaction - no desync
    # between the database row and what a resumed graph thread would load.
    with session_scope() as session:
        investigation = session.query(Investigation).filter_by(case_id=case_id).one()
        assert investigation.transaction_id == TXN_DUPLICATE
        for_defective = session.query(Investigation).filter_by(transaction_id=TXN_DEFECTIVE).count()
        assert for_defective == 0


def test_unknown_transaction_raises(seeded):
    with session_scope() as session:
        with pytest.raises(TransactionNotFound):
            intake(
                session,
                idempotency_key="idem-3",
                transaction_id="99999999-9999-4999-8999-999999999999",
                dispute_reason="other",
                description="x",
            )


def test_fingerprint_is_deterministic_and_sensitive_to_content():
    a = fingerprint(TXN_DUPLICATE, "duplicate_charge", "same text")
    b = fingerprint(TXN_DUPLICATE, "duplicate_charge", "same text")
    c = fingerprint(TXN_DUPLICATE, "duplicate_charge", "different text")
    assert a == b
    assert a != c


def test_losing_a_genuine_concurrent_insert_race_falls_back_cleanly(seeded, monkeypatch):
    """Reproduces exactly what a true concurrent race produces: this
    session's own `session.get()` check (performed first, like intake()
    does) returns None - no winner visible yet - but by the time its insert
    actually reaches the database, a *different* session has already
    committed a row under the same key. That is forced here by monkeypatching
    the IdempotencyKey probe specifically, rather than relying on real
    thread timing, so the IntegrityError-handling branch is exercised on
    every run, not just when the scheduler happens to interleave two threads
    unluckily (see tests/test_api.py for a real two-thread version of this
    same race, which this unit test complements rather than replaces).
    """
    winner_case_id, created = _commit_intake(
        idempotency_key="race-key", transaction_id=TXN_DUPLICATE, dispute_reason="duplicate_charge", description="A"
    )
    assert created

    real_get = OrmSession.get
    calls = {"n": 0}

    def fake_get(self, entity, ident, *args, **kwargs):
        # Blind only the *first* lookup - a real race only makes intake()'s
        # own initial check miss the not-yet-committed winner. The second
        # lookup is intake()'s own post-IntegrityError re-check, which must
        # see the now-genuinely-committed winner, exactly as it would in a
        # real race (the winner's transaction has to have already committed
        # for our insert to have conflicted at all).
        if entity.__name__ == "IdempotencyKey" and ident == "race-key":
            calls["n"] += 1
            if calls["n"] == 1:
                return None
        return real_get(self, entity, ident, *args, **kwargs)

    monkeypatch.setattr(OrmSession, "get", fake_get)

    # Identical payload to the winner -> should fall back to "existing key found".
    with session_scope() as loser_session:
        case_id, created = intake(
            loser_session, idempotency_key="race-key", transaction_id=TXN_DUPLICATE, dispute_reason="duplicate_charge", description="A"
        )
    assert created is False
    assert case_id == winner_case_id

    with session_scope() as session:
        assert session.query(Investigation).filter_by(case_id=winner_case_id).count() == 1


def test_losing_a_genuine_concurrent_insert_race_with_conflicting_payload(seeded, monkeypatch):
    """Same forced race as above, but the loser's payload *differs* from the
    winner's - must raise IdempotencyConflict, not silently succeed, and
    must not leave a second case behind."""
    winner_case_id, created = _commit_intake(
        idempotency_key="race-key-2", transaction_id=TXN_DUPLICATE, dispute_reason="duplicate_charge", description="A"
    )
    assert created

    real_get = OrmSession.get
    calls = {"n": 0}

    def fake_get(self, entity, ident, *args, **kwargs):
        if entity.__name__ == "IdempotencyKey" and ident == "race-key-2":
            calls["n"] += 1
            if calls["n"] == 1:
                return None
        return real_get(self, entity, ident, *args, **kwargs)

    monkeypatch.setattr(OrmSession, "get", fake_get)

    with session_scope() as loser_session:
        with pytest.raises(IdempotencyConflict) as exc_info:
            intake(
                loser_session,
                idempotency_key="race-key-2",
                transaction_id=TXN_DEFECTIVE,  # different payload than the winner
                dispute_reason="defective_merchandise",
                description="B",
            )
    assert exc_info.value.existing_case_id == winner_case_id

    with session_scope() as session:
        assert session.query(Investigation).filter_by(transaction_id=TXN_DEFECTIVE).count() == 0


def _commit_intake(**kwargs) -> tuple[str, bool]:
    with session_scope() as session:
        return intake(session, **kwargs)
