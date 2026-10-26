"""Opt-in integration suite: real PostgreSQL + real Qdrant, no mocks on the
storage layer (the LLM stays mocked - LLM_PROVIDER=mock - deterministic
output is what makes the assertions below exact).

Skipped by default. The offline suite (tests/test_api.py, test_graph.py,
test_retrieval.py) is what runs in CI and does not need this file.

To run it:
    docker compose up -d postgres qdrant
    RUN_INTEGRATION_TESTS=1 pytest tests/test_integration.py -v

It talks to whatever Postgres/Qdrant `docker compose` starts on
127.0.0.1:5432 / 127.0.0.1:6333 (override with INTEGRATION_DATABASE_URL /
INTEGRATION_QDRANT_URL). This is a local dev convenience, not an isolated
test database: it reuses the `paymentops` Postgres database and adds rows
with fresh UUIDs each run, and indexes policies into a dedicated
`integration_test_policies` Qdrant collection so it never touches the
`dispute_policies` collection the demo/API use. Do not point it at a
database whose data you care about.

Deliberately `127.0.0.1`, not `localhost`: on this Windows/Docker
Desktop/WSL2 setup, `localhost` resolves to the IPv6 loopback (`::1`)
first, and since `docker-compose.yml` now publishes these ports bound to
`127.0.0.1` only (IPv4), psycopg3/libpq's connection attempt to `::1:5432`
does not fail fast - it was measured hanging 20s+ before any IPv4
fallback, confirmed reproducible, and confirmed resolved by using
`127.0.0.1` explicitly. curl/httpx against the API's port 8000 were not
affected the same way, but the Postgres client driver specifically was.
"""

from __future__ import annotations

import os
import uuid

import pytest
import sqlalchemy
from sqlalchemy import select

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION_TESTS"),
    reason="opt-in: set RUN_INTEGRATION_TESTS=1 with `docker compose up -d postgres qdrant` running",
)

DEFAULT_DATABASE_URL = "postgresql+psycopg://paymentops:paymentops@127.0.0.1:5432/paymentops"
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"


@pytest.fixture
def integration_env(monkeypatch):
    from app.config import reset_settings
    from app.db import reset_engine
    from app.retrieval import reset_client

    monkeypatch.setenv("DATABASE_URL", os.environ.get("INTEGRATION_DATABASE_URL", DEFAULT_DATABASE_URL))
    monkeypatch.setenv("QDRANT_URL", os.environ.get("INTEGRATION_QDRANT_URL", DEFAULT_QDRANT_URL))
    monkeypatch.setenv("QDRANT_COLLECTION", "integration_test_policies")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("REVIEWER_API_KEYS", "integration-reviewer:integration-test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reset_settings()
    reset_engine()
    reset_client()

    from app.db import get_engine

    try:
        with get_engine().connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - any connection failure means "not running"
        pytest.skip(f"PostgreSQL not reachable: {exc}")

    from app.retrieval import get_client

    try:
        get_client().get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qdrant not reachable: {exc}")

    yield

    reset_settings()
    reset_engine()
    reset_client()


@pytest.fixture
def seeded_pg(integration_env):
    from app.seed import seed_all

    return seed_all()


def test_retrieval_round_trips_through_real_qdrant(seeded_pg):
    from app.retrieval import count_policies, search

    assert count_policies() > 0
    hits = search("dispute filed after the authorisation window", k=3)
    assert hits
    assert any(h.policy_id == "DP-POL-001" for h in hits)


def test_full_lifecycle_with_real_postgres_checkpointer_and_process_restart(seeded_pg):
    """The real PostgresSaver, a real retrieval round-trip, an interrupt that
    survives a brand-new checkpointer/graph object (the restart proxy), and a
    resume - then the full audit trail checked straight from PostgreSQL.

    Uses a brand-new Transaction row (not one of the fixed seed transactions)
    so the test is safely re-runnable against the persistent compose
    PostgreSQL: reusing a seed transaction across repeated runs would trip
    the real `duplicate_prior_approved_dispute` eligibility rule on the
    second run, since the first run's approval is still sitting in the
    database - that would be this test fighting the feature it's meant to
    exercise, not a bug.
    """
    from datetime import UTC, datetime, timedelta

    from app.db import session_scope
    from app.graph import build_checkpointer, build_graph, case_view, intake, start_case, submit_decision
    from app.models import AuditEvent, Transaction

    transaction_id = str(uuid.uuid4())
    with session_scope() as session:
        session.add(
            Transaction(
                id=transaction_id,
                card_last4="9999",
                merchant_name="Integration Test Merchant",
                merchant_category="retail",
                amount_minor=134_900,
                currency="USD",
                occurred_at=datetime.now(UTC) - timedelta(days=10),
            )
        )

    idem_key = f"integration-{uuid.uuid4()}"
    with session_scope() as session:
        case_id, created = intake(
            session,
            idempotency_key=idem_key,
            transaction_id=transaction_id,
            dispute_reason="defective_merchandise",
            description="Arrived broken, merchant confirmed a return label.",
        )
    assert created

    checkpointer_a = build_checkpointer()
    graph_a = build_graph(checkpointer_a)
    try:
        start_case(graph_a, case_id)
        view = case_view(case_id)
        assert view["status"] == "PENDING_APPROVAL"
        assert view["policy_citations"], "retrieval must have hit the real Qdrant collection"
    finally:
        checkpointer_a.conn.close()  # closes the ConnectionPool - simulates process exit
    del graph_a, checkpointer_a

    # --- simulate a process restart: brand new PostgresSaver, brand new pool ---
    checkpointer_b = build_checkpointer()
    graph_b = build_graph(checkpointer_b)
    try:
        resumed_state = graph_b.get_state({"configurable": {"thread_id": case_id}})
        assert resumed_state.interrupts, "the interrupt must have survived the simulated restart"
        assert resumed_state.interrupts[0].value["transaction"]["id"] == transaction_id

        submit_decision(
            graph_b, case_id, {"decision": "approve", "actor": "integration-reviewer", "reason": "Confirmed via test."}
        )
    finally:
        checkpointer_b.conn.close()

    final_view = case_view(case_id)
    assert final_view["status"] == "APPROVED"
    assert final_view["decided_by"] == "integration-reviewer"
    assert final_view["funds_moved"] is False
    assert final_view["decision_id"]

    with session_scope() as session:
        events = session.scalars(
            select(AuditEvent.event_type).where(AuditEvent.case_id == case_id).order_by(AuditEvent.created_at, AuditEvent.id)
        ).all()
    assert events == ["investigation_opened", "assessment_recorded", "decision_recorded", "settlement_simulated"]


def test_database_rejects_a_second_approval_on_the_same_transaction_in_postgres(seeded_pg):
    """fix #1 against real PostgreSQL: the partial unique index is the
    actual enforcement mechanism in production, not just in SQLite (which
    the offline suite already covers in tests/test_graph.py) - confirms the
    DDL this project generates for Postgres behaves the same way."""
    from datetime import UTC, datetime, timedelta

    from app.db import session_scope
    from app.graph import DuplicateApprovalError, build_checkpointer, build_graph, case_view, intake, start_case, submit_decision
    from app.models import Transaction

    transaction_id = str(uuid.uuid4())
    with session_scope() as session:
        session.add(
            Transaction(
                id=transaction_id,
                card_last4="8888",
                merchant_name="Integration Duplicate-Approval Merchant",
                merchant_category="retail",
                amount_minor=5_000,
                currency="USD",
                occurred_at=datetime.now(UTC) - timedelta(days=3),
            )
        )

    with session_scope() as session:
        case_a, _ = intake(
            session,
            idempotency_key=f"integration-dup-a-{uuid.uuid4()}",
            transaction_id=transaction_id,
            dispute_reason="duplicate_charge",
            description="Same amount charged twice.",
        )
        case_b, _ = intake(
            session,
            idempotency_key=f"integration-dup-b-{uuid.uuid4()}",
            transaction_id=transaction_id,
            dispute_reason="defective_merchandise",
            description="Arrived broken, return requested.",
        )

    checkpointer = build_checkpointer()
    graph = build_graph(checkpointer)
    try:
        start_case(graph, case_a)
        start_case(graph, case_b)
        submit_decision(graph, case_a, {"decision": "approve", "actor": "integration-reviewer", "reason": "A first."})
        assert case_view(case_a)["status"] == "APPROVED"

        try:
            submit_decision(graph, case_b, {"decision": "approve", "actor": "integration-reviewer", "reason": "B too."})
            raise AssertionError("expected DuplicateApprovalError")
        except DuplicateApprovalError:
            pass

        assert case_view(case_b)["status"] == "FAILED"
        assert case_view(case_b)["failure_reason"] == "duplicate_approval_conflict"
    finally:
        checkpointer.conn.close()


def test_crash_recovery_against_real_postgres(seeded_pg):
    """fix #2 against real PostgreSQL: both recovery windows (stuck
    RUNNING, abandoned DECIDING), exercised with the same real
    checkpointer/connection-pool machinery the Docker Compose deployment
    actually uses, and recovered via the real advisory-lock ownership
    mechanism rather than any staleness heuristic."""
    from datetime import UTC, datetime, timedelta

    from app.db import session_scope
    from app.graph import (
        begin_deciding,
        build_checkpointer,
        build_graph,
        case_ownership,
        case_view,
        intake,
        resolve_stuck_state,
        start_case,
    )
    from app.models import Investigation, Transaction, utcnow

    transaction_id = str(uuid.uuid4())
    with session_scope() as session:
        session.add(
            Transaction(
                id=transaction_id,
                card_last4="7777",
                merchant_name="Integration Crash-Recovery Merchant",
                merchant_category="retail",
                amount_minor=12_000,
                currency="USD",
                occurred_at=datetime.now(UTC) - timedelta(days=4),
            )
        )

    checkpointer = build_checkpointer()
    graph = build_graph(checkpointer)
    try:
        # --- stuck RUNNING: no one currently holds ownership (the
        # simulated crashed worker never actually acquired the real
        # advisory lock here), so a fresh acquisition succeeds and
        # resolve_stuck_state safely re-drives it. claimed_at is backdated
        # only to document that it plays no role in the decision. ---
        with session_scope() as session:
            case_id, _ = intake(
                session,
                idempotency_key=f"integration-crash-running-{uuid.uuid4()}",
                transaction_id=transaction_id,
                dispute_reason="duplicate_charge",
                description="Same amount charged twice.",
            )
            session.query(Investigation).filter_by(case_id=case_id).one().claimed_at = utcnow() - timedelta(seconds=60)

        with case_ownership(case_id) as token:
            assert token is not None
            resolve_stuck_state(graph, case_id)
        assert case_view(case_id)["status"] == "PENDING_APPROVAL"

        # --- abandoned DECIDING claim ---
        with session_scope() as session:
            begin_deciding(
                session, case_id, "crashed-worker-token", {"decision": "approve", "actor": "integration-reviewer", "reason": "Pre-crash."}
            )
            session.query(Investigation).filter_by(case_id=case_id).one().claimed_at = utcnow() - timedelta(seconds=60)

        with case_ownership(case_id) as token:
            assert token is not None
            resolve_stuck_state(graph, case_id)
        view = case_view(case_id)
        assert view["status"] == "APPROVED"
        assert view["decided_by"] == "integration-reviewer"
    finally:
        checkpointer.conn.close()


def test_advisory_lock_exclusion_and_release_on_close_against_real_postgres(seeded_pg):
    """Test E (part 1). Direct verification of the actual primitive
    `app.graph.case_ownership` relies on, with two genuinely separate
    PostgreSQL connections - the same empirical check used to validate the
    design before writing it, kept here as a permanent regression test."""
    import psycopg

    from app.config import get_settings
    from app.graph import _advisory_lock_key, _psycopg_uri

    settings = get_settings()
    dsn = _psycopg_uri(settings.database_url)
    key = _advisory_lock_key(f"integration-advisory-lock-{uuid.uuid4()}")

    conn_a = psycopg.connect(dsn, autocommit=True)
    conn_b = psycopg.connect(dsn, autocommit=True)
    try:
        with conn_a.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (key,))
            assert cur.fetchone()[0] is True

        with conn_b.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (key,))
            assert cur.fetchone()[0] is False, "a second connection must not acquire an already-held lock"

        conn_a.close()  # simulates the holder's process terminating

        conn_c = psycopg.connect(dsn, autocommit=True)
        try:
            with conn_c.cursor() as cur:
                cur.execute("SELECT pg_try_advisory_lock(%s)", (key,))
                assert cur.fetchone()[0] is True, "closing the holder's connection must release the lock immediately"
        finally:
            conn_c.close()
    finally:
        conn_b.close()
        # conn_a already closed above; closing twice would raise.


def test_regression_e_concurrent_workers_against_real_postgres_ownership(seeded_pg, monkeypatch):
    """Test E (part 2). The ownership mechanism end-to-end against the
    *authoritative* PostgreSQL backend (not the SQLite threading.Lock
    fallback the offline suite uses), with two real OS threads and real,
    separate database connections - the same scenario as
    tests/test_graph.py's Test A, repeated here specifically because the
    task requires PostgreSQL to be the authoritative proof, not just
    SQLite. Uses a synchronisation barrier (threading.Event), not timing.
    """
    import threading
    from datetime import UTC, datetime, timedelta

    from app.db import session_scope
    from app.graph import build_checkpointer, build_graph, case_ownership, case_view, intake, start_case
    from app.models import Transaction

    transaction_id = str(uuid.uuid4())
    with session_scope() as session:
        session.add(
            Transaction(
                id=transaction_id,
                card_last4="6666",
                merchant_name="Integration Ownership-Race Merchant",
                merchant_category="retail",
                amount_minor=9_900,
                currency="USD",
                occurred_at=datetime.now(UTC) - timedelta(days=5),
            )
        )
    with session_scope() as session:
        case_id, _ = intake(
            session,
            idempotency_key=f"integration-ownership-{uuid.uuid4()}",
            transaction_id=transaction_id,
            dispute_reason="duplicate_charge",
            description="Same amount charged twice.",
        )

    checkpointer = build_checkpointer()
    graph = build_graph(checkpointer)
    try:
        import app.graph as graph_module

        worker_inside = threading.Event()
        release_worker = threading.Event()
        calls = {"n": 0}
        original_assess = graph_module.llm_assess

        def slow_assess(case_facts, citations):
            calls["n"] += 1
            worker_inside.set()
            assert release_worker.wait(timeout=10), "test harness did not release the worker in time"
            return original_assess(case_facts, citations)

        monkeypatch.setattr(graph_module, "llm_assess", slow_assess)

        def worker_one():
            with case_ownership(case_id) as token:
                assert token is not None
                start_case(graph, case_id)

        t1 = threading.Thread(target=worker_one)
        t1.start()
        try:
            assert worker_inside.wait(timeout=10), "worker 1 must have entered the slow call"

            # A second "worker" - a genuinely separate PostgreSQL
            # connection, attempting ownership while worker 1's dedicated
            # lock-holding connection is still open.
            with case_ownership(case_id) as token2:
                assert token2 is None, "must not acquire ownership while worker 1 is still alive on real PostgreSQL"
        finally:
            release_worker.set()
            t1.join(timeout=10)

        assert calls["n"] == 1, "no overlapping execution against the real PostgreSQL-backed graph"
        assert case_view(case_id)["status"] == "PENDING_APPROVAL"
    finally:
        checkpointer.conn.close()


def test_checkpoint_tables_exist_in_postgres(seeded_pg):
    """Confirms the PostgresSaver actually wrote to PostgreSQL, not SQLite."""
    from app.db import session_scope

    with session_scope() as session:
        rows = session.execute(
            sqlalchemy.text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name LIKE 'checkpoint%'"
            )
        ).all()
    table_names = {r[0] for r in rows}
    assert {"checkpoints", "checkpoint_blobs", "checkpoint_writes"}.issubset(table_names)
