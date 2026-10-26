"""LangGraph workflow: load -> retrieve -> assess -> interrupt -> record.

    START -> load_case -> retrieve_policies -> assess_risk -> await_decision -> record_outcome -> END

There is no branch that skips `await_decision`. Every case that completes
analysis reaches a real `interrupt()` before any terminal decision is
recorded - there is no automatic-approval path. A case can only end without
a human decision by failing analysis first (see "Fail-closed grounding"
below), which raises before the graph ever reaches `await_decision` and is
a `FAILED` case, not a decided one.

Root-cause fix for the graph/database case-ID mismatch
--------------------------------------------------------
The only value ever passed into a fresh graph invocation is `case_id`.
`load_case` re-derives the transaction and dispute from the database by
following `case_id -> Investigation.dispute_id -> Dispute.transaction_id`
on every run. There is no second channel (no `transaction_id`, no intake
payload) through which a later, possibly-conflicting request could inject
different data into an *existing* thread: `app.api` only ever calls
`graph.invoke` with a fresh `case_id` once, at the moment `intake()` creates
a brand-new case (see `intake()` below and `tests/test_api.py::
test_regression_conflicting_retry_does_not_desync_graph_and_database`).

Node re-execution
-----------------
LangGraph re-enters an interrupted node *from its first line* when a thread
resumes, so a node that calls `interrupt()` must not perform side effects
before the call. `await_decision` therefore contains nothing but the
`interrupt()` - no writes, no logging, no LLM call. `load_case` is pure
read. `assess_risk` and `record_outcome` each check the persisted row before
doing any write, so a replay (or a crash-restart) of either is a no-op if
that node's work was already committed.

Two LangGraph resume facts this module relies on (verified empirically
against the installed langgraph version, not assumed from memory):
  1. A *plain* `graph.invoke(input, config)` on an existing thread ALWAYS
     restarts the run from START, re-running every node up to whatever is
     pending - even a node that already committed once (harmless here
     because every node checks before writing) and even a node that is
     currently sitting on a live interrupt (it abandons that interrupt and
     raises a fresh one with the same value). This is what makes it safe to
     call `start_case` again on a genuinely stuck `RUNNING` case: nothing
     about calling it twice is special-cased, so it only ever has to be
     *safe*, which the per-node guards already make it.
  2. `graph.invoke(Command(resume=value), config)` is different: if the
     thread has no pending interrupt to deliver `value` to (because the
     checkpoint is lagging behind - see the crash window in
     `submit_decision_reliably`), it continues the Pregel loop from whatever
     node is actually pending (re-running only that node, not restarting
     from START) and the supplied `value` is silently unused. If the thread
     is already at `END` (fully decided), it is a safe no-op that returns
     the final state. Both facts are what let `submit_decision_reliably` and
     `reconcile_case` resume a thread without knowing in advance which of
     these situations they are actually in.

Fail-closed grounding
----------------------
`assess_risk` fails the case (status `FAILED`, never a silent guess) when:
  - no policy citation effective as of the dispute's filing date was
    retrieved at all (see "Policy effective dates" below);
  - the configured LLM provider raises or returns something that does not
    parse into `LLMAssessment` (see app.llm.LLMError);
  - the LLM cites a `policy_id@version` ref that was not in the retrieved
    set - it may only reason about the policy text it was actually given.
A failed case never reaches `await_decision`; there is nothing for a human
to approve until grounding and the model response are both trustworthy.

Policy effective dates
-----------------------
`retrieve_policies` passes the dispute's filing date (`dispute.created_at`,
not "today" - the rules that govern a case are the ones in effect when it
was opened) to `app.retrieval.search` as `as_of`, which excludes any policy
chunk not yet effective or already expired on that date before `assess_risk`
ever sees it. A future-dated or superseded policy can score however well it
likes lexically; it is never handed to the LLM, and never fills a citation
slot that an in-effect policy would otherwise take.

Eligibility is deterministic, not the LLM's call
--------------------------------------------------
`compute_eligibility` is plain code: the filing-window check and the
duplicate-approved-dispute check never touch the LLM. `assess_risk` persists
the result; `app.api.decide` refuses an `approve` decision when a *live*
re-check (not the assess-time snapshot, which can go stale - see
`refresh_eligibility_for_decision`) shows `eligible_for_approval` is False
(409, before the graph is ever resumed), and `record_outcome` repeats the
static check as a defence-in-depth guard in case the graph is ever invoked
directly.

One approved dispute per transaction, enforced in the database
------------------------------------------------------------------
The live re-check above closes the *sequential* race (case B is decided
after case A's approval has already committed) with a clean 409 and a fully
recoverable case B. It cannot close the *concurrent* race, where two
different investigations on the same transaction are both approved at
almost the same instant and each one's live re-check runs before the other's
write commits - no read-then-act check, however fresh, is airtight against
that without serializable isolation or an explicit lock. `record_outcome`'s
actual `decision='approve'` write is therefore guarded by
`ux_one_approved_dispute_per_transaction` (app/models.py), a partial unique
index - the database itself rejects the second commit, with no window for
either side to observe a stale answer. Because LangGraph's interrupt is
one-shot (see above), the investigation that loses this race cannot be
"corrected" with a second resume; `record_outcome` catches the
`IntegrityError` and durably marks that case `FAILED` with
`failure_reason="duplicate_approval_conflict"` instead, in its own clean
transaction (never inside the one that just rolled back).

Exclusive per-case ownership - not a time-based heuristic
--------------------------------------------------------------
An earlier version of this module gated crash recovery on a `claimed_at`
lease timestamp: "if nobody renewed this claim in N seconds, it's safe to
take over." That is wrong, and was shown to be wrong: a *slow but alive*
worker can exceed any chosen N while still correctly finishing its own
work, and a timestamp cannot tell those two cases apart. A fast worker and
a dead one can both leave the same `claimed_at` behind.

What actually proves a worker is gone - and only this - is that a
PostgreSQL session-level advisory lock for the case becomes acquirable.
`case_ownership(case_id)` acquires one (`pg_try_advisory_lock`) on a
dedicated connection and holds it for the full duration of whatever graph
operation is about to happen, including any cleanup on failure; the lock
is released only when that `with` block exits (success or exception) or,
if the process is killed outright, the instant the connection drops -
PostgreSQL enforces that, not application code. A second worker's own
`case_ownership(case_id)` call simply cannot succeed while the first is
still inside its `with` block, no matter how slow it is or how stale
`claimed_at` looks; `claimed_at` is kept purely as a diagnostic "last
touched" timestamp now, never as a safety gate.

Every place that touches the graph or a `RUNNING`/`DECIDING` claim goes
through this: `start_case` (initial execution, and the re-invoke that
recovers a stuck `RUNNING` case), `submit_decision_reliably` (decision
submission and resume, including reconciling an abandoned `DECIDING`
claim), and the `GET`-triggered reconciliation in `app.api`. See
`resolve_stuck_state`, which performs the actual RUNNING/DECIDING recovery
once ownership is held - it needs no staleness check of its own, because
successfully holding the lock already proves no one else can be live on
this case.

Defence in depth: `lock_token`, a fresh id generated per acquisition and
stored on the row for the duration of a `DECIDING` claim. `revert_claim`
only clears a claim `WHERE lock_token = :token`, so even if lock discipline
were ever violated by a bug elsewhere, a worker's own (stale) cleanup can
never erase a *different* worker's claim or its `pending_decision` - it
would simply match zero rows. The lock is what makes the old race
structurally impossible; the token is what keeps a cleanup path honest even
if that structure were ever bypassed.

SQLite lock fallback (offline tests / local single-process dev only): there
is no advisory-lock primitive, so `case_ownership` falls back to a plain
in-process `threading.Lock` keyed by `case_id`. That is correct mutual
exclusion for concurrent *threads in this one process* - exactly what the
offline, `ThreadPoolExecutor`-based tests exercise - but it is not a
liveness proof across processes: a second process gets its own empty lock
table and would not see it at all. PostgreSQL is the only backend this
mechanism is correct for under real multi-worker/multi-container
concurrency; the opt-in integration suite (`tests/test_integration.py`) is
what actually verifies it there.

Crash recovery - two windows, one mechanism
------------------------------------------------
  1. A case stuck `RUNNING` - crashed before `start_case` was ever called,
     or between `assess_risk`'s database commit and its own commit never
     happening at all. Triggered by a same-key intake retry
     (`app.api.create_investigation` finding `created=False`): acquire
     ownership, then `resolve_stuck_state` safely re-drives it with a plain
     re-invoke (LangGraph resume fact 1 below - every node guards itself).
  2. An abandoned `DECIDING` claim - crashed after a decision request
     claimed the row but before `record_outcome` ran to completion. The
     pending decision is persisted at claim time (`pending_decision`)
     specifically so recovery does not have to guess what to resume with;
     `resolve_stuck_state` (triggered from `GET` and from a new decision
     request, both after acquiring ownership) replays exactly that payload.

In both cases, acquiring the lock is the only precondition - there is no
"is this stale enough yet" wait, because staleness was never the right
question.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, NotRequired, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as OrmSession

from app.config import get_settings
from app.db import get_engine, session_scope
from app.llm import LLMError
from app.llm import assess as llm_assess
from app.models import (
    INV_APPROVED,
    INV_DECIDING,
    INV_FAILED,
    INV_PENDING_APPROVAL,
    INV_REJECTED,
    INV_RUNNING,
    Dispute,
    IdempotencyKey,
    Investigation,
    Transaction,
    dispute_to_dict,
    log_audit,
    new_id,
    transaction_to_dict,
    utcnow,
)
from app.retrieval import search
from app.schemas import PolicyCitation

SYSTEM_ACTOR = "system:workflow"


class CaseState(TypedDict):
    """Typed workflow state. Only JSON-serialisable values - it gets checkpointed.

    `case_id` is the only input a fresh invocation ever carries - see the
    module docstring. Field names here are private workflow plumbing (they
    feed the `interrupt()` payload) and are independent of the public API
    shape in app.schemas.CaseResponse, which app.graph.case_view builds
    straight from the database, never from this state.
    """

    case_id: str
    transaction: NotRequired[dict[str, Any]]
    dispute: NotRequired[dict[str, Any]]
    citations: NotRequired[list[dict[str, Any]]]
    assessment: NotRequired[dict[str, Any]]
    approval_request: NotRequired[dict[str, Any]]
    decision: NotRequired[dict[str, Any]]
    outcome: NotRequired[dict[str, Any]]


class TransactionNotFound(LookupError):
    pass


class IdempotencyConflict(RuntimeError):
    """Same Idempotency-Key, different request payload."""

    def __init__(self, existing_case_id: str) -> None:
        super().__init__(f"Idempotency-Key already used for a different request (case {existing_case_id})")
        self.existing_case_id = existing_case_id


class AssessmentFailedError(RuntimeError):
    """Raised after a case has been durably marked `FAILED`. See module docstring."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class IneligibleDecisionError(RuntimeError):
    """Defence-in-depth guard: `app.api.decide` should never let this fire."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__(f"Investigation is not eligible for approval: {', '.join(reasons)}")
        self.reasons = reasons


class DuplicateApprovalError(RuntimeError):
    """Raised when the database's one-approval-per-transaction index rejects
    this case's commit because a *different* investigation on the same
    transaction won the race. The case is durably `FAILED`; see module
    docstring "One approved dispute per transaction, enforced in the
    database"."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Duplicate approval conflict: {reason}")
        self.reason = reason


class CaseLockError(RuntimeError):
    """The ownership mechanism itself failed - e.g. could not open a
    connection to attempt the PostgreSQL advisory lock. Distinct from the
    lock simply being held by another worker (that yields `None` from
    `case_ownership`, it does not raise)."""


# --------------------------------------------------------------------------- #
# Intake (idempotent, outside the graph)
# --------------------------------------------------------------------------- #


def fingerprint(transaction_id: str, dispute_reason: str, description: str) -> str:
    """Canonical hash of an intake payload. Used to detect a conflicting retry."""
    canonical = json.dumps(
        {
            "transaction_id": str(transaction_id),
            "dispute_reason": dispute_reason,
            "description": description,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def intake(
    session,
    *,
    idempotency_key: str,
    transaction_id: str,
    dispute_reason: str,
    description: str,
) -> tuple[str, bool]:
    """Create a case for one Idempotency-Key, exactly once - even under a
    genuine concurrent race on the same key.

    Returns (case_id, created). `created` is False for a replayed identical
    request - the caller must then skip `graph.invoke` entirely, so no path
    ever hands new intake data to an existing thread. Raises
    `IdempotencyConflict` for a replay with a different payload (before
    anything is written) and `TransactionNotFound` if `transaction_id` does
    not exist.

    Concurrency: two requests sharing a key can both pass the initial
    `session.get()` check (neither has committed yet) and both attempt the
    insert. Only one commits; the loser's insert raises `IntegrityError` on
    the `idempotency_keys` primary key. That is not a 500 - it means a
    winner now exists, so the loser rolls back its own half-built rows and
    falls back to the exact same "existing key found" logic it would have
    taken had it simply lost a slower race.
    """
    fp = fingerprint(transaction_id, dispute_reason, description)
    existing = session.get(IdempotencyKey, idempotency_key)
    if existing is not None:
        if existing.request_fingerprint != fp:
            raise IdempotencyConflict(existing.case_id)
        return existing.case_id, False

    transaction = session.get(Transaction, str(transaction_id))
    if transaction is None:
        raise TransactionNotFound(str(transaction_id))

    case_id = str(uuid4())  # canonical dashed form - matches `str(UUID)` when a path param is re-stringified
    dispute = Dispute(transaction_id=transaction.id, reason=dispute_reason, description=description)
    session.add(dispute)
    session.flush()  # populates dispute.id (random uuid4 hex PK - no collision risk) before it's used as a FK below

    investigation = Investigation(
        case_id=case_id,
        dispute_id=dispute.id,
        transaction_id=transaction.id,
        status=INV_RUNNING,
        claimed_at=utcnow(),
    )
    session.add(investigation)
    session.add(IdempotencyKey(key=idempotency_key, request_fingerprint=fp, case_id=case_id))

    try:
        session.flush()
    except IntegrityError:
        # session.rollback() undoes *this entire transaction*, including the
        # dispute.id flush above - no orphaned Dispute row survives for the loser.
        session.rollback()
        winner = session.get(IdempotencyKey, idempotency_key)
        if winner is None:  # pragma: no cover - conflict was on something else entirely
            raise
        if winner.request_fingerprint != fp:
            raise IdempotencyConflict(winner.case_id) from None
        return winner.case_id, False

    log_audit(
        session,
        case_id=case_id,
        event_type="investigation_opened",
        actor=SYSTEM_ACTOR,
        transaction_id=transaction.id,
        dispute_id=dispute.id,
        payload={"dispute_reason": dispute_reason},
    )
    return case_id, True


# --------------------------------------------------------------------------- #
# Deterministic eligibility (never the LLM's call)
# --------------------------------------------------------------------------- #


def _has_other_approved_dispute(session, *, transaction_id: str, exclude_case_id: str) -> bool:
    return (
        session.scalar(
            select(Investigation)
            .where(
                Investigation.transaction_id == transaction_id,
                Investigation.decision == "approve",
                Investigation.case_id != exclude_case_id,
            )
            .limit(1)
        )
        is not None
    )


def compute_eligibility(
    session, *, case_id: str, transaction: dict[str, Any], dispute: dict[str, Any]
) -> tuple[bool, list[str], int]:
    """Returns (eligible, reasons, days_since_transaction). Plain code, no LLM.

    This is the assess-time snapshot, persisted on the Investigation row. It
    can go stale for the duplicate-approval reason specifically (a *different*
    investigation on the same transaction can be approved later) - see
    `refresh_eligibility_for_decision` for the live re-check used at decision
    time, and the module docstring for why only the database constraint is
    actually airtight against a concurrent race.
    """
    settings = get_settings()
    occurred_at = datetime.fromisoformat(transaction["occurred_at"])
    filed_at = datetime.fromisoformat(dispute["created_at"])
    days_since = max(0, (filed_at - occurred_at).days)

    reasons: list[str] = []
    if days_since > settings.dispute_filing_window_days:
        reasons.append("filed_after_dispute_window")
    if _has_other_approved_dispute(session, transaction_id=transaction["id"], exclude_case_id=case_id):
        reasons.append("duplicate_prior_approved_dispute")

    return len(reasons) == 0, reasons, days_since


def refresh_eligibility_for_decision(
    session, *, case_id: str, transaction_id: str, stored_reasons: list[str]
) -> tuple[bool, list[str]]:
    """Live re-check used by `app.api.decide` right after claiming the row,
    before the graph is ever resumed. Only the duplicate-approval reason is
    time-varying (the filing-window reason is fixed at assessment time and
    cannot go stale), so that is the only thing re-queried; everything else
    carries over from the stored snapshot unchanged."""
    reasons = list(stored_reasons)
    if "duplicate_prior_approved_dispute" not in reasons and _has_other_approved_dispute(
        session, transaction_id=transaction_id, exclude_case_id=case_id
    ):
        reasons.append("duplicate_prior_approved_dispute")
    return len(reasons) == 0, reasons


# --------------------------------------------------------------------------- #
# Exclusive per-case ownership (see module docstring)
# --------------------------------------------------------------------------- #

_sqlite_case_locks: dict[str, threading.Lock] = {}
_sqlite_locks_guard = threading.Lock()


def _advisory_lock_key(case_id: str) -> int:
    """Deterministic signed 64-bit key for pg_advisory_lock, derived from
    case_id. pg_advisory_lock(bigint) accepts the full signed 64-bit range,
    so no masking is needed."""
    digest = hashlib.sha256(case_id.encode()).digest()[:8]
    return int.from_bytes(digest, "big", signed=True)


def _pg_try_acquire(case_id: str, database_url: str):
    """Open a dedicated connection (never the SQLAlchemy pool - this
    connection runs no business queries and holds no business transaction,
    purely the advisory lock) and attempt it. Always returns a connection
    the caller must close, whether or not it acquired - except on
    `CaseLockError`, where connecting itself failed and there is nothing to
    close.

    ponytail: one fresh connection per acquisition, no pooling for this -
    simplest correct thing, and cheap enough at this app's scale. If case
    operations ever become hot enough for connection-churn to matter, add a
    small dedicated pool here; nothing above this function would change.
    """
    import psycopg

    try:
        conn = psycopg.connect(_psycopg_uri(database_url), autocommit=True)
    except Exception as exc:  # noqa: BLE001 - any connection failure is a mechanism failure
        raise CaseLockError(f"could not connect to acquire the case lock for {case_id}: {exc}") from exc
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (_advisory_lock_key(case_id),))
            acquired = bool(cur.fetchone()[0])
        return acquired, conn
    except Exception as exc:
        conn.close()
        raise CaseLockError(f"case lock query failed for {case_id}: {exc}") from exc


def _sqlite_lock_for(case_id: str) -> threading.Lock:
    with _sqlite_locks_guard:
        return _sqlite_case_locks.setdefault(case_id, threading.Lock())


@contextmanager
def case_ownership(case_id: str):
    """True mutual exclusion per case_id. Acquire *before* touching the
    graph or a RUNNING/DECIDING claim in any way; hold it for the entire
    operation, including any failure cleanup. See the module docstring for
    the full reasoning and the SQLite fallback's limits.

    Yields a fresh ownership token (str) on success, or `None` if another
    worker currently owns this case - a caller that gets `None` must not
    touch the graph or mutate the claim at all. Raises `CaseLockError` if
    the mechanism itself fails (e.g. PostgreSQL unreachable) - deliberately
    distinct from "busy", so a caller that needs ownership to proceed
    safely can tell "try again shortly" apart from "something is broken".
    """
    settings = get_settings()
    token = uuid4().hex
    if settings.database_url.startswith(("postgresql", "postgres")):
        acquired, conn = _pg_try_acquire(case_id, settings.database_url)
        try:
            yield token if acquired else None
        finally:
            # Closing releases every advisory lock this connection ever
            # took - as effective as an explicit pg_advisory_unlock, and
            # it's what happens anyway if this process is killed outright.
            conn.close()
    else:
        acquired = _sqlite_lock_for(case_id).acquire(blocking=False)
        try:
            yield token if acquired else None
        finally:
            if acquired:
                _sqlite_lock_for(case_id).release()


def begin_deciding(session, case_id: str, token: str, pending_decision: dict[str, Any]) -> None:
    """Transition PENDING_APPROVAL -> DECIDING. Only ever called while this
    worker holds `case_ownership(case_id)`, so there is no concurrent
    writer to race against here - a plain UPDATE is correct. The `WHERE
    status=...` clause is a defensive sanity check, not what provides
    safety; the lock already does."""
    session.execute(
        update(Investigation)
        .where(Investigation.case_id == case_id, Investigation.status == INV_PENDING_APPROVAL)
        .values(status=INV_DECIDING, claimed_at=utcnow(), lock_token=token, pending_decision=pending_decision)
        .execution_options(synchronize_session=False)
    )


def revert_claim(case_id: str, token: str) -> None:
    """Revert a DECIDING claim back to PENDING_APPROVAL - fenced by `token`
    so this can never clear a *different* worker's claim or its
    pending_decision, even though holding `case_ownership` already makes
    that structurally impossible under correct lock discipline. Defence in
    depth, not the primary safety mechanism - see the module docstring."""
    with session_scope() as session:
        session.execute(
            update(Investigation)
            .where(Investigation.case_id == case_id, Investigation.status == INV_DECIDING, Investigation.lock_token == token)
            .values(status=INV_PENDING_APPROVAL, claimed_at=None, lock_token=None, pending_decision=None)
            .execution_options(synchronize_session=False)
        )


def resolve_stuck_state(graph, case_id: str) -> None:
    """Finish whatever a previous owner left behind. Must only be called
    while holding `case_ownership(case_id)` - successfully acquiring that
    lock already proves no other worker can currently be live on this
    case, however long `claimed_at` has sat unrenewed, so whatever status
    is found here is unconditionally safe to act on; there is no staleness
    check left to make.

    Never raises: AssessmentFailedError / IneligibleDecisionError /
    DuplicateApprovalError from finishing the job just means the case is
    now correctly terminal in the database - callers read the current
    state afterward (`case_view`) rather than needing this to report it.
    """
    with session_scope() as session:
        investigation = session.query(Investigation).filter_by(case_id=case_id).one_or_none()
        if investigation is None:
            return
        status = investigation.status
        pending = investigation.pending_decision
    try:
        if status == INV_RUNNING:
            start_case(graph, case_id)
        elif status == INV_DECIDING and pending is not None:
            submit_decision_reliably(graph, case_id, pending)
    except (AssessmentFailedError, IneligibleDecisionError, DuplicateApprovalError):
        pass


# --------------------------------------------------------------------------- #
# View builders (DB -> flat API shape). Pure, no side effects.
# --------------------------------------------------------------------------- #


def _assessment_fields(investigation: Investigation) -> dict[str, Any]:
    return {
        "recommendation": investigation.recommendation,
        "risk_score": investigation.risk_score,
        "risk_flags": investigation.risk_flags or [],
        "policy_citations": investigation.policy_citations or [],
        "rationale": investigation.rationale,
        "eligible_for_approval": investigation.eligible_for_approval,
        "eligibility_reasons": investigation.eligibility_reasons or [],
    }


def _question_for(transaction: dict[str, Any], dispute: dict[str, Any]) -> str:
    return (
        f"Decide the {dispute['reason']} dispute on transaction {transaction['id']} "
        f"({transaction['merchant_name']}, {transaction['amount']} {transaction['currency']})? "
        "Recording a decision only - no funds move."
    )


def _approval_request_payload(
    investigation: Investigation, transaction: dict[str, Any], dispute: dict[str, Any]
) -> dict[str, Any]:
    """The rich interrupt payload a human reviewer actually needs - distinct
    from (and a superset of) the flat API fields `case_view` exposes; this is
    private workflow state, checkpointed as the `interrupt()` value."""
    return {
        "question": _question_for(transaction, dispute),
        "transaction": transaction,
        "dispute": dispute,
        **_assessment_fields(investigation),
    }


def _decision_fields(investigation: Investigation) -> dict[str, Any]:
    return {
        "decision_id": investigation.decision_id,
        "decision": investigation.decision,
        "decided_by": investigation.decided_by,
        "decision_reason": investigation.decision_reason or "",
        "decided_at": investigation.decided_at.isoformat() if investigation.decided_at else None,
        "simulated": True,
        "funds_moved": False,
    }


def case_view(case_id: str) -> dict[str, Any] | None:
    """Build the full, flat API response purely from the database.

    Deliberately independent of `graph.get_state`: the database row is the
    single source of truth for case status (including `FAILED`, which the
    graph's own checkpoint does not represent), so GET never has to
    reconcile two stores.
    """
    with session_scope() as session:
        investigation = session.query(Investigation).filter_by(case_id=case_id).one_or_none()
        if investigation is None:
            return None
        dispute = session.get(Dispute, investigation.dispute_id)
        transaction = session.get(Transaction, dispute.transaction_id)
        transaction_dict = transaction_to_dict(transaction)
        dispute_dict = dispute_to_dict(dispute)

        status = investigation.status
        view: dict[str, Any] = {
            "case_id": case_id,
            "transaction_id": transaction.id,
            # DECIDING is an internal CAS state (app.api.decide) - never surface it externally.
            "status": INV_PENDING_APPROVAL if status == INV_DECIDING else status,
            "failure_reason": investigation.failure_reason,
            "question": None,
            "recommendation": None,
            "risk_score": None,
            "risk_flags": [],
            "policy_citations": [],
            "rationale": None,
            "eligible_for_approval": None,
            "eligibility_reasons": [],
            "decision_id": None,
            "decision": None,
            "decided_by": None,
            "decision_reason": None,
            "decided_at": None,
            "simulated": True,
            "funds_moved": False,
        }
        if investigation.recommendation is not None:
            view.update(_assessment_fields(investigation))
        if status in (INV_PENDING_APPROVAL, INV_DECIDING):
            view["question"] = _question_for(transaction_dict, dispute_dict)
        if status in (INV_APPROVED, INV_REJECTED):
            view.update(_decision_fields(investigation))
        return view


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def load_case(state: CaseState) -> dict[str, Any]:
    """Re-derive the transaction and dispute from the database. Pure read."""
    with session_scope() as session:
        investigation = session.query(Investigation).filter_by(case_id=state["case_id"]).one()
        dispute = session.get(Dispute, investigation.dispute_id)
        transaction = session.get(Transaction, dispute.transaction_id)
        return {
            "transaction": transaction_to_dict(transaction),
            "dispute": dispute_to_dict(dispute),
        }


def retrieve_policies(state: CaseState) -> dict[str, Any]:
    """Qdrant lookup, filtered to policies in effect as of the dispute's
    filing date. Read-only, so replay-safe by construction."""
    dispute = state["dispute"]
    transaction = state["transaction"]
    query = (
        f"{dispute['reason']} dispute on a {transaction['merchant_category']} transaction "
        f"of {transaction['amount']} {transaction['currency']} with {transaction['merchant_name']}. "
        f"{dispute['description']}"
    )
    as_of = datetime.fromisoformat(dispute["created_at"]).date()
    citations = search(query, as_of=as_of)
    return {"citations": [c.model_dump(mode="json") for c in citations]}


def _case_facts(transaction: dict[str, Any], dispute: dict[str, Any], days_since: int) -> dict[str, Any]:
    return {
        "reason": dispute["reason"],
        "description": dispute["description"],
        "amount_minor": transaction["amount_minor"],
        "currency": transaction["currency"],
        "merchant_name": transaction["merchant_name"],
        "merchant_category": transaction["merchant_category"],
        "days_since_transaction": days_since,
    }


def assess_risk(state: CaseState) -> dict[str, Any]:
    """LLM assessment + fail-closed grounding + deterministic eligibility.

    Writes happen in two short, independent transactions (read-check, then
    persist-result) with no raise inside either `with` block, so a fail-
    closed result is always committed *before* this function raises -
    unlike raising mid-transaction, which would roll the failure write back.
    """
    case_id = state["case_id"]
    transaction = state["transaction"]
    dispute = state["dispute"]

    with session_scope() as session:
        investigation = session.query(Investigation).filter_by(case_id=case_id).one()
        if investigation.status == INV_FAILED:
            raise AssessmentFailedError(investigation.failure_reason or "analysis_failed")
        if investigation.recommendation is not None:
            return {
                "assessment": _assessment_fields(investigation),
                "approval_request": _approval_request_payload(investigation, transaction, dispute),
            }

    citations = [PolicyCitation(**c) for c in state.get("citations", [])]
    failure_reason: str | None = None
    llm_result = assessed_by = None

    if not citations:
        failure_reason = "no_applicable_policy_retrieved"
    else:
        # Own short-lived session for this read-only query, bracketed so it
        # closes before the (possibly slow) LLM call below - never hold a
        # database connection open across a network request.
        with OrmSession(bind=get_engine()) as elig_session:
            eligible, eligibility_reasons, days_since = compute_eligibility(
                elig_session, case_id=case_id, transaction=transaction, dispute=dispute
            )
        try:
            llm_result, assessed_by = llm_assess(_case_facts(transaction, dispute, days_since), citations)
        except LLMError as exc:
            failure_reason = f"llm_error: {exc}"
        else:
            retrieved_refs = {c.ref for c in citations}
            unknown_refs = sorted(set(llm_result.cited_policy_refs) - retrieved_refs)
            if unknown_refs:
                failure_reason = f"llm_cited_unretrieved_policy_refs: {unknown_refs}"

    with session_scope() as session:
        investigation = session.query(Investigation).filter_by(case_id=case_id).one()
        if investigation.status == INV_FAILED:
            failure_reason = investigation.failure_reason
        elif investigation.recommendation is not None:
            failure_reason = None  # a concurrent call already finished successfully
            success_view = (
                _assessment_fields(investigation),
                _approval_request_payload(investigation, transaction, dispute),
            )
        elif failure_reason is not None:
            investigation.status = INV_FAILED
            investigation.failure_reason = failure_reason
            log_audit(
                session,
                case_id=case_id,
                event_type="analysis_failed",
                actor=SYSTEM_ACTOR,
                transaction_id=transaction["id"],
                dispute_id=dispute["id"],
                payload={"reason": failure_reason},
            )
        else:
            risk_score = max(0.0, min(1.0, float(llm_result.risk_score)))
            investigation.risk_score = risk_score
            investigation.recommendation = llm_result.recommendation
            investigation.rationale = llm_result.rationale
            investigation.risk_flags = list(llm_result.risk_flags)
            investigation.policy_citations = state.get("citations", [])
            investigation.assessed_by = assessed_by
            investigation.eligible_for_approval = eligible
            investigation.eligibility_reasons = eligibility_reasons
            investigation.status = INV_PENDING_APPROVAL
            log_audit(
                session,
                case_id=case_id,
                event_type="assessment_recorded",
                actor=assessed_by,
                transaction_id=transaction["id"],
                dispute_id=dispute["id"],
                payload={
                    "risk_score": risk_score,
                    "recommendation": llm_result.recommendation,
                    "risk_flags": list(llm_result.risk_flags),
                    "eligible_for_approval": eligible,
                    "eligibility_reasons": eligibility_reasons,
                },
            )
            success_view = (
                _assessment_fields(investigation),
                _approval_request_payload(investigation, transaction, dispute),
            )

    if failure_reason is not None:
        raise AssessmentFailedError(failure_reason)
    assessment, approval_request = success_view
    return {"assessment": assessment, "approval_request": approval_request}


def await_decision(state: CaseState) -> dict[str, Any]:
    """Pause for a human.

    NOTHING may precede the `interrupt()` call: on resume LangGraph re-enters
    this node at line one, so any statement above it would run twice.
    """
    decision = interrupt(state["approval_request"])
    return {"decision": decision}


def _fail_record_outcome(session, case_id: str, reason: str, payload: dict[str, Any]) -> None:
    """Commit a terminal FAILED state for this case, in the caller's current
    (post-rollback-if-any, otherwise still-open) transaction. Shared by both
    failure paths below so neither one raises before its own write commits -
    see module docstring "Node re-execution" and `assess_risk`'s identical
    two-phase pattern. Both call sites are in `record_outcome`, which is
    itself inside `with session_scope()`, so returning here just lets that
    block exit normally; the raise happens in `record_outcome` after that."""
    investigation = session.query(Investigation).filter_by(case_id=case_id).one()
    investigation.status = INV_FAILED
    investigation.failure_reason = reason
    investigation.lock_token = None
    investigation.pending_decision = None
    dispute = session.get(Dispute, investigation.dispute_id)
    log_audit(
        session,
        case_id=case_id,
        event_type="decision_conflict",
        actor=SYSTEM_ACTOR,
        transaction_id=dispute.transaction_id,
        dispute_id=dispute.id,
        payload=payload,
    )


def record_outcome(state: CaseState) -> dict[str, Any]:
    """Record the decision. Simulated only - this never initiates a transfer.

    Both failure paths below (an ineligible approve slipping past
    `app.api.decide`'s own checks, or losing the database's one-approval-
    per-transaction race) commit a durable `FAILED` case *before* raising,
    for the same reason `assess_risk` does: LangGraph's interrupt is
    one-shot (see module docstring), so once this node is reached there is
    no way to ask the human again on this thread - leaving the row at
    `DECIDING` with no terminal marker, as an earlier version of this
    function did for the ineligible-approve path, means every future touch
    reclaims the lease, fails the same way, and leaves the case silently
    stuck forever. A clean `FAILED` is a wrong decision to have to tell a
    human to re-investigate, but the case will never quietly disappear.
    """
    case_id = state["case_id"]
    decision_input = state["decision"]
    decision = decision_input["decision"]
    actor = decision_input["actor"]
    reason = decision_input.get("reason") or ""

    with session_scope() as session:
        investigation = session.query(Investigation).filter_by(case_id=case_id).one()
        if investigation.decided_at is not None:
            # Replay after a crash: the decision already exists, do not re-record it.
            return {"outcome": _decision_fields(investigation)}
        if investigation.status == INV_FAILED:
            # A prior attempt on this thread already lost a race below and is
            # terminally failed; which exception to re-raise is recorded in
            # failure_reason itself.
            failure_reason = investigation.failure_reason or "decision_conflict"
            if failure_reason == "ineligible_approval_attempted":
                raise IneligibleDecisionError(investigation.eligibility_reasons or [])
            raise DuplicateApprovalError(failure_reason)

        # Fetched *before* any mutation below: a `session.get()` issued while
        # `investigation`'s pending changes are dirty triggers an autoflush
        # that would otherwise send this decision's UPDATE to the database
        # before the code reaches its own explicit, guarded `session.flush()`
        # - meaning a unique-constraint violation would raise as a raw
        # IntegrityError straight out of this function instead of being
        # caught below and turned into a clean `DuplicateApprovalError`.
        dispute = session.get(Dispute, investigation.dispute_id)

        if decision == "approve" and not investigation.eligible_for_approval:
            # Defence in depth - app.api.decide must already have blocked this.
            reasons = investigation.eligibility_reasons or []
            _fail_record_outcome(
                session,
                case_id,
                "ineligible_approval_attempted",
                {"reason": "ineligible_approval_attempted", "eligibility_reasons": reasons, "attempted_decision": decision},
            )
            outcome_kind, outcome_payload = "ineligible", reasons
        else:
            now = utcnow()
            investigation.decision_id = new_id()
            investigation.decision = decision
            investigation.decided_by = actor
            investigation.decision_reason = reason
            investigation.decided_at = now
            investigation.status = INV_APPROVED if decision == "approve" else INV_REJECTED
            investigation.completed_at = now
            investigation.lock_token = None
            investigation.pending_decision = None

            log_audit(
                session,
                case_id=case_id,
                event_type="decision_recorded",
                actor=actor,
                transaction_id=dispute.transaction_id,
                dispute_id=dispute.id,
                payload={"decision": decision, "reason": reason, "recommendation": investigation.recommendation},
            )
            log_audit(
                session,
                case_id=case_id,
                event_type="settlement_simulated",
                actor=SYSTEM_ACTOR,
                transaction_id=dispute.transaction_id,
                dispute_id=dispute.id,
                payload={
                    "funds_moved": False,
                    "note": "Decision recorded only. Execute in the core banking platform separately.",
                },
            )

            outcome_kind, outcome_payload = "ok", None
            if decision == "approve":
                try:
                    session.flush()
                except IntegrityError:
                    session.rollback()
                    # A *different* investigation on the same transaction won
                    # the race (see module docstring). session.rollback()
                    # above undid everything this branch just set, including
                    # status - _fail_record_outcome re-fetches and re-applies
                    # a fresh, correct FAILED write on top of that reverted state.
                    conflict_reason = "duplicate_approval_conflict"
                    _fail_record_outcome(
                        session, case_id, conflict_reason, {"reason": conflict_reason, "attempted_decision": decision}
                    )
                    outcome_kind, outcome_payload = "conflict", conflict_reason
            if outcome_kind == "ok":
                outcome_payload = _decision_fields(investigation)

    if outcome_kind == "conflict":
        raise DuplicateApprovalError(outcome_payload)
    if outcome_kind == "ineligible":
        raise IneligibleDecisionError(outcome_payload)
    return {"outcome": outcome_payload}


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def _psycopg_uri(database_url: str) -> str:
    return database_url.replace("+psycopg2", "").replace("+psycopg", "")


def build_checkpointer():
    """Durable checkpoint store matching the configured database.

    Strict deserialization: `pickle_fallback=False` (the default, passed
    explicitly here so it cannot silently drift) plus
    `LANGGRAPH_STRICT_MSGPACK=true`, which restricts checkpoint
    deserialization to a built-in allowlist of safe types. Both savers only
    ever deserialize bytes this process itself wrote, but a checkpoint store
    is still an untrusted-input boundary if the database is ever compromised
    independently of the API - defence in depth, not a response to an
    observed exploit.
    """
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    serde = JsonPlusSerializer(pickle_fallback=False)
    settings = get_settings()
    if settings.database_url.startswith(("postgresql", "postgres")):
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(
            conninfo=_psycopg_uri(settings.database_url),
            max_size=10,
            kwargs={"autocommit": True, "row_factory": dict_row},
            open=True,
        )
        saver = PostgresSaver(pool, serde=serde)
        saver.setup()
        return saver

    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(settings.checkpoint_db, check_same_thread=False)
    saver = SqliteSaver(conn, serde=serde)
    saver.setup()
    return saver


def build_graph(checkpointer):
    builder = StateGraph(CaseState)
    builder.add_node("load_case", load_case)
    builder.add_node("retrieve_policies", retrieve_policies)
    builder.add_node("assess_risk", assess_risk)
    builder.add_node("await_decision", await_decision)
    builder.add_node("record_outcome", record_outcome)

    builder.add_edge(START, "load_case")
    builder.add_edge("load_case", "retrieve_policies")
    builder.add_edge("retrieve_policies", "assess_risk")
    builder.add_edge("assess_risk", "await_decision")
    builder.add_edge("await_decision", "record_outcome")
    builder.add_edge("record_outcome", END)
    return builder.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------- #
# Driver helpers (shared by the API and the tests)
# --------------------------------------------------------------------------- #


def thread_config(case_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": case_id}}


def start_case(graph, case_id: str) -> None:
    """First-ever invocation of a freshly created case, or a recovery
    re-invoke of one stuck `RUNNING` (see `recover_stuck_running`). Safe to
    call more than once on the same thread - see LangGraph resume fact 1 in
    the module docstring plus every node's own replay guard."""
    graph.invoke({"case_id": case_id}, config=thread_config(case_id))


def submit_decision(graph, case_id: str, decision: dict[str, Any]) -> None:
    graph.invoke(Command(resume=decision), config=thread_config(case_id))


def submit_decision_reliably(graph, case_id: str, decision: dict[str, Any]) -> None:
    """Resume the thread, with one bounded catch-up retry for the crash
    window between `assess_risk`'s database commit and its own LangGraph
    checkpoint commit (see LangGraph resume fact 2 in the module docstring).

    In that window the checkpoint is still waiting on `assess_risk`, not on
    the interrupt, so the first resume re-runs (the already-cached)
    `assess_risk` and lands on a *fresh* interrupt without ever consuming
    `decision` - `record_outcome` never runs, and the call returns normally
    (no exception) because that fresh interrupt is a normal, successful
    pause, not a failure. Detectable because the case's status is still
    neither decided nor failed afterwards; exactly one more resume then
    finds a genuine pending interrupt and completes it. Any exception from
    the first call (AssessmentFailedError / IneligibleDecisionError /
    DuplicateApprovalError) propagates immediately - no retry, since the
    graph has already reached a definitive outcome.
    """
    submit_decision(graph, case_id, decision)
    with session_scope() as session:
        investigation = session.query(Investigation).filter_by(case_id=case_id).one()
        still_pending = investigation.status not in (INV_APPROVED, INV_REJECTED, INV_FAILED)
    if still_pending:
        submit_decision(graph, case_id, decision)
