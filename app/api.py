"""FastAPI surface over the dispute investigation workflow.

Endpoints are plain `def`, so FastAPI runs them in a worker thread - the graph,
SQLAlchemy and Qdrant clients here are all synchronous.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from sqlalchemy import select

from app.config import get_settings
from app.db import init_db, session_scope
from app.graph import (
    AssessmentFailedError,
    CaseLockError,
    DuplicateApprovalError,
    IdempotencyConflict,
    IneligibleDecisionError,
    TransactionNotFound,
    begin_deciding,
    build_checkpointer,
    build_graph,
    case_ownership,
    case_view,
    intake,
    refresh_eligibility_for_decision,
    resolve_stuck_state,
    revert_claim,
    start_case,
    submit_decision_reliably,
)
from app.models import (
    INV_APPROVED,
    INV_DECIDING,
    INV_FAILED,
    INV_PENDING_APPROVAL,
    INV_REJECTED,
    INV_RUNNING,
    AuditEvent,
    Investigation,
    Transaction,
    transaction_to_dict,
)
from app.retrieval import count_policies, ensure_index
from app.schemas import (
    AuditEventOut,
    CaseResponse,
    CreateInvestigationRequest,
    DecisionRequest,
    HealthResponse,
    TransactionOut,
)
from app.seed import POLICY_DOCS


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Idempotent; also the only way the embedded in-memory Qdrant gets populated.
    ensure_index(POLICY_DOCS)
    app.state.graph = build_graph(build_checkpointer())
    yield


app = FastAPI(
    title="PaymentOps AI - Dispute Investigations",
    version="0.4.0",
    summary="Policy-grounded card dispute investigation with human-in-the-loop decisions.",
    description=(
        "Decisions recorded by this service are SIMULATED. No credit is ever issued, "
        "no transfer is ever initiated, and no funds are ever moved."
    ),
    lifespan=lifespan,
)


def require_reviewer(authorization: str | None = Header(default=None)) -> str:
    """Resolve the caller's reviewer identity from a configured API key.

    Fails closed: an empty REVIEWER_API_KEYS configuration rejects every
    decision request rather than letting one through unauthenticated.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    reviewer_id = get_settings().reviewer_key_map().get(token)
    if not reviewer_id:
        raise HTTPException(status_code=401, detail="Unrecognised reviewer API key")
    return reviewer_id


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        database=settings.database_url.split("://", 1)[0],
        qdrant=settings.qdrant_url or "embedded:memory",
        llm_provider=settings.llm_provider,
        policies_indexed=count_policies(),
    )


@app.get("/v1/transactions", response_model=list[TransactionOut])
def list_transactions() -> list[TransactionOut]:
    with session_scope() as session:
        rows = session.scalars(select(Transaction).order_by(Transaction.id)).all()
        return [TransactionOut.model_validate(transaction_to_dict(t)) for t in rows]


@app.post("/v1/investigations", response_model=CaseResponse)
def create_investigation(
    request: CreateInvestigationRequest,
    response: Response,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=200),
) -> CaseResponse:
    with session_scope() as session:
        try:
            case_id, created = intake(
                session,
                idempotency_key=idempotency_key,
                transaction_id=str(request.transaction_id),
                dispute_reason=request.dispute_reason,
                description=request.description,
            )
        except TransactionNotFound as exc:
            raise HTTPException(status_code=404, detail=f"Unknown transaction {exc}") from exc
        except IdempotencyConflict as exc:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Idempotency-Key was already used with a different request payload "
                    f"(existing case {exc.existing_case_id})"
                ),
            ) from exc

    if not created:
        # Identical retry: never hand new intake data to an existing thread.
        # Only bother acquiring exclusive ownership if there is actually
        # something that might need finishing (RUNNING) - avoids a
        # PostgreSQL round trip on the common "retry of an already-settled
        # case" path. If ownership can't be acquired, another worker is
        # genuinely still live on it right now - leave it alone and just
        # report the current state.
        with session_scope() as session:
            still_running = session.query(Investigation).filter_by(case_id=case_id).one().status == INV_RUNNING
        if still_running:
            try:
                with case_ownership(case_id) as token:
                    if token is not None:
                        resolve_stuck_state(app.state.graph, case_id)
            except CaseLockError:
                pass  # best-effort; the client can simply retry
        response.status_code = 200
        return CaseResponse.model_validate(case_view(case_id))

    try:
        with case_ownership(case_id) as token:
            if token is None:  # pragma: no cover - a brand-new case_id has no contender
                raise HTTPException(
                    status_code=503, detail="Could not acquire exclusive processing for this case; try again."
                )
            start_case(app.state.graph, case_id)
    except AssessmentFailedError:
        # Case is durably `FAILED` in the database - app.graph.assess_risk commits
        # the failure before raising - so it is still a normal 200-readable
        # resource. The create call itself reports the upstream failure.
        raise HTTPException(
            status_code=502,
            detail={"case_id": case_id, "message": "Analysis could not be grounded; case marked failed."},
        ) from None
    except CaseLockError as exc:
        raise HTTPException(status_code=503, detail=f"Case lock mechanism unavailable: {exc}") from exc

    response.status_code = 201
    return CaseResponse.model_validate(case_view(case_id))


@app.get("/v1/investigations/{case_id}", response_model=CaseResponse)
def get_investigation(case_id: UUID) -> CaseResponse:
    cid = str(case_id)
    with session_scope() as session:
        investigation = session.query(Investigation).filter_by(case_id=cid).one_or_none()
        if investigation is None:
            raise HTTPException(status_code=404, detail=f"Unknown case {case_id}")
        maybe_stuck = investigation.status in (INV_RUNNING, INV_DECIDING)

    if maybe_stuck:
        # Only attempt ownership when there's actually something that might
        # need reconciling - a plain read of a normal pending/decided case
        # never touches PostgreSQL for the lock. If another worker holds
        # ownership right now, it is by definition still live - leave it
        # alone and just return whatever the view currently shows.
        try:
            with case_ownership(cid) as token:
                if token is not None:
                    resolve_stuck_state(app.state.graph, cid)
        except CaseLockError:
            pass

    view = case_view(cid)
    if view is None:  # pragma: no cover - already confirmed to exist above
        raise HTTPException(status_code=404, detail=f"Unknown case {case_id}")
    return CaseResponse.model_validate(view)


@app.post("/v1/investigations/{case_id}/decision", response_model=CaseResponse)
def submit_investigation_decision(
    case_id: UUID,
    request: DecisionRequest,
    reviewer_id: str = Depends(require_reviewer),
) -> CaseResponse:
    """Concurrency-safe terminal decision.

    `case_ownership` is the actual safety mechanism (see app/graph.py's
    module docstring): while this request holds it, no other worker - no
    matter how it got there - can be concurrently touching this case's
    graph thread or claim, so the claim transition below is a plain write,
    not a compare-and-swap racing anyone. A request that fails to acquire
    ownership gets a `409` and touches nothing.

    Eligibility is re-checked live, under that ownership, rather than
    trusting the snapshot cached at assessment time - a *different*
    investigation on the same transaction may have been approved since
    then. Even that live check cannot close a genuinely concurrent race
    between two *different* investigations' decisions; the database's
    one-approval-per-transaction index is what actually closes that one
    (see app.graph.record_outcome) - ownership here only ever serializes
    operations on *this one* case_id.
    """
    cid = str(case_id)
    try:
        with case_ownership(cid) as token:
            if token is None:
                raise HTTPException(
                    status_code=409,
                    detail="This case is currently being processed by another request; try again shortly.",
                )
            return _decide_with_ownership(cid, token, request, reviewer_id)
    except CaseLockError as exc:
        raise HTTPException(status_code=503, detail=f"Case lock mechanism unavailable: {exc}") from exc


def _decide_with_ownership(cid: str, token: str, request: DecisionRequest, reviewer_id: str) -> CaseResponse:
    """The actual decision logic, run only while `case_ownership(cid)` is held."""
    resolve_stuck_state(app.state.graph, cid)  # finish any dead worker's leftover first - safe, we hold the lock
    normalized_reason = (request.reason or "").strip()
    pending_decision = {"decision": request.decision, "actor": reviewer_id, "reason": request.reason}

    with session_scope() as session:
        investigation = session.query(Investigation).filter_by(case_id=cid).one_or_none()
        if investigation is None:
            raise HTTPException(status_code=404, detail=f"Unknown case {cid}")

        status = investigation.status
        if status in (INV_APPROVED, INV_REJECTED):
            decision_snapshot = (
                investigation.decision,
                investigation.decided_by,
                (investigation.decision_reason or "").strip(),
            )
            if decision_snapshot == (request.decision, reviewer_id, normalized_reason):
                return CaseResponse.model_validate(case_view(cid))
            raise HTTPException(status_code=409, detail="A different decision is already recorded for this case")
        if status == INV_FAILED:
            raise HTTPException(status_code=409, detail="Case failed analysis and cannot be decided")
        if status != INV_PENDING_APPROVAL:
            raise HTTPException(status_code=409, detail=f"Case is {status}, not awaiting approval")

        eligible, reasons = True, []
        if request.decision == "approve":
            eligible, reasons = refresh_eligibility_for_decision(
                session,
                case_id=cid,
                transaction_id=investigation.transaction_id,
                stored_reasons=list(investigation.eligibility_reasons or []),
            )
            if not eligible:
                raise HTTPException(
                    status_code=409,
                    detail={"message": "Investigation is not eligible for approval", "eligibility_reasons": reasons},
                )

        begin_deciding(session, cid, token, pending_decision)

    try:
        submit_decision_reliably(app.state.graph, cid, pending_decision)
    except DuplicateApprovalError as exc:
        # The database's one-approval-per-transaction index caught a genuine
        # concurrent race *across two different cases*; app.graph.record_outcome
        # already persisted this case as FAILED - nothing to revert.
        raise HTTPException(
            status_code=409,
            detail=f"A different investigation on this transaction was already approved: {exc.reason}",
        ) from None
    except IneligibleDecisionError as exc:
        # Residual defence-in-depth: the live re-check above should already
        # have caught this. app.graph.record_outcome has still persisted a
        # clean FAILED case rather than leaving it stuck - nothing to revert.
        raise HTTPException(
            status_code=409,
            detail={"message": "Investigation is not eligible for approval", "eligibility_reasons": exc.reasons},
        ) from None
    except Exception as exc:  # noqa: BLE001 - must revert the claim no matter what else failed
        revert_claim(cid, token)
        raise HTTPException(status_code=502, detail="Failed to record the decision") from exc

    return CaseResponse.model_validate(case_view(cid))


@app.get("/v1/investigations/{case_id}/audit", response_model=list[AuditEventOut])
def get_audit_trail(case_id: UUID) -> list[AuditEventOut]:
    cid = str(case_id)
    with session_scope() as session:
        events = session.scalars(
            select(AuditEvent).where(AuditEvent.case_id == cid).order_by(AuditEvent.created_at, AuditEvent.id)
        ).all()
        if not events:
            raise HTTPException(status_code=404, detail=f"No audit trail for case {case_id}")
        return [AuditEventOut.model_validate(e, from_attributes=True) for e in events]
