# PaymentOps AI - Dispute Investigations

Policy-grounded investigation of synthetic card-payment disputes, with a durable,
concurrency-safe, crash-recoverable, human-in-the-loop decision step.

A cardholder's dispute against a transaction is loaded, the dispute-policy corpus is
searched in Qdrant (filtered to policies actually in effect on the filing date), an LLM
produces a cited risk assessment, deterministic eligibility rules are applied independently
of the LLM, and **every** case pauses for a named human reviewer before any terminal
decision - there is no automatic-approval path. The pause is a real LangGraph `interrupt()`
over a durable PostgreSQL checkpointer, so a paused case survives a process restart, and an
abandoned claim (the process crashed mid-decision) is reconciled automatically on the next
touch.

> **Safety boundary.** This service records decisions. It never issues a credit, never
> initiates a transfer, and never contacts a banking system. Every recorded outcome carries
> `"funds_moved": false`, and policy clause DP-POL-008 §8.2 states the same thing to the
> model. Treat an approval here as a signed-off recommendation to be executed elsewhere.

> **Scope, up front.** (1) All transactions, disputes and policies are **synthetic demo
> data** - see "Demo transactions" below for the fixed set this ships with. (2) Policy
> retrieval uses a **deterministic lexical hashing vectoriser**, not a trained semantic
> embedding model - see "Known limitations" #1. (3) The `LLM_PROVIDER=anthropic` path has
> been verified only with **mocked Anthropic SDK-boundary tests**
> (`tests/test_llm_anthropic_boundary.py`), never a live API call in this environment - see
> "Known limitations" #2 before relying on it.

---

## Architecture

```
POST /v1/investigations                                 POST /v1/investigations/{case_id}/decision
      |                                                              |
      v                                                              v
 +-----------+   +-------------------+   +-------------+   +----------------+   +----------------+
 | load_case |-->| retrieve_policies |-->| assess_risk |-->| await_decision |-->| record_outcome |
 +-----------+   +-------------------+   +-------------+   +----------------+   +----------------+
  DB read only     Qdrant top-k           LLM + fail-closed   interrupt()         simulated
  (case_id only     policy citations      grounding + det.    (nothing else)      decision + audit
   -> re-derives    filtered to as-of     eligibility                                   ^
   txn/dispute)     the filing date            |                                        |
                                           on failure: status=FAILED,              every case reaches
                                           raise (no interrupt reached)            this - no auto path
                                                                                   UPDATE guarded by a
                                                                                   partial unique index:
                                                                                   one APPROVED per txn

 checkpointer: PostgresSaver (PostgreSQL) | SqliteSaver (local/dev) -- keyed by case_id, strict msgpack
```

| File | Role |
|---|---|
| `app/config.py` | `Settings` from env/`.env`. No secret has a default; reviewer keys fail closed empty. |
| `app/models.py` | `transactions`, `disputes`, `investigations`, `idempotency_keys`, `audit_events`, plus the partial unique index enforcing one approval per transaction. |
| `app/db.py` | Lazy engine + `session_scope()`. |
| `app/retrieval.py` | Deterministic hashing embeddings + Qdrant index/search, filtered by policy version/effective-date/expiry. |
| `app/llm.py` | Claude via `messages.parse` structured output, plus a deterministic rule-based mock. |
| `app/graph.py` | The LangGraph workflow, intake idempotency (race-safe), deterministic eligibility, crash recovery, checkpointer, view builders. |
| `app/api.py` | FastAPI endpoints: idempotent intake, reviewer auth, concurrency-safe decisions, self-healing reconciliation. |
| `app/schemas.py` | Strict, flat Pydantic request/response + the LLM output contract. |
| `app/seed.py` | 11-clause dispute-policy corpus (including a future-effective and an expired policy) + 6 synthetic transactions + one pre-existing decided case. |

### The only input a graph thread ever receives is `case_id`

`load_case` re-derives the transaction and dispute from the database by following
`case_id -> Investigation.dispute_id -> Dispute.transaction_id` on every run. `graph.invoke`
is only ever called with `{"case_id": ...}` - there is no second channel through which a
later, conflicting request could inject different data into an existing thread. The
dispute itself (`transaction_id`, reason, description) is created exactly once, by
`intake()`, under a database transaction that commits *before* the graph is ever invoked.
See `tests/test_api.py::test_regression_conflicting_retry_does_not_desync_graph_and_database`.

### Every case reaches a real interrupt - no automatic approval

`assess_risk -> await_decision -> record_outcome` is a straight line; there is no
conditional edge that skips `await_decision`. A case can only end without a human decision
by **failing analysis first** (see below), which raises before the graph ever reaches
`await_decision` and leaves the case `FAILED`, never decided.

### Node re-execution and the interrupt

LangGraph re-enters an interrupted node **at its first line** when the thread resumes.
`await_decision` therefore contains **only** the `interrupt()` call - no write, no logging,
no LLM call. The billable LLM call lives in `assess_risk`, which completes and checkpoints
*before* the pause, so a resume never repeats it
(`tests/test_graph.py::test_interrupt_does_not_re_run_the_llm_or_duplicate_audit_rows`).
`load_case` is a pure read. `record_outcome` checks `decided_at` before writing, so a
crash-replay of it is a no-op.

Two LangGraph resume facts the crash-recovery code below relies on, verified empirically
against the installed LangGraph version rather than assumed (see `app/graph.py`'s module
docstring for the detail): a **plain** `graph.invoke(input, config)` on an existing thread
always restarts the run from `START`, safely, because every node guards itself before
writing; a **`Command(resume=...)`** call with no pending interrupt to deliver it to instead
just continues the Pregel loop from whatever node is actually pending, and is a safe no-op
if the thread has already reached `END`.

### Fail-closed policy grounding

`assess_risk` marks a case `FAILED` (never a silent guess) when:

- **no policy citation effective as of the dispute's filing date** was retrieved at all
  (see "Policy effective dates" below);
- **the LLM response does not parse** into the typed `LLMAssessment` contract, or the
  provider call itself errors (`app.llm.LLMError`);
- **the LLM cites a `policy_id@version` ref that was not in the retrieved set.**

A failed case is terminal and is never offered to a reviewer.

### Policy effective dates (fix for dated/superseded policy leakage)

Every policy chunk carries `effective_date` and an optional `expires_date`.
`retrieve_policies` passes the **dispute's filing date** (`dispute.created_at`, not
"today" - the rules that govern a case are the ones in effect when it was opened) to
`app.retrieval.search` as `as_of`, which excludes anything not yet effective or already
expired on that date **before** `assess_risk` ever sees it - a future-dated or superseded
policy can score however well it likes lexically; it is never handed to the LLM, and never
fills a citation slot an in-effect policy would otherwise take. If nothing in-effect
survives the filter, the case fails closed exactly as if nothing had been indexed at all.
See `tests/test_retrieval.py` (the filtering mechanism) and
`tests/test_graph.py::test_retrieve_policies_passes_the_dispute_filing_date_as_of` (the
wiring).

### Eligibility is deterministic, never the LLM's call

`compute_eligibility` is plain code: the filing-window check and the duplicate-approved-
dispute check never touch the LLM. `app.api.decide` refuses an `approve` decision with a
**409** when a **live** re-check (not the assess-time snapshot, which can go stale - see
below) shows `eligible_for_approval` is `False`, before the graph is ever resumed - no side
effect, no stuck thread.

### One approved dispute per transaction - enforced in the database

Two different investigations on the same transaction must never both end up `APPROVED`,
whether the two decisions happen seconds apart or at the same instant:

- **Sequential race** (case B is decided after case A's approval has already committed):
  `app.api.decide` re-checks `refresh_eligibility_for_decision` **live** against the
  database, under the claim, right before resuming the graph - not the cached snapshot from
  assessment time, which is exactly what goes stale here. Catches it with a clean `409`;
  case B is left perfectly usable (`PENDING_APPROVAL`, can still be rejected or retried).
- **Concurrent race** (both commits land at almost the same instant): no read-then-act
  check, however fresh, is airtight against this without serializable isolation or an
  explicit lock. The actual enforcement is
  `ux_one_approved_dispute_per_transaction` (`app/models.py`) - a partial unique index on
  `(transaction_id) WHERE decision = 'approve'`. `record_outcome`'s write is guarded by it;
  the database rejects the loser's commit outright, with no window for either side to
  observe a stale answer. Because LangGraph's interrupt is one-shot, the loser cannot be
  "corrected" with a second resume - `record_outcome` catches the `IntegrityError` and
  durably marks that case `FAILED` (`failure_reason="duplicate_approval_conflict"`) in its
  own clean transaction instead of a stuck, silently-retrying thread.

Tested directly against the database constraint with no API-layer check in the way at all
in `tests/test_graph.py::test_database_rejects_a_second_approval_on_the_same_transaction`
(and again in `tests/test_integration.py` against real PostgreSQL); tested through the live
HTTP surface, including a genuine two-thread race, in
`tests/test_api.py::test_concurrent_approvals_on_the_same_transaction_exactly_one_wins`.

**PostgreSQL business records are the source of truth - not the LangGraph checkpoint.**
`case_view` (what every API response is built from) reads only the `investigations` row;
it never inspects `graph.get_state`. This matters specifically for the losing side of a
duplicate-approval conflict: `record_outcome` raises `DuplicateApprovalError` *before*
returning an `outcome` state update, so the checkpoint for that thread never commits
anything success-shaped for that superstep in the first place - there is no stale
checkpoint value anywhere that a client-facing view could misread as "approved."
`tests/test_graph.py::test_duplicate_approval_conflict_leaves_no_misleading_success_state`
checks this directly: the PostgreSQL row (`FAILED`, no `decision_id`, no `decision`), the
API-shaped view built from it, and the raw checkpoint's `outcome` value (absent) all agree.

### Exclusive per-case ownership - not a time-based heuristic

An earlier version of this project gated crash recovery on a `claimed_at` lease timestamp:
"if nobody renewed this claim in N seconds, it's safe to take over." **That was wrong, and
was shown to be wrong**: a *slow but alive* worker can exceed any chosen N while still
correctly finishing its own work, and a timestamp cannot tell those two cases apart - a
fast worker and a dead one can leave the exact same `claimed_at` behind. Gating recovery on
staleness let a second worker reclaim a case out from under a worker that was still alive,
and that first worker's own eventual cleanup could then clobber the second worker's claim.

What actually proves a worker is gone - and only this does - is that a **PostgreSQL
session-level advisory lock** for the case becomes acquirable. `app.graph.case_ownership
(case_id)` acquires one (`pg_try_advisory_lock`) on a **dedicated connection** (never the
SQLAlchemy pool, so holding it never ties up a business-query connection and never requires
any business transaction to stay open) and holds it for the full duration of whatever graph
operation is about to happen, including cleanup on failure. The lock is released only when
that `with` block exits, or - if the process is killed outright - the instant the
connection drops; PostgreSQL enforces that, not application code. A second worker's own
`case_ownership(case_id)` call simply cannot succeed while the first is still inside its
`with` block, no matter how slow it is or how stale `claimed_at` looks. `claimed_at` is kept
only as a diagnostic "last touched" timestamp now - it gates nothing.

Every place that touches the graph or a `RUNNING`/`DECIDING` claim acquires this first:
`start_case` (initial execution and the re-invoke that recovers a stuck `RUNNING` case),
`submit_decision_reliably` (decision submission and resume, including replaying an
abandoned `DECIDING` claim), and the `GET`-triggered reconciliation in `app.api`.
`resolve_stuck_state` performs the actual recovery once ownership is held - it needs no
staleness check of its own, because successfully holding the lock already proves no one
else can be live on this case.

**Defence in depth:** `lock_token`, a fresh id generated per acquisition and stored on the
row for the duration of a `DECIDING` claim. `revert_claim` only clears a claim
`WHERE lock_token = :token`, so even if lock discipline were ever violated by a bug
elsewhere, a worker's own (stale) cleanup can never erase a *different* worker's claim or
its `pending_decision` - it would simply match zero rows.

**SQLite fallback (offline tests / local single-process dev only):** there is no
advisory-lock primitive, so `case_ownership` falls back to a plain in-process
`threading.Lock` keyed by `case_id`. That is correct mutual exclusion for concurrent
*threads in one process* - exactly what the offline `ThreadPoolExecutor`-based tests
exercise - but it is **not** a liveness proof across processes: a second process gets its
own empty lock table and would run straight through it. **PostgreSQL is the only backend
this mechanism is claimed correct for under real multi-worker concurrency**; the opt-in
integration suite is what verifies it there.

Two recovery windows, both gated purely on acquiring ownership - no staleness wait of any
kind:

1. **A case stuck `RUNNING`** - crashed before `start_case` was ever called, or before
   `assess_risk` committed anything. Triggered by a **same-key intake retry** finding
   `created=False` on a `RUNNING` row: acquire ownership, then `resolve_stuck_state` safely
   re-drives it with a plain re-invoke.
2. **An abandoned `DECIDING` claim** - crashed after a decision request claimed the row but
   before `record_outcome` ran to completion. The pending decision is persisted **at claim
   time** (`Investigation.pending_decision`) specifically so recovery does not have to guess
   what to resume with; `resolve_stuck_state` (after acquiring ownership from both `GET` and
   a new decision request) replays exactly that payload.

A third, narrower window - the database commit inside `assess_risk` lands but its LangGraph
checkpoint commit doesn't - is handled by `submit_decision_reliably`'s single bounded
catch-up resume (see the module docstring for the exact mechanics); it never loops.

Tested with real `threading.Event` synchronisation barriers (never timing assumptions) that
prove a *slow but alive* worker retains exclusive ownership and causes no overlapping
execution, that a stale cleanup cannot clobber a successor's claim, and that a *genuinely*
abandoned case can still be recovered, in `tests/test_graph.py` (unit level, SQLite
fallback) and again against the **authoritative PostgreSQL backend**, with real separate
connections and real OS threads, in
`tests/test_integration.py::test_regression_e_concurrent_workers_against_real_postgres_ownership`
and `test_advisory_lock_exclusion_and_release_on_close_against_real_postgres`.

### Concurrent idempotency-key insertion

Two requests sharing an `Idempotency-Key` can both pass `intake()`'s initial
"does this key exist" check before either commits - the loser's insert then raises a
primary-key `IntegrityError`, not a 500. `intake()` catches it, rolls back its own
half-built rows, and falls back to the same "existing key found" logic it would have taken
had it simply lost a slower race: identical payload -> `200` with the winner's case;
conflicting payload -> `IdempotencyConflict` -> `409`. Deterministic white-box test (forces
the exact race window every run, not just when the scheduler gets unlucky) in
`tests/test_graph.py`; real two-thread test through the live HTTP surface in
`tests/test_api.py::test_concurrent_intake_with_the_same_idempotency_key_never_500s`.

### Reviewer authentication

Decisions require `Authorization: Bearer <key>`, resolved against `REVIEWER_API_KEYS`
(`"name1:key1,name2:key2"`). The resolved name becomes `decided_by` - a caller cannot
assert someone else's identity through the request body. Empty configuration fails closed:
no key means no decision is ever accepted.

### Checkpoint security

`build_checkpointer()` passes `JsonPlusSerializer(pickle_fallback=False)` explicitly and
sets `LANGGRAPH_STRICT_MSGPACK=true`, restricting checkpoint deserialization to a built-in
allowlist of safe types.

### Money and identifiers

`amount_minor: BigInteger` + `Decimal(n).scaleb(-2)` accessor - no float ever touches an
amount. `case_id` is a canonical UUID string, validated as a path parameter (a malformed
one is `422`). A whitespace-only `description` is rejected (`422`) the same as an empty one.

---

## Quick start (no Docker)

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python -m app.seed                                  # synthetic data + policy index
REVIEWER_API_KEYS="cfo@example.com:demo-reviewer-key" uvicorn app.api:app --port 8000   # terminal 1
PAYMENTOPS_URL=http://localhost:8000 python demo.py                                     # terminal 2
```

Defaults are SQLite + embedded in-memory Qdrant + the deterministic mock assessor, so this
runs with no services and no API key. `demo.py` is safe to run repeatedly without resetting
the database - see "Demonstration" below.

## Quick start (Docker Compose: PostgreSQL + Qdrant + API)

```bash
cp .env.example .env            # optional; only needed to switch to a real LLM or reviewer keys
docker compose up --build
curl http://localhost:8000/healthz
PAYMENTOPS_URL=http://localhost:8000 python demo.py
```

Compose seeds the database and policy index on startup, and sets a demo reviewer key
(`cfo@example.com:demo-reviewer-key` - override via `REVIEWER_API_KEYS` in `.env`; replace
before any real deployment). To use Claude instead of the mock:

```bash
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... docker compose up --build
```

The key is read from your shell or `.env` and passed through as an environment variable -
never written into the image, the compose file, or the repository.

**This is a local demo stack, not a deployment.** All three published ports
(`127.0.0.1:8000`, `127.0.0.1:5432`, `127.0.0.1:6333`) are bound to localhost only in
`docker-compose.yml` - nothing here is reachable from another machine by default. The
`POSTGRES_PASSWORD` and the default `REVIEWER_API_KEYS` value are demo credentials checked
into the compose file for convenience; **replace both, and reconsider the port bindings,
before running any of this outside a single developer's own machine.**

---

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `sqlite+pysqlite:///./paymentops.db` | SQLAlchemy URL. A `postgresql+psycopg://` URL also switches the checkpointer to `PostgresSaver`. |
| `CHECKPOINT_DB` | `./checkpoints.sqlite` | Checkpoint file, used only when the database is not PostgreSQL. |
| `QDRANT_URL` | *(empty)* | Empty means embedded in-memory Qdrant. Set to `http://qdrant:6333` for a server. |
| `QDRANT_COLLECTION` | `dispute_policies` | Collection name. |
| `EMBEDDING_DIM` | `256` | Hashing-vectoriser dimension. |
| `TOP_K` | `4` | Policy chunks retrieved per case. |
| `LLM_PROVIDER` | `mock` | `mock` (deterministic, offline) or `anthropic`. |
| `ANTHROPIC_MODEL` | `claude-opus-5` | Any Claude model id. |
| `ANTHROPIC_API_KEY` | *(unset)* | Falls back to the SDK's own credential resolution if unset. |
| `ANTHROPIC_BASE_URL` | *(unset)* | For a gateway or proxy. |
| `LLM_MAX_TOKENS` | `8192` | Response cap for the assessment call. |
| `DISPUTE_FILING_WINDOW_DAYS` | `60` | Deterministic eligibility: filing-window cutoff. |
| `HIGH_VALUE_MINOR` | `500000` | 5,000.00 in minor units. Advisory risk flag threshold (not an eligibility gate). |
| `REVIEWER_API_KEYS` | *(empty)* | `"name1:key1,name2:key2"`. Empty => every decision is `401` (fail closed). |
| `STUCK_CLAIM_TIMEOUT_SECONDS` | `20` | Diagnostic only - not a safety gate. See "Exclusive per-case ownership" above. |

---

## API

**Response shape.** Every case endpoint returns one flat object: `case_id`,
`transaction_id`, `status`, `failure_reason`, `question`, `recommendation`, `risk_score`,
`risk_flags`, `policy_citations`, `rationale`, `eligible_for_approval`,
`eligibility_reasons`, `decision_id`, `decision`, `decided_by`, `decision_reason`,
`decided_at`, `simulated`, `funds_moved`. Fields not yet applicable to the case's current
status are `null` / empty rather than omitted.

`status` is one of `RUNNING`, `PENDING_APPROVAL`, `APPROVED`, `REJECTED`, `FAILED`.
`decision` (and the LLM's advisory `recommendation`) is `approve` or `reject`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Liveness + which backends and provider are wired up. |
| `GET` | `/v1/transactions` | All synthetic transactions. |
| `POST` | `/v1/investigations` | Open a case. Requires `Idempotency-Key` header. `201` new, `200` identical replay (or a recovered stuck case), `409` conflicting replay, `404` unknown transaction, `422` strict-validation failure, `502` analysis failed to ground (case is still `FAILED`-readable). |
| `GET` | `/v1/investigations/{case_id}` | Current state. Also reconciles an abandoned `DECIDING` claim before returning. |
| `POST` | `/v1/investigations/{case_id}/decision` | Requires `Authorization: Bearer <reviewer key>`. `200` decided or replayed, `401` auth failure, `404` unknown case, `409` conflicting/concurrent decision **or an ineligible approval attempt**, `422` malformed input. |
| `GET` | `/v1/investigations/{case_id}/audit` | Ordered, immutable audit trail for the case. |

Interactive docs: `http://localhost:8000/docs`.

### Reproducible example requests

Transaction ids are fixed by the seed, so these run verbatim after `python -m app.seed`
(on a **fresh** database - see "Demonstration" for what changes on a repeat run).

**Routine, eligible dispute** - duplicate charge, corroborated, well within the filing
window:

```bash
curl -s -X POST http://localhost:8000/v1/investigations \
  -H 'Idempotency-Key: demo-1' -H 'Content-Type: application/json' \
  -d '{"transaction_id": "11111111-1111-4111-8111-111111111111", "dispute_reason": "duplicate_charge", "description": "Same amount charged twice by this merchant."}'
```

```json
{
  "case_id": "...",
  "status": "PENDING_APPROVAL",
  "recommendation": "approve",
  "risk_score": 0.1,
  "eligible_for_approval": true,
  "eligibility_reasons": [],
  "policy_citations": [{"policy_id": "DP-POL-002", "version": "1.0", "effective_date": "2025-01-01", "expires_date": null, "...": "..."}]
}
```

**A dispute no one can approve** - filed 90 days after the transaction, outside the
60-day window:

```bash
curl -s -X POST http://localhost:8000/v1/investigations \
  -H 'Idempotency-Key: demo-2' -H 'Content-Type: application/json' \
  -d '{"transaction_id": "33333333-3333-4333-8333-333333333333", "dispute_reason": "credit_not_processed", "description": "Merchant promised a refund that never arrived."}'
```

`eligible_for_approval` is `false`, `eligibility_reasons: ["filed_after_dispute_window"]`.
Attempting to approve it anyway:

```bash
curl -s -X POST http://localhost:8000/v1/investigations/<case_id>/decision \
  -H 'Authorization: Bearer demo-reviewer-key' \
  -d '{"decision": "approve", "reason": "approve anyway"}'
# -> 409 {"detail": {"message": "Investigation is not eligible for approval", "eligibility_reasons": ["filed_after_dispute_window"]}}
```

Rejection is always allowed regardless of eligibility.

**Record a human decision:**

```bash
curl -s -X POST http://localhost:8000/v1/investigations/<case_id>/decision \
  -H 'Authorization: Bearer demo-reviewer-key' -H 'Content-Type: application/json' \
  -d '{"decision": "approve", "reason": "Confirmed duplicate with the cardholder statement."}'
```

**Audit trail:**

```bash
curl -s http://localhost:8000/v1/investigations/<case_id>/audit
```

```
investigation_opened   system:workflow
assessment_recorded    mock-rules/v1
decision_recorded      cfo@example.com
settlement_simulated   system:workflow   {"funds_moved": false, ...}
```

### Demo transactions

| Transaction | Amount | Why it is interesting |
|---|---|---|
| `11111111-...1111` | 18.50 USD | Routine, corroborated duplicate charge -> eligible, low risk. |
| `22222222-...2222` | 1,349.00 USD | Defective merchandise, return referenced -> eligible. |
| `33333333-...3333` | 24.99 USD | Occurred 90 days ago -> outside the filing window, ineligible. |
| `44444444-...4444` | 320.00 USD | Already has a prior **approved** dispute (seeded fixture) -> a second dispute is ineligible (no double credit). |
| `55555555-...5555` | 6,120.00 USD | Over the enhanced-verification threshold -> flagged, still eligible. |
| `66666666-...6666` | 450.00 USD | Reason `other` -> unclassified, still eligible. |

---

## Tests

### Offline suite (no network, no Docker, no API key)

```bash
python -m pytest -q
```

```
89 passed in 11.32s
```

(The opt-in integration suite is excluded by default - see below.) Covers retrieval
(including the version/effective-date/expiry filter), the always-interrupt workflow,
fail-closed grounding, deterministic eligibility, the one-approval-per-transaction database
constraint (both the direct database-level test and the live HTTP concurrency test), crash
recovery for both bounded-lease windows, the concurrent idempotency-key insertion race,
durability across a rebuilt graph+checkpointer, node re-execution safety, reviewer auth, and
strict validation (`422` without state mutation, including whitespace-only descriptions)
across the whole HTTP surface.

### Mocked Anthropic SDK-boundary tests

```bash
python -m pytest tests/test_llm_anthropic_boundary.py -q
```

```
6 passed in 2.11s
```

Monkeypatches `anthropic.Anthropic` itself - verifies `app.llm`'s structured-output
handling and error translation without ever calling the real API. **Not live
verification** - see Known limitations.

### Opt-in integration suite (real PostgreSQL + real Qdrant)

```bash
docker compose up -d postgres qdrant
RUN_INTEGRATION_TESTS=1 python -m pytest tests/test_integration.py -v
```

```
tests/test_integration.py::test_retrieval_round_trips_through_real_qdrant PASSED
tests/test_integration.py::test_full_lifecycle_with_real_postgres_checkpointer_and_process_restart PASSED
tests/test_integration.py::test_database_rejects_a_second_approval_on_the_same_transaction_in_postgres PASSED
tests/test_integration.py::test_crash_recovery_against_real_postgres PASSED
tests/test_integration.py::test_checkpoint_tables_exist_in_postgres PASSED
5 passed in 4.84s
```

Skipped automatically unless `RUN_INTEGRATION_TESTS=1` (and skipped, not failed, if
Postgres/Qdrant aren't reachable). Verifies a real Qdrant round-trip; a real
`PostgresSaver`-backed interrupt that survives a brand-new checkpointer/graph object; the
one-approval-per-transaction partial unique index against real PostgreSQL, not just SQLite;
both crash-recovery windows against the real checkpointer and connection pool; and the
resulting audit trail read straight back out of PostgreSQL. All of it uses freshly-created
rows per run (not fixed seed ones) so the whole file is safely re-runnable against the same
persistent compose database without manual cleanup.

---

## Demonstration (full Docker stack)

```bash
docker compose up -d --build
curl -s http://localhost:8000/healthz
PAYMENTOPS_URL=http://localhost:8000 python demo.py
```

`demo.py` is safe to run again immediately, as many times as you like, without resetting
the database - every intake uses a fresh `Idempotency-Key`, and every response is checked
for its actual status code before any success-only field is read. The one place a second
run looks different from the first is deliberate: once the demo's routine-approval
transaction already has an approved dispute from an earlier run, the one-approval-per-
transaction rule correctly refuses a *second* one with a `409` - the script detects that and
shows the refusal (then rejects the case instead) rather than crashing on it. This was
verified by running the demo three times in a row against one persistent PostgreSQL volume
with no reset in between.

Restart-durability proof:

```bash
CASE=$(curl -s -X POST localhost:8000/v1/investigations -H 'Idempotency-Key: restart-1' \
  -d '{"transaction_id":"55555555-5555-4555-8555-555555555555","dispute_reason":"unauthorized","description":"..."}' \
  | python -c "import sys,json;print(json.load(sys.stdin)['case_id'])")
docker compose restart api
curl -s localhost:8000/v1/investigations/$CASE   # -> still PENDING_APPROVAL
curl -s -X POST localhost:8000/v1/investigations/$CASE/decision \
  -H 'Authorization: Bearer demo-reviewer-key' -d '{"decision":"approve","reason":"..."}'
```

Concurrency proof (two conflicting decisions on the same case, fired at once):

```bash
curl -s -X POST localhost:8000/v1/investigations/$CASE/decision -H 'Authorization: Bearer demo-reviewer-key' -d '{"decision":"approve","reason":"A"}' &
curl -s -X POST localhost:8000/v1/investigations/$CASE/decision -H 'Authorization: Bearer demo-reviewer-key' -d '{"decision":"reject","reason":"B"}' &
wait
# -> one 200, one 409; audit trail shows exactly one decision_recorded row
```

One-approval-per-transaction proof (two *different* investigations on the *same*
transaction, approved at once):

```bash
# open case A and case B against the same transaction_id with different idempotency keys...
curl -s -X POST localhost:8000/v1/investigations/$CASE_A/decision -H 'Authorization: Bearer demo-reviewer-key' -d '{"decision":"approve","reason":"A"}' &
curl -s -X POST localhost:8000/v1/investigations/$CASE_B/decision -H 'Authorization: Bearer demo-reviewer-key' -d '{"decision":"approve","reason":"B"}' &
wait
# -> one 200 (APPROVED), one 409 (FAILED) - never both APPROVED
```

Slow-but-alive-worker proof (the exclusive-ownership mechanism itself, with a real
synchronisation barrier rather than curl's limited timing control - this is what
`tests/test_integration.py::test_regression_e_concurrent_workers_against_real_postgres_ownership`
actually runs, against this same live stack): a worker holding ownership of a case for
several real seconds - deliberately slow, not dead - is confirmed to block a second
worker's `pg_try_advisory_lock` attempt for the same case the entire time, with zero
overlapping graph execution, then correctly completes once released.

---

## Known limitations

1. **Embeddings are lexical, not semantic (deliberate MVP scope).** `app/retrieval.py`
   uses a deterministic hashing vectoriser (stdlib + numpy), not a trained embedding
   model, so indexing stays offline, free and perfectly reproducible - appropriate for the
   11-chunk policy corpus this demo ships. It will miss paraphrases a real embedding model
   would catch. Swap `embed()` for a hosted embedding model (and bump `EMBEDDING_DIM` to
   match it) if the corpus grows past a few dozen clauses, or if recall on reworded
   queries becomes a real problem - `search()` already returns the same `PolicyCitation`
   shape regardless of how `embed()` is implemented.
2. **The Anthropic provider path is covered by mocked-SDK-boundary tests, not live
   verification.** No `ANTHROPIC_API_KEY` was available in this environment. The mock
   provider (default, used by every offline and integration test here) is a real rule
   engine, not a stub, and exercises the same grounding/eligibility code paths the real
   provider would. Before trusting `LLM_PROVIDER=anthropic` in anything beyond a demo: run
   it once against the live API and confirm structured output and citation-copying behave
   as `app.graph.assess_risk`'s fail-closed checks expect.
3. **The graph-level defence-in-depth guards (`IneligibleDecisionError`,
   `DuplicateApprovalError`) are not retryable if they ever fire** through a caller that
   bypasses `app.api.decide`'s own pre-checks (LangGraph's interrupt is one-shot - see
   `app/graph.py`'s module docstring). Both now commit a clean terminal `FAILED` state
   before raising, so the case is never left silently stuck - but it is still a wrong
   decision for a human to have to re-investigate from scratch, not a graceful retry.
4. **A crash before any client ever retries leaves a case waiting indefinitely** - recovery
   is triggered by a same-key intake retry, a `GET`, or a new decision request; nothing
   proactively sweeps for stuck cases on its own. This is unrelated to staleness now (there
   is no staleness check left to satisfy) - it is purely "nobody has touched this case
   since it broke." No background sweeper is implemented - it would be speculative
   infrastructure for a case this project has not needed in practice.
5. **The advisory-lock mechanism opens one dedicated PostgreSQL connection per ownership
   acquisition attempt, with no pooling** (`app.graph._pg_try_acquire`) - simplest correct
   thing at this app's scale (see the `ponytail:` comment there for the upgrade path if
   case operations ever become hot enough for connection churn to matter). It is also only
   as prompt as PostgreSQL's own dead-connection detection: if a worker's process is killed
   in a way that doesn't cleanly close its TCP connection (unlike the graceful
   `conn.close()` this code always runs), the lock can appear held for a TCP-keepalive-
   dependent interval after the crash, not instantly - strictly better than the old
   heuristic (which could wait the same or longer *while also* being wrong about live
   workers), never worse, but not instantaneous either.
6. **The SQLite fallback (`threading.Lock`) is a same-process mutual-exclusion mechanism,
   not a liveness proof across processes** - correct for the offline test suite's own
   concurrency (real threads, one process), explicitly not claimed correct for a real
   multi-worker deployment on SQLite. PostgreSQL is the only backend this project claims
   the ownership guarantee for.
7. **No migrations.** `Base.metadata.create_all()` on startup, with no in-place upgrade
   path for an existing database's schema - a schema change (a new column, a new index)
   against a database that already has the old schema needs a `docker compose down -v`
   (or an equivalent manual migration) rather than being picked up automatically. Add
   Alembic when the schema needs to change under real data.
8. **No authn/authz on intake** (`POST /v1/investigations`), only on decisions.
9. **Reviewer API keys are a flat shared-secret map**, not a real identity provider.
