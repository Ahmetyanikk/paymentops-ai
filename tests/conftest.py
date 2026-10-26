"""Test fixtures.

The offline suite runs entirely without network or real services: SQLite
instead of PostgreSQL, embedded in-memory Qdrant instead of a server, and the
deterministic mock assessor instead of a real Claude call. No offline test
result depends on a network call or an API key.

See tests/test_integration.py for the opt-in suite that exercises real
PostgreSQL + Qdrant containers, and tests/test_llm_anthropic_boundary.py for
mocked-at-the-SDK-boundary coverage of the `anthropic` provider path.
"""

from __future__ import annotations

import pytest

from app.config import reset_settings
from app.db import reset_engine
from app.retrieval import reset_client

TEST_REVIEWER = "reviewer-1"
TEST_REVIEWER_KEY = "test-reviewer-key-123"
OTHER_REVIEWER = "reviewer-2"
OTHER_REVIEWER_KEY = "test-reviewer-key-456"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("CHECKPOINT_DB", (tmp_path / "checkpoints.sqlite").as_posix())
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_policies")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv(
        "REVIEWER_API_KEYS", f"{TEST_REVIEWER}:{TEST_REVIEWER_KEY},{OTHER_REVIEWER}:{OTHER_REVIEWER_KEY}"
    )
    # Short, but tests that need a "stale" claim still explicitly backdate
    # `claimed_at` rather than relying on wall-clock sleep - this just keeps
    # a "not yet stale" assertion fast too.
    monkeypatch.setenv("STUCK_CLAIM_TIMEOUT_SECONDS", "2")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    reset_settings()
    reset_engine()
    reset_client()
    yield tmp_path
    reset_settings()
    reset_engine()
    reset_client()


@pytest.fixture
def seeded(env):
    from app.seed import seed_all

    return seed_all()


@pytest.fixture
def graph(seeded):
    """A compiled graph over a durable (file-backed) SQLite checkpoint store."""
    from app.graph import build_checkpointer, build_graph

    checkpointer = build_checkpointer()
    yield build_graph(checkpointer)
    checkpointer.conn.close()


@pytest.fixture
def reviewer_auth_header() -> dict[str, str]:
    return {"Authorization": f"Bearer {TEST_REVIEWER_KEY}"}
