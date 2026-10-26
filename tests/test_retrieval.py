"""Policy indexing and retrieval, including the version/effective-date metadata."""

from __future__ import annotations

import math
from datetime import date

import pytest

from app.retrieval import count_policies, embed, ensure_index, search
from app.seed import POLICY_DOCS


def test_index_is_populated_and_idempotent(env):
    first = ensure_index(POLICY_DOCS)
    assert first == len(POLICY_DOCS)
    # Re-indexing upserts the same stable point ids rather than duplicating.
    assert ensure_index(POLICY_DOCS) == len(POLICY_DOCS)
    assert count_policies() == len(POLICY_DOCS)


def test_embedding_is_deterministic_and_normalised(env):
    a = embed("dispute filed after the authorisation window")
    b = embed("dispute filed after the authorisation window")
    assert a == b
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-5)
    assert embed("") == [0.0] * len(a)


@pytest.mark.parametrize(
    ("query", "expected_policy_id"),
    [
        ("dispute filed more than 60 days after the transaction", "DP-POL-001"),
        ("duplicate charge matching merchant and amount", "DP-POL-002"),
        ("unauthorized transaction over 5000 requires enhanced verification", "DP-POL-003"),
        ("product never arrived no delivery date referenced", "DP-POL-004"),
    ],
)
def test_search_finds_the_right_policy(env, query, expected_policy_id):
    ensure_index(POLICY_DOCS)
    hits = search(query, k=3)
    assert hits, "expected at least one hit"
    assert expected_policy_id in {h.policy_id for h in hits}


def test_citations_carry_version_and_effective_date(env):
    ensure_index(POLICY_DOCS)
    hits = search("dispute filing window", k=1)
    assert hits
    citation = hits[0]
    assert citation.version == "1.0"
    assert citation.effective_date.isoformat() == "2025-01-01"
    assert citation.ref == f"{citation.policy_id}@{citation.version}"


def test_search_returns_empty_when_nothing_indexed(env):
    assert search("anything at all") == []


# --------------------------------------------------------------------------- #
# Effective dates: exclude future-effective and expired policies (fix #4)
# --------------------------------------------------------------------------- #


def test_future_effective_policy_is_excluded_as_of_today(env):
    ensure_index(POLICY_DOCS)
    query = "dispute must be verified against the cardholder's biometric signature"

    # Without a date filter, the future policy is genuinely the best lexical
    # match - proves exclusion below is really the date filter, not low rank.
    undated = search(query, k=3)
    assert "DP-POL-009" in {h.policy_id for h in undated}

    dated = search(query, k=3, as_of=date.today())
    assert "DP-POL-009" not in {h.policy_id for h in dated}


def test_expired_policy_is_excluded_as_of_today(env):
    ensure_index(POLICY_DOCS)
    query = "a dispute credit requires a signed paper dispute form mailed to the processing centre"

    undated = search(query, k=3)
    assert "DP-POL-010" in {h.policy_id for h in undated}

    dated = search(query, k=3, as_of=date.today())
    assert "DP-POL-010" not in {h.policy_id for h in dated}


def test_as_of_exactly_on_effective_date_includes_the_policy(env):
    """effective_date <= as_of: a policy is in effect on its own effective date."""
    ensure_index(POLICY_DOCS)
    hits = search("dispute filing window", k=5, as_of=date(2025, 1, 1))
    assert "DP-POL-001" in {h.policy_id for h in hits}


def test_as_of_before_effective_date_excludes_the_policy(env):
    ensure_index(POLICY_DOCS)
    hits = search("dispute filing window", k=5, as_of=date(2024, 12, 31))
    assert "DP-POL-001" not in {h.policy_id for h in hits}


def test_as_of_exactly_on_expiry_date_excludes_the_policy(env):
    """expires_date is exclusive: a policy is NOT in effect on its own expiry date."""
    ensure_index(POLICY_DOCS)
    expired = next(d for d in POLICY_DOCS if d["policy_id"] == "DP-POL-010")
    hits = search(
        "a dispute credit requires a signed paper dispute form mailed to the processing centre",
        k=5,
        as_of=expired["expires_date"],
    )
    assert "DP-POL-010" not in {h.policy_id for h in hits}
