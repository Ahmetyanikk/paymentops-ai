"""Mocked-SDK-boundary tests for the LLM_PROVIDER=anthropic path.

IMPORTANT: these tests never make a real network call and never require
ANTHROPIC_API_KEY. `anthropic.Anthropic` itself is monkeypatched at the SDK
boundary (`app.llm._anthropic_assess` constructs a client and calls
`.messages.parse(...)` - that call is what is faked here), so they verify
`app.llm`'s structured-output handling and error translation, not Claude's
actual behaviour. They are a *substitute* for live verification, not a
replacement for it - see README "Known limitations" for what live
verification against the real API would still need to confirm before this
provider path is trusted in production.
"""

from __future__ import annotations

import anthropic
import httpx2
import pytest

from app.llm import LLMError, assess
from app.schemas import LLMAssessment, PolicyCitation

CITATION = PolicyCitation(
    policy_id="DP-POL-001",
    version="1.0",
    effective_date="2025-01-01",
    title="Dispute Filing Window",
    clause="1.1",
    score=0.9,
    excerpt="A dispute must be filed within 60 days of the transaction date.",
)

CASE = {
    "reason": "duplicate_charge",
    "description": "Same amount charged twice.",
    "amount_minor": 1850,
    "currency": "USD",
    "merchant_name": "Cornerstone Coffee Roasters",
    "merchant_category": "food_and_drink",
    "days_since_transaction": 5,
}


class _FakeMessages:
    def __init__(self, result):
        self._result = result

    def parse(self, **kwargs):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _FakeClient:
    def __init__(self, result, **_kwargs):
        self.messages = _FakeMessages(result)


def _fake_anthropic(monkeypatch, result) -> None:
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: _FakeClient(result, **kwargs))


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


class _Response:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output


@pytest.fixture
def anthropic_env(env, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    from app.config import reset_settings

    reset_settings()
    yield
    reset_settings()


# --------------------------------------------------------------------------- #
# Structured output: the success path
# --------------------------------------------------------------------------- #


def test_valid_structured_output_is_returned(anthropic_env, monkeypatch):
    expected = LLMAssessment(
        risk_score=0.2,
        recommendation="approve",
        rationale="Routine duplicate charge, corroborated by description.",
        risk_flags=[],
        cited_policy_refs=["DP-POL-001@1.0"],
    )
    _fake_anthropic(monkeypatch, _Response(expected))

    result, assessed_by = assess(CASE, [CITATION])
    assert result == expected
    assert assessed_by == "anthropic:claude-opus-5"


# --------------------------------------------------------------------------- #
# Error handling at the SDK boundary
# --------------------------------------------------------------------------- #


def test_sdk_connection_error_becomes_llm_error(anthropic_env, monkeypatch):
    _fake_anthropic(monkeypatch, anthropic.APIConnectionError(request=_request()))

    with pytest.raises(LLMError, match="Anthropic request failed"):
        assess(CASE, [CITATION])


def test_sdk_rate_limit_error_becomes_llm_error(anthropic_env, monkeypatch):
    response = httpx2.Response(429, request=_request())
    _fake_anthropic(monkeypatch, anthropic.RateLimitError("rate limited", response=response, body=None))

    with pytest.raises(LLMError, match="Anthropic request failed"):
        assess(CASE, [CITATION])


def test_sdk_bad_request_error_becomes_llm_error(anthropic_env, monkeypatch):
    response = httpx2.Response(400, request=_request())
    _fake_anthropic(monkeypatch, anthropic.BadRequestError("invalid request", response=response, body=None))

    with pytest.raises(LLMError, match="Anthropic request failed"):
        assess(CASE, [CITATION])


def test_no_parsed_output_becomes_llm_error(anthropic_env, monkeypatch):
    """`messages.parse` can return a response whose `parsed_output` is None
    (e.g. the model's final turn was a refusal) - must fail closed, not
    silently proceed with no assessment."""
    _fake_anthropic(monkeypatch, _Response(None))

    with pytest.raises(LLMError, match="no parsable assessment"):
        assess(CASE, [CITATION])


# --------------------------------------------------------------------------- #
# These failures propagate into the graph's fail-closed grounding path -
# exercised end to end (with the mock provider) in tests/test_graph.py;
# this file only verifies the app.llm <-> SDK boundary in isolation.
# --------------------------------------------------------------------------- #


def test_unconfigured_api_key_still_uses_sdk_default_resolution(anthropic_env, monkeypatch):
    """No ANTHROPIC_API_KEY is set (see tests/conftest.py); app.llm must not
    pass an explicit api_key= to the client in that case, leaving the SDK's
    own credential resolution (env var / OAuth profile / WIF) in control."""
    captured: dict = {}

    class _CapturingClient(_FakeClient):
        def __init__(self, result, **kwargs):
            captured.update(kwargs)
            super().__init__(result, **kwargs)

    expected = LLMAssessment(
        risk_score=0.1,
        recommendation="approve",
        rationale="ok",
        risk_flags=[],
        cited_policy_refs=["DP-POL-001@1.0"],
    )
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: _CapturingClient(_Response(expected), **kwargs))

    assess(CASE, [CITATION])
    assert "api_key" not in captured
