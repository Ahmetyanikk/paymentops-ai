"""Qdrant policy index + retrieval.

KNOWN LIMITATION (deliberate, MVP scope): embeddings use a deterministic
lexical hashing vectorizer (stdlib + numpy), not a trained embedding model.
That keeps indexing offline, free and perfectly reproducible, which is what
the test suite and the demo need, and the small dispute-policy corpus is
small enough that lexical overlap is a reliable retrieval signal.

Policy chunks carry `effective_date` and an optional `expires_date`; `search`
excludes anything not in effect as of the date it is given (see `as_of`
below) - a future-dated or superseded policy is never handed to the LLM,
regardless of how well it scores lexically.

ponytail: this is lexical, not semantic - it will miss paraphrases a real
embedding model would catch (e.g. "chargeback" vs "dispute"). Swap `embed()`
for a hosted embedding model (and bump EMBEDDING_DIM to match it) if the
policy corpus grows past a few dozen clauses, or if recall on reworded
queries becomes a real problem. Nothing else in this module, or its callers,
would need to change - `search()` already returns the same PolicyCitation
shape regardless of how `embed()` is implemented.
"""

from __future__ import annotations

import re
import zlib
from datetime import date
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import get_settings
from app.schemas import PolicyCitation

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_client: QdrantClient | None = None


def _tokens(text: str) -> list[str]:
    words = _TOKEN_RE.findall(text.lower())
    # Bigrams give the hashing vectorizer a little word-order signal.
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def embed(text: str) -> list[float]:
    """Deterministic L2-normalised hashing vector. Stable across processes."""
    dim = get_settings().embedding_dim
    vec = np.zeros(dim, dtype=np.float32)
    for token in _tokens(text):
        # zlib.crc32 is stable; Python's hash() is salted per process.
        vec[zlib.crc32(token.encode()) % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm:
        vec /= norm
    return vec.tolist()


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        url = get_settings().qdrant_url
        _client = QdrantClient(url=url) if url else QdrantClient(":memory:")
    return _client


def reset_client() -> None:
    global _client
    if _client is not None:
        _client.close()
    _client = None


def _point_id(doc: dict[str, Any]) -> str:
    return str(uuid5(NAMESPACE_URL, f"{doc['policy_id']}:{doc['version']}:{doc['clause']}"))


def ensure_index(docs: list[dict[str, Any]]) -> int:
    """Create the collection if missing and upsert every policy chunk.

    Upsert is keyed by a stable UUID, so running this on every startup is safe.
    """
    settings = get_settings()
    client = get_client()
    collection = settings.qdrant_collection

    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )

    points = [
        PointStruct(
            id=_point_id(doc),
            vector=embed(f"{doc['title']} {doc['clause']} {doc['text']}"),
            payload={
                **doc,
                "effective_date": doc["effective_date"].isoformat(),
                "expires_date": doc["expires_date"].isoformat() if doc.get("expires_date") else None,
            },
        )
        for doc in docs
    ]
    if points:
        client.upsert(collection_name=collection, points=points, wait=True)
    return count_policies()


def count_policies() -> int:
    client = get_client()
    collection = get_settings().qdrant_collection
    if not client.collection_exists(collection):
        return 0
    return client.count(collection_name=collection, exact=True).count


def search(query: str, k: int | None = None, as_of: date | None = None) -> list[PolicyCitation]:
    """Retrieve the top-k policy chunks for a query. Empty list if none indexed
    or if nothing in-effect-as-of survives the date filter.

    `as_of` (when given) excludes a policy chunk that is not yet effective or
    has already expired on that date - the corpus is small, so this filters
    in Python after an over-fetch rather than needing a Qdrant payload filter.
    Callers must fail closed on an empty result - see
    app.graph.AssessmentFailedError - rather than letting an ungrounded
    assessment through.
    """
    settings = get_settings()
    client = get_client()
    if not client.collection_exists(settings.qdrant_collection):
        return []
    limit = k or settings.top_k
    # Over-fetch when date-filtering so a chunk excluded by date doesn't just
    # shrink the result below `limit` - the whole corpus here is tiny, so
    # fetching generously and filtering in Python costs nothing.
    fetch_limit = max(limit * 5, 50) if as_of is not None else limit
    hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=embed(query),
        limit=fetch_limit,
        with_payload=True,
    ).points

    citations = [
        PolicyCitation(
            policy_id=hit.payload["policy_id"],
            version=hit.payload["version"],
            effective_date=hit.payload["effective_date"],
            expires_date=hit.payload.get("expires_date"),
            title=hit.payload["title"],
            clause=hit.payload["clause"],
            score=round(float(hit.score), 4),
            excerpt=hit.payload["text"],
        )
        for hit in hits
    ]

    if as_of is not None:
        citations = [
            c
            for c in citations
            if c.effective_date <= as_of and (c.expires_date is None or as_of < c.expires_date)
        ]

    return citations[:limit]
