"""Runtime configuration. All values come from the environment or .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Storage
    database_url: str = "sqlite+pysqlite:///./paymentops.db"
    checkpoint_db: str = "./checkpoints.sqlite"
    qdrant_url: str = ""  # empty => embedded in-memory Qdrant
    qdrant_collection: str = "dispute_policies"

    # Retrieval
    embedding_dim: int = 256
    top_k: int = 4

    # LLM
    llm_provider: str = "mock"  # "mock" | "anthropic"
    anthropic_model: str = "claude-opus-5"
    anthropic_api_key: str | None = None
    anthropic_base_url: str | None = None
    llm_max_tokens: int = 8192

    # Deterministic eligibility rules (never decided by the LLM).
    dispute_filing_window_days: int = 60
    high_value_minor: int = 500_000  # 5,000.00 - triggers enhanced verification flag

    # Reviewer authentication. Format: "name1:key1,name2:key2".
    # Empty by default => every decision request is rejected (fail closed, not open).
    reviewer_api_keys: str = ""

    # Diagnostic only - NOT a safety gate. Real exclusive ownership of a case
    # is the PostgreSQL advisory lock in app.graph.case_ownership; a `RUNNING`
    # or `DECIDING` row whose `claimed_at` is older than this is *displayed*
    # as possibly stuck, but recovery never acts on that timestamp alone -
    # only on successfully acquiring the lock, which is the one thing that
    # actually proves the previous owner is gone (see app.graph's module
    # docstring, "Exclusive per-case ownership").
    stuck_claim_timeout_seconds: int = 20

    def reviewer_key_map(self) -> dict[str, str]:
        """key -> reviewer_id."""
        result: dict[str, str] = {}
        for pair in self.reviewer_api_keys.split(","):
            pair = pair.strip()
            if not pair:
                continue
            name, _, key = pair.partition(":")
            if name and key:
                result[key] = name
        return result


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings() -> None:
    """Drop the cached Settings so tests can re-read the environment."""
    get_settings.cache_clear()
