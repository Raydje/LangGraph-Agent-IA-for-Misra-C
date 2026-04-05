"""
Integration test configuration.

Provides session-scoped, real service fixtures for the integration test suite.
These fixtures replicate the lifespan startup from main.py — initialising the
same singletons (MongoDBService, PineconeService, EmbeddingService) once per
test session and tearing them down cleanly afterwards.

Design principles
-----------------
- Real API keys are injected by CI via GitHub Secrets (env vars).  If any
  required key is absent the entire session is skipped with a human-readable
  message rather than crashing with a cryptic import error.
- get_settings() lru_cache is NOT patched here — integration tests must run
  against the real Settings populated from the environment.
- The root tests/conftest.py (unit test conftest) uses os.environ.setdefault,
  so it will NOT overwrite env vars that are already set when integration tests
  run.  Running pytest tests/integration/ directly avoids the autouse fixture
  in that file because pytest only applies fixtures from conftest files in the
  test path hierarchy, not from sibling directories.
  When both suites must run together, the real secrets must be present first so
  setdefault does not replace them with dummy values.
"""

import os

import pytest

# ---------------------------------------------------------------------------
# Guard — skip the entire session early if required secrets are absent.
# This produces a single, readable skip message in the CI log rather than a
# cascade of connection errors across every test.
# ---------------------------------------------------------------------------
_REQUIRED_ENV_VARS = ("GEMINI_API_KEY", "PINECONE_API_KEY", "MONGODB_URI", "JWT_SECRET_KEY")
_DUMMY_VALUES = frozenset(
    {
        "dummy_key",
        "test-fake-gemini-key",
        "test-fake-pinecone-key",
        "test-only-insecure-jwt-secret-do-not-use-in-prod",
    }
)


def _check_secrets() -> str | None:
    """Return an error message if any required secret is missing or is a known dummy value."""
    for var in _REQUIRED_ENV_VARS:
        value = os.environ.get(var, "")
        if not value:
            return f"Required environment variable '{var}' is not set."
        if value in _DUMMY_VALUES:
            return (
                f"Environment variable '{var}' is set to a known dummy/test value "
                f"('{value}'). Integration tests require real credentials."
            )
    return None


_secrets_error = _check_secrets()


# Collect-time skip: if secrets are invalid, mark every test in this package
# with a module-level skip so CI reports them as skipped (not failed).
def pytest_collection_modifyitems(items: list, config: object) -> None:  # noqa: ARG001
    if not _secrets_error:
        return
    skip_marker = pytest.mark.skip(reason=f"Integration secrets unavailable: {_secrets_error}")
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip_marker)


# ---------------------------------------------------------------------------
# Real Settings — loaded once from the environment (no lru_cache override).
# ---------------------------------------------------------------------------

# Import after the guard so that Settings() is never constructed with dummies
# on a cache miss if the cache was already primed by the unit conftest.
from app.config import get_settings  # noqa: E402
from app.services.embedding_service import EmbeddingService  # noqa: E402
from app.services.mongodb_service import MongoDBService  # noqa: E402
from app.services.pinecone_service import PineconeService  # noqa: E402


# ---------------------------------------------------------------------------
# Session-scoped service fixtures
# Mirrors the startup block in main.py's lifespan context manager.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_settings():
    """Return the real Settings instance populated from environment variables."""
    # Clear any stale cache entry that might have been primed with dummy values
    # by the unit conftest (if both suites are run in the same process).
    get_settings.cache_clear()
    settings = get_settings()
    return settings


@pytest.fixture(scope="session")
def mongodb_service(real_settings):  # noqa: ARG001
    """
    Provide a live MongoDBService instance, shared across the entire session.

    We intentionally do NOT call await mongodb.connect() — MongoDBService uses
    Motor which establishes its connection lazily on the first query.  This
    matches the behaviour in main.py where the service is instantiated in the
    synchronous part of lifespan before the first await.
    """
    service = MongoDBService()
    yield service
    service.close()


@pytest.fixture(scope="session")
def pinecone_service(real_settings):  # noqa: ARG001
    """Provide a live PineconeService instance, shared across the entire session."""
    service = PineconeService()
    yield service
    # PineconeService has no explicit close method — the HTTP client is managed
    # internally by the Pinecone SDK and cleaned up by the GC.


@pytest.fixture(scope="session")
def embedding_service(real_settings):  # noqa: ARG001
    """Provide a live EmbeddingService instance, shared across the entire session."""
    service = EmbeddingService()
    yield service


@pytest.fixture(scope="session")
def rag_config(mongodb_service, pinecone_service, embedding_service):
    """
    Build the LangGraph RunnableConfig dict expected by rag_node.

    Mirrors the config construction in app/api/v1/routes.py so that the node
    under test receives its dependencies in exactly the same shape as it does
    in production.
    """
    return {
        "configurable": {
            "mongo_db": mongodb_service,
            "pinecone_service": pinecone_service,
            "embedding_service": embedding_service,
        }
    }
