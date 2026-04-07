"""
Global test configuration.

Sets dummy environment variables **before** any app module is imported,
then overrides ``get_settings()`` so unit tests never need real API keys
or external services.  Safe for GitHub Actions / any CI environment.
"""

import os

# ── Set env vars at import time (before Settings / lru_cache can fire) ──
os.environ.setdefault("GEMINI_API_KEY", "test-fake-gemini-key")
os.environ.setdefault("PINECONE_API_KEY", "test-fake-pinecone-key")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017/test_db")
os.environ.setdefault("JWT_SECRET_KEY", "test-only-insecure-jwt-secret-do-not-use-in-prod")

from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

from app.config import Settings, get_settings  # noqa: E402


def _build_test_settings() -> Settings:
    """Return a ``Settings`` instance with dummy values for every required field."""
    return Settings(
        gemini_api_key="test-fake-gemini-key",
        pinecone_api_key="test-fake-pinecone-key",
        mongodb_uri="mongodb://localhost:27017/test_db",
        jwt_secret_key="test-only-insecure-jwt-secret-do-not-use-in-prod",
    )


@pytest.fixture(autouse=True, scope="session")
def override_settings():
    """Replace ``get_settings()`` globally for the entire test session.

    Patches ``get_settings.__wrapped__`` — the raw function that ``lru_cache``
    delegates to on a cache miss.  Because every module holds a reference to
    the *same* ``lru_cache``-decorated object (not to ``app.config.get_settings``
    by name), this single patch intercepts all call sites regardless of how or
    when each module imported the function.

    Call order:
      1. Clear the cache so no stale real-``Settings`` instance survives.
      2. Swap ``__wrapped__`` so the next cache-miss calls our factory.
      3. Clear again to force that first miss immediately.
      4. Yield — every ``get_settings()`` call during the session hits the cache
         and returns ``test_settings``.
      5. Restore and clear on teardown.
    """
    test_settings = _build_test_settings()
    get_settings.cache_clear()
    with patch.object(get_settings, "__wrapped__", return_value=test_settings):
        get_settings.cache_clear()  # force re-population via the patched __wrapped__
        yield test_settings
    get_settings.cache_clear()
