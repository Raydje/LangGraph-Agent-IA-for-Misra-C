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

import pytest  # noqa: E402
from unittest.mock import patch  # noqa: E402

from app.config import Settings, get_settings  # noqa: E402


def _build_test_settings() -> Settings:
    """Return a ``Settings`` instance with dummy values for every required field."""
    return Settings(
        gemini_api_key="test-fake-gemini-key",
        pinecone_api_key="test-fake-pinecone-key",
        mongodb_uri="mongodb://localhost:27017/test_db",
    )


@pytest.fixture(autouse=True, scope="session")
def override_settings():
    """Replace ``get_settings()`` globally for the entire test session.

    Also clears the lru_cache so any module that already called
    ``get_settings()`` at import time will pick up the patched version
    on next call.
    """
    test_settings = _build_test_settings()
    get_settings.cache_clear()
    with patch("app.config.get_settings", return_value=test_settings):
        yield test_settings
    get_settings.cache_clear()
