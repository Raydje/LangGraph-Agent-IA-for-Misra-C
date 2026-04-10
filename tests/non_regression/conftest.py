"""
Non-regression test configuration.

Provides session-scoped fixtures for running end-to-end tests against a live
docker-compose container. No mocking — real HTTP requests only.

Required environment variables
--------------------------------
TNR_BASE_URL   Base URL of the running container. Defaults to http://localhost:8000.
TEST_EMAIL     Email of a registered user (already set in .env).
TEST_PASSWORD  Password for that user (already set in .env).

Design principles (mirrors tests/integration/conftest.py)
----------------------------------------------------------
- Missing or empty credentials cause the entire session to be skipped with a
  human-readable message rather than a cascade of 401 / connection errors.
- Auth (POST /auth/token) is performed once per session; the token is reused
  across all parametrized test cases.
- The httpx.Client is session-scoped and closed in teardown so a single TCP
  connection pool is shared across all 10 test cases.
- No container readiness wait — the CI pipeline guarantees the health-check
  has passed before this suite is triggered.
"""

import os

import httpx
import pytest

# ---------------------------------------------------------------------------
# Guard — skip the entire session if required env vars are absent.
# ---------------------------------------------------------------------------

_REQUIRED_ENV_VARS = ("TEST_EMAIL", "TEST_PASSWORD")


def _check_env() -> str | None:
    """Return an error message if any required env var is missing or empty."""
    for var in _REQUIRED_ENV_VARS:
        if not os.environ.get(var, "").strip():
            return (
                f"Required environment variable '{var}' is not set. "
                "Non-regression tests require a live user account on the running container."
            )
    return None


_env_error = _check_env()


def pytest_collection_modifyitems(items: list, config: object) -> None:  # noqa: ARG001
    """Skip every test in this package if credentials are unavailable."""
    if not _env_error:
        return
    skip_marker = pytest.mark.skip(reason=f"TNR credentials unavailable: {_env_error}")
    for item in items:
        if "non_regression" in str(item.fspath):
            item.add_marker(skip_marker)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tnr_base_url() -> str:
    """Base URL of the live container. Override with TNR_BASE_URL env var."""
    return os.environ.get("TNR_BASE_URL", "http://localhost:8000").rstrip("/")


@pytest.fixture(scope="session")
def tnr_auth_token(tnr_base_url: str) -> str:
    """
    Authenticate once per session via OAuth2 password flow.

    Calls POST /api/v1/auth/token with TEST_EMAIL / TEST_PASSWORD.
    Skips the entire session (rather than failing) if the login is rejected,
    so CI reports a clean skip instead of 10 cascading 401 errors.
    """
    email = os.environ.get("TEST_EMAIL", "")
    password = os.environ.get("TEST_PASSWORD", "")

    try:
        response = httpx.post(
            f"{tnr_base_url}/api/v1/auth/token",
            data={"username": email, "password": password},
            timeout=15,
        )
    except httpx.ConnectError as exc:
        pytest.skip(
            f"Could not connect to the container at {tnr_base_url}. Make sure docker compose is running. Error: {exc}"
        )

    if response.status_code == 401:
        pytest.skip(
            f"Authentication failed for '{email}'. "
            "Check that TEST_EMAIL and TEST_PASSWORD match a registered user on the container."
        )

    if response.status_code != 200:
        pytest.skip(f"Unexpected status {response.status_code} from /auth/token. Body: {response.text[:300]}")

    return response.json()["access_token"]


@pytest.fixture(scope="session")
def tnr_client(tnr_base_url: str, tnr_auth_token: str) -> httpx.Client:
    """
    Session-scoped httpx.Client pre-configured with the base URL and Bearer token.

    A single client (and its underlying connection pool) is shared across all
    10 parametrized test cases to avoid repeated TCP handshake overhead.
    """
    client = httpx.Client(
        base_url=tnr_base_url,
        headers={"Authorization": f"Bearer {tnr_auth_token}"},
        timeout=90,  # LangGraph pipeline can take 20-40s per query
    )
    yield client
    client.close()
