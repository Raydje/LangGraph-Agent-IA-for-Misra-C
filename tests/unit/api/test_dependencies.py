# tests/unit/api/test_dependencies.py
"""
Unit tests for app/api/dependencies.py.

All dependency getter functions are trivial attribute lookups on request.app.state,
so we use a minimal MagicMock request rather than a full FastAPI TestClient.

The module-level `limiter` fallback branch (lines 49-50) is covered by reloading
the module with get_settings patched to raise, then restoring it via a fixture.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(**state_attrs) -> MagicMock:
    """Build a minimal mock Request whose app.state carries the given attrs."""
    req = MagicMock()
    for key, val in state_attrs.items():
        setattr(req.app.state, key, val)
    return req


# ---------------------------------------------------------------------------
# Getter functions — each returns the corresponding app.state attribute
# ---------------------------------------------------------------------------


def test_get_mongodb_service_returns_state_mongodb():
    from app.api.dependencies import get_mongodb_service

    svc = MagicMock()
    req = _make_request(mongodb=svc)
    assert get_mongodb_service(req) is svc


def test_get_mongodb_checkpoint_service_returns_state_mongodb_checkpoint():
    from app.api.dependencies import get_mongodb_checkpoint_service

    svc = MagicMock()
    req = _make_request(mongodb_checkpoint=svc)
    assert get_mongodb_checkpoint_service(req) is svc


def test_get_pinecone_service_returns_state_pinecone():
    from app.api.dependencies import get_pinecone_service

    svc = MagicMock()
    req = _make_request(pinecone=svc)
    assert get_pinecone_service(req) is svc


def test_get_embedding_service_returns_state_embedding():
    from app.api.dependencies import get_embedding_service

    svc = MagicMock()
    req = _make_request(embedding=svc)
    assert get_embedding_service(req) is svc


def test_get_compiled_graph_returns_state_graph():
    from app.api.dependencies import get_compiled_graph

    graph = MagicMock()
    req = _make_request(graph=graph)
    assert get_compiled_graph(req) is graph


def test_get_mongodb_database_returns_state_mongodb_db():
    from app.api.dependencies import get_mongodb_database

    db = MagicMock()
    svc = MagicMock()
    svc.db = db
    req = _make_request(mongodb=svc)
    assert get_mongodb_database(req) is db


def test_get_pinecone_index_returns_state_pinecone_index():
    from app.api.dependencies import get_pinecone_index

    index = MagicMock()
    svc = MagicMock()
    svc.index = index
    req = _make_request(pinecone=svc)
    assert get_pinecone_index(req) is index


# ---------------------------------------------------------------------------
# get_real_ip
# ---------------------------------------------------------------------------


def test_get_real_ip_returns_first_forwarded_for_ip():
    from app.api.dependencies import get_real_ip

    req = MagicMock()
    req.headers.get.return_value = "203.0.113.5, 10.0.0.1, 172.16.0.1"
    assert get_real_ip(req) == "203.0.113.5"
    req.headers.get.assert_called_once_with("X-Forwarded-For")


def test_get_real_ip_falls_back_to_get_remote_address_when_no_header():
    from app.api.dependencies import get_real_ip

    req = MagicMock()
    req.headers.get.return_value = None

    with patch("app.api.dependencies.get_remote_address", return_value="192.168.1.1") as mock_gra:
        result = get_real_ip(req)

    assert result == "192.168.1.1"
    mock_gra.assert_called_once_with(req)


# ---------------------------------------------------------------------------
# _redis_reachable
# ---------------------------------------------------------------------------


def test_redis_reachable_returns_true_when_ping_succeeds():
    from app.api.dependencies import _redis_reachable

    mock_client = MagicMock()
    mock_redis_module = MagicMock()
    mock_redis_module.from_url.return_value = mock_client

    with patch.dict("sys.modules", {"redis": mock_redis_module}):
        result = _redis_reachable("redis://localhost:6379")

    assert result is True
    mock_client.ping.assert_called_once()
    mock_client.close.assert_called_once()


def test_redis_reachable_returns_false_when_ping_raises():
    from app.api.dependencies import _redis_reachable

    mock_client = MagicMock()
    mock_client.ping.side_effect = Exception("connection refused")
    mock_redis_module = MagicMock()
    mock_redis_module.from_url.return_value = mock_client

    with patch.dict("sys.modules", {"redis": mock_redis_module}):
        result = _redis_reachable("redis://localhost:6379")

    assert result is False


# ---------------------------------------------------------------------------
# Module-level limiter fallback branch (lines 49-50)
#
# The `try` block at module level calls get_settings() and _redis_reachable().
# We reload the module with get_settings patched to raise so the `except`
# branch executes and creates the fallback Limiter.
# The fixture restores the module to its normal state after the test.
# ---------------------------------------------------------------------------


@pytest.fixture()
def restored_dependencies_module():
    """Reload app.api.dependencies after the test to restore normal state."""
    yield
    import app.api.dependencies as dep_mod

    importlib.reload(dep_mod)


def test_limiter_fallback_created_when_get_settings_raises(restored_dependencies_module):
    import app.api.dependencies as dep_mod

    with patch("app.api.dependencies.get_settings", side_effect=Exception("no settings")):
        importlib.reload(dep_mod)

    # The fallback limiter must exist and be a Limiter instance
    from slowapi import Limiter

    assert isinstance(dep_mod.limiter, Limiter)
