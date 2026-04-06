# tests/unit/auth/test_auth_dependencies.py
"""
Unit tests for app/auth/dependencies.py.

All DB interactions are replaced by AsyncMock; JWT creation uses real
tokens so we test the actual decode path.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.security import SecurityScopes

from app.auth.dependencies import _build_401, _resolve_jwt, get_current_principal
from app.auth.models import Principal
from app.auth.service import create_access_token, create_refresh_token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(db_coro_side_effect=None, key_doc=None, user_doc=None) -> MagicMock:
    """Fabricate a minimal FastAPI Request mock whose app.state.mongodb.db is a mock."""
    db = MagicMock()

    async def _find_one(query, *args, **kwargs):
        # key_doc or user_doc depending on which collection is queried
        if "key_id" in query:
            return key_doc
        return user_doc

    db["api_keys"].find_one = AsyncMock(side_effect=_find_one)
    db["users"].find_one = AsyncMock(return_value=user_doc)
    db["api_keys"].update_one = AsyncMock(return_value=None)

    request = MagicMock()
    request.app.state.mongodb.db = db
    return request


def _valid_access_token(user_id: str = "u1", email: str = "u@test.com", scopes: list[str] | None = None) -> str:
    token, _ = create_access_token(user_id, email, scopes or ["query:read"])
    return token


def _no_scopes() -> SecurityScopes:
    return SecurityScopes([])


def _with_scopes(*scopes: str) -> SecurityScopes:
    return SecurityScopes(list(scopes))


# ---------------------------------------------------------------------------
# _resolve_jwt — happy path
# ---------------------------------------------------------------------------

def test_resolve_jwt_happy_path_returns_principal():
    token = _valid_access_token("user-99", "user@example.com", ["query:read"])
    principal = _resolve_jwt(token)
    assert isinstance(principal, Principal)
    assert principal.user_id == "user-99"
    assert principal.email == "user@example.com"
    assert principal.auth_method == "jwt"


def test_resolve_jwt_scopes_propagated():
    token = _valid_access_token(scopes=["query:read", "admin:seed"])
    principal = _resolve_jwt(token)
    assert "admin:seed" in principal.scopes


# ---------------------------------------------------------------------------
# _resolve_jwt — error paths
# ---------------------------------------------------------------------------

def test_resolve_jwt_invalid_token_raises_401():
    with pytest.raises(HTTPException) as exc_info:
        _resolve_jwt("not.a.valid.token")
    assert exc_info.value.status_code == 401


def test_resolve_jwt_refresh_token_rejected():
    """A refresh token must not be accepted as an access token."""
    refresh = create_refresh_token("user-1")
    with pytest.raises(HTTPException) as exc_info:
        _resolve_jwt(refresh)
    assert exc_info.value.status_code == 401
    assert "Not an access token" in exc_info.value.detail


# ---------------------------------------------------------------------------
# get_current_principal — no token
# ---------------------------------------------------------------------------

async def test_get_current_principal_no_token_raises_401():
    request = _make_request()
    with pytest.raises(HTTPException) as exc_info:
        await get_current_principal(request, _no_scopes(), token=None)
    assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# get_current_principal — JWT path dispatch
# ---------------------------------------------------------------------------

async def test_get_current_principal_dispatches_jwt_for_bearer_token():
    token = _valid_access_token()
    request = _make_request()
    principal = await get_current_principal(request, _no_scopes(), token=token)
    assert principal.auth_method == "jwt"


# ---------------------------------------------------------------------------
# get_current_principal — API key path dispatch
# ---------------------------------------------------------------------------

async def test_get_current_principal_dispatches_api_key_for_ak_prefix():
    from app.auth.service import generate_api_key

    full_key, key_id, key_hash = generate_api_key()
    key_doc = {
        "key_id": key_id,
        "key_hash": key_hash,
        "user_id": "user-10",
        "scopes": ["query:read"],
        "is_active": True,
        "expires_at": None,
    }
    user_doc = {"_id": "user-10", "email": "apikey@example.com"}

    # Wire up the DB so api_keys.find_one returns key_doc and users.find_one returns user_doc
    db = MagicMock()
    api_keys_coll = MagicMock()
    api_keys_coll.find_one = AsyncMock(return_value=key_doc)
    api_keys_coll.update_one = AsyncMock(return_value=None)
    users_coll = MagicMock()
    users_coll.find_one = AsyncMock(return_value=user_doc)
    db.__getitem__ = MagicMock(side_effect=lambda name: api_keys_coll if name == "api_keys" else users_coll)
    request = MagicMock()
    request.app.state.mongodb.db = db

    principal = await get_current_principal(request, _no_scopes(), token=full_key)
    assert principal.auth_method == "api_key"
    assert principal.key_id == key_id


# ---------------------------------------------------------------------------
# Scope enforcement
# ---------------------------------------------------------------------------

async def test_scope_enforcement_missing_scope_raises_403():
    token = _valid_access_token(scopes=["query:read"])
    request = _make_request()
    with pytest.raises(HTTPException) as exc_info:
        await get_current_principal(request, _with_scopes("admin:seed"), token=token)
    assert exc_info.value.status_code == 403


async def test_scope_enforcement_admin_all_bypasses_check():
    token = _valid_access_token(scopes=["admin:all"])
    request = _make_request()
    # Should NOT raise even though we require "admin:seed"
    principal = await get_current_principal(request, _with_scopes("admin:seed"), token=token)
    assert principal is not None


async def test_scope_enforcement_exact_scope_passes():
    token = _valid_access_token(scopes=["admin:seed"])
    request = _make_request()
    principal = await get_current_principal(request, _with_scopes("admin:seed"), token=token)
    assert principal is not None


# ---------------------------------------------------------------------------
# API key error paths
# ---------------------------------------------------------------------------

async def test_resolve_api_key_malformed_key_raises_401():
    request = _make_request()
    with pytest.raises(HTTPException) as exc_info:
        await get_current_principal(request, _no_scopes(), token="ak_bad_no_proper_format_x")
    # parse_api_key("ak_bad_no_proper_format_x") will succeed (3 segments), but
    # the DB lookup will return None → 401
    assert exc_info.value.status_code == 401


async def test_resolve_api_key_not_found_in_db_raises_401():
    request = _make_request(key_doc=None)
    # Generate a valid-format key so parse_api_key won't fail
    from app.auth.service import generate_api_key
    full_key, _, _ = generate_api_key()
    # Override api_keys find_one to always return None
    request.app.state.mongodb.db["api_keys"].find_one = AsyncMock(return_value=None)
    with pytest.raises(HTTPException) as exc_info:
        await get_current_principal(request, _no_scopes(), token=full_key)
    assert exc_info.value.status_code == 401


async def test_resolve_api_key_expired_raises_401():
    from app.auth.service import generate_api_key
    full_key, key_id, key_hash = generate_api_key()
    # expires_at is in the past (naive datetime — service makes it aware)
    expired_at = datetime(2000, 1, 1)
    key_doc = {
        "key_id": key_id,
        "key_hash": key_hash,
        "user_id": "u1",
        "scopes": ["query:read"],
        "is_active": True,
        "expires_at": expired_at,
    }
    request = _make_request(key_doc=key_doc)
    request.app.state.mongodb.db["api_keys"].find_one = AsyncMock(return_value=key_doc)
    with pytest.raises(HTTPException) as exc_info:
        await get_current_principal(request, _no_scopes(), token=full_key)
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()


async def test_resolve_api_key_wrong_secret_raises_401():
    from app.auth.service import generate_api_key
    full_key, key_id, key_hash = generate_api_key()
    # A key_hash from a *different* key — secret won't match
    _, _, different_hash = generate_api_key()
    key_doc = {
        "key_id": key_id,
        "key_hash": different_hash,  # wrong hash!
        "user_id": "u1",
        "scopes": ["query:read"],
        "is_active": True,
        "expires_at": None,
    }
    request = _make_request(key_doc=key_doc)
    request.app.state.mongodb.db["api_keys"].find_one = AsyncMock(return_value=key_doc)
    with pytest.raises(HTTPException) as exc_info:
        await get_current_principal(request, _no_scopes(), token=full_key)
    assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# _build_401 helper
# ---------------------------------------------------------------------------

def test_build_401_without_scopes_uses_plain_bearer():
    exc = _build_401(_no_scopes(), "Not authenticated")
    assert exc.status_code == 401
    assert exc.headers["WWW-Authenticate"] == "Bearer"


def test_build_401_with_scopes_includes_scope_str():
    exc = _build_401(_with_scopes("query:read"), "Not authenticated")
    assert 'scope="query:read"' in exc.headers["WWW-Authenticate"]
