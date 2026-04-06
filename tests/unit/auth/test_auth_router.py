# tests/unit/auth/test_auth_router.py
"""
Unit tests for app/auth/router.py.

Strategy: call route handler functions directly with a mocked Request.
app.state.mongodb.db is a MagicMock with AsyncMock DB methods.
get_current_principal is patched to return a pre-built Principal so the
auth dependency chain is not exercised here (tested separately).
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.security import SecurityScopes

from app.auth.models import APIKeyCreate, Principal, RefreshRequest, UserCreate
from app.auth.router import (
    create_api_key,
    list_api_keys,
    login,
    refresh,
    register,
    revoke_api_key,
)
from app.auth.service import create_refresh_token, hash_password


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(
    *,
    existing_user: dict | None = None,
    login_user: dict | None = None,
    refresh_user: dict | None = None,
    api_keys: list[dict] | None = None,
    update_result_matched: int = 1,
) -> MagicMock:
    db = MagicMock()

    # users collection
    users = MagicMock()
    users.find_one = AsyncMock(return_value=existing_user or login_user or refresh_user)
    users.insert_one = AsyncMock(return_value=MagicMock(inserted_id="fake-id"))
    users.update_one = AsyncMock(return_value=None)
    db.__getitem__ = MagicMock(side_effect=lambda name: {
        "users": users,
        "api_keys": _make_api_keys_coll(api_keys, update_result_matched),
    }[name])
    return db


def _make_api_keys_coll(api_keys: list[dict] | None, matched: int) -> MagicMock:
    coll = MagicMock()
    coll.insert_one = AsyncMock(return_value=None)

    # list_api_keys uses async for — need an async iterator
    async def _async_iter(*args, **kwargs):
        for doc in (api_keys or []):
            yield doc

    coll.find = MagicMock(return_value=_async_iter())

    result = MagicMock()
    result.matched_count = matched
    coll.update_one = AsyncMock(return_value=result)
    return coll


def _make_request(db: MagicMock) -> MagicMock:
    request = MagicMock()
    request.app.state.mongodb.db = db
    return request


def _principal(scopes: list[str] | None = None) -> Principal:
    return Principal(
        user_id="user-test",
        email="test@test.com",
        scopes=scopes or ["query:read"],
        auth_method="jwt",
    )


def _make_form(username: str = "user@test.com", password: str = "password123") -> MagicMock:
    form = MagicMock()
    form.username = username
    form.password = password
    return form


# ---------------------------------------------------------------------------
# POST /auth/register
# ---------------------------------------------------------------------------

async def test_register_success_returns_user_info():
    db = _make_db(existing_user=None)
    db["users"].find_one = AsyncMock(return_value=None)
    request = _make_request(db)

    result = await register(UserCreate(email="new@user.com", password="securepass"), request)

    assert result["email"] == "new@user.com"
    assert "user_id" in result
    assert result["scopes"] == ["query:read"]


async def test_register_duplicate_email_raises_400():
    db = _make_db(existing_user={"email": "existing@user.com"})
    db["users"].find_one = AsyncMock(return_value={"email": "existing@user.com"})
    request = _make_request(db)

    with pytest.raises(HTTPException) as exc_info:
        await register(UserCreate(email="existing@user.com", password="securepass"), request)
    assert exc_info.value.status_code == 400


async def test_register_with_valid_admin_token_grants_admin_scopes():
    db = _make_db(existing_user=None)
    db["users"].find_one = AsyncMock(return_value=None)
    request = _make_request(db)

    with patch("app.auth.router.get_settings") as mock_settings:
        mock_settings.return_value.admin_registration_token = "super-secret-admin"
        result = await register(
            UserCreate(email="admin@test.com", password="securepass", admin_token="super-secret-admin"),
            request,
        )

    assert "admin:all" in result["scopes"]


async def test_register_with_wrong_admin_token_raises_403():
    db = _make_db(existing_user=None)
    db["users"].find_one = AsyncMock(return_value=None)
    request = _make_request(db)

    with patch("app.auth.router.get_settings") as mock_settings:
        mock_settings.return_value.admin_registration_token = "real-secret"
        with pytest.raises(HTTPException) as exc_info:
            await register(
                UserCreate(email="evil@test.com", password="securepass", admin_token="wrong-token"),
                request,
            )
    assert exc_info.value.status_code == 403


async def test_register_admin_token_when_feature_disabled_raises_403():
    db = _make_db(existing_user=None)
    db["users"].find_one = AsyncMock(return_value=None)
    request = _make_request(db)

    with patch("app.auth.router.get_settings") as mock_settings:
        mock_settings.return_value.admin_registration_token = ""
        with pytest.raises(HTTPException) as exc_info:
            await register(
                UserCreate(email="x@test.com", password="securepass", admin_token="anything"),
                request,
            )
    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# POST /auth/token
# ---------------------------------------------------------------------------

async def test_login_success_returns_token_response():
    user = {
        "_id": "user-1",
        "email": "u@test.com",
        "hashed_password": hash_password("mypassword"),
        "scopes": ["query:read"],
        "is_active": True,
    }
    db = _make_db(login_user=user)
    db["users"].find_one = AsyncMock(return_value=user)
    db["users"].update_one = AsyncMock(return_value=None)
    request = _make_request(db)

    result = await login(request, _make_form(username="u@test.com", password="mypassword"))

    assert result.access_token
    assert result.refresh_token
    assert result.token_type == "bearer"


async def test_login_wrong_password_raises_401():
    user = {
        "_id": "user-1",
        "email": "u@test.com",
        "hashed_password": hash_password("correct"),
        "scopes": ["query:read"],
        "is_active": True,
    }
    db = _make_db(login_user=user)
    db["users"].find_one = AsyncMock(return_value=user)
    request = _make_request(db)

    with pytest.raises(HTTPException) as exc_info:
        await login(request, _make_form(password="wrong"))
    assert exc_info.value.status_code == 401


async def test_login_user_not_found_raises_401():
    db = _make_db(login_user=None)
    db["users"].find_one = AsyncMock(return_value=None)
    request = _make_request(db)

    with pytest.raises(HTTPException) as exc_info:
        await login(request, _make_form())
    assert exc_info.value.status_code == 401


async def test_login_inactive_account_raises_403():
    user = {
        "_id": "user-1",
        "email": "u@test.com",
        "hashed_password": hash_password("pass"),
        "scopes": [],
        "is_active": False,
    }
    db = _make_db(login_user=user)
    db["users"].find_one = AsyncMock(return_value=user)
    request = _make_request(db)

    with pytest.raises(HTTPException) as exc_info:
        await login(request, _make_form(password="pass"))
    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# POST /auth/refresh
# ---------------------------------------------------------------------------

async def test_refresh_success_rotates_tokens():
    user_id = "user-refresh"
    old_refresh = create_refresh_token(user_id)
    user = {
        "_id": user_id,
        "email": "r@test.com",
        "scopes": ["query:read"],
        "refresh_tokens": [{"token": old_refresh}],
    }
    db = _make_db(refresh_user=user)
    db["users"].find_one = AsyncMock(return_value=user)
    db["users"].update_one = AsyncMock(return_value=None)
    request = _make_request(db)

    result = await refresh(RefreshRequest(refresh_token=old_refresh), request)

    assert result.access_token
    assert result.refresh_token != old_refresh  # rotation happened


async def test_refresh_invalid_token_raises_401():
    db = _make_db()
    request = _make_request(db)

    with pytest.raises(HTTPException) as exc_info:
        await refresh(RefreshRequest(refresh_token="not.a.jwt"), request)
    assert exc_info.value.status_code == 401


async def test_refresh_token_not_in_db_raises_401():
    user_id = "user-refresh"
    token = create_refresh_token(user_id)
    db = _make_db()
    db["users"].find_one = AsyncMock(return_value=None)  # token not found in DB
    request = _make_request(db)

    with pytest.raises(HTTPException) as exc_info:
        await refresh(RefreshRequest(refresh_token=token), request)
    assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# POST /auth/api-keys
# ---------------------------------------------------------------------------

async def test_create_api_key_returns_full_key_once():
    db = _make_db()
    db["api_keys"].insert_one = AsyncMock(return_value=None)
    request = _make_request(db)
    principal = _principal(["query:read"])

    result = await create_api_key(
        APIKeyCreate(name="my-key", scopes=["query:read"]),
        request,
        principal,
    )

    assert result.full_key.startswith("ak_")
    assert result.name == "my-key"


async def test_create_api_key_strips_unpermitted_scopes():
    """Caller cannot grant scopes they don't hold."""
    db = _make_db()
    db["api_keys"].insert_one = AsyncMock(return_value=None)
    request = _make_request(db)
    principal = _principal(["query:read"])  # does NOT hold admin:seed

    result = await create_api_key(
        APIKeyCreate(name="key", scopes=["query:read", "admin:seed"]),
        request,
        principal,
    )

    assert "admin:seed" not in result.scopes
    assert "query:read" in result.scopes


async def test_create_api_key_no_valid_scopes_raises_400():
    db = _make_db()
    request = _make_request(db)
    principal = _principal(["query:read"])

    with pytest.raises(HTTPException) as exc_info:
        await create_api_key(
            APIKeyCreate(name="key", scopes=["admin:seed"]),  # caller doesn't have this
            request,
            principal,
        )
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# GET /auth/api-keys
# ---------------------------------------------------------------------------

async def test_list_api_keys_returns_active_keys():
    keys = [
        {
            "key_id": "abc12345",
            "name": "key-1",
            "scopes": ["query:read"],
            "expires_at": None,
            "last_used_at": None,
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
        }
    ]
    db = MagicMock()

    async def _async_iter(*args, **kwargs):
        for doc in keys:
            yield doc

    api_keys_coll = MagicMock()
    api_keys_coll.find = MagicMock(return_value=_async_iter())
    db.__getitem__ = MagicMock(return_value=api_keys_coll)

    request = _make_request(db)
    principal = _principal()

    result = await list_api_keys(request, principal)

    assert len(result) == 1
    assert result[0].key_id == "abc12345"


# ---------------------------------------------------------------------------
# DELETE /auth/api-keys/{key_id}
# ---------------------------------------------------------------------------

async def test_revoke_api_key_success_returns_none():
    db = _make_db(update_result_matched=1)
    request = _make_request(db)
    principal = _principal()

    # Should not raise
    result = await revoke_api_key("abc12345", request, principal)
    assert result is None


async def test_revoke_api_key_not_owned_raises_404():
    db = _make_db(update_result_matched=0)
    request = _make_request(db)
    principal = _principal()

    with pytest.raises(HTTPException) as exc_info:
        await revoke_api_key("nonexistent-key", request, principal)
    assert exc_info.value.status_code == 404
