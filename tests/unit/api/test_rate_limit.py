# tests/unit/api/test_rate_limit.py
"""
Unit tests for app/api/rate_limit.py.

Tests both enforce_user_rate_limit and enforce_user_budget dependencies
with mocked Redis, mocked UsageService, and pre-built Principal objects.
No real network calls are made.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.rate_limit import enforce_user_budget, enforce_user_rate_limit
from app.auth.models import Principal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _principal(scopes: list[str] | None = None) -> Principal:
    return Principal(
        user_id="user-123",
        email="u@test.com",
        scopes=scopes or ["query:read"],
        auth_method="jwt",
    )


def _admin_principal() -> Principal:
    return Principal(
        user_id="admin-1",
        email="admin@test.com",
        scopes=["query:read", "admin:seed", "admin:replay", "admin:all"],
        auth_method="jwt",
    )


def _make_request(redis_client=None, usage_service=None) -> MagicMock:
    request = MagicMock()
    request.app.state.redis = redis_client
    request.app.state.usage_service = usage_service
    return request


def _make_response() -> MagicMock:
    response = MagicMock()
    response.headers = {}
    return response


def _make_redis(zcard_result: int = 0) -> MagicMock:
    """Build a mock async Redis client whose pipeline returns controlled zcard."""
    pipe = AsyncMock()
    pipe.zremrangebyscore = AsyncMock()
    pipe.zcard = AsyncMock()
    pipe.zadd = AsyncMock()
    pipe.expire = AsyncMock()
    # pipeline().__aenter__ returns pipe; execute returns [None, zcard_result, None, None]
    pipe.execute = AsyncMock(return_value=[None, zcard_result, None, None])
    pipe.__aenter__ = AsyncMock(return_value=pipe)
    pipe.__aexit__ = AsyncMock(return_value=False)

    redis = MagicMock()
    redis.pipeline = MagicMock(return_value=pipe)
    return redis


def _make_usage_service(within_budget: bool = True, current_cost: float = 0.0) -> MagicMock:
    svc = MagicMock()
    svc.check_budget = AsyncMock(return_value=(within_budget, current_cost))
    return svc


# ---------------------------------------------------------------------------
# enforce_user_rate_limit — admin bypass
# ---------------------------------------------------------------------------


async def test_rate_limit_admin_bypasses_redis():
    redis = _make_redis()
    request = _make_request(redis_client=redis)
    response = _make_response()

    await enforce_user_rate_limit(request=request, response=response, principal=_admin_principal())

    redis.pipeline.assert_not_called()


# ---------------------------------------------------------------------------
# enforce_user_rate_limit — within limit
# ---------------------------------------------------------------------------


async def test_rate_limit_within_limit_does_not_raise():
    redis = _make_redis(zcard_result=5)  # 5 requests, limit is 20
    request = _make_request(redis_client=redis)
    response = _make_response()

    await enforce_user_rate_limit(request=request, response=response, principal=_principal())
    # No exception → test passes


async def test_rate_limit_sets_response_headers():
    redis = _make_redis(zcard_result=3)
    request = _make_request(redis_client=redis)
    response = _make_response()

    with patch("app.api.rate_limit.get_settings") as mock_settings:
        mock_settings.return_value.user_rate_limit_per_minute = 20
        await enforce_user_rate_limit(request=request, response=response, principal=_principal())

    assert response.headers["X-RateLimit-Limit"] == "20"
    assert response.headers["X-RateLimit-Remaining"] == "16"  # 20 - 3 - 1
    assert "X-RateLimit-Window" in response.headers


# ---------------------------------------------------------------------------
# enforce_user_rate_limit — over limit
# ---------------------------------------------------------------------------


async def test_rate_limit_exceeded_raises_429():
    redis = _make_redis(zcard_result=20)  # At limit of 20
    request = _make_request(redis_client=redis)
    response = _make_response()

    with patch("app.api.rate_limit.get_settings") as mock_settings:
        mock_settings.return_value.user_rate_limit_per_minute = 20
        with pytest.raises(HTTPException) as exc_info:
            await enforce_user_rate_limit(request=request, response=response, principal=_principal())

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["error"] == "rate_limit_exceeded"


async def test_rate_limit_exceeded_sets_retry_after_header():
    redis = _make_redis(zcard_result=25)
    request = _make_request(redis_client=redis)
    response = _make_response()

    with patch("app.api.rate_limit.get_settings") as mock_settings:
        mock_settings.return_value.user_rate_limit_per_minute = 20
        with pytest.raises(HTTPException):
            await enforce_user_rate_limit(request=request, response=response, principal=_principal())

    assert "Retry-After" in response.headers


# ---------------------------------------------------------------------------
# enforce_user_rate_limit — Redis unavailable (graceful degradation)
# ---------------------------------------------------------------------------


async def test_rate_limit_no_redis_degrades_gracefully():
    """When Redis is None, request should be allowed without error."""
    request = _make_request(redis_client=None)
    response = _make_response()

    await enforce_user_rate_limit(request=request, response=response, principal=_principal())
    # No exception raised → degraded to pass-through


async def test_rate_limit_redis_exception_degrades_gracefully():
    """When Redis pipeline raises, request should be allowed without error."""
    redis = MagicMock()
    pipe = AsyncMock()
    pipe.__aenter__ = AsyncMock(return_value=pipe)
    pipe.__aexit__ = AsyncMock(return_value=False)
    pipe.execute = AsyncMock(side_effect=ConnectionError("Redis down"))
    redis.pipeline = MagicMock(return_value=pipe)

    request = _make_request(redis_client=redis)
    response = _make_response()

    await enforce_user_rate_limit(request=request, response=response, principal=_principal())
    # No exception raised


# ---------------------------------------------------------------------------
# enforce_user_budget — admin bypass
# ---------------------------------------------------------------------------


async def test_budget_admin_bypasses_check():
    svc = _make_usage_service()
    request = _make_request(usage_service=svc)
    response = _make_response()

    await enforce_user_budget(request=request, response=response, principal=_admin_principal())

    svc.check_budget.assert_not_called()


# ---------------------------------------------------------------------------
# enforce_user_budget — within budget
# ---------------------------------------------------------------------------


async def test_budget_within_limit_does_not_raise():
    svc = _make_usage_service(within_budget=True, current_cost=1.5)
    request = _make_request(usage_service=svc)
    response = _make_response()

    with patch("app.api.rate_limit.get_settings") as mock_settings:
        mock_settings.return_value.user_max_budget = 5.0
        await enforce_user_budget(request=request, response=response, principal=_principal())


async def test_budget_sets_response_headers():
    svc = _make_usage_service(within_budget=True, current_cost=2.0)
    request = _make_request(usage_service=svc)
    response = _make_response()

    with patch("app.api.rate_limit.get_settings") as mock_settings:
        mock_settings.return_value.user_max_budget = 5.0
        await enforce_user_budget(request=request, response=response, principal=_principal())

    assert response.headers["X-Budget-Limit"] == "5.000000"
    assert response.headers["X-Budget-Used"] == "2.000000"
    assert response.headers["X-Budget-Remaining"] == "3.000000"


# ---------------------------------------------------------------------------
# enforce_user_budget — over budget
# ---------------------------------------------------------------------------


async def test_budget_exceeded_raises_429():
    svc = _make_usage_service(within_budget=False, current_cost=5.5)
    request = _make_request(usage_service=svc)
    response = _make_response()

    with patch("app.api.rate_limit.get_settings") as mock_settings:
        mock_settings.return_value.user_max_budget = 5.0
        with pytest.raises(HTTPException) as exc_info:
            await enforce_user_budget(request=request, response=response, principal=_principal())

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["error"] == "budget_exceeded"
    assert exc_info.value.detail["current_cost"] == 5.5
    assert exc_info.value.detail["max_budget"] == 5.0


async def test_budget_exceeded_remaining_header_is_zero():
    svc = _make_usage_service(within_budget=False, current_cost=7.0)
    request = _make_request(usage_service=svc)
    response = _make_response()

    with patch("app.api.rate_limit.get_settings") as mock_settings:
        mock_settings.return_value.user_max_budget = 5.0
        with pytest.raises(HTTPException):
            await enforce_user_budget(request=request, response=response, principal=_principal())

    # Remaining should be clamped to 0, not negative
    assert response.headers["X-Budget-Remaining"] == "0.000000"


# ---------------------------------------------------------------------------
# enforce_user_budget — graceful degradation
# ---------------------------------------------------------------------------


async def test_budget_no_usage_service_degrades_gracefully():
    """When usage_service is not on app.state, request should be allowed."""
    request = _make_request(usage_service=None)
    response = _make_response()

    await enforce_user_budget(request=request, response=response, principal=_principal())
    # No exception raised


async def test_budget_check_exception_degrades_gracefully():
    """When check_budget raises, request should be allowed."""
    svc = MagicMock()
    svc.check_budget = AsyncMock(side_effect=Exception("DB timeout"))
    request = _make_request(usage_service=svc)
    response = _make_response()

    with patch("app.api.rate_limit.get_settings") as mock_settings:
        mock_settings.return_value.user_max_budget = 5.0
        await enforce_user_budget(request=request, response=response, principal=_principal())
    # No exception raised
