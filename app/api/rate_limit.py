"""
Per-user rate limiting and budget enforcement — FastAPI dependencies.

Two reusable dependencies:

  enforce_user_rate_limit
    Uses a Redis sorted-set sliding window keyed by user_id.
    Returns 429 + Retry-After header when the per-minute call quota is exceeded.
    Degrades gracefully (allows request) when Redis is unavailable.

  enforce_user_budget
    Reads the denormalized total_cost field from the users collection.
    Returns 429 when the user has already reached their lifetime budget cap.

Admin users (principal.scopes contains "admin:all") bypass both checks entirely.

Both dependencies are designed to be injected via FastAPI's Depends() on any
route that requires authentication.  They consume the already-resolved Principal
object — no duplicate DB/auth overhead.
"""

from __future__ import annotations

import time
import uuid

from fastapi import HTTPException, Request, Response, Security

from app.auth.dependencies import get_current_principal
from app.auth.models import Principal
from app.config import get_settings
from app.services.usage_service import UsageService
from app.utils import logger

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_admin(principal: Principal) -> bool:
    return "admin:all" in principal.scopes


# ---------------------------------------------------------------------------
# enforce_user_rate_limit
# ---------------------------------------------------------------------------


async def enforce_user_rate_limit(
    request: Request,
    response: Response,
    principal: Principal = Security(get_current_principal, scopes=["query:read"]),
) -> None:
    """Sliding-window rate limiter keyed by user_id (Redis sorted set).

    Window: 1 minute.  Limit: settings.user_rate_limit_per_minute.
    Admin users (admin:all scope) are exempt.

    Algorithm (atomic via Redis pipeline):
      1. Remove all entries older than (now - 60s) from the sorted set.
      2. Count remaining entries.
      3. If count >= limit → 429.
      4. Otherwise, add current timestamp and set TTL on the key.

    Raises:
        HTTPException 429 if the user has exceeded their per-minute quota.
    """
    if _is_admin(principal):
        return

    settings = get_settings()
    limit = settings.user_rate_limit_per_minute
    window_seconds = 60
    redis_key = f"rate:user:{principal.user_id}"
    now = time.time()
    window_start = now - window_seconds

    redis_client = getattr(request.app.state, "redis", None)

    if redis_client is None:
        # Redis not available — degrade gracefully
        logger.warning(
            "[RateLimit] Redis unavailable, skipping per-user rate check",
            user_id=principal.user_id,
        )
        return

    try:
        async with redis_client.pipeline(transaction=True) as pipe:
            # Remove timestamps outside the current window
            pipe.zremrangebyscore(redis_key, "-inf", window_start)
            # Count requests still in the window
            pipe.zcard(redis_key)
            # Add current request timestamp (score=timestamp, member=unique string)
            pipe.zadd(redis_key, {f"{now}:{uuid.uuid4()}": now})
            # Ensure key expires (avoids orphaned keys for inactive users)
            pipe.expire(redis_key, window_seconds * 2)
            results = await pipe.execute()

        current_count: int = results[1]  # zcard result (before adding current)

        remaining = max(0, limit - current_count - 1)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = f"{window_seconds}s"

        if current_count >= limit:
            retry_after = window_seconds
            response.headers["Retry-After"] = str(retry_after)
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"You have exceeded {limit} requests per minute. Please wait {retry_after}s.",
                    "limit": limit,
                    "window": f"{window_seconds}s",
                },
            )

    except HTTPException:
        raise
    except Exception as exc:
        # Any unexpected Redis error → degrade gracefully
        logger.warning(
            "[RateLimit] Redis error during rate check, allowing request",
            user_id=principal.user_id,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# enforce_user_budget
# ---------------------------------------------------------------------------


async def enforce_user_budget(
    request: Request,
    response: Response,
    principal: Principal = Security(get_current_principal, scopes=["query:read"]),
) -> None:
    """Pre-execution budget guard: blocks requests when the user has exhausted
    their lifetime cost allowance.

    Admin users (admin:all scope) are exempt.

    The check reads the denormalized ``total_cost`` field on the user document
    (O(1) index lookup) — no aggregation across usage_logs needed.

    Raises:
        HTTPException 429 if the user has reached or exceeded their budget cap.
    """
    if _is_admin(principal):
        return

    settings = get_settings()
    max_budget = settings.user_max_budget

    usage_service: UsageService | None = getattr(request.app.state, "usage_service", None)

    if usage_service is None:
        # Service not initialized — fail open and log
        logger.warning(
            "[BudgetCheck] UsageService not available on app.state, skipping budget check",
            user_id=principal.user_id,
        )
        return

    try:
        is_within_budget, current_cost = await usage_service.check_budget(principal.user_id, max_budget)
    except Exception as exc:
        logger.warning(
            "[BudgetCheck] Budget check failed, allowing request",
            user_id=principal.user_id,
            error=str(exc),
        )
        return

    budget_remaining = max(0.0, max_budget - current_cost)
    response.headers["X-Budget-Limit"] = f"{max_budget:.6f}"
    response.headers["X-Budget-Used"] = f"{current_cost:.6f}"
    response.headers["X-Budget-Remaining"] = f"{budget_remaining:.6f}"

    if not is_within_budget:
        logger.info(
            "[BudgetCheck] User exceeded budget",
            user_id=principal.user_id,
            current_cost=current_cost,
            max_budget=max_budget,
        )
        raise HTTPException(
            status_code=429,
            detail={
                "error": "budget_exceeded",
                "message": (
                    f"You have reached your lifetime usage budget of ${max_budget:.4f}. "
                    "Contact an administrator to increase your limit."
                ),
                "current_cost": round(current_cost, 6),
                "max_budget": max_budget,
            },
        )
