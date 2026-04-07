"""
UsageService — per-user cost tracking and budget enforcement.

Responsibilities:
  - check_budget(user_id)     : fast O(1) pre-check against users.total_cost
  - record_usage(user_id, …)  : insert usage_logs doc + atomically $inc user totals
  - get_user_usage(user_id)   : return summary for a given user

All methods are async (Motor / AsyncIOMotorDatabase).

Lifecycle:
  Instantiated once in main.py lifespan and stored on app.state.usage_service.
  Retrieved in routes via get_usage_service() dependency.
"""

from __future__ import annotations

from datetime import UTC, datetime

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.utils import logger


class UsageService:
    """Service for persisting per-request usage data and enforcing budget limits."""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check_budget(self, user_id: str, max_budget: float) -> tuple[bool, float]:
        """Check whether the user is still within their lifetime budget.

        Returns:
            (is_within_budget, current_total_cost)

        Uses the denormalized ``total_cost`` field on the user document for an
        O(1) read — no aggregation across usage_logs is required.
        """
        user = await self._db["users"].find_one(
            {"_id": user_id},
            {"total_cost": 1},
        )
        if user is None:
            # Unknown user — let auth layer handle it; don't block here
            return True, 0.0

        current_cost: float = user.get("total_cost", 0.0)
        return current_cost < max_budget, current_cost

    async def record_usage(
        self,
        *,
        user_id: str,
        endpoint: str,
        method: str,
        thread_id: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        estimated_cost: float,
        critique_iterations: int = 0,
        nodes_visited: list[str] | None = None,
        status_code: int,
    ) -> None:
        """Persist one usage log entry and atomically update the user's running totals.

        The two writes are intentionally NOT wrapped in a transaction:
          - usage_logs insert is append-only; a duplicate is harmless.
          - $inc on users is idempotent only in the happy path, but a partial
            failure (log inserted, user not updated) is recoverable via
            re-aggregation from usage_logs.  A transaction would require a
            replica set, which is not assumed in the local dev environment.

        Additional metadata: critique_iterations (number of critique loops),

        nodes_visited (list of graph node names executed in order).
        """
        log_entry = {
            "user_id": user_id,
            "endpoint": endpoint,
            "method": method,
            "timestamp": datetime.now(UTC),
            "thread_id": thread_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost": estimated_cost,
            "critique_iterations": critique_iterations,
            "nodes_visited": nodes_visited,
            "status_code": status_code,
        }

        try:
            await self._db["usage_logs"].insert_one(log_entry)
        except Exception as exc:
            logger.error("[UsageService] Failed to insert usage log", user_id=user_id, error=str(exc))
            return  # Do not crash the request; log failure is non-fatal

        try:
            await self._db["users"].update_one(
                {"_id": user_id},
                {
                    "$inc": {
                        "total_cost": estimated_cost,
                        "total_requests": 1,
                    }
                },
            )
        except Exception as exc:
            logger.error(
                "[UsageService] Failed to update user totals after usage log insert",
                user_id=user_id,
                error=str(exc),
            )

    async def get_user_usage(self, user_id: str) -> dict:
        """Return the usage summary for a user (totals + recent log entries).

        Combines the fast denormalized totals from the user document with the
        last 20 detailed entries from usage_logs (newest-first).
        """
        user = await self._db["users"].find_one(
            {"_id": user_id},
            {"total_cost": 1, "total_requests": 1, "email": 1},
        )
        if user is None:
            return {}

        cursor = self._db["usage_logs"].find({"user_id": user_id}, {"_id": 0}).sort("timestamp", -1).limit(20)
        recent_logs = await cursor.to_list(length=20)

        return {
            "user_id": user_id,
            "email": user.get("email"),
            "total_cost": user.get("total_cost", 0.0),
            "total_requests": user.get("total_requests", 0),
            "recent_logs": recent_logs,
        }

    # ------------------------------------------------------------------
    # Index creation (called once at startup, idempotent)
    # ------------------------------------------------------------------

    async def create_indexes(self) -> None:
        """Ensure MongoDB indexes exist on the usage_logs collection."""
        await self._db["usage_logs"].create_index("user_id")
        await self._db["usage_logs"].create_index("timestamp")
        # Compound index for the most common query pattern: user logs in time order
        await self._db["usage_logs"].create_index(
            [("user_id", 1), ("timestamp", -1)],
            name="user_id_timestamp_desc",
        )
        logger.info("[Startup] UsageService indexes ensured (usage_logs.user_id, usage_logs.timestamp)")
