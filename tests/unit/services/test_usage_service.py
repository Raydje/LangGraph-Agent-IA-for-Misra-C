# tests/unit/services/test_usage_service.py
"""
Unit tests for app/services/usage_service.py.

All MongoDB interactions are mocked via AsyncMock — no real DB connection required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from app.services.usage_service import UsageService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(
    user_doc: dict | None = None,
    insert_raises: Exception | None = None,
    update_raises: Exception | None = None,
    find_docs: list[dict] | None = None,
) -> MagicMock:
    """Build a mock AsyncIOMotorDatabase with pre-configured collection behaviour."""
    db = MagicMock()

    # users collection
    users_coll = MagicMock()
    users_coll.find_one = AsyncMock(return_value=user_doc)
    users_coll.update_one = AsyncMock(side_effect=update_raises) if update_raises else AsyncMock()
    users_coll.create_index = AsyncMock()

    # usage_logs collection
    logs_coll = MagicMock()
    if insert_raises:
        logs_coll.insert_one = AsyncMock(side_effect=insert_raises)
    else:
        logs_coll.insert_one = AsyncMock()
    logs_coll.create_index = AsyncMock()

    # Cursor for get_user_usage
    cursor = MagicMock()
    cursor.sort = MagicMock(return_value=cursor)
    cursor.limit = MagicMock(return_value=cursor)
    cursor.to_list = AsyncMock(return_value=find_docs or [])
    logs_coll.find = MagicMock(return_value=cursor)

    # Route __getitem__ to the right collection
    def _getitem(name: str):
        if name == "users":
            return users_coll
        if name == "usage_logs":
            return logs_coll
        return MagicMock()

    db.__getitem__ = MagicMock(side_effect=_getitem)
    return db


def _make_service(db: MagicMock) -> UsageService:
    return UsageService(db=db)


_RECORD_KWARGS = {
    "user_id": "user-123",
    "endpoint": "/api/v1/query",
    "method": "POST",
    "thread_id": "thread-abc",
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150,
    "estimated_cost": 0.003,
    "critique_iterations": 0,
    "nodes_visited": None,
    "status_code": 200,
}


# ---------------------------------------------------------------------------
# check_budget
# ---------------------------------------------------------------------------


async def test_check_budget_within_limit_returns_true():
    db = _make_db(user_doc={"_id": "user-123", "total_cost": 1.5})
    svc = _make_service(db)
    within, cost = await svc.check_budget("user-123", max_budget=5.0)
    assert within is True
    assert cost == 1.5


async def test_check_budget_exactly_at_limit_returns_false():
    db = _make_db(user_doc={"_id": "user-123", "total_cost": 5.0})
    svc = _make_service(db)
    within, cost = await svc.check_budget("user-123", max_budget=5.0)
    assert within is False
    assert cost == 5.0


async def test_check_budget_over_limit_returns_false():
    db = _make_db(user_doc={"_id": "user-123", "total_cost": 7.42})
    svc = _make_service(db)
    within, cost = await svc.check_budget("user-123", max_budget=5.0)
    assert within is False
    assert cost == 7.42


async def test_check_budget_missing_total_cost_field_defaults_to_zero():
    # User exists but was created before total_cost field was added
    db = _make_db(user_doc={"_id": "user-123"})
    svc = _make_service(db)
    within, cost = await svc.check_budget("user-123", max_budget=5.0)
    assert within is True
    assert cost == 0.0


async def test_check_budget_unknown_user_returns_true():
    # Unknown user — let auth layer handle it; budget check passes
    db = _make_db(user_doc=None)
    svc = _make_service(db)
    within, cost = await svc.check_budget("ghost-user", max_budget=5.0)
    assert within is True
    assert cost == 0.0


async def test_check_budget_queries_correct_user():
    db = _make_db(user_doc={"_id": "user-123", "total_cost": 0.0})
    svc = _make_service(db)
    await svc.check_budget("user-123", max_budget=5.0)
    db["users"].find_one.assert_called_once_with({"_id": "user-123"}, {"total_cost": 1})


# ---------------------------------------------------------------------------
# record_usage
# ---------------------------------------------------------------------------


async def test_record_usage_inserts_log_and_increments_user():
    db = _make_db()
    svc = _make_service(db)
    await svc.record_usage(**_RECORD_KWARGS)

    db["usage_logs"].insert_one.assert_called_once()
    inserted = db["usage_logs"].insert_one.call_args[0][0]
    assert inserted["user_id"] == "user-123"
    assert inserted["estimated_cost"] == 0.003
    assert inserted["thread_id"] == "thread-abc"
    assert inserted["status_code"] == 200

    db["users"].update_one.assert_called_once_with(
        {"_id": "user-123"},
        {"$inc": {"total_cost": 0.003, "total_requests": 1}},
    )


async def test_record_usage_insert_failure_does_not_raise():
    """A failed insert_one should log a warning but not crash the caller."""
    db = _make_db(insert_raises=Exception("MongoDB unavailable"))
    svc = _make_service(db)
    # Must not raise
    await svc.record_usage(**_RECORD_KWARGS)
    # update_one should NOT be called if insert failed
    db["users"].update_one.assert_not_called()


async def test_record_usage_update_failure_does_not_raise():
    """A failed update_one should log a warning but not crash the caller."""
    db = _make_db(update_raises=Exception("MongoDB timeout"))
    svc = _make_service(db)
    # Must not raise
    await svc.record_usage(**_RECORD_KWARGS)
    # insert_one was called successfully before the update failure
    db["usage_logs"].insert_one.assert_called_once()


async def test_record_usage_null_thread_id_is_stored():
    db = _make_db()
    svc = _make_service(db)
    kwargs = {**_RECORD_KWARGS, "thread_id": None}
    await svc.record_usage(**kwargs)
    inserted = db["usage_logs"].insert_one.call_args[0][0]
    assert inserted["thread_id"] is None


async def test_record_usage_log_contains_all_required_fields():
    db = _make_db()
    svc = _make_service(db)
    await svc.record_usage(**_RECORD_KWARGS)
    inserted = db["usage_logs"].insert_one.call_args[0][0]
    required_fields = {
        "user_id",
        "endpoint",
        "method",
        "timestamp",
        "thread_id",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_cost",
        "critique_iterations",
        "nodes_visited",
        "status_code",
    }
    assert required_fields.issubset(inserted.keys())


# ---------------------------------------------------------------------------
# get_user_usage
# ---------------------------------------------------------------------------


async def test_get_user_usage_returns_combined_summary():
    log_entry = {"user_id": "user-123", "estimated_cost": 0.003, "critique_iterations": 0, "nodes_visited": None}
    db = _make_db(
        user_doc={"_id": "user-123", "email": "u@test.com", "total_cost": 1.5, "total_requests": 3},
        find_docs=[log_entry],
    )
    svc = _make_service(db)
    result = await svc.get_user_usage("user-123")

    assert result["user_id"] == "user-123"
    assert result["email"] == "u@test.com"
    assert result["total_cost"] == 1.5
    assert result["total_requests"] == 3
    assert result["recent_logs"] == [log_entry]


async def test_get_user_usage_unknown_user_returns_empty():
    db = _make_db(user_doc=None)
    svc = _make_service(db)
    result = await svc.get_user_usage("ghost")
    assert result == {}


# ---------------------------------------------------------------------------
# create_indexes
# ---------------------------------------------------------------------------


async def test_create_indexes_creates_all_three_indexes():
    db = _make_db()
    svc = _make_service(db)
    await svc.create_indexes()

    calls = db["usage_logs"].create_index.call_args_list
    # Should have been called 3 times: user_id, timestamp, compound
    assert len(calls) == 3
    index_args = [c[0][0] for c in calls]
    assert "user_id" in index_args
    assert "timestamp" in index_args
    # Third call is the compound index (list of tuples)
    compound_calls = [a for a in index_args if isinstance(a, list)]
    assert len(compound_calls) == 1
