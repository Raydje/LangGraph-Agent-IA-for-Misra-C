# tests/unit/api/test_routes.py
"""
Unit tests for app/api/v1/routes.py.

Strategy: call route handler functions directly with a mocked Request and
pre-built Principal.  All FastAPI dependencies (graph, services, principal)
are passed as explicit arguments — no DI container involved.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import app.api.v1.routes as _routes_mod
from app.auth.models import Principal

# Each route handler is wrapped by @limiter.limit which rejects non-starlette
# Request objects.  We test handler *logic* here, not rate limiting, so we
# reach through to the unwrapped function via __wrapped__.
_build_response = _routes_mod._build_response
health_check = _routes_mod.health_check.__wrapped__
query_compliance = _routes_mod.query_compliance.__wrapped__
seed_database = _routes_mod.seed_database.__wrapped__
replay_from_checkpoint = _routes_mod.replay_from_checkpoint.__wrapped__
get_thread_history = _routes_mod.get_thread_history.__wrapped__
get_usage = _routes_mod.get_usage.__wrapped__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _principal(scopes: list[str] | None = None) -> Principal:
    return Principal(
        user_id="u1",
        email="u@test.com",
        scopes=scopes or ["query:read", "admin:seed", "admin:replay"],
        auth_method="jwt",
    )


def _make_request() -> MagicMock:
    return MagicMock()


def _make_response() -> MagicMock:
    resp = MagicMock()
    resp.headers = {}
    return resp


def _make_usage_service() -> MagicMock:
    svc = MagicMock()
    svc.record_usage = AsyncMock()
    return svc


def _make_graph(result: dict | None = None) -> MagicMock:
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value=result or _minimal_result())
    graph.aget_state = AsyncMock(return_value=MagicMock(values={"query": "q"}))
    return graph


def _minimal_result() -> dict:
    return {
        "intent": "validate",
        "final_response": "Compliant.",
        "is_compliant": True,
        "confidence_score": 0.9,
        "cited_rules": ["MISRA-1.1"],
        "iteration_count": 1,
        "critique_approved": True,
        "critique_history": [],
        "retrieved_rules": [{"rule_id": "MISRA-1.1"}],
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "orchestrator_tokens": 5,
        "validation_tokens": 5,
        "critique_tokens": 0,
        "remediation_tokens": 0,
        "estimated_cost": 0.001,
        "error": None,
        "fixed_code_snippet": None,
        "remediation_explanation": None,
    }


def _mock_db(ping_raises: bool = False) -> MagicMock:
    db = MagicMock()
    if ping_raises:
        db.command = AsyncMock(side_effect=Exception("ping failed"))
    else:
        db.command = AsyncMock(return_value={"ok": 1})
    return db


def _mock_index(stats_raises: bool = False) -> MagicMock:
    index = MagicMock()
    if stats_raises:
        index.describe_index_stats = MagicMock(side_effect=Exception("stats failed"))
    else:
        index.describe_index_stats = MagicMock(return_value={})
    return index


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


async def test_health_check_healthy_when_both_services_ok():
    result = await health_check(_make_request(), db=_mock_db(), index=_mock_index())
    assert result.status == "healthy"
    assert result.mongodb_connected is True
    assert result.pinecone_connected is True


async def test_health_check_degraded_when_mongo_fails():
    result = await health_check(_make_request(), db=_mock_db(ping_raises=True), index=_mock_index())
    assert result.status == "degraded"
    assert result.mongodb_connected is False
    assert result.pinecone_connected is True


async def test_health_check_degraded_when_pinecone_fails():
    result = await health_check(_make_request(), db=_mock_db(), index=_mock_index(stats_raises=True))
    assert result.status == "degraded"
    assert result.mongodb_connected is True
    assert result.pinecone_connected is False


async def test_health_check_none_services_marks_degraded():
    result = await health_check(_make_request(), db=None, index=None)
    assert result.status == "degraded"
    assert result.mongodb_connected is False
    assert result.pinecone_connected is False


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


async def test_query_compliance_happy_path_returns_response():
    from app.api.v1.requests import ComplianceQueryRequest

    graph = _make_graph()
    body = ComplianceQueryRequest(query="Does this comply?", standard="MISRA C:2023")

    result = await query_compliance(
        request=_make_request(),
        response=_make_response(),
        body=body,
        graph=graph,
        embedding_service=MagicMock(),
        mongo_db=MagicMock(),
        pinecone_service=MagicMock(),
        usage_service=_make_usage_service(),
        principal=_principal(),
    )

    assert result.intent == "validate"
    assert result.is_compliant is True
    graph.ainvoke.assert_called_once()


async def test_query_compliance_graph_exception_raises_500():
    from app.api.v1.requests import ComplianceQueryRequest

    graph = MagicMock()
    graph.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
    body = ComplianceQueryRequest(query="Test?", standard="MISRA C:2023")

    with pytest.raises(HTTPException) as exc_info:
        await query_compliance(
            request=_make_request(),
            response=_make_response(),
            body=body,
            graph=graph,
            embedding_service=MagicMock(),
            mongo_db=MagicMock(),
            pinecone_service=MagicMock(),
            usage_service=_make_usage_service(),
            principal=_principal(),
        )
    assert exc_info.value.status_code == 500


async def test_query_compliance_thread_id_preserved_when_provided():
    from app.api.v1.requests import ComplianceQueryRequest

    graph = _make_graph()
    body = ComplianceQueryRequest(query="Test?", standard="MISRA C:2023", thread_id="my-thread-123")

    result = await query_compliance(
        request=_make_request(),
        response=_make_response(),
        body=body,
        graph=graph,
        embedding_service=MagicMock(),
        mongo_db=MagicMock(),
        pinecone_service=MagicMock(),
        usage_service=_make_usage_service(),
        principal=_principal(),
    )

    assert result.thread_id == "my-thread-123"


# ---------------------------------------------------------------------------
# POST /seed
# ---------------------------------------------------------------------------


async def test_seed_database_calls_ingest_and_returns_response():
    with patch("app.api.v1.routes.ingest", new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = {"rules_ingested": 42, "vectors_upserted": 42}

        result = await seed_database(
            request=_make_request(),
            response=_make_response(),
            principal=_principal(["admin:seed"]),
            embedding_service=MagicMock(),
            mongo_db=MagicMock(),
            pinecone_service=MagicMock(),
        )

    assert result.rules_ingested == 42
    assert result.vectors_upserted == 42
    mock_ingest.assert_called_once()


# ---------------------------------------------------------------------------
# POST /replay/{thread_id}/{checkpoint_id}
# ---------------------------------------------------------------------------


async def test_replay_happy_path_returns_response():
    graph = _make_graph()

    result = await replay_from_checkpoint(
        request=_make_request(),
        response=_make_response(),
        thread_id="t1",
        checkpoint_id="c1",
        graph=graph,
        embedding_service=MagicMock(),
        mongo_db=MagicMock(),
        pinecone_service=MagicMock(),
        principal=_principal(),
    )

    assert result.thread_id == "t1"


async def test_replay_missing_checkpoint_raises_404():
    graph = MagicMock()
    # aget_state returns an object with empty values
    empty_state = MagicMock()
    empty_state.values = {}
    graph.aget_state = AsyncMock(return_value=empty_state)

    with pytest.raises(HTTPException) as exc_info:
        await replay_from_checkpoint(
            request=_make_request(),
            response=_make_response(),
            thread_id="t1",
            checkpoint_id="missing",
            graph=graph,
            embedding_service=MagicMock(),
            mongo_db=MagicMock(),
            pinecone_service=MagicMock(),
            principal=_principal(),
        )
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# GET /history/{thread_id}
# ---------------------------------------------------------------------------


async def test_get_thread_history_returns_history():
    graph = MagicMock()

    checkpoint = MagicMock()
    checkpoint.config = {"configurable": {"checkpoint_id": "ckpt-1"}}
    checkpoint.next = ("rag",)
    checkpoint.values = {"intent": "validate"}

    async def _state_history(config):
        yield checkpoint

    graph.aget_state_history = _state_history

    result = await get_thread_history(
        request=_make_request(),
        response=_make_response(),
        thread_id="t1",
        graph=graph,
        principal=_principal(),
    )

    assert result.thread_id == "t1"
    assert len(result.history) == 1
    assert result.history[0].checkpoint_id == "ckpt-1"


async def test_get_thread_history_empty_raises_404():
    graph = MagicMock()

    async def _empty_history(config):
        return
        yield  # make it an async generator

    graph.aget_state_history = _empty_history

    with pytest.raises(HTTPException) as exc_info:
        await get_thread_history(
            request=_make_request(),
            response=_make_response(),
            thread_id="t-empty",
            graph=graph,
            principal=_principal(),
        )
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# GET /usage
# ---------------------------------------------------------------------------


async def test_get_usage_returns_usage_data():
    usage_service = _make_usage_service()
    mock_data = {
        "user_id": "u1",
        "email": "u@test.com",
        "total_cost": 1.5,
        "total_requests": 3,
        "recent_logs": [
            {
                "user_id": "u1",
                "endpoint": "/api/v1/query",
                "method": "POST",
                "timestamp": "2026-04-07T00:00:00Z",
                "thread_id": "thread-abc",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "estimated_cost": 0.003,
                "critique_iterations": 2,
                "nodes_visited": ["orchestrator", "rag", "validation", "critique", "validation"],
                "status_code": 200,
            }
        ],
    }
    usage_service.get_user_usage = AsyncMock(return_value=mock_data)

    result = await get_usage(
        request=_make_request(),
        response=_make_response(),
        usage_service=usage_service,
        principal=_principal(),
    )

    assert result.user_id == "u1"
    assert result.total_cost == 1.5
    assert result.total_requests == 3
    assert len(result.recent_logs) == 1
    assert result.recent_logs[0].endpoint == "/api/v1/query"
    assert result.recent_logs[0].critique_iterations == 2
    assert result.recent_logs[0].nodes_visited == ["orchestrator", "rag", "validation", "critique", "validation"]


async def test_get_usage_user_not_found_raises_404():
    usage_service = _make_usage_service()
    usage_service.get_user_usage = AsyncMock(return_value={})

    with pytest.raises(HTTPException) as exc_info:
        await get_usage(
            request=_make_request(),
            response=_make_response(),
            usage_service=usage_service,
            principal=_principal(),
        )
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# _build_response (pure function)
# ---------------------------------------------------------------------------


def test_build_response_maps_result_to_response():
    response = _build_response("thread-xyz", _minimal_result())
    assert response.thread_id == "thread-xyz"
    assert response.intent == "validate"
    assert response.is_compliant is True
    assert response.retrieved_rule_ids == ["MISRA-1.1"]
    assert response.total_tokens_usage.total_tokens == 15


def test_build_response_defaults_for_missing_keys():
    response = _build_response("t1", {})
    assert response.intent == "unknown"
    assert response.final_response == ""
    assert response.cited_rules == []
    assert response.retrieved_rule_ids == []
