# tests/unit/services/test_pinecone_service.py
"""
Unit tests for app/services/pinecone_service.py.

Uses object.__new__ to bypass __init__ (which calls the real Pinecone SDK).
The index attribute is replaced with a MagicMock in each test.

asyncio.to_thread is patched module-wide so the session-scoped event loop
never spawns real thread-pool coroutines (avoids pytest-asyncio teardown warnings).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.services.pinecone_service import PineconeService

# ---------------------------------------------------------------------------
# Module-wide to_thread shim — keeps tests in-loop, no real threads
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_to_thread():
    async def _fake(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch("app.services.pinecone_service.asyncio.to_thread", new=_fake):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service() -> PineconeService:
    """Instantiate PineconeService without calling __init__."""
    svc = object.__new__(PineconeService)
    svc.index = MagicMock()
    return svc


def _pinecone_matches(*id_score_pairs: tuple[str, float]) -> MagicMock:
    result = MagicMock()
    result.matches = [MagicMock(id=rid, score=score, metadata={"rule": rid}) for rid, score in id_score_pairs]
    return result


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


async def test_query_happy_path_returns_matches():
    svc = _make_service()
    pinecone_result = _pinecone_matches(("MISRA-1.1", 0.95), ("MISRA-2.2", 0.80))
    svc.index.query = MagicMock(return_value=pinecone_result)

    result = await svc.query(vector=[0.1] * 768, top_k=5)

    assert len(result["matches"]) == 2
    assert result["matches"][0]["id"] == "MISRA-1.1"
    assert result["matches"][0]["score"] == 0.95
    assert result["matches"][1]["id"] == "MISRA-2.2"


async def test_query_passes_filter_and_top_k():
    svc = _make_service()
    svc.index.query = MagicMock(return_value=_pinecone_matches())

    await svc.query(vector=[0.0] * 768, top_k=3, filter={"scope": "MISRA C:2023"})

    call_kwargs = svc.index.query.call_args[1]
    assert call_kwargs["top_k"] == 3
    assert call_kwargs["filter"] == {"scope": "MISRA C:2023"}
    assert call_kwargs["include_metadata"] is True


async def test_query_timeout_returns_empty_matches():
    svc = _make_service()

    # Simulate asyncio.wait_for raising TimeoutError
    with patch("app.services.pinecone_service.asyncio.wait_for", side_effect=asyncio.TimeoutError):
        result = await svc.query(vector=[0.1] * 768)

    assert result == {"matches": []}


# ---------------------------------------------------------------------------
# upsert_vectors
# ---------------------------------------------------------------------------


async def test_upsert_vectors_returns_total_count():
    svc = _make_service()
    svc.index.upsert = MagicMock(return_value=None)

    vectors = [{"id": f"v{i}", "values": [0.1] * 768, "metadata": {}} for i in range(5)]
    total = await svc.upsert_vectors(vectors)

    assert total == 5


async def test_upsert_vectors_batches_in_groups_of_100():
    svc = _make_service()
    call_sizes: list[int] = []

    def _sync_upsert(vectors, **kwargs):
        call_sizes.append(len(vectors))

    svc.index.upsert = _sync_upsert

    # 250 vectors → 3 batches: 100 + 100 + 50
    vectors = [{"id": f"v{i}", "values": [0.0], "metadata": {}} for i in range(250)]
    total = await svc.upsert_vectors(vectors)

    assert total == 250
    assert len(call_sizes) == 3
    assert call_sizes == [100, 100, 50]


async def test_upsert_vectors_empty_list_returns_zero():
    svc = _make_service()
    total = await svc.upsert_vectors([])
    assert total == 0
    svc.index.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# PineconeService.__init__
# ---------------------------------------------------------------------------


def test_init_uses_existing_index_without_creating():
    """When index already exists in list_indexes, create_index must NOT be called."""
    mock_settings = MagicMock()
    mock_settings.pinecone_api_key = "fake-key"
    mock_settings.pinecone_index_name = "misra-index"

    existing_idx = MagicMock()
    existing_idx.name = "misra-index"

    mock_pc = MagicMock()
    mock_pc.list_indexes.return_value = [existing_idx]
    mock_pc.Index.return_value = MagicMock()

    with (
        patch("app.services.pinecone_service.get_settings", return_value=mock_settings),
        patch("app.services.pinecone_service.Pinecone", return_value=mock_pc),
    ):
        svc = PineconeService()

    mock_pc.create_index.assert_not_called()
    assert svc.index is mock_pc.Index.return_value


def test_init_creates_index_when_not_existing():
    """When index is absent from list_indexes, create_index must be called once."""
    mock_settings = MagicMock()
    mock_settings.pinecone_api_key = "fake-key"
    mock_settings.pinecone_index_name = "misra-index"
    mock_settings.embedding_dimensions = 768
    mock_settings.pinecone_cloud = "aws"
    mock_settings.pinecone_region = "us-east-1"

    mock_pc = MagicMock()
    mock_pc.list_indexes.return_value = []  # index does not exist
    mock_pc.Index.return_value = MagicMock()

    with (
        patch("app.services.pinecone_service.get_settings", return_value=mock_settings),
        patch("app.services.pinecone_service.Pinecone", return_value=mock_pc),
        patch("app.services.pinecone_service.ServerlessSpec"),
    ):
        svc = PineconeService()

    mock_pc.create_index.assert_called_once()
    call_kwargs = mock_pc.create_index.call_args[1]
    assert call_kwargs["name"] == "misra-index"
    assert call_kwargs["dimension"] == 768
    assert call_kwargs["metric"] == "cosine"
    assert svc.index is mock_pc.Index.return_value
