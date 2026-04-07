# tests/unit/services/test_embedding_service.py
"""
Unit tests for app/services/embedding_service.py.

Uses object.__new__ to bypass __init__ (avoids real GoogleGenerativeAIEmbeddings
instantiation). The embeddings attribute is replaced with an AsyncMock.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from app.services.embedding_service import EmbeddingService
from app.services.pinecone_service import PineconeService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(embedding_vector: list[float] | None = None) -> EmbeddingService:
    """Bypass __init__ and inject a mock embeddings client."""
    svc = object.__new__(EmbeddingService)
    vec = embedding_vector or [0.1] * 768

    mock_embeddings = MagicMock()
    mock_embeddings.aembed_query = AsyncMock(return_value=vec)
    mock_embeddings.aembed_documents = AsyncMock(return_value=[vec])
    svc.embeddings = mock_embeddings
    return svc


def _make_pinecone() -> PineconeService:
    svc = object.__new__(PineconeService)
    svc.upsert_vectors = AsyncMock(return_value=1)
    return svc


def _make_rule(section: int = 1, rule_number: int = 1, rule_type: str = "RULE") -> dict:
    return {
        "scope": "MISRA C:2023",
        "rule_type": rule_type,
        "section": section,
        "rule_number": rule_number,
        "category": "Required",
        "full_text": f"Rule {section}.{rule_number} text.",
    }


# ---------------------------------------------------------------------------
# get_embedding
# ---------------------------------------------------------------------------


async def test_get_embedding_delegates_to_aembed_query():
    vec = [0.5] * 768
    svc = _make_service(embedding_vector=vec)

    result = await svc.get_embedding("test query")

    svc.embeddings.aembed_query.assert_called_once_with("test query")
    assert result == vec


# ---------------------------------------------------------------------------
# embed_and_store
# ---------------------------------------------------------------------------


async def test_embed_and_store_empty_rules_returns_zero():
    svc = _make_service()
    pinecone = _make_pinecone()

    result = await svc.embed_and_store([], pinecone)

    assert result == 0
    svc.embeddings.aembed_documents.assert_not_called()


async def test_embed_and_store_builds_correct_vector_ids():
    vec = [0.1] * 768
    svc = _make_service(vec)
    svc.embeddings.aembed_documents = AsyncMock(return_value=[vec])
    pinecone = _make_pinecone()

    rules = [_make_rule(section=15, rule_number=1, rule_type="RULE")]
    await svc.embed_and_store(rules, pinecone)

    upsert_call = pinecone.upsert_vectors.call_args[0][0]
    assert upsert_call[0]["id"] == "MISRA_RULE_15.1"


async def test_embed_and_store_builds_correct_metadata():
    vec = [0.2] * 768
    svc = _make_service(vec)
    svc.embeddings.aembed_documents = AsyncMock(return_value=[vec])
    pinecone = _make_pinecone()

    rules = [_make_rule(section=4, rule_number=1, rule_type="DIR")]
    await svc.embed_and_store(rules, pinecone)

    vectors = pinecone.upsert_vectors.call_args[0][0]
    meta = vectors[0]["metadata"]
    assert meta["scope"] == "MISRA C:2023"
    assert meta["rule_type"] == "DIR"
    assert meta["section"] == 4
    assert meta["rule_number"] == 1


async def test_embed_and_store_delegates_to_pinecone_upsert():
    vec = [0.3] * 768
    svc = _make_service(vec)
    svc.embeddings.aembed_documents = AsyncMock(return_value=[vec, vec])
    pinecone = _make_pinecone()
    pinecone.upsert_vectors = AsyncMock(return_value=2)

    rules = [_make_rule(1, 1), _make_rule(1, 2)]
    result = await svc.embed_and_store(rules, pinecone)

    pinecone.upsert_vectors.assert_called_once()
    assert result == 2


async def test_embed_and_store_uses_full_text_for_embedding():
    vec = [0.4] * 768
    svc = _make_service(vec)
    svc.embeddings.aembed_documents = AsyncMock(return_value=[vec])
    pinecone = _make_pinecone()

    rule = _make_rule()
    rule["full_text"] = "unique rule text content"
    await svc.embed_and_store([rule], pinecone)

    texts_embedded = svc.embeddings.aembed_documents.call_args[0][0]
    assert texts_embedded == ["unique rule text content"]


# ---------------------------------------------------------------------------
# EmbeddingService.__init__
# ---------------------------------------------------------------------------


def test_init_creates_embeddings_from_settings():
    mock_settings = MagicMock()
    mock_settings.gemini_embedding_model = "models/text-embedding-004"
    mock_settings.gemini_api_key = "fake-api-key"
    mock_settings.embedding_dimensions = 768

    mock_embeddings_instance = MagicMock()

    with (
        patch("app.services.embedding_service.get_settings", return_value=mock_settings),
        patch(
            "app.services.embedding_service.GoogleGenerativeAIEmbeddings", return_value=mock_embeddings_instance
        ) as mock_cls,
    ):
        svc = EmbeddingService()

    mock_cls.assert_called_once_with(
        model=mock_settings.gemini_embedding_model,
        google_api_key=mock_settings.gemini_api_key,
        output_dimensionality=mock_settings.embedding_dimensions,
    )
    assert svc.embeddings is mock_embeddings_instance
