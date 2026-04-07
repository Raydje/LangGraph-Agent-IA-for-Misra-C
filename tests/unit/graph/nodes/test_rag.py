# tests/unit/graph/nodes/test_rag.py

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.graph.nodes.rag import rag_node

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(query: str = "check pointer arithmetic") -> dict:
    return {"query": query}


def _make_config(embed_return, pinecone_return, mongo_return=None):
    """Build a LangGraph-style config dict with mocked services."""
    mock_embed_svc = MagicMock()
    mock_embed_svc.get_embedding = AsyncMock(return_value=embed_return)

    mock_pinecone_svc = MagicMock()
    mock_pinecone_svc.query = AsyncMock(return_value=pinecone_return)

    mock_mongo_svc = MagicMock()
    mock_mongo_svc.get_misra_rules_by_pinecone_ids = AsyncMock(return_value=mongo_return or [])

    config = {
        "configurable": {
            "embedding_service": mock_embed_svc,
            "pinecone_service": mock_pinecone_svc,
            "mongo_db": mock_mongo_svc,
        }
    }
    return config, mock_embed_svc, mock_pinecone_svc, mock_mongo_svc


def _pinecone_result(*rule_ids_and_scores: tuple[str, float]) -> dict:
    """Build a minimal Pinecone response dict."""
    return {"matches": [{"id": rid, "score": score} for rid, score in rule_ids_and_scores]}


def _mongo_doc(
    rule_id: str,
    *,
    section: str = "1.1",
    category: str = "N/A",
    title: str = "",
    full_text: str = "some rule text",
) -> dict:
    return {
        "rule_id": rule_id,
        "section": section,
        "category": category,
        "title": title,
        "full_text": full_text,
    }


FAKE_VECTOR = [0.1] * 768


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_happy_path_returns_retrieved_rules():
    config, _, _, _ = _make_config(
        FAKE_VECTOR,
        _pinecone_result(("MISRA-R1", 0.9), ("MISRA-R2", 0.7)),
        [_mongo_doc("MISRA-R1", title="Rule 1"), _mongo_doc("MISRA-R2", title="Rule 2")],
    )

    result = await rag_node(_make_state("pointer arithmetic"), config)

    assert len(result["retrieved_rules"]) == 2
    assert result["retrieved_rules"][0]["rule_id"] == "MISRA-R1"
    assert result["retrieved_rules"][0]["relevance_score"] == 0.9
    assert result["retrieved_rules"][1]["rule_id"] == "MISRA-R2"
    assert result["retrieved_rules"][1]["relevance_score"] == 0.7


async def test_returns_correct_state_keys():
    config, _, _, _ = _make_config(
        FAKE_VECTOR,
        _pinecone_result(("MISRA-R1", 0.8)),
        [_mongo_doc("MISRA-R1")],
    )

    result = await rag_node(_make_state(), config)

    assert set(result.keys()) == {"retrieved_rules", "rag_query_used", "metadata_filters_applied"}


async def test_metadata_filter_is_always_misra_c():
    config, _, _, _ = _make_config(FAKE_VECTOR, {"matches": []})

    result = await rag_node(_make_state(), config)

    assert result["metadata_filters_applied"] == {"scope": "MISRA C:2023"}


async def test_rag_query_used_equals_state_query():
    config, _, _, _ = _make_config(FAKE_VECTOR, {"matches": []})

    result = await rag_node(_make_state("my specific query"), config)

    assert result["rag_query_used"] == "my specific query"


async def test_empty_pinecone_matches_skips_mongo():
    config, _, _, mock_mongo_svc = _make_config(FAKE_VECTOR, {"matches": []})

    result = await rag_node(_make_state(), config)

    mock_mongo_svc.get_misra_rules_by_pinecone_ids.assert_not_called()
    assert result["retrieved_rules"] == []


async def test_missing_matches_key_in_pinecone_response():
    config, _, _, mock_mongo_svc = _make_config(FAKE_VECTOR, {})  # no "matches" key

    result = await rag_node(_make_state(), config)

    mock_mongo_svc.get_misra_rules_by_pinecone_ids.assert_not_called()
    assert result["retrieved_rules"] == []


async def test_retrieved_rules_sorted_by_score_descending():
    config, _, _, _ = _make_config(
        FAKE_VECTOR,
        _pinecone_result(("MISRA-R3", 0.5), ("MISRA-R1", 0.95), ("MISRA-R2", 0.75)),
        [_mongo_doc("MISRA-R3"), _mongo_doc("MISRA-R1"), _mongo_doc("MISRA-R2")],
    )

    result = await rag_node(_make_state(), config)

    scores = [r["relevance_score"] for r in result["retrieved_rules"]]
    assert scores == sorted(scores, reverse=True)


async def test_rule_standard_is_hardcoded_to_misra_c():
    config, _, _, _ = _make_config(
        FAKE_VECTOR,
        _pinecone_result(("MISRA-R1", 0.8)),
        [_mongo_doc("MISRA-R1")],
    )

    result = await rag_node(_make_state(), config)

    assert result["retrieved_rules"][0]["standard"] == "MISRA C:2023"


async def test_full_text_fallback_to_text_field():
    """If 'full_text' is absent, 'text' key should be used as fallback."""
    config, _, _, _ = _make_config(
        FAKE_VECTOR,
        _pinecone_result(("MISRA-R1", 0.8)),
        [{"rule_id": "MISRA-R1", "text": "fallback text content"}],
    )

    result = await rag_node(_make_state(), config)

    assert result["retrieved_rules"][0]["full_text"] == "fallback text content"


async def test_missing_optional_mongo_fields_use_defaults():
    """Mongo doc with only rule_id → all optional fields fall back to defaults."""
    config, _, _, _ = _make_config(
        FAKE_VECTOR,
        _pinecone_result(("MISRA-R1", 0.6)),
        [{"rule_id": "MISRA-R1"}],
    )

    result = await rag_node(_make_state(), config)

    rule = result["retrieved_rules"][0]
    assert rule["section"] == ""
    assert rule["category"] == "N/A"
    assert rule["title"] == "Rule MISRA-R1"
    assert rule["full_text"] == ""


async def test_partial_mongo_results_are_accepted():
    """Pinecone returns 3 IDs but MongoDB only resolves 2 — no crash."""
    config, _, _, _ = _make_config(
        FAKE_VECTOR,
        _pinecone_result(("MISRA-R1", 0.9), ("MISRA-R2", 0.8), ("MISRA-R3", 0.7)),
        [_mongo_doc("MISRA-R1"), _mongo_doc("MISRA-R2")],
    )

    result = await rag_node(_make_state(), config)

    assert len(result["retrieved_rules"]) == 2


async def test_pinecone_called_with_misra_c_filter_and_top_k_5():
    config, _, mock_pinecone_svc, _ = _make_config(FAKE_VECTOR, {"matches": []})

    await rag_node(_make_state("null pointer"), config)

    mock_pinecone_svc.query.assert_called_once_with(
        vector=FAKE_VECTOR,
        top_k=5,
        filter={"scope": "MISRA C:2023"},
    )


async def test_embedding_called_with_query_text():
    config, mock_embed_svc, _, _ = _make_config(FAKE_VECTOR, {"matches": []})

    await rag_node(_make_state("array bounds check"), config)

    mock_embed_svc.get_embedding.assert_called_once_with("array bounds check")


async def test_empty_query_string_is_handled():
    """State with no 'query' key should default to empty string without crash."""
    config, _, _, _ = _make_config(FAKE_VECTOR, {"matches": []})

    result = await rag_node({}, config)

    assert result["rag_query_used"] == ""
    assert result["retrieved_rules"] == []


# ---------------------------------------------------------------------------
# None-guard tests (lines 22-27)
# ---------------------------------------------------------------------------


async def test_raises_value_error_if_embedding_service_is_none():
    config = {"configurable": {"embedding_service": None, "pinecone_service": MagicMock(), "mongo_db": MagicMock()}}
    with pytest.raises(ValueError, match="embedding_service is not configured"):
        await rag_node(_make_state(), config)


async def test_raises_value_error_if_pinecone_service_is_none():
    config = {"configurable": {"embedding_service": MagicMock(), "pinecone_service": None, "mongo_db": MagicMock()}}
    with pytest.raises(ValueError, match="pinecone_service is not configured"):
        await rag_node(_make_state(), config)


async def test_raises_value_error_if_mongo_db_is_none():
    config = {"configurable": {"embedding_service": MagicMock(), "pinecone_service": MagicMock(), "mongo_db": None}}
    with pytest.raises(ValueError, match="mongo_db is not configured"):
        await rag_node(_make_state(), config)
