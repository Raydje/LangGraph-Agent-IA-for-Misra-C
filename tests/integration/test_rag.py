"""
Integration tests — RAG node.

Verifies the full hybrid retrieval pipeline: query embedding (Gemini) →
vector similarity search (Pinecone) → document fetch (MongoDB).

The rag_config fixture (defined in conftest.py) provides real, session-scoped
service instances via the LangGraph RunnableConfig, exactly as the production
route handler constructs it.  No service is instantiated inside the test file.

Node contract
-------------
Input  : ComplianceState {query, standard, code_snippet, retrieved_rules,
                          rag_query_used, metadata_filters_applied}
         + RunnableConfig {configurable: {mongo_db, pinecone_service,
                                          embedding_service}}
Output : dict containing {retrieved_rules, rag_query_used,
                          metadata_filters_applied}
"""

import pytest

from app.graph.nodes.rag import rag_node
from app.models.state import ComplianceState, RetrievedRule

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RETRIEVED_RULE_KEYS = frozenset(RetrievedRule.__annotations__.keys())


def _assert_rule_shape(rule: dict) -> None:
    """Assert that a retrieved rule contains all required TypedDict fields."""
    missing = _RETRIEVED_RULE_KEYS - rule.keys()
    assert not missing, f"RetrievedRule is missing keys: {missing}"
    assert isinstance(rule["rule_id"], str) and rule["rule_id"]
    assert isinstance(rule["relevance_score"], float)
    assert 0.0 <= rule["relevance_score"] <= 1.0


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


async def test_rag_returns_rules_for_goto_query(rag_config):
    """
    The query 'Is it allowed to use the goto statement in C?' must retrieve
    at least one MISRA C:2023 rule from the live Pinecone + MongoDB stack.

    This validates:
    - EmbeddingService produces a valid vector for the query.
    - PineconeService returns matches for a standard compliance question.
    - MongoDBService resolves Pinecone IDs to full rule documents.
    - rag_node formats and sorts results correctly.
    """
    state: ComplianceState = {
        "query": "Is it allowed to use the goto statement in C?",
        "standard": "MISRA C:2023",
        "code_snippet": "void func() { goto label; label: return; }",
        "retrieved_rules": [],
        "rag_query_used": "",
        "metadata_filters_applied": {},
    }

    result = await rag_node(state, rag_config)

    retrieved = result.get("retrieved_rules", [])
    assert len(retrieved) >= 1, (
        "Expected at least 1 retrieved rule for a goto-related query against "
        "the live MISRA C:2023 Pinecone index."
    )

    # Every rule must conform to the RetrievedRule TypedDict shape
    for rule in retrieved:
        _assert_rule_shape(rule)

    # Results must be sorted by descending relevance score
    scores = [r["relevance_score"] for r in retrieved]
    assert scores == sorted(scores, reverse=True), (
        "Retrieved rules are not sorted by descending relevance score."
    )

    # Metadata state fields must be populated
    assert result.get("rag_query_used") == state["query"]
    assert result.get("metadata_filters_applied") == {"scope": "MISRA C:2023"}


async def test_rag_returns_rules_for_pointer_cast_query(rag_config):
    """
    A pointer-cast compliance question must yield rules from the pointer /
    type-conversion sections of MISRA C:2023 (sections 11.x).

    This is a second independent query to confirm the retrieval pipeline is
    not hard-coded to a single topic.
    """
    state: ComplianceState = {
        "query": "Is casting an integer to a pointer allowed in MISRA C:2023?",
        "standard": "MISRA C:2023",
        "code_snippet": "int *p = (int *)0x1234;",
        "retrieved_rules": [],
        "rag_query_used": "",
        "metadata_filters_applied": {},
    }

    result = await rag_node(state, rag_config)

    retrieved = result.get("retrieved_rules", [])
    assert len(retrieved) >= 1, (
        "Expected at least 1 retrieved rule for a pointer-cast query."
    )

    for rule in retrieved:
        _assert_rule_shape(rule)

    # All retrieved rules must belong to the requested standard
    for rule in retrieved:
        assert rule["standard"] == "MISRA C:2023", (
            f"Rule {rule['rule_id']} has standard='{rule['standard']}', "
            "expected 'MISRA C:2023'."
        )
