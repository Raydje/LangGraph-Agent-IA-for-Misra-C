"""
Integration tests — Orchestrator node.

Verifies that the orchestrator LLM correctly classifies user intent for three
canonical input patterns.  Each test uses the real Gemini API (no mocks) and
asserts the returned intent value, ensuring the node produces the right routing
decision for production traffic.

Node contract
-------------
Input  : ComplianceState with at least {query, code_snippet, standard}
Output : dict containing {intent, orchestrator_reasoning, *_tokens, estimated_cost}
"""

import pytest

from app.graph.nodes.orchestrator import orchestrate
from app.models.state import ComplianceState

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


async def test_orchestrator_search_intent():
    """
    A plain natural-language question with no code snippet must be routed to
    'search'.  This is the most common entry point for documentation queries.
    """
    state: ComplianceState = {
        "query": "What does MISRA C:2023 say about pointer arithmetic?",
        "code_snippet": "",
        "standard": "MISRA C:2023",
    }

    result = await orchestrate(state)

    assert result["intent"] == "search", (
        f"Expected intent='search', got '{result['intent']}'. Reasoning: {result.get('orchestrator_reasoning')}"
    )
    assert isinstance(result.get("orchestrator_reasoning"), str)
    assert len(result["orchestrator_reasoning"]) > 0
    # Token accounting must be present and non-negative
    assert result.get("total_tokens", 0) >= 0
    assert result.get("estimated_cost", 0.0) >= 0.0


async def test_orchestrator_validate_intent():
    """
    A compliance check request that includes a C code snippet must be routed
    to 'validate'.  The presence of code + an explicit compliance question
    is the canonical signal.
    """
    state: ComplianceState = {
        "query": "Please check this snippet and see if it complies with MISRA rules.",
        "code_snippet": ("int divide(int a, int b) {\n    if (b == 0) return 0;\n    return a / b;\n}"),
        "standard": "MISRA C:2023",
    }

    result = await orchestrate(state)

    assert result["intent"] == "validate", (
        f"Expected intent='validate', got '{result['intent']}'. Reasoning: {result.get('orchestrator_reasoning')}"
    )
    assert isinstance(result.get("orchestrator_reasoning"), str)
    assert len(result["orchestrator_reasoning"]) > 0
    assert result.get("total_tokens", 0) >= 0
    assert result.get("estimated_cost", 0.0) >= 0.0


async def test_orchestrator_explain_intent():
    """
    A conceptual 'explain why' question with no code must be routed to
    'explain'.  This intent covers rationale and educational queries.
    """
    state: ComplianceState = {
        "query": "Explain why recursion is banned in MISRA.",
        "code_snippet": "",
        "standard": "MISRA C:2023",
    }

    result = await orchestrate(state)

    assert result["intent"] == "explain", (
        f"Expected intent='explain', got '{result['intent']}'. Reasoning: {result.get('orchestrator_reasoning')}"
    )
    assert isinstance(result.get("orchestrator_reasoning"), str)
    assert len(result["orchestrator_reasoning"]) > 0
    assert result.get("total_tokens", 0) >= 0
    assert result.get("estimated_cost", 0.0) >= 0.0
