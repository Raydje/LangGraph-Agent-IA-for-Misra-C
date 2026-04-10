import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.graph.nodes.orchestrator import OrchestratorOutput, orchestrate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_result(intent: str, reasoning: str) -> dict:
    """Build the dict that chain.ainvoke() returns when include_raw=True."""
    raw_mock = MagicMock()
    raw_mock.usage_metadata = {"input_tokens": 0, "output_tokens": 0}
    return {
        "parsed": OrchestratorOutput(intent=intent, reasoning=reasoning),
        "raw": raw_mock,
    }


def _setup_mocks(mock_get_structured_llm, mock_template, intent: str, reasoning: str) -> MagicMock:
    """
    Wire up the mock chain:
      get_structured_llm() -> structured_llm_mock
      ChatPromptTemplate.from_messages(...) -> prompt_mock
      prompt_mock | structured_llm_mock -> chain
      chain.ainvoke(...) -> {"parsed": ..., "raw": ...}
    """
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value=_make_raw_result(intent, reasoning))

    structured_llm_mock = MagicMock()
    mock_get_structured_llm.return_value = structured_llm_mock

    prompt_mock = MagicMock()
    prompt_mock.__or__ = MagicMock(return_value=chain)
    mock_template.from_messages.return_value = prompt_mock

    return chain


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_returns_intent_and_reasoning_from_llm():
    with (
        patch("app.graph.nodes.orchestrator.get_structured_llm") as mock_get_structured_llm,
        patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template,
    ):
        _setup_mocks(mock_get_structured_llm, mock_template, "search", "User is asking about rules.")

        result = await orchestrate({"query": "What are pointer rules?", "code_snippet": ""})

    assert result["intent"] == "search"
    assert result["orchestrator_reasoning"] == "User is asking about rules."


async def test_standard_is_always_hardcoded_to_misra():
    with (
        patch("app.graph.nodes.orchestrator.get_structured_llm") as mock_get_structured_llm,
        patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template,
    ):
        _setup_mocks(mock_get_structured_llm, mock_template, "validate", "Code snippet present.")

        result = await orchestrate({"query": "Check this code", "code_snippet": "int x = 0;"})

    assert result["standard"] == "MISRA C:2023"


async def test_explain_intent_propagated():
    with (
        patch("app.graph.nodes.orchestrator.get_structured_llm") as mock_get_structured_llm,
        patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template,
    ):
        _setup_mocks(mock_get_structured_llm, mock_template, "explain", "User wants an explanation.")

        result = await orchestrate({"query": "Explain rule 15.5", "code_snippet": ""})

    assert result["intent"] == "explain"


async def test_returns_exactly_three_state_keys():
    """Verify that at minimum the three LangGraph-relevant state keys are present."""
    with (
        patch("app.graph.nodes.orchestrator.get_structured_llm") as mock_get_structured_llm,
        patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template,
    ):
        _setup_mocks(mock_get_structured_llm, mock_template, "search", "reason")

        result = await orchestrate({"query": "Find rules", "code_snippet": ""})

    assert {"intent", "orchestrator_reasoning", "standard"}.issubset(result.keys())


async def test_chain_invoked_with_query_and_code():
    with (
        patch("app.graph.nodes.orchestrator.get_structured_llm") as mock_get_structured_llm,
        patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template,
    ):
        chain = _setup_mocks(mock_get_structured_llm, mock_template, "validate", "Code provided.")

        await orchestrate({"query": "Validate code", "code_snippet": "void foo() {}"})

    chain.ainvoke.assert_called_once()
    call_kwargs = chain.ainvoke.call_args[0][0]
    assert call_kwargs["query"] == "Validate code"
    assert call_kwargs["code"] == "void foo() {}"


async def test_no_code_snippet_passes_none_provided_string():
    with (
        patch("app.graph.nodes.orchestrator.get_structured_llm") as mock_get_structured_llm,
        patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template,
    ):
        chain = _setup_mocks(mock_get_structured_llm, mock_template, "search", "No code.")

        await orchestrate({"query": "Find memory rules", "code_snippet": ""})

    call_kwargs = chain.ainvoke.call_args[0][0]
    assert call_kwargs["code"] == "None provided."


async def test_get_structured_llm_called_with_orchestrator_output_schema():
    with (
        patch("app.graph.nodes.orchestrator.get_structured_llm") as mock_get_structured_llm,
        patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template,
    ):
        _setup_mocks(mock_get_structured_llm, mock_template, "search", "r")

        await orchestrate({"query": "q", "code_snippet": ""})

    mock_get_structured_llm.assert_called_once()
    assert mock_get_structured_llm.call_args[0][0] == OrchestratorOutput


async def test_timeout_returns_default_search_intent():
    """asyncio.TimeoutError during chain.ainvoke → default 'search' response with zeros."""
    with (
        patch("app.graph.nodes.orchestrator.get_structured_llm") as mock_get_structured_llm,
        patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template,
        patch("app.graph.nodes.orchestrator.asyncio.wait_for", side_effect=asyncio.TimeoutError),
    ):
        _setup_mocks(mock_get_structured_llm, mock_template, "search", "r")
        result = await orchestrate({"query": "q", "code_snippet": ""})

    assert result["intent"] == "search"
    assert "timed out" in result["orchestrator_reasoning"].lower()
    assert result["prompt_tokens"] == 0
    assert result["total_tokens"] == 0
    assert result["estimated_cost"] == 0.0


async def test_parse_failure_returns_default_search_intent():
    """When chain returns {'parsed': None}, the ValueError path returns search defaults."""
    with (
        patch("app.graph.nodes.orchestrator.get_structured_llm") as mock_get_structured_llm,
        patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template,
    ):
        chain = _setup_mocks(mock_get_structured_llm, mock_template, "search", "r")
        # Override ainvoke to return a result with parsed=None
        chain.ainvoke = AsyncMock(return_value={"parsed": None, "raw": MagicMock()})
        result = await orchestrate({"query": "q", "code_snippet": ""})

    assert result["intent"] == "search"
    assert "failed" in result["orchestrator_reasoning"].lower()
    assert result["prompt_tokens"] == 0
    assert result["estimated_cost"] == 0.0
