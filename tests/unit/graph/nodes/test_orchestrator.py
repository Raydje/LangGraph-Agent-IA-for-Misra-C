from unittest.mock import MagicMock, AsyncMock, patch
from app.graph.nodes.orchestrator import orchestrate, OrchestratorOutput


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


def _setup_mocks(mock_get_llm, mock_template, intent: str, reasoning: str) -> MagicMock:
    """
    Wire up the mock chain:
      get_llm() -> llm_mock
      llm_mock.with_structured_output(...) -> structured_llm_mock
      ChatPromptTemplate.from_messages(...) -> prompt_mock
      prompt_mock | structured_llm_mock -> chain
      chain.ainvoke(...) -> {"parsed": ..., "raw": ...}
    """
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value=_make_raw_result(intent, reasoning))

    structured_llm_mock = MagicMock()
    llm_mock = MagicMock()
    llm_mock.with_structured_output.return_value = structured_llm_mock
    mock_get_llm.return_value = llm_mock

    prompt_mock = MagicMock()
    prompt_mock.__or__ = MagicMock(return_value=chain)
    mock_template.from_messages.return_value = prompt_mock

    return chain


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_returns_intent_and_reasoning_from_llm():
    with patch("app.graph.nodes.orchestrator.get_llm") as mock_get_llm, \
         patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template:
        _setup_mocks(mock_get_llm, mock_template, "search", "User is asking about rules.")

        result = await orchestrate({"query": "What are pointer rules?", "code_snippet": ""})

    assert result["intent"] == "search"
    assert result["orchestrator_reasoning"] == "User is asking about rules."


async def test_standard_is_always_hardcoded_to_misra():
    with patch("app.graph.nodes.orchestrator.get_llm") as mock_get_llm, \
         patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template:
        _setup_mocks(mock_get_llm, mock_template, "validate", "Code snippet present.")

        result = await orchestrate({"query": "Check this code", "code_snippet": "int x = 0;"})

    assert result["standard"] == "MISRA C:2023"


async def test_explain_intent_propagated():
    with patch("app.graph.nodes.orchestrator.get_llm") as mock_get_llm, \
         patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template:
        _setup_mocks(mock_get_llm, mock_template, "explain", "User wants an explanation.")

        result = await orchestrate({"query": "Explain rule 15.5", "code_snippet": ""})

    assert result["intent"] == "explain"


async def test_returns_exactly_three_state_keys():
    """Verify that at minimum the three LangGraph-relevant state keys are present."""
    with patch("app.graph.nodes.orchestrator.get_llm") as mock_get_llm, \
         patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template:
        _setup_mocks(mock_get_llm, mock_template, "search", "reason")

        result = await orchestrate({"query": "Find rules", "code_snippet": ""})

    assert {"intent", "orchestrator_reasoning", "standard"}.issubset(result.keys())


async def test_chain_invoked_with_query_and_code():
    with patch("app.graph.nodes.orchestrator.get_llm") as mock_get_llm, \
         patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template:
        chain = _setup_mocks(mock_get_llm, mock_template, "validate", "Code provided.")

        await orchestrate({"query": "Validate code", "code_snippet": "void foo() {}"})

    chain.ainvoke.assert_called_once()
    call_kwargs = chain.ainvoke.call_args[0][0]
    assert call_kwargs["query"] == "Validate code"
    assert call_kwargs["code"] == "void foo() {}"


async def test_no_code_snippet_passes_none_provided_string():
    with patch("app.graph.nodes.orchestrator.get_llm") as mock_get_llm, \
         patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template:
        chain = _setup_mocks(mock_get_llm, mock_template, "search", "No code.")

        await orchestrate({"query": "Find memory rules", "code_snippet": ""})

    call_kwargs = chain.ainvoke.call_args[0][0]
    assert call_kwargs["code"] == "None provided."


async def test_get_llm_called_with_zero_temperature():
    with patch("app.graph.nodes.orchestrator.get_llm") as mock_get_llm, \
         patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template:
        _setup_mocks(mock_get_llm, mock_template, "search", "r")

        await orchestrate({"query": "q", "code_snippet": ""})

    mock_get_llm.assert_called_once_with(temperature=0.0)
