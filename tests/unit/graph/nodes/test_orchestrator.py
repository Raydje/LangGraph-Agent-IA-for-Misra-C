from unittest.mock import MagicMock, patch
from app.graph.nodes.orchestrator import orchestrate, OrchestratorOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_chain(intent: str, reasoning: str) -> MagicMock:
    chain = MagicMock()
    chain.invoke.return_value = OrchestratorOutput(intent=intent, reasoning=reasoning)
    return chain


def _patch_template(mock_template: MagicMock, chain: MagicMock) -> None:
    """Make ChatPromptTemplate.from_messages(...) | llm return our chain."""
    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=chain)
    mock_template.from_messages.return_value = mock_prompt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_intent_and_reasoning_from_llm():
    chain = _make_mock_chain("search", "User is asking about rules.")
    with patch("app.graph.nodes.orchestrator.get_structured_llm") as mock_llm, \
         patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template:
        _patch_template(mock_template, chain)
        mock_llm.return_value = MagicMock()

        result = orchestrate({"query": "What are pointer rules?", "code_snippet": ""})

    assert result["intent"] == "search"
    assert result["orchestrator_reasoning"] == "User is asking about rules."


def test_standard_is_always_hardcoded_to_misra():
    chain = _make_mock_chain("validate", "Code snippet present.")
    with patch("app.graph.nodes.orchestrator.get_structured_llm"), \
         patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template:
        _patch_template(mock_template, chain)

        result = orchestrate({"query": "Check this code", "code_snippet": "int x = 0;"})

    assert result["standard"] == "MISRA C:2023"


def test_explain_intent_propagated():
    chain = _make_mock_chain("explain", "User wants an explanation.")
    with patch("app.graph.nodes.orchestrator.get_structured_llm"), \
         patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template:
        _patch_template(mock_template, chain)

        result = orchestrate({"query": "Explain rule 15.5", "code_snippet": ""})

    assert result["intent"] == "explain"


def test_returns_exactly_three_state_keys():
    chain = _make_mock_chain("search", "reason")
    with patch("app.graph.nodes.orchestrator.get_structured_llm"), \
         patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template:
        _patch_template(mock_template, chain)

        result = orchestrate({"query": "Find rules", "code_snippet": ""})

    assert set(result.keys()) == {"intent", "orchestrator_reasoning", "standard"}


def test_chain_invoked_with_query_and_code():
    chain = _make_mock_chain("validate", "Code provided.")
    with patch("app.graph.nodes.orchestrator.get_structured_llm"), \
         patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template:
        _patch_template(mock_template, chain)

        orchestrate({"query": "Validate code", "code_snippet": "void foo() {}"})

    chain.invoke.assert_called_once()
    call_kwargs = chain.invoke.call_args[0][0]
    assert call_kwargs["query"] == "Validate code"
    assert call_kwargs["code"] == "void foo() {}"


def test_no_code_snippet_passes_none_provided_string():
    chain = _make_mock_chain("search", "No code.")
    with patch("app.graph.nodes.orchestrator.get_structured_llm"), \
         patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template:
        _patch_template(mock_template, chain)

        orchestrate({"query": "Find memory rules", "code_snippet": ""})

    call_kwargs = chain.invoke.call_args[0][0]
    assert call_kwargs["code"] == "None provided."


def test_get_structured_llm_called_with_zero_temperature():
    chain = _make_mock_chain("search", "r")
    with patch("app.graph.nodes.orchestrator.get_structured_llm") as mock_get_llm, \
         patch("app.graph.nodes.orchestrator.ChatPromptTemplate") as mock_template:
        _patch_template(mock_template, chain)
        mock_get_llm.return_value = MagicMock()

        orchestrate({"query": "q", "code_snippet": ""})

    mock_get_llm.assert_called_once_with(OrchestratorOutput, temperature=0.0)
