# tests/unit/graph/nodes/test_remedier.py

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from app.graph.nodes.remedier import remediate_code, RemediationOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    code_snippet: str = "char *p = malloc(n);",
    validation_result: str = "Rule 21.3 violated: malloc used.",
    cited_rules: list[str] | None = None,
    retrieved_rules: list[dict] | None = None,
) -> dict:
    return {
        "code_snippet": code_snippet,
        "validation_result": validation_result,
        "cited_rules": cited_rules or [],
        "retrieved_rules": retrieved_rules or [],
    }


def _make_rule(rule_id: str, category: str = "Required", title: str = "A rule", full_text: str = "Rule text.") -> dict:
    return {"rule_id": rule_id, "category": category, "title": title, "full_text": full_text}


def _mock_structured_llm(
    parsed: RemediationOutput | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> MagicMock:
    """Mock for get_structured_llm() — returns a Runnable with .ainvoke() directly."""
    raw = MagicMock()
    raw.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}

    structured_chain = MagicMock()
    structured_chain.ainvoke = AsyncMock(return_value={
        "raw": raw,
        "parsed": parsed,
        "parsing_error": None if parsed else ValueError("parse failed"),
    })
    return structured_chain


VALID_OUTPUT = RemediationOutput(
    fixed_code_snippet="char *p = NULL; p = malloc(n); if (!p) return -1;",
    remediation_explanation="Rule 21.3 (Required): malloc used without null check → added null check.",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("app.graph.nodes.remedier.get_structured_llm")
async def test_happy_path_returns_fixed_code(mock_get_structured_llm):
    mock_get_structured_llm.return_value = _mock_structured_llm(VALID_OUTPUT)

    state = _make_state(cited_rules=["MISRA-21.3"], retrieved_rules=[_make_rule("MISRA-21.3")])
    result = await remediate_code(state)

    assert result["fixed_code_snippet"] == VALID_OUTPUT.fixed_code_snippet
    assert result["remediation_explanation"] == VALID_OUTPUT.remediation_explanation


@patch("app.graph.nodes.remedier.get_structured_llm")
async def test_returns_correct_state_keys(mock_get_structured_llm):
    mock_get_structured_llm.return_value = _mock_structured_llm(VALID_OUTPUT)

    result = await remediate_code(_make_state())

    expected_keys = {
        "fixed_code_snippet",
        "remediation_explanation",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "remediation_tokens",
        "estimated_cost",
    }
    assert set(result.keys()) == expected_keys


@patch("app.graph.nodes.remedier.get_structured_llm")
async def test_parse_failure_returns_original_code(mock_get_structured_llm):
    mock_get_structured_llm.return_value = _mock_structured_llm(parsed=None)

    code = "int x = 1;"
    result = await remediate_code(_make_state(code_snippet=code))

    assert result["fixed_code_snippet"] == code
    assert "System failed to generate a fix automatically" in result["remediation_explanation"]
    assert result["prompt_tokens"] == 0
    assert result["completion_tokens"] == 0
    assert result["estimated_cost"] == 0.0


@patch("app.graph.nodes.remedier.get_structured_llm")
async def test_only_cited_rules_appear_in_llm_context(mock_get_structured_llm):
    """Only cited rules should be forwarded in the prompt; uncited rules must be excluded."""
    mock_get_structured_llm.return_value = _mock_structured_llm(VALID_OUTPUT)

    state = _make_state(
        cited_rules=["MISRA-1.1"],
        retrieved_rules=[
            _make_rule("MISRA-1.1", title="Cited rule"),
            _make_rule("MISRA-2.2", title="Uncited rule"),
        ],
    )
    await remediate_code(state)

    call_args = mock_get_structured_llm.return_value.ainvoke.call_args[0][0]
    human_content = next(
        m.content for m in call_args
        if hasattr(m, "content") and ("Cited rule" in m.content or "Uncited rule" in m.content)
    )
    assert "Cited rule" in human_content
    assert "Uncited rule" not in human_content


@patch("app.graph.nodes.remedier.get_structured_llm")
async def test_fallback_to_all_rules_when_no_cited_match(mock_get_structured_llm):
    """When cited_rules IDs don't match retrieved_rules, all rules are shown as fallback."""
    mock_get_structured_llm.return_value = _mock_structured_llm(VALID_OUTPUT)

    state = _make_state(
        cited_rules=["MISRA-99.99"],  # no match
        retrieved_rules=[_make_rule("MISRA-1.1", title="Fallback rule")],
    )
    await remediate_code(state)

    call_args = mock_get_structured_llm.return_value.ainvoke.call_args[0][0]
    human_content = next(m.content for m in call_args if hasattr(m, "content") and ("Fallback rule" in m.content or "No rule details" in m.content))
    assert "Fallback rule" in human_content


@patch("app.graph.nodes.remedier.get_structured_llm")
async def test_no_retrieved_rules_uses_default_message(mock_get_structured_llm):
    """When retrieved_rules is empty and cited_rules is empty, fallback text is used."""
    mock_get_structured_llm.return_value = _mock_structured_llm(VALID_OUTPUT)

    state = _make_state(cited_rules=[], retrieved_rules=[])
    await remediate_code(state)

    call_args = mock_get_structured_llm.return_value.ainvoke.call_args[0][0]
    human_content = next(m.content for m in call_args if hasattr(m, "content") and "No rule details available" in m.content)
    assert "No rule details available." in human_content


@patch("app.graph.nodes.remedier.get_structured_llm")
async def test_empty_state_does_not_crash(mock_get_structured_llm):
    mock_get_structured_llm.return_value = _mock_structured_llm(VALID_OUTPUT)

    result = await remediate_code({})

    assert "fixed_code_snippet" in result


@patch("app.graph.nodes.remedier.get_structured_llm")
async def test_token_counts_from_usage_metadata(mock_get_structured_llm):
    mock_get_structured_llm.return_value = _mock_structured_llm(VALID_OUTPUT, input_tokens=200, output_tokens=80)

    result = await remediate_code(_make_state())

    assert result["prompt_tokens"] == 200
    assert result["completion_tokens"] == 80
    assert result["total_tokens"] == 280
    assert result["remediation_tokens"] == 280


@patch("app.graph.nodes.remedier.get_structured_llm")
async def test_missing_usage_metadata_defaults_to_zero(mock_get_structured_llm):
    raw = MagicMock(spec=[])  # no usage_metadata attribute
    structured_chain = MagicMock()
    structured_chain.ainvoke = AsyncMock(return_value={
        "raw": raw,
        "parsed": VALID_OUTPUT,
        "parsing_error": None,
    })
    mock_get_structured_llm.return_value = structured_chain

    result = await remediate_code(_make_state())

    assert result["prompt_tokens"] == 0
    assert result["completion_tokens"] == 0


@patch("app.graph.nodes.remedier.get_structured_llm")
async def test_get_structured_llm_called_with_remediation_output_schema(mock_get_structured_llm):
    mock_get_structured_llm.return_value = _mock_structured_llm(VALID_OUTPUT)

    await remediate_code(_make_state())

    mock_get_structured_llm.assert_called_once()
    assert mock_get_structured_llm.call_args[0][0] == RemediationOutput


@patch("app.graph.nodes.remedier.get_structured_llm")
async def test_llm_invoked_with_system_and_human_messages(mock_get_structured_llm):
    from langchain_core.messages import SystemMessage, HumanMessage

    mock_get_structured_llm.return_value = _mock_structured_llm(VALID_OUTPUT)

    await remediate_code(_make_state())

    messages = mock_get_structured_llm.return_value.ainvoke.call_args[0][0]
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)


@patch("app.graph.nodes.remedier.get_structured_llm")
async def test_human_message_contains_code_snippet(mock_get_structured_llm):
    mock_get_structured_llm.return_value = _mock_structured_llm(VALID_OUTPUT)

    code = "volatile int *ptr = (int*)0xDEAD;"
    await remediate_code(_make_state(code_snippet=code))

    messages = mock_get_structured_llm.return_value.ainvoke.call_args[0][0]
    assert code in messages[1].content


@patch("app.graph.nodes.remedier.get_structured_llm")
async def test_human_message_contains_validation_result(mock_get_structured_llm):
    mock_get_structured_llm.return_value = _mock_structured_llm(VALID_OUTPUT)

    validation = "Violated Rule 11.3: cast from int to pointer."
    await remediate_code(_make_state(validation_result=validation))

    messages = mock_get_structured_llm.return_value.ainvoke.call_args[0][0]
    assert validation in messages[1].content


@patch("app.graph.nodes.remedier.calculate_gemini_cost")
@patch("app.graph.nodes.remedier.get_structured_llm")
async def test_estimated_cost_uses_calculate_gemini_cost(mock_get_structured_llm, mock_cost):
    mock_get_structured_llm.return_value = _mock_structured_llm(VALID_OUTPUT, input_tokens=100, output_tokens=50)
    mock_cost.return_value = 0.0042

    result = await remediate_code(_make_state())

    mock_cost.assert_called_with(100, 50)
    assert result["estimated_cost"] == 0.0042
