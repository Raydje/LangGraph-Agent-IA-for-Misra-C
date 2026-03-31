# tests/unit/graph/nodes/test_remedier.py

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from app.graph.nodes.remedier import remediate_code


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


def _mock_llm_response(content: dict, input_tokens: int = 100, output_tokens: int = 50) -> MagicMock:
    response = MagicMock()
    response.content = json.dumps(content)
    response.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    return response


def _mock_llm(response: MagicMock) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=response)
    return llm


VALID_RESULT = {
    "fixed_code_snippet": "char *p = NULL; p = malloc(n); if (!p) return -1;",
    "remediation_explanation": "Rule 21.3 (Required): malloc used without null check → added null check.",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("app.graph.nodes.remedier.get_llm")
async def test_happy_path_returns_fixed_code(mock_get_llm):
    mock_get_llm.return_value = _mock_llm(_mock_llm_response(VALID_RESULT))

    state = _make_state(cited_rules=["MISRA-21.3"], retrieved_rules=[_make_rule("MISRA-21.3")])
    result = await remediate_code(state)

    assert result["fixed_code_snippet"] == VALID_RESULT["fixed_code_snippet"]
    assert result["remediation_explanation"] == VALID_RESULT["remediation_explanation"]


@patch("app.graph.nodes.remedier.get_llm")
async def test_returns_correct_state_keys(mock_get_llm):
    mock_get_llm.return_value = _mock_llm(_mock_llm_response(VALID_RESULT))

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


@patch("app.graph.nodes.remedier.get_llm")
async def test_json_parse_failure_returns_original_code(mock_get_llm):
    bad_response = MagicMock()
    bad_response.content = "Not valid JSON at all"
    bad_response.usage_metadata = {}
    mock_get_llm.return_value = _mock_llm(bad_response)

    code = "int x = 1;"
    result = await remediate_code(_make_state(code_snippet=code))

    assert result["fixed_code_snippet"] == code
    assert "System failed to generate a fix automatically" in result["remediation_explanation"]
    assert result["prompt_tokens"] == 0
    assert result["completion_tokens"] == 0
    assert result["estimated_cost"] == 0.0


@patch("app.graph.nodes.remedier.get_llm")
async def test_json_parse_key_error_returns_original_code(mock_get_llm):
    """parse_json_response succeeds but result dict is missing expected keys — should not crash."""
    mock_get_llm.return_value = _mock_llm(_mock_llm_response({"unexpected": "keys"}))

    code = "int y = 0;"
    result = await remediate_code(_make_state(code_snippet=code))

    # .get() with default empty string means it returns gracefully with empty strings
    assert result["fixed_code_snippet"] == ""
    assert result["remediation_explanation"] == ""


@patch("app.graph.nodes.remedier.get_llm")
async def test_parse_json_raises_value_error_triggers_fallback(mock_get_llm):
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="irrelevant"))
    mock_get_llm.return_value = llm

    with patch("app.graph.nodes.remedier.parse_json_response", side_effect=ValueError("bad json")):
        result = await remediate_code(_make_state(code_snippet="int z;"))

    assert result["fixed_code_snippet"] == "int z;"
    assert "bad json" in result["remediation_explanation"]


@patch("app.graph.nodes.remedier.get_llm")
async def test_only_cited_rules_appear_in_llm_context(mock_get_llm):
    """Only cited rules should be forwarded in the prompt; uncited rules must be excluded."""
    mock_get_llm.return_value = _mock_llm(_mock_llm_response(VALID_RESULT))

    state = _make_state(
        cited_rules=["MISRA-1.1"],
        retrieved_rules=[
            _make_rule("MISRA-1.1", title="Cited rule"),
            _make_rule("MISRA-2.2", title="Uncited rule"),
        ],
    )
    await remediate_code(state)

    call_args = mock_get_llm.return_value.ainvoke.call_args[0][0]
    human_content = next(
        m.content for m in call_args
        if hasattr(m, "content") and ("Cited rule" in m.content or "Uncited rule" in m.content)
    )
    assert "Cited rule" in human_content
    assert "Uncited rule" not in human_content


@patch("app.graph.nodes.remedier.get_llm")
async def test_fallback_to_all_rules_when_no_cited_match(mock_get_llm):
    """When cited_rules IDs don't match retrieved_rules, all rules are shown as fallback."""
    mock_get_llm.return_value = _mock_llm(_mock_llm_response(VALID_RESULT))

    state = _make_state(
        cited_rules=["MISRA-99.99"],  # no match
        retrieved_rules=[_make_rule("MISRA-1.1", title="Fallback rule")],
    )
    await remediate_code(state)

    call_args = mock_get_llm.return_value.ainvoke.call_args[0][0]
    human_content = next(m.content for m in call_args if hasattr(m, "content") and ("Fallback rule" in m.content or "No rule details" in m.content))
    assert "Fallback rule" in human_content


@patch("app.graph.nodes.remedier.get_llm")
async def test_no_retrieved_rules_uses_default_message(mock_get_llm):
    """When retrieved_rules is empty and cited_rules is empty, fallback text is used."""
    mock_get_llm.return_value = _mock_llm(_mock_llm_response(VALID_RESULT))

    state = _make_state(cited_rules=[], retrieved_rules=[])
    await remediate_code(state)

    call_args = mock_get_llm.return_value.ainvoke.call_args[0][0]
    human_content = next(m.content for m in call_args if hasattr(m, "content") and "No rule details available" in m.content)
    assert "No rule details available." in human_content


@patch("app.graph.nodes.remedier.get_llm")
async def test_empty_state_does_not_crash(mock_get_llm):
    mock_get_llm.return_value = _mock_llm(_mock_llm_response(VALID_RESULT))

    result = await remediate_code({})

    assert "fixed_code_snippet" in result


@patch("app.graph.nodes.remedier.get_llm")
async def test_token_counts_from_usage_metadata(mock_get_llm):
    mock_get_llm.return_value = _mock_llm(_mock_llm_response(VALID_RESULT, input_tokens=200, output_tokens=80))

    result = await remediate_code(_make_state())

    assert result["prompt_tokens"] == 200
    assert result["completion_tokens"] == 80
    assert result["total_tokens"] == 280
    assert result["remediation_tokens"] == 280


@patch("app.graph.nodes.remedier.get_llm")
async def test_missing_usage_metadata_defaults_to_zero(mock_get_llm):
    response = MagicMock(spec=[])  # no usage_metadata attribute
    response.content = json.dumps(VALID_RESULT)
    mock_get_llm.return_value = _mock_llm(response)

    result = await remediate_code(_make_state())

    assert result["prompt_tokens"] == 0
    assert result["completion_tokens"] == 0


@patch("app.graph.nodes.remedier.get_llm")
async def test_llm_called_with_temperature_02(mock_get_llm):
    mock_get_llm.return_value = _mock_llm(_mock_llm_response(VALID_RESULT))

    await remediate_code(_make_state())

    mock_get_llm.assert_called_once_with(temperature=0.2)


@patch("app.graph.nodes.remedier.get_llm")
async def test_llm_invoked_with_system_and_human_messages(mock_get_llm):
    from langchain_core.messages import SystemMessage, HumanMessage

    mock_get_llm.return_value = _mock_llm(_mock_llm_response(VALID_RESULT))

    await remediate_code(_make_state())

    messages = mock_get_llm.return_value.ainvoke.call_args[0][0]
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)


@patch("app.graph.nodes.remedier.get_llm")
async def test_human_message_contains_code_snippet(mock_get_llm):
    mock_get_llm.return_value = _mock_llm(_mock_llm_response(VALID_RESULT))

    code = "volatile int *ptr = (int*)0xDEAD;"
    await remediate_code(_make_state(code_snippet=code))

    messages = mock_get_llm.return_value.ainvoke.call_args[0][0]
    assert code in messages[1].content


@patch("app.graph.nodes.remedier.get_llm")
async def test_human_message_contains_validation_result(mock_get_llm):
    mock_get_llm.return_value = _mock_llm(_mock_llm_response(VALID_RESULT))

    validation = "Violated Rule 11.3: cast from int to pointer."
    await remediate_code(_make_state(validation_result=validation))

    messages = mock_get_llm.return_value.ainvoke.call_args[0][0]
    assert validation in messages[1].content


@patch("app.graph.nodes.remedier.calculate_gemini_cost")
@patch("app.graph.nodes.remedier.get_llm")
async def test_estimated_cost_uses_calculate_gemini_cost(mock_get_llm, mock_cost):
    mock_get_llm.return_value = _mock_llm(_mock_llm_response(VALID_RESULT, input_tokens=100, output_tokens=50))
    mock_cost.return_value = 0.0042

    result = await remediate_code(_make_state())

    mock_cost.assert_called_with(100, 50)
    assert result["estimated_cost"] == 0.0042
