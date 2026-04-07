import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.graph.nodes.validation import ValidationOutput, validation_node

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_structured_llm(
    parsed: ValidationOutput | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> MagicMock:
    """Mock for get_structured_llm() — returns a Runnable with .ainvoke() directly."""
    raw = MagicMock()
    raw.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}

    structured_chain = MagicMock()
    structured_chain.ainvoke = AsyncMock(
        return_value={
            "raw": raw,
            "parsed": parsed,
            "parsing_error": None if parsed else ValueError("parse failed"),
        }
    )
    return structured_chain


def _base_state(**overrides) -> dict:
    state = {
        "query": "Check for memory issues",
        "code_snippet": "char *p = malloc(n);",
        "retrieved_rules": [
            {
                "rule_id": "MISRA_21.3",
                "title": "Memory allocation",
                "full_text": "The memory allocation and deallocation functions...",
                "category": "Required",
            }
        ],
        "critique_feedback": "",
        "iteration_count": 0,
    }
    state.update(overrides)
    return state


VALID_OUTPUT = ValidationOutput(
    is_compliant=False,
    validation_result="malloc used",
    confidence_score=0.9,
    cited_rules=["MISRA_21.3"],
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_valid_response_maps_all_fields():
    with patch("app.graph.nodes.validation.get_structured_llm", return_value=_mock_structured_llm(VALID_OUTPUT)):
        result = await validation_node(_base_state())

    assert result["is_compliant"] is False
    assert result["validation_result"] == "malloc used"
    assert result["confidence_score"] == 0.9
    assert result["cited_rules"] == ["MISRA_21.3"]


async def test_iteration_count_incremented_from_zero():
    with patch("app.graph.nodes.validation.get_structured_llm", return_value=_mock_structured_llm(VALID_OUTPUT)):
        result = await validation_node(_base_state(iteration_count=0))

    assert result["iteration_count"] == 1


async def test_iteration_count_incremented_from_nonzero():
    with patch("app.graph.nodes.validation.get_structured_llm", return_value=_mock_structured_llm(VALID_OUTPUT)):
        result = await validation_node(_base_state(iteration_count=2))

    assert result["iteration_count"] == 3


async def test_parse_failure_returns_error_defaults():
    with patch("app.graph.nodes.validation.get_structured_llm", return_value=_mock_structured_llm(parsed=None)):
        result = await validation_node(_base_state())

    assert result["is_compliant"] is False
    assert result["confidence_score"] == 0.0
    assert result["cited_rules"] == []
    assert "failed" in result["validation_result"].lower()


async def test_parse_failure_still_increments_iteration():
    with patch("app.graph.nodes.validation.get_structured_llm", return_value=_mock_structured_llm(parsed=None)):
        result = await validation_node(_base_state(iteration_count=3))

    assert result["iteration_count"] == 4


async def test_critique_feedback_on_second_iteration_does_not_crash():
    with patch("app.graph.nodes.validation.get_structured_llm", return_value=_mock_structured_llm(VALID_OUTPUT)):
        result = await validation_node(_base_state(iteration_count=1, critique_feedback="Rule hallucination found."))

    assert result["iteration_count"] == 2


async def test_empty_retrieved_rules_no_crash():
    with patch("app.graph.nodes.validation.get_structured_llm", return_value=_mock_structured_llm(VALID_OUTPUT)):
        result = await validation_node(_base_state(retrieved_rules=[]))

    assert "validation_result" in result


async def test_get_structured_llm_called_with_validation_output_schema():
    with patch("app.graph.nodes.validation.get_structured_llm") as mock_get_structured_llm:
        mock_get_structured_llm.return_value = _mock_structured_llm(VALID_OUTPUT)
        await validation_node(_base_state())

    mock_get_structured_llm.assert_called_once()
    assert mock_get_structured_llm.call_args[0][0] == ValidationOutput


async def test_token_counts_propagated():
    with patch(
        "app.graph.nodes.validation.get_structured_llm",
        return_value=_mock_structured_llm(VALID_OUTPUT, input_tokens=100, output_tokens=50),
    ):
        result = await validation_node(_base_state())

    assert result["prompt_tokens"] == 100
    assert result["completion_tokens"] == 50
    assert result["total_tokens"] == 150
    assert result["validation_tokens"] == 150


async def test_timeout_returns_not_compliant_and_zero_tokens():
    """asyncio.TimeoutError during LLM call → is_compliant False, all tokens zero."""
    with (
        patch("app.graph.nodes.validation.get_structured_llm", return_value=_mock_structured_llm(VALID_OUTPUT)),
        patch("app.graph.nodes.validation.asyncio.wait_for", side_effect=asyncio.TimeoutError),
    ):
        result = await validation_node(_base_state())

    assert result["is_compliant"] is False
    assert "timed out" in result["validation_result"].lower()
    assert result["confidence_score"] == 0.0
    assert result["cited_rules"] == []
    assert result["estimated_cost"] == 0.0
