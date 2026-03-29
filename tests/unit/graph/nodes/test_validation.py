from unittest.mock import MagicMock, patch
from app.graph.nodes.validation import validation_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_llm(content: str) -> MagicMock:
    llm = MagicMock()
    response = MagicMock()
    response.content = content
    llm.invoke.return_value = response
    return llm


def _base_state(**overrides) -> dict:
    state = {
        "query": "Check for memory issues",
        "code_snippet": "char *p = malloc(n);",
        "retrieved_rules": [
            {
                "rule_id": "MISRA_21.3",
                "title": "Memory allocation",
                "full_text": "The memory allocation and deallocation functions...",
                "dal_level": "Required",
            }
        ],
        "critique_feedback": "",
        "iteration_count": 0,
    }
    state.update(overrides)
    return state


VALID_JSON = (
    '{"is_compliant": false, "validation_result": "malloc used",'
    ' "confidence_score": 0.9, "cited_rules": ["MISRA_21.3"]}'
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_valid_response_maps_all_fields():
    with patch("app.graph.nodes.validation.get_llm", return_value=_mock_llm(VALID_JSON)):
        result = validation_node(_base_state())

    assert result["is_compliant"] is False
    assert result["validation_result"] == "malloc used"
    assert result["confidence_score"] == 0.9
    assert result["cited_rules"] == ["MISRA_21.3"]


def test_iteration_count_incremented_from_zero():
    with patch("app.graph.nodes.validation.get_llm", return_value=_mock_llm(VALID_JSON)):
        result = validation_node(_base_state(iteration_count=0))

    assert result["iteration_count"] == 1


def test_iteration_count_incremented_from_nonzero():
    with patch("app.graph.nodes.validation.get_llm", return_value=_mock_llm(VALID_JSON)):
        result = validation_node(_base_state(iteration_count=2))

    assert result["iteration_count"] == 3


def test_parse_failure_returns_error_defaults():
    with patch("app.graph.nodes.validation.get_llm", return_value=_mock_llm("NOT JSON AT ALL")):
        result = validation_node(_base_state())

    assert result["is_compliant"] is False
    assert result["confidence_score"] == 0.0
    assert result["cited_rules"] == []
    assert "failed" in result["validation_result"].lower()


def test_parse_failure_still_increments_iteration():
    with patch("app.graph.nodes.validation.get_llm", return_value=_mock_llm("BAD")):
        result = validation_node(_base_state(iteration_count=3))

    assert result["iteration_count"] == 4


def test_fenced_json_is_parsed():
    fenced = (
        "```json\n"
        '{"is_compliant": true, "validation_result": "ok",'
        ' "confidence_score": 1.0, "cited_rules": []}\n'
        "```"
    )
    with patch("app.graph.nodes.validation.get_llm", return_value=_mock_llm(fenced)):
        result = validation_node(_base_state())

    assert result["is_compliant"] is True
    assert result["confidence_score"] == 1.0


def test_critique_feedback_on_second_iteration_does_not_crash():
    with patch("app.graph.nodes.validation.get_llm", return_value=_mock_llm(VALID_JSON)):
        result = validation_node(
            _base_state(iteration_count=1, critique_feedback="Rule hallucination found.")
        )

    assert result["iteration_count"] == 2


def test_empty_retrieved_rules_no_crash():
    with patch("app.graph.nodes.validation.get_llm", return_value=_mock_llm(VALID_JSON)):
        result = validation_node(_base_state(retrieved_rules=[]))

    assert "validation_result" in result


def test_missing_json_fields_default_to_safe_values():
    partial = '{"is_compliant": true}'
    with patch("app.graph.nodes.validation.get_llm", return_value=_mock_llm(partial)):
        result = validation_node(_base_state())

    assert result["validation_result"] == ""
    assert result["confidence_score"] == 0.0
    assert result["cited_rules"] == []


def test_get_llm_called_with_point_one_temperature():
    with patch("app.graph.nodes.validation.get_llm", return_value=_mock_llm(VALID_JSON)) as mock_get_llm:
        validation_node(_base_state())

    mock_get_llm.assert_called_once_with(temperature=0.1)
