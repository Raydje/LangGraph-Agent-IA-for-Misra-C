from unittest.mock import MagicMock, patch
from app.graph.nodes.critique import critique_node


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
        "code_snippet": "int x = 0;",
        "retrieved_rules": [{"rule_id": "MISRA_1.1", "title": "No dead code"}],
        "validation_result": "Code is compliant with all retrieved rules.",
        "cited_rules": ["MISRA_1.1"],
        "is_compliant": True,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_approved_sets_critique_approved_true():
    response = '{"approved": true, "feedback": "Pass"}'
    with patch("app.graph.nodes.critique.get_llm", return_value=_mock_llm(response)):
        result = critique_node(_base_state())

    assert result["critique_approved"] is True
    assert result["critique_feedback"] == "Pass"


def test_rejected_sets_critique_approved_false_with_feedback():
    response = '{"approved": false, "feedback": "Criteria 1 failed: hallucinated rule."}'
    with patch("app.graph.nodes.critique.get_llm", return_value=_mock_llm(response)):
        result = critique_node(_base_state())

    assert result["critique_approved"] is False
    assert "Criteria 1 failed" in result["critique_feedback"]


def test_parse_failure_sets_not_approved():
    with patch("app.graph.nodes.critique.get_llm", return_value=_mock_llm("INVALID JSON")):
        result = critique_node(_base_state())

    assert result["critique_approved"] is False
    assert isinstance(result["critique_feedback"], str)
    assert len(result["critique_feedback"]) > 0


def test_fenced_json_is_parsed():
    response = '```json\n{"approved": true, "feedback": "Pass"}\n```'
    with patch("app.graph.nodes.critique.get_llm", return_value=_mock_llm(response)):
        result = critique_node(_base_state())

    assert result["critique_approved"] is True


def test_returns_only_expected_state_keys():
    response = '{"approved": true, "feedback": "Pass"}'
    with patch("app.graph.nodes.critique.get_llm", return_value=_mock_llm(response)):
        result = critique_node(_base_state())

    assert set(result.keys()) == {"critique_approved", "critique_feedback"}


def test_empty_retrieved_rules_no_crash():
    response = '{"approved": true, "feedback": "Pass"}'
    with patch("app.graph.nodes.critique.get_llm", return_value=_mock_llm(response)):
        result = critique_node(_base_state(retrieved_rules=[]))

    assert result["critique_approved"] is True


def test_non_compliant_state_no_crash():
    response = '{"approved": false, "feedback": "Logical inconsistency."}'
    with patch("app.graph.nodes.critique.get_llm", return_value=_mock_llm(response)):
        result = critique_node(
            _base_state(is_compliant=False, validation_result="Recursion found.")
        )

    assert result["critique_approved"] is False


def test_get_llm_called_with_zero_temperature():
    response = '{"approved": true, "feedback": "Pass"}'
    with patch("app.graph.nodes.critique.get_llm", return_value=_mock_llm(response)) as mock_get_llm:
        critique_node(_base_state())

    mock_get_llm.assert_called_once_with(temperature=0.0)
