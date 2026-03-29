"""
Unit tests for assemble_node (defined inline in graph/builder.py).
No LLM calls — purely a state formatting function.
"""
from app.graph.builder import assemble_node


# --- Helpers ---

def _validate_state(compliant, confidence, cited, details, standard="MISRA C:2023", error=None):
    return {
        "intent": "validate",
        "is_compliant": compliant,
        "confidence_score": confidence,
        "cited_rules": cited,
        "validation_result": details,
        "standard": standard,
        "error": error,
    }


# --- validate intent ---

def test_assemble_compliant_verdict_contains_key_fields():
    state = _validate_state(True, 0.95, ["Rule 1.1", "Rule 2.3"], "All rules satisfied.")
    text = assemble_node(state)["final_response"]
    assert "Compliant: True" in text
    assert "95%" in text
    assert "Rule 1.1, Rule 2.3" in text
    assert "All rules satisfied." in text


def test_assemble_noncompliant_verdict():
    state = _validate_state(False, 0.80, ["Rule 15.5"], "Recursion detected.")
    text = assemble_node(state)["final_response"]
    assert "Compliant: False" in text
    assert "80%" in text
    assert "Rule 15.5" in text


def test_assemble_validate_empty_cited_rules_shows_none():
    state = _validate_state(True, 1.0, [], "Fully compliant.")
    text = assemble_node(state)["final_response"]
    assert "none" in text


def test_assemble_validate_standard_name_in_output():
    state = _validate_state(True, 0.9, [], "ok", standard="MISRA C:2023")
    text = assemble_node(state)["final_response"]
    assert "MISRA C:2023" in text


# --- search intent ---

def test_assemble_search_with_rules_lists_them():
    state = {
        "intent": "search",
        "retrieved_rules": [
            {"rule_id": "MISRA_1.1", "section": 1, "title": "No dead code"},
            {"rule_id": "MISRA_2.3", "section": 2, "title": "Unused vars"},
        ],
        "standard": "MISRA C:2023",
        "error": None,
    }
    text = assemble_node(state)["final_response"]
    assert "MISRA_1.1" in text
    assert "No dead code" in text
    assert "MISRA_2.3" in text


def test_assemble_search_no_rules_returns_not_found_message():
    state = {"intent": "search", "retrieved_rules": [], "standard": "MISRA C:2023", "error": None}
    text = assemble_node(state)["final_response"]
    assert "No matching rules found" in text


# --- explain intent ---

def test_assemble_explain_with_rules():
    state = {
        "intent": "explain",
        "retrieved_rules": [{"rule_id": "MISRA_15.5", "title": "No recursion"}],
        "standard": "MISRA C:2023",
        "error": None,
    }
    text = assemble_node(state)["final_response"]
    assert "MISRA_15.5" in text
    assert "No recursion" in text


def test_assemble_explain_no_rules_returns_not_found_message():
    state = {"intent": "explain", "retrieved_rules": [], "standard": "MISRA C:2023", "error": None}
    text = assemble_node(state)["final_response"]
    assert "No rules found" in text


# --- error state ---

def test_assemble_error_overrides_normal_logic():
    state = {"error": "Upstream service failed", "intent": "search"}
    text = assemble_node(state)["final_response"]
    assert "Upstream service failed" in text


def test_assemble_returns_final_response_key():
    state = {"intent": "search", "retrieved_rules": [], "standard": "MISRA C:2023", "error": None}
    result = assemble_node(state)
    assert "final_response" in result
