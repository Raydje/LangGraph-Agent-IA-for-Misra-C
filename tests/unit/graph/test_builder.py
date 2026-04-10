"""
Unit tests for assemble_node (defined inline in graph/builder.py).
No LLM calls — purely a state formatting function.
"""

from unittest.mock import MagicMock, patch

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


async def test_assemble_compliant_verdict_contains_key_fields():
    state = _validate_state(True, 0.95, ["Rule 1.1", "Rule 2.3"], "All rules satisfied.")
    text = (await assemble_node(state))["final_response"]
    assert "Compliant: True" in text
    assert "95%" in text
    assert "Rule 1.1, Rule 2.3" in text
    assert "All rules satisfied." in text


async def test_assemble_noncompliant_verdict():
    state = _validate_state(False, 0.80, ["Rule 15.5"], "Recursion detected.")
    text = (await assemble_node(state))["final_response"]
    assert "Compliant: False" in text
    assert "80%" in text
    assert "Rule 15.5" in text


async def test_assemble_validate_empty_cited_rules_shows_none():
    state = _validate_state(True, 1.0, [], "Fully compliant.")
    text = (await assemble_node(state))["final_response"]
    assert "none" in text


async def test_assemble_validate_standard_name_in_output():
    state = _validate_state(True, 0.9, [], "ok", standard="MISRA C:2023")
    text = (await assemble_node(state))["final_response"]
    assert "MISRA C:2023" in text


# --- search intent ---


async def test_assemble_search_with_rules_lists_them():
    state = {
        "intent": "search",
        "retrieved_rules": [
            {"rule_id": "MISRA_1.1", "section": 1, "title": "No dead code"},
            {"rule_id": "MISRA_2.3", "section": 2, "title": "Unused vars"},
        ],
        "standard": "MISRA C:2023",
        "error": None,
    }
    text = (await assemble_node(state))["final_response"]
    assert "MISRA_1.1" in text
    assert "No dead code" in text
    assert "MISRA_2.3" in text


async def test_assemble_search_no_rules_returns_not_found_message():
    state = {"intent": "search", "retrieved_rules": [], "standard": "MISRA C:2023", "error": None}
    text = (await assemble_node(state))["final_response"]
    assert "No matching rules found" in text


# --- explain intent ---


async def test_assemble_explain_with_rules():
    state = {
        "intent": "explain",
        "retrieved_rules": [{"rule_id": "MISRA_15.5", "title": "No recursion"}],
        "standard": "MISRA C:2023",
        "error": None,
    }
    text = (await assemble_node(state))["final_response"]
    assert "MISRA_15.5" in text
    assert "No recursion" in text


async def test_assemble_explain_no_rules_returns_not_found_message():
    state = {"intent": "explain", "retrieved_rules": [], "standard": "MISRA C:2023", "error": None}
    text = (await assemble_node(state))["final_response"]
    assert "No rules found" in text


# --- error state ---


async def test_assemble_error_overrides_normal_logic():
    state = {"error": "Upstream service failed", "intent": "search"}
    text = (await assemble_node(state))["final_response"]
    assert "Upstream service failed" in text


async def test_assemble_returns_final_response_key():
    state = {"intent": "search", "retrieved_rules": [], "standard": "MISRA C:2023", "error": None}
    result = await assemble_node(state)
    assert "final_response" in result


# ---------------------------------------------------------------------------
# build_graph — wiring tests (no LLM, no real MongoDB)
# ---------------------------------------------------------------------------


async def test_build_graph_returns_compiled_graph():
    """build_graph() must return the object produced by workflow.compile()."""
    mock_checkpointer = MagicMock()
    mock_compiled = MagicMock()
    mock_workflow = MagicMock()
    mock_workflow.compile.return_value = mock_compiled

    with patch("app.graph.builder.StateGraph", return_value=mock_workflow):
        from app.graph.builder import build_graph

        result = await build_graph(mock_checkpointer)

    assert result is mock_compiled
    mock_workflow.compile.assert_called_once_with(checkpointer=mock_checkpointer)


async def test_build_graph_registers_all_six_nodes():
    mock_checkpointer = MagicMock()
    mock_workflow = MagicMock()

    with patch("app.graph.builder.StateGraph", return_value=mock_workflow):
        from app.graph.builder import build_graph

        await build_graph(mock_checkpointer)

    added_names = [c[0][0] for c in mock_workflow.add_node.call_args_list]
    assert set(added_names) == {"orchestrator", "rag", "validation", "critique", "remedier", "assemble"}


async def test_build_graph_adds_linear_edges():
    from langgraph.graph import END, START

    mock_checkpointer = MagicMock()
    mock_workflow = MagicMock()

    with patch("app.graph.builder.StateGraph", return_value=mock_workflow):
        from app.graph.builder import build_graph

        await build_graph(mock_checkpointer)

    edge_calls = [c[0] for c in mock_workflow.add_edge.call_args_list]
    assert (START, "orchestrator") in edge_calls
    assert ("orchestrator", "rag") in edge_calls
    assert ("remedier", "assemble") in edge_calls
    assert ("validation", "critique") in edge_calls
    assert ("assemble", END) in edge_calls


async def test_build_graph_adds_conditional_edges_after_rag_and_critique():
    mock_checkpointer = MagicMock()
    mock_workflow = MagicMock()

    with patch("app.graph.builder.StateGraph", return_value=mock_workflow):
        from app.graph.builder import build_graph

        await build_graph(mock_checkpointer)

    conditional_sources = [c[0][0] for c in mock_workflow.add_conditional_edges.call_args_list]
    assert "rag" in conditional_sources
    assert "critique" in conditional_sources


async def test_build_graph_initialises_state_graph_with_compliance_state():
    from app.models.state import ComplianceState

    mock_checkpointer = MagicMock()

    with patch("app.graph.builder.StateGraph") as mock_sg_cls:
        mock_sg_cls.return_value = MagicMock()
        from app.graph.builder import build_graph

        await build_graph(mock_checkpointer)

    mock_sg_cls.assert_called_once_with(ComplianceState)
