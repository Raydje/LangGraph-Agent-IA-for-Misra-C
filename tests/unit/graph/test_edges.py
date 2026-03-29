from app.graph.edges import route_after_rag, should_loop_or_finish


# --- route_after_rag ---

def test_validate_intent_routes_to_validation_node():
    assert route_after_rag({"intent": "validate"}) == "validation_node"


def test_search_intent_routes_to_assemble_node():
    assert route_after_rag({"intent": "search"}) == "assemble_node"


def test_explain_intent_routes_to_assemble_node():
    assert route_after_rag({"intent": "explain"}) == "assemble_node"


# --- should_loop_or_finish ---

def test_approved_critique_goes_to_assemble():
    state = {"critique_approved": True, "iteration_count": 1, "max_iterations": 4}
    assert should_loop_or_finish(state) == "assemble_node"


def test_rejected_under_max_loops_back_to_validation():
    state = {"critique_approved": False, "iteration_count": 2, "max_iterations": 4}
    assert should_loop_or_finish(state) == "validation_node"


def test_rejected_at_max_iterations_goes_to_assemble():
    state = {"critique_approved": False, "iteration_count": 4, "max_iterations": 4}
    assert should_loop_or_finish(state) == "assemble_node"


def test_rejected_over_max_iterations_goes_to_assemble():
    state = {"critique_approved": False, "iteration_count": 5, "max_iterations": 4}
    assert should_loop_or_finish(state) == "assemble_node"


def test_iteration_one_below_max_still_loops():
    state = {"critique_approved": False, "iteration_count": 3, "max_iterations": 4}
    assert should_loop_or_finish(state) == "validation_node"
