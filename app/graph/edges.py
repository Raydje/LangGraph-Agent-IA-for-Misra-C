from app.models.state import ComplianceState


def route_after_rag(state: ComplianceState) -> str:
    """
    Determines the next step after retrieving context via RAG.
    """
    if state["intent"] == "validate":
        return "validation_node"
    return "assemble_node"


def should_loop_or_finish(state: ComplianceState) -> str:
    """
    Evaluates the critique's output to decide whether to refine the validation or finish.
    """
    if state["critique_approved"]:
        return "remedier_node" if not state["is_compliant"] else "assemble_node"
    if state["iteration_count"] < state["max_iterations"]:
        return "validation_node"
    return "assemble_node"
