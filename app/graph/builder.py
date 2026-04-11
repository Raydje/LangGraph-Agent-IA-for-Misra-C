from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.edges import route_after_rag, should_loop_or_finish
from app.graph.nodes.critique import critique_node
from app.graph.nodes.orchestrator import orchestrate
from app.graph.nodes.rag import rag_node
from app.graph.nodes.remedier import remediate_code
from app.graph.nodes.validation import validation_node
from app.models.state import ComplianceState
from app.utils import logger


async def assemble_node(state: ComplianceState) -> dict[str, str]:
    """
    The 5th node: Formats the final output based on the intent and validation results.
    """
    logger.info("--- NODE: ASSEMBLE ---")

    if state.get("error"):
        return {"final_response": f"An error occurred: {state['error']}"}

    intent = state.get("intent", "search")
    standard = state.get("standard", "")

    if intent == "validate":
        compliant = state.get("is_compliant")
        confidence = state.get("confidence_score")
        cited = state.get("cited_rules") or []
        details = state.get("validation_result", "No details available.")
        cited_text = ", ".join(cited) if cited else "none"

        confidence_text = (
            f"Confidence: {confidence:.0%}\n" if confidence is not None else "error: confidence score missing\n"
        )

        final_answer = (
            f"Validation Complete.\n"
            f"Standard: {standard}\n"
            f"Compliant: {compliant}\n"
            f"{confidence_text}"
            f"Cited rules: {cited_text}\n"
            f"Details: {details}"
        )
    elif intent == "explain":
        rules = state.get("retrieved_rules", [])
        if not rules:
            final_answer = f"No rules found for your query in {standard}."
        else:
            rules_text = "\n".join(f"[{r['rule_id']}] {r['title']}" for r in rules)
            final_answer = f"Here is an explanation of relevant {standard} rules:\n{rules_text}"
    else:  # search
        rules = state.get("retrieved_rules", [])
        if not rules:
            final_answer = f"No matching rules found in {standard}."
        else:
            rules_text = "\n".join(f"[{r['rule_id']}] ({r['section']}) {r['title']}" for r in rules)
            final_answer = f"Relevant {standard} rules:\n{rules_text}"

    return {"final_response": final_answer}


async def build_graph(checkpointer: MongoDBSaver) -> CompiledStateGraph:
    """
    Compiles the LangGraph state machine.
    """
    workflow = StateGraph(ComplianceState)

    workflow.add_node("orchestrator", orchestrate)
    workflow.add_node("rag", rag_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("remedier", remediate_code)
    workflow.add_node("assemble", assemble_node)

    workflow.add_edge(START, "orchestrator")
    workflow.add_edge("orchestrator", "rag")
    workflow.add_edge("remedier", "assemble")

    # Bypass validation entirely for 'search' and 'explain' intents to save tokens and execution time.
    workflow.add_conditional_edges(
        "rag", route_after_rag, {"validation_node": "validation", "assemble_node": "assemble"}
    )

    workflow.add_edge("validation", "critique")

    # The critique node acts as an adversarial gatekeeper. Rejected validations are fed back
    # into the validation node with feedback for self-correction until max loops are hit.
    workflow.add_conditional_edges(
        "critique",
        should_loop_or_finish,
        {"validation_node": "validation", "assemble_node": "assemble", "remedier_node": "remedier"},
    )

    workflow.add_edge("assemble", END)

    # Attach the MongoDB checkpointer here to enable durable state persistence across nodes,
    # allowing granular session resumption and time-travel replay.
    return workflow.compile(checkpointer=checkpointer)
