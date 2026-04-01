from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from app.models.state import ComplianceState

# Import your nodes
from app.graph.nodes.orchestrator import orchestrate
from app.graph.nodes.rag import rag_node
from app.graph.nodes.validation import validation_node
from app.graph.nodes.critique import critique_node
from app.graph.nodes.remedier import remediate_code
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite

# Import your routing edges
from app.graph.edges import route_after_rag, should_loop_or_finish

# Import utilities
from app.utils import logger


async def assemble_node(state: ComplianceState) -> dict:
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
        final_answer = (
            f"Validation Complete.\n"
            f"Standard: {standard}\n"
            f"Compliant: {compliant}\n"
            f"Confidence: {confidence:.0%}\n"
            f"Cited rules: {cited_text}\n"
            f"Details: {details}"
        )
    elif intent == "explain":
        rules = state.get("retrieved_rules", [])
        if not rules:
            final_answer = f"No rules found for your query in {standard}."
        else:
            rules_text = "\n".join(
                f"[{r['rule_id']}] {r['title']}" for r in rules
            )
            final_answer = f"Here is an explanation of relevant {standard} rules:\n{rules_text}"
    else:  # search
        rules = state.get("retrieved_rules", [])
        if not rules:
            final_answer = f"No matching rules found in {standard}."
        else:
            rules_text = "\n".join(
                f"[{r['rule_id']}] ({r['section']}) {r['title']}" for r in rules
            )
            final_answer = f"Relevant {standard} rules:\n{rules_text}"

    return {"final_response": final_answer}


async def build_graph(conn: aiosqlite.Connection) -> CompiledStateGraph:
    """
    Compiles the LangGraph state machine.
    """
    # 1. Initialize the StateGraph with your TypedDict
    workflow = StateGraph(ComplianceState)

    # 2. Add all the nodes to the graph
    workflow.add_node("orchestrator", orchestrate)
    workflow.add_node("rag", rag_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("remedier", remediate_code)
    workflow.add_node("assemble", assemble_node)

    # 3. Define the standard, linear edges
    workflow.add_edge(START, "orchestrator")
    workflow.add_edge("orchestrator", "rag")  # RAG always happens after the orchestrator figures out the intent
    workflow.add_edge("remedier", "assemble")  # If remediation is needed, it happens after critique and before final assembly

    # 4. Define the conditional routing AFTER RAG
    # Depending on the intent, it either goes to validate the code, or skip straight to assembling an answer
    workflow.add_conditional_edges(
        "rag",
        route_after_rag,
        {
            "validation_node": "validation",
            "assemble_node": "assemble"
        }
    )

    # 5. Connect Validation to Critique
    workflow.add_edge("validation", "critique")

    # 6. Define the self-correction loop AFTER CRITIQUE
    # If critique fails, it loops back to validation. If it passes (or max loops hit), it goes to assemble.
    workflow.add_conditional_edges(
        "critique",
        should_loop_or_finish,
        {
            "validation_node": "validation",
            "assemble_node": "assemble",
            "remedier_node": "remedier"
        }
    )
    

    # 7. End the graph
    workflow.add_edge("assemble", END)
    
    # 8. Initialize the async SQLite saver with the injected connection
    memory = AsyncSqliteSaver(conn)

    # 9. Compile!
    return workflow.compile(checkpointer=memory)