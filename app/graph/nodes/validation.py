import json
from langchain_core.messages import SystemMessage, HumanMessage
from app.models.state import ComplianceState
from app.services.llm_service import get_llm
from app.config import get_settings
from app.utils import parse_json_response, calculate_gemini_cost, logger


def validation_node(state: ComplianceState) -> dict:
    """
    Evaluates the provided code snippet against the retrieved MISRA C:2023 rules.
    Takes critique feedback into account if this is a subsequent iteration.
    """
    logger.info("--- NODE: VALIDATION ---")

    settings = get_settings()
    llm = get_llm(temperature=settings.validation_temperature)

    code = state.get("code_snippet", "No code provided.")
    query = state.get("query", "")
    rules = state.get("retrieved_rules", [])
    critique_feedback = state.get("critique_feedback", "")
    iteration = state.get("iteration_count", 0)
    logger.info("Validation_node", query=query, code_snippet=code, iteration=iteration)
    
    # Format retrieved rules — MISRA C:2023 IDs are either "Dir X.Y" or "Rule MISRA_X.Y"
    rules_context = "\n\n".join(
        [f"Rule ID: {r['rule_id']}\nCategory: {r.get('category', 'Unknown')}\nTitle: {r['title']}\nText: {r['full_text']}"
         for r in rules]
    )

    system_prompt = """You are a strict, expert compliance auditor for MISRA C:2023.
Your task is to validate the provided C/C++ code against the provided MISRA C:2023 rules.

MISRA C:2023 rule IDs follow these formats:
- Directives: "Dir X.Y" (e.g., "Dir 4.1")
- Rules: "Rule MISRA_X.Y" (e.g., "Rule MISRA_15.5")
Categories are: Mandatory, Required, or Advisory.

You MUST respond with a valid JSON object matching this schema exactly:
{
  "is_compliant": bool,
  "validation_result": "string",
  "confidence_score": float,
  "cited_rules": ["string"]
}

Field details:
- "is_compliant": true only if the code fully satisfies all applicable retrieved rules.
- "validation_result": detailed explanation of each violation or confirmation of compliance. Reference specific lines when possible.
  IMPORTANT: Every rule you mention in this field MUST be written as "Rule ID (Category)" — for example:
  "Rule MISRA_15.5 (Required): ..." or "Dir 4.1 (Mandatory): ...".
  The category (Mandatory, Required, or Advisory) is provided for each rule in the context. Never omit it.
- "confidence_score": float between 0.0 and 1.0.
- "cited_rules": list of MISRA C:2023 rule IDs used in the evaluation (e.g., ["Rule MISRA_15.5", "Dir 4.1"]).

Do not include any text outside the JSON block."""

    critique_section = ""
    if iteration > 0 and critique_feedback:
        critique_section = f"""
Previous Critique Feedback (iteration {iteration}):
{critique_feedback}

Address all points raised above in your revised evaluation.
"""

    human_content = f"""User Query: {query}

Retrieved MISRA C:2023 Rules:
{rules_context if rules_context else "No specific rules retrieved. Apply general MISRA C:2023 knowledge strictly relevant to the query."}

Code to Validate:
```c
{code}
```
{critique_section}
Respond with the JSON verdict only."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]

    response = llm.invoke(messages)
    try:
        result = parse_json_response(response.content)
    except (json.JSONDecodeError, ValueError):
        return {
            "validation_result": "Validation failed: LLM returned unparseable output.",
            "is_compliant": False,
            "confidence_score": 0.0,
            "cited_rules": [],
            "iteration_count": iteration + 1,
            "estimated_cost": 0.0,
        }
    # Track validation tokens used
    validation_usage = response.usage_metadata if hasattr(response, "usage_metadata") else {}
    _input_tokens = validation_usage.get("input_tokens", 0)
    _output_tokens = validation_usage.get("output_tokens", 0)
    logger.info("Validation_node_result", validation_result=result.get("validation_result", ""), is_compliant=result.get("is_compliant", False), confidence_score=result.get("confidence_score", 0.0), cited_rules=result.get("cited_rules", []), input_tokens=_input_tokens, output_tokens=_output_tokens)
    logger.info("Validation_node_cost", estimated_cost=calculate_gemini_cost(_input_tokens, _output_tokens))
    return {
        "validation_result": result.get("validation_result", ""),
        "is_compliant": result.get("is_compliant", False),
        "confidence_score": result.get("confidence_score", 0.0),
        "cited_rules": result.get("cited_rules", []),
        "iteration_count": iteration + 1,
        "prompt_tokens": _input_tokens,
        "completion_tokens": _output_tokens,
        "total_tokens": _input_tokens + _output_tokens,
        "validation_tokens": _input_tokens + _output_tokens,
        "estimated_cost": calculate_gemini_cost(_input_tokens, _output_tokens),
    }
