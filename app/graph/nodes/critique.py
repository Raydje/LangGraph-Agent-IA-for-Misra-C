import json
from langchain_core.messages import SystemMessage, HumanMessage
from app.models.state import ComplianceState, CritiqueEntry
from app.services.llm_service import get_llm
from app.config import get_settings
from app.utils import parse_json_response, calculate_gemini_cost, logger


async def critique_node(state: ComplianceState) -> dict:
    """
    Meta-reviewer that detects hallucinations or logical flaws in the validation result.
    Evaluates the output against 5 strict criteria specific to MISRA C:2023.
    """
    logger.info("--- NODE: CRITIQUE ---")

    settings = get_settings()
    llm = get_llm(temperature=settings.critique_temperature)

    code = state.get("code_snippet", "No code provided.")
    rules = state.get("retrieved_rules", [])
    validation_result = state.get("validation_result", "")
    cited_rules = state.get("cited_rules", [])
    is_compliant = state.get("is_compliant", False)

    actual_retrieved_rule_ids = [r["rule_id"] for r in rules]
    logger.info("Critique_node", validation_result=validation_result, cited_rules=cited_rules, is_compliant=is_compliant)

    system_prompt = """You are a Senior Quality Assurance Reviewer for MISRA C:2023 compliance.
Your job is to review the validation report produced by a junior AI agent and determine if it is accurate, logical, and free of hallucinations.

Evaluate the junior agent's output against these 5 CRITERIA:
1. Rule Hallucination: Did the agent cite a MISRA C:2023 rule ID that was NOT in the 'Actually Retrieved Rules' list?
2. Logical Consistency: Does the explanation match the 'is_compliant' boolean? (e.g., it cannot say "code is compliant" while is_compliant is false).
3. Code Grounding: Does the explanation specifically reference the provided C/C++ code, or is it too generic?
4. Standard Accuracy: Does the agent correctly use MISRA C:2023 rule ID formats ("Dir X.Y" or "Rule MISRA_X.Y") AND include the rule category (Mandatory, Required, or Advisory) in the explanation for each cited rule using the format "Rule ID (Category): ..."? Reject if any cited rule in the validation_result text is missing its category.
5. Completeness: Did the agent address the actual violation or compliance question implied by the code?

You MUST respond with a valid JSON object matching this schema exactly:
{
  "approved": bool,
  "feedback": "string"
}

- "approved": true only if the validation passes ALL 5 criteria.
- "feedback": "Pass" if approved. If rejected, provide a specific explanation of which criteria failed and why, so the agent can correct it.

Do not include any text outside the JSON block."""

    human_content = f"""--- INPUT DATA ---
Code to Validate:
```c
{code}
```

Actually Retrieved MISRA C:2023 Rule IDs (the only valid IDs): {actual_retrieved_rule_ids}

--- JUNIOR AGENT'S OUTPUT TO REVIEW ---
Validation Result Text: "{validation_result}"
Cited Rules by Agent: {cited_rules}
Agent's 'is_compliant' Verdict: {is_compliant}

Based on the 5 criteria, generate your JSON verdict."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]

    response = await llm.ainvoke(messages)
    try:
        result = parse_json_response(response.content)
        approved = result.get("approved", False)
        feedback = result.get("feedback", "Failed to parse critique feedback.")
        critique_entry: CritiqueEntry = {
            "iteration": state.get("iteration_count", 0),
            "issues_found": [feedback] if not approved else [],
            "approved": approved,
        }
    except (json.JSONDecodeError, ValueError):
        logger.error("Critique node failed to parse JSON.")
        approved = False
        feedback = "Critique system failed to output valid JSON. Please simplify your validation output."
        critique_entry : CritiqueEntry = {
            "iteration": state.get("iteration_count", 0) + 1,
            "issues_found": [feedback],
            "approved": False,
        }
    #Track critique tokens used
    critique_usage = getattr(response, "usage_metadata", None) or {}
    _input_tokens = critique_usage.get("input_tokens", 0)
    _output_tokens = critique_usage.get("output_tokens", 0)
    logger.info("Critique_node_result", approved=approved, feedback=feedback, input_tokens=_input_tokens, output_tokens=_output_tokens)
    logger.info("Critique_node_cost", estimated_cost=calculate_gemini_cost(_input_tokens, _output_tokens))
    
    return {
        "critique_approved": approved,
        "critique_feedback": feedback,
        "critique_history": [critique_entry],
        "prompt_tokens": _input_tokens,
        "completion_tokens": _output_tokens,
        "total_tokens": _input_tokens + _output_tokens,
        "critique_tokens": _input_tokens + _output_tokens,
        "estimated_cost": calculate_gemini_cost(_input_tokens, _output_tokens),
    }
