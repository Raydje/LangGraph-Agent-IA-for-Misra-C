import json
from langchain_core.messages import SystemMessage, HumanMessage
from app.models.state import ComplianceState
from app.services.llm_service import get_llm
from app.utils import parse_json_response


def critique_node(state: ComplianceState) -> dict:
    """
    Meta-reviewer that detects hallucinations or logical flaws in the validation result.
    Evaluates the output against 5 strict criteria specific to MISRA C:2023.
    """
    print("--- NODE: CRITIQUE ---")

    llm = get_llm(temperature=0.0)

    code = state.get("code_snippet", "No code provided.")
    rules = state.get("retrieved_rules", [])
    validation_result = state.get("validation_result", "")
    cited_rules = state.get("cited_rules", [])
    is_compliant = state.get("is_compliant", False)

    actual_retrieved_rule_ids = [r["rule_id"] for r in rules]

    system_prompt = """You are a Senior Quality Assurance Reviewer for MISRA C:2023 compliance.
Your job is to review the validation report produced by a junior AI agent and determine if it is accurate, logical, and free of hallucinations.

Evaluate the junior agent's output against these 5 CRITERIA:
1. Rule Hallucination: Did the agent cite a MISRA C:2023 rule ID that was NOT in the 'Actually Retrieved Rules' list?
2. Logical Consistency: Does the explanation match the 'is_compliant' boolean? (e.g., it cannot say "code is compliant" while is_compliant is false).
3. Code Grounding: Does the explanation specifically reference the provided C/C++ code, or is it too generic?
4. Standard Accuracy: Does the agent correctly use MISRA C:2023 rule ID formats ("Dir X.Y" or "Rule X.Y") and valid categories (Mandatory, Required, Advisory)?
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

    response = llm.invoke(messages)
    try:
        result = parse_json_response(response.content)
        approved = result.get("approved", False)
        feedback = result.get("feedback", "Failed to parse critique feedback.")
    except (json.JSONDecodeError, ValueError):
        print("Critique node failed to parse JSON.")
        approved = False
        feedback = "Critique system failed to output valid JSON. Please simplify your validation output."

    return {
        "critique_approved": approved,
        "critique_feedback": feedback,
    }
