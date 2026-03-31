import json
from langchain_core.messages import SystemMessage, HumanMessage
from app.models.state import ComplianceState
from app.services.llm_service import get_llm
from app.config import get_settings
from app.utils import calculate_gemini_cost, parse_json_response, logger


async def remediate_code(state: ComplianceState) -> dict:
    """
    Remediation Node: Takes a non-compliant C code snippet and attempts to fix it
    based on the cited rules and validation results.
    """
    logger.info("--- NODE: REMEDIATION ---")

    settings = get_settings()
    llm = get_llm(temperature=settings.remediation_temperature)

    code_snippet = state.get("code_snippet", "")
    validation_result = state.get("validation_result", "")
    cited_rules = state.get("cited_rules", [])
    retrieved_rules = state.get("retrieved_rules", [])

    # Build a rich rules context filtered to only the rules actually cited
    cited_set = set(cited_rules)
    cited_rules_context = "\n\n".join(
        f"Rule ID: {r['rule_id']}\nCategory: {r.get('category', 'Unknown')}\nTitle: {r['title']}\nText: {r['full_text']}"
        for r in retrieved_rules
        if r["rule_id"] in cited_set
    )
    if not cited_rules_context:
        # Fallback: show all retrieved rules if none match (e.g. ID format mismatch)
        cited_rules_context = "\n\n".join(
            f"Rule ID: {r['rule_id']}\nCategory: {r.get('category', 'Unknown')}\nTitle: {r['title']}\nText: {r['full_text']}"
            for r in retrieved_rules
        ) or "No rule details available."

    system_prompt = """You are a Senior Embedded C Software Engineer and an expert in the MISRA C:2023 standard.
Your task is to fix the provided C code snippet so that it complies with the cited MISRA C:2023 rules.

CONSTRAINTS — follow all of them without exception:
1. Apply ONLY the minimal changes necessary to resolve each violation. Do not refactor, rename, or restructure code beyond what is strictly required.
2. Preserve the original functionality exactly. The fixed code must behave identically to the original in all compliant scenarios.
3. Your fix must NOT introduce any new MISRA C:2023 violations.
4. Treat rule categories with the correct priority:
   - Mandatory: MUST be fixed. There is no deviation permitted.
   - Required: MUST be fixed unless a formal deviation has been documented (assume none here).
   - Advisory: SHOULD be fixed. If fixing it would break functionality or introduce a Required/Mandatory violation, note it but leave it unfixed.

You MUST respond with a valid JSON object matching this schema exactly:
{
  "fixed_code_snippet": "/* The corrected C code */",
  "remediation_explanation": "string"
}

Field details:
- "fixed_code_snippet": the complete corrected C code, ready to compile.
- "remediation_explanation": a per-rule breakdown. For each violation addressed, use the format:
  "Rule ID (Category): what was wrong on which line → what was changed and why."
  If an Advisory rule was intentionally left unfixed, state the reason explicitly.

Do not include any text outside the JSON block."""

    human_content = f"""### Original Non-Compliant Code:
```c
{code_snippet}
```

### Violated Rules (with full context):
{cited_rules_context}

### Validation Report (Why it failed):
{validation_result}

### Instructions:
Fix the code strictly according to the CONSTRAINTS in the system prompt.
Respond with the JSON object only."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]

    response = await llm.ainvoke(messages)
    try:
        result = parse_json_response(response.content)
        fixed_code = result.get("fixed_code_snippet", "")
        explanation = result.get("remediation_explanation", "")
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.error("Remediation node failed to parse JSON.", error=str(e))
        return {
            "fixed_code_snippet": code_snippet,
            "remediation_explanation": f"System failed to generate a fix automatically. Error: {str(e)}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "remediation_tokens": 0,
            "estimated_cost": 0.0,
        }

    remediation_usage = getattr(response, "usage_metadata", None) or {}
    _input_tokens = remediation_usage.get("input_tokens", 0)
    _output_tokens = remediation_usage.get("output_tokens", 0)
    logger.info("Remediation_node_result", fixed_code_snippet=fixed_code, input_tokens=_input_tokens, output_tokens=_output_tokens)
    logger.info("Remediation_node_cost", estimated_cost=calculate_gemini_cost(_input_tokens, _output_tokens))

    return {
        "fixed_code_snippet": fixed_code,
        "remediation_explanation": explanation,
        "prompt_tokens": _input_tokens,
        "completion_tokens": _output_tokens,
        "total_tokens": _input_tokens + _output_tokens,
        "remediation_tokens": _input_tokens + _output_tokens,
        "estimated_cost": calculate_gemini_cost(_input_tokens, _output_tokens),
    }
