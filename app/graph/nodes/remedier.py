import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import get_settings
from app.models.state import ComplianceState
from app.services.llm_service import get_structured_llm
from app.utils import extracting_tokens_metadata, logger


# Structured output schema — guarantees valid typed output from the LLM
class RemediationOutput(BaseModel):
    fixed_code_snippet: str = Field(description="The complete corrected C code, ready to compile.")
    remediation_explanation: str = Field(
        description=(
            "A per-rule breakdown. For each violation addressed, use the format: "
            "'Rule ID (Category): what was wrong on which line → what was changed and why.' "
            "If an Advisory rule was intentionally left unfixed, state the reason explicitly."
        )
    )


async def remediate_code(state: ComplianceState) -> dict[str, Any]:
    """
    Remediation Node: Takes a non-compliant C code snippet and attempts to fix it
    based on the cited rules and validation results.
    """
    logger.info("--- NODE: REMEDIATION ---")

    settings = get_settings()
    structured_llm = get_structured_llm(RemediationOutput, temperature=settings.remediation_temperature)

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
        cited_rules_context = (
            "\n\n".join(
                f"Rule ID: {r['rule_id']}\nCategory: {r.get('category', 'Unknown')}\nTitle: {r['title']}\nText: {r['full_text']}"
                for r in retrieved_rules
            )
            or "No rule details available."
        )

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

Field details:
- "fixed_code_snippet": the complete corrected C code, ready to compile.
- "remediation_explanation": a per-rule breakdown. For each violation addressed, use the format:
  "Rule ID (Category): what was wrong on which line → what was changed and why."
  If an Advisory rule was intentionally left unfixed, state the reason explicitly."""

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
Provide your structured remediation output."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]

    # Use with_structured_output for guaranteed Pydantic-validated output
    try:
        raw_result = await asyncio.wait_for(structured_llm.ainvoke(messages), timeout=settings.llm_timeout)
    except TimeoutError:
        logger.error("Remediation LLM call timed out after seconds.", timeout=settings.llm_timeout)
        return {
            "fixed_code_snippet": code_snippet,
            "remediation_explanation": "Remediation timed out: LLM did not respond in time.",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "remediation_tokens": 0,
            "estimated_cost": 0.0,
        }

    try:
        result: RemediationOutput = raw_result["parsed"]
        if result is None:
            raise ValueError("Structured output parsing returned None")
        fixed_code = result.fixed_code_snippet
        explanation = result.remediation_explanation
    except (KeyError, ValueError, AttributeError) as e:
        logger.error("Remediation failed to parse structured output.", error=str(e))
        return {
            "fixed_code_snippet": code_snippet,
            "remediation_explanation": f"System failed to generate a fix automatically. Error: {str(e)}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "remediation_tokens": 0,
            "estimated_cost": 0.0,
        }

    tokens_metadata = extracting_tokens_metadata(raw_result)
    logger.info(
        "Remediation_node_result",
        fixed_code_snippet=fixed_code,
        input_tokens=tokens_metadata["prompt_tokens"],
        output_tokens=tokens_metadata["completion_tokens"],
    )

    return {
        "fixed_code_snippet": fixed_code,
        "remediation_explanation": explanation,
        "prompt_tokens": tokens_metadata["prompt_tokens"],
        "completion_tokens": tokens_metadata["completion_tokens"],
        "total_tokens": tokens_metadata["total_tokens"],
        "remediation_tokens": tokens_metadata["total_tokens"],
        "estimated_cost": tokens_metadata["estimated_cost"],
    }
