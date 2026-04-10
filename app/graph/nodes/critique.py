import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import get_settings
from app.models.state import ComplianceState, CritiqueEntry
from app.services.llm_service import get_structured_llm
from app.utils import extracting_tokens_metadata, logger


# Structured output schema — guarantees valid typed output from the LLM
class CritiqueOutput(BaseModel):
    approved: bool = Field(description="True only if the validation passes ALL 5 criteria.")
    feedback: str = Field(
        description=(
            "'Pass' if approved. If rejected, a specific explanation of which "
            "criteria failed and why, so the agent can correct it."
        )
    )


async def critique_node(state: ComplianceState) -> dict[str, Any]:
    """
    Meta-reviewer that detects hallucinations or logical flaws in the validation result.
    Evaluates the output against 5 strict criteria specific to MISRA C:2023.
    """
    logger.info("--- NODE: CRITIQUE ---")

    settings = get_settings()
    structured_llm = get_structured_llm(CritiqueOutput, temperature=settings.critique_temperature)

    code = state.get("code_snippet", "No code provided.")
    rules = state.get("retrieved_rules", [])
    validation_result = state.get("validation_result", "")
    cited_rules = state.get("cited_rules", [])
    is_compliant = state.get("is_compliant", False)

    actual_retrieved_rule_ids = [r["rule_id"] for r in rules]
    logger.info(
        "Critique_node", validation_result=validation_result, cited_rules=cited_rules, is_compliant=is_compliant
    )

    system_prompt = """You are a Senior Quality Assurance Reviewer for MISRA C:2023 compliance.
Your job is to review the validation report produced by a junior AI agent and determine if it is accurate, logical, and free of hallucinations.

Evaluate the junior agent's output against these 5 CRITERIA:
1. Rule Hallucination: Did the agent cite a MISRA C:2023 rule ID that was NOT in the 'Actually Retrieved Rules' list?
2. Logical Consistency: Does the explanation match the 'is_compliant' boolean? (e.g., it cannot say "code is compliant" while is_compliant is false).
3. Code Grounding: Does the explanation specifically reference the provided C/C++ code, or is it too generic?
4. Standard Accuracy: Does the agent correctly use MISRA C:2023 rule ID formats ("MISRA_DIR_4.7" or "MISRA_RULE_17.4") AND include the rule category (Mandatory, Required, or Advisory) in the explanation for each cited rule using the format "Rule ID (Category): ..."? Reject if any cited rule in the validation_result text is missing its category.
5. Completeness: Did the agent address the actual violation or compliance question implied by the code?

- "approved": true only if the validation passes ALL 5 criteria.
- "feedback": "Pass" if approved. If rejected, provide a specific explanation of which criteria failed and why, so the agent can correct it."""

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

Based on the 5 criteria, generate your structured verdict."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]

    # Use with_structured_output for guaranteed Pydantic-validated output
    try:
        raw_result = await asyncio.wait_for(structured_llm.ainvoke(messages), timeout=settings.llm_timeout)
    except TimeoutError:
        logger.error("Critique LLM call timed out after seconds.", timeout=settings.llm_timeout)
        return {
            "critique_approved": False,
            "critique_feedback": "Critique timed out; validation will be re-run.",
            "critique_history": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "critique_tokens": 0,
            "estimated_cost": 0.0,
        }

    try:
        result: CritiqueOutput = raw_result["parsed"]
        if result is None:
            raise ValueError("Structured output parsing returned None")
        approved = result.approved
        feedback = result.feedback
        critique_entry: CritiqueEntry = {
            "iteration": state.get("iteration_count", 0),
            "issues_found": [feedback] if not approved else [],
            "approved": approved,
        }
    except (KeyError, ValueError, AttributeError) as e:
        logger.error("Critique failed to parse structured output.", error=str(e))
        approved = False
        feedback = "Critique system failed to produce valid output. Please simplify your validation output."
        critique_entry = {
            "iteration": state.get("iteration_count", 0) + 1,
            "issues_found": [feedback],
            "approved": False,
        }

    # Track critique tokens used
    tokens_metadata = extracting_tokens_metadata(raw_result)
    logger.info("Critique_node_result", approved=approved, feedback=feedback, tokens_metadata=tokens_metadata)

    return {
        "critique_approved": approved,
        "critique_feedback": feedback,
        "critique_history": [critique_entry],
        "prompt_tokens": tokens_metadata["prompt_tokens"],
        "completion_tokens": tokens_metadata["completion_tokens"],
        "total_tokens": tokens_metadata["total_tokens"],
        "critique_tokens": tokens_metadata["total_tokens"],
        "estimated_cost": tokens_metadata["estimated_cost"],
    }
