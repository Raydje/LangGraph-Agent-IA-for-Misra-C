import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import get_settings
from app.models.state import ComplianceState
from app.services.llm_service import get_structured_llm
from app.utils import extracting_tokens_metadata, logger


class ValidationOutput(BaseModel):
    is_compliant: bool = Field(description="True only if the code fully satisfies all applicable retrieved rules.")
    validation_result: str = Field(
        description=(
            "Detailed explanation of each violation or confirmation of compliance. "
            "Reference specific lines when possible. Every rule mentioned MUST use "
            "the format 'Rule ID (Category): ...' — e.g., 'MISRA_RULE_17.4 (Required): ...'."
        )
    )
    confidence_score: float = Field(description="Float between 0.0 and 1.0 indicating confidence in the assessment.")
    cited_rules: list[str] = Field(
        description=(
            "List of MISRA C:2023 rule IDs used in the evaluation (e.g., ['MISRA_RULE_17.4', 'MISRA_DIR_4.7'])."
        )
    )


async def validation_node(state: ComplianceState) -> dict[str, Any]:
    """
    Evaluates the provided code snippet against the retrieved MISRA C:2023 rules.
    Takes critique feedback into account if this is a subsequent iteration.
    """
    logger.info("--- NODE: VALIDATION ---")

    settings = get_settings()
    structured_llm = get_structured_llm(ValidationOutput, temperature=settings.validation_temperature)

    code = state.get("code_snippet", "No code provided.")
    query = state.get("query", "")
    rules = state.get("retrieved_rules", [])
    critique_feedback = state.get("critique_feedback", "")
    iteration = state.get("iteration_count", 0)
    logger.info("Validation_node", query=query, code_snippet=code, iteration=iteration)

    # The LLM must ground its validation against specific text rather than relying on its
    # internal weights, so we inject the full retrieved rules into the context window here.
    rules_context = "\n\n".join(
        [
            f"Rule ID: {r['rule_id']}\nCategory: {r.get('category', 'Unknown')}\nTitle: {r['title']}\nText: {r['full_text']}"
            for r in rules
        ]
    )

    system_prompt = """You are a strict, expert compliance auditor for MISRA C:2023.
Your task is to validate the provided C/C++ code against the provided MISRA C:2023 rules.

MISRA C:2023 rule IDs follow these formats:
- Directives: "MISRA_DIR_X.Y" (e.g., "MISRA_DIR_4.7 (Mandatory)")
- Rules: "MISRA_RULE_X.Y" (e.g., "MISRA_RULE_17.4 (Required), "MISRA_RULE_1.2 (Advisory)")
Categories are: Mandatory, Required, or Advisory.

Field details:
- "is_compliant": true only if the code fully satisfies all applicable retrieved rules.
- "validation_result": detailed explanation of each violation or confirmation of compliance. Reference specific lines when possible.
  IMPORTANT: Every rule you mention in this field MUST be written as "Rule ID (Category)" — for example:
  "MISRA_RULE_17.4 (Required): ..." or "MISRA_DIR_4.7 (Mandatory): ...".
  The category (Mandatory, Required, or Advisory) is provided for each rule in the context. Never omit it.
- "confidence_score": float between 0.0 and 1.0.
- "cited_rules": list of MISRA C:2023 rule IDs used in the evaluation (e.g., ["MISRA_RULE_17.4", "MISRA_DIR_4.7"])."""

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
Provide your structured validation verdict."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]

    # Use with_structured_output for guaranteed Pydantic-validated output
    try:
        raw_result = await asyncio.wait_for(structured_llm.ainvoke(messages), timeout=settings.llm_timeout)
    except TimeoutError:
        logger.error("Validation LLM call timed out after seconds.", timeout=settings.llm_timeout)
        return {
            "validation_result": "Validation timed out: LLM did not respond in time.",
            "is_compliant": False,
            "confidence_score": 0.0,
            "cited_rules": [],
            "iteration_count": iteration + 1,
            "estimated_cost": 0.0,
        }

    try:
        result: ValidationOutput = raw_result["parsed"]
        if result is None:
            raise ValueError("Structured output parsing returned None")
    except (KeyError, ValueError, AttributeError) as e:
        logger.error("Validation failed to parse structured output.", error=str(e))
        return {
            "validation_result": "Validation failed: LLM returned unparseable output.",
            "is_compliant": False,
            "confidence_score": 0.0,
            "cited_rules": [],
            "iteration_count": iteration + 1,
            "estimated_cost": 0.0,
        }

    tokens_metadata = extracting_tokens_metadata(raw_result)
    logger.info(
        "Validation_node_result",
        validation_result=result.validation_result,
        is_compliant=result.is_compliant,
        confidence_score=result.confidence_score,
        cited_rules=result.cited_rules,
        tokens_metadata=tokens_metadata,
    )

    return {
        "validation_result": result.validation_result,
        "is_compliant": result.is_compliant,
        "confidence_score": result.confidence_score,
        "cited_rules": result.cited_rules,
        "iteration_count": iteration + 1,
        "prompt_tokens": tokens_metadata["prompt_tokens"],
        "completion_tokens": tokens_metadata["completion_tokens"],
        "total_tokens": tokens_metadata["total_tokens"],
        "validation_tokens": tokens_metadata["total_tokens"],
        "estimated_cost": tokens_metadata["estimated_cost"],
    }
