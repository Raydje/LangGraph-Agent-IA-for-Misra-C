from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from app.models.state import ComplianceState
from app.services.llm_service import get_structured_llm
from app.config import get_settings
from app.utils import calculate_gemini_cost, logger

# 1. Define the desired structured output schema
class OrchestratorOutput(BaseModel):
    intent: Literal["search", "validate", "explain"] = Field(
        description="The classified intent of the user's request."
    )
    reasoning: str = Field(
        description="A brief explanation (1-2 sentences) of why this intent was chosen."
    )

# 2. Define the Orchestrator Node function
async def orchestrate(state: ComplianceState) -> dict:
    """
    Analyzes the user's query and code snippet to determine the workflow intent.
    Currently focused entirely on MISRA C:2023 compliance.
    Updates the state with 'intent' and 'orchestrator_reasoning'.
    """
    query = state.get("query", "")
    code_snippet = state.get("code_snippet", "")
    logger.info("Orchestrator_node", query=query, code_snippet=code_snippet)
    # Initialize the base LLM (temperature=0.0 for deterministic classification)
    settings = get_settings()
    structured_llm = get_structured_llm(OrchestratorOutput, temperature=settings.orchestrator_temperature)

    # Create the prompt instructing the LLM on how to classify for MISRA C:2023
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the intelligent routing orchestrator for a C/C++ static analysis AI agent.
        Your job is to analyze the user's request regarding the MISRA C:2023 standard and classify it into one of three intents:

        1. "search": The user is looking for specific MISRA C:2023 rules, guidelines, or documentation.
           (e.g., "Find rules about dead code", "What does MISRA say about pointer arithmetic?")
        2. "validate": The user has provided C/C++ code and wants to check if it complies with MISRA C:2023 rules.
           (e.g., "Check this C code snippet against MISRA C:2023", "Does this function violate any MISRA directives?")
        3. "explain": The user wants a detailed, conceptual explanation of a specific MISRA C:2023 rule or why a practice is banned.
           (e.g., "Explain why recursion is banned in MISRA", "What is the rationale behind rule 11.4?")

        Analyze the inputs carefully and output the intent and your reasoning.
        """),
        ("human", "User Query: {query}\n\nProvided Code (if any):\n{code}")
    ])

    # Chain the prompt and the structured LLM together
    # include_raw=True returns {"raw": AIMessage, "parsed": OrchestratorOutput, "parsing_error": ...}
    chain = prompt | structured_llm

    # Invoke the chain with the current state data
    raw_result = await chain.ainvoke({
        "query": query,
        "code": code_snippet if code_snippet else "None provided."
    })

    try:
        result: OrchestratorOutput = raw_result["parsed"]
        if result is None:
            raise ValueError("Structured output parsing returned None")
    except (KeyError, ValueError, AttributeError) as e:
        logger.error("Orchestrator failed to parse structured output.", error=str(e))
        return {
            "intent": "search",
            "orchestrator_reasoning": "Orchestrator failed to parse LLM output; defaulting to search.",
            "standard": state.get("standard", "MISRA C:2023"),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "orchestrator_tokens": 0,
            "estimated_cost": 0.0,
        }

    # Extract token usage from the raw AIMessage (usage_metadata uses input_tokens/output_tokens)
    usage = getattr(raw_result.get("raw"), "usage_metadata", None) or {}
    _input_tokens = usage.get("input_tokens", 0)
    _output_tokens = usage.get("output_tokens", 0)
    logger.info("Orchestrator_node_result", intent=result.intent, reasoning=result.reasoning, input_tokens=_input_tokens, output_tokens=_output_tokens)
    logger.info("Orchestrator_node_cost", estimated_cost=calculate_gemini_cost(_input_tokens, _output_tokens))

    # LangGraph nodes must return a dictionary containing the keys of the State to update
    return {
        "intent": result.intent,
        "orchestrator_reasoning": result.reasoning,
        "standard": state.get("standard", "MISRA C:2023"),
        "prompt_tokens": _input_tokens,
        "completion_tokens": _output_tokens,
        "total_tokens": _input_tokens + _output_tokens,
        "orchestrator_tokens": _input_tokens + _output_tokens,
        "estimated_cost": calculate_gemini_cost(_input_tokens, _output_tokens),
    }