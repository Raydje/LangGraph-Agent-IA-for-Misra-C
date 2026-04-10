import asyncio
import os

from dotenv import load_dotenv

# Ensure environment variables are loaded before importing app modules
# This will load GEMINI_API_KEY and any settings from .env
load_dotenv()

# Adjust the imports based on your actual State implementation
from app.graph.nodes.orchestrator import orchestrate
from app.models.state import (
    ComplianceState,  # noqa: F401 — used for state type hints below
)


async def run_live_tests():
    print("==================================================")
    print("🚀 Starting Live Test for Orchestrator Node...")
    print("==================================================\n")

    # ---------------------------------------------------------
    # Test Case 1: SEARCH Intent
    # ---------------------------------------------------------
    print("--- Test Case 1: SEARCH Intent ---")
    state_search: ComplianceState = {
        "query": "What does MISRA C:2023 say about pointer arithmetic?",
        "code_snippet": "",
        "standard": "MISRA C:2023",
    }

    result_search = await orchestrate(state_search)
    print("Expected : search")
    print(f"Got      : {result_search.get('intent')}")
    print(f"Reasoning: {result_search.get('orchestrator_reasoning')}")
    print(f"Tokens   : {result_search.get('total_tokens')} (Cost: ${result_search.get('estimated_cost', 0):.6f})\n")

    # ---------------------------------------------------------
    # Test Case 2: VALIDATE Intent
    # ---------------------------------------------------------
    print("--- Test Case 2: VALIDATE Intent ---")
    state_validate: ComplianceState = {
        "query": "Please check this snippet and see if it complies with MISRA rules.",
        "code_snippet": """
        int divide(int a, int b) {
            if (b == 0) return 0;
            return a / b;
        }
        """,
        "standard": "MISRA C:2023",
    }

    result_validate = await orchestrate(state_validate)
    print("Expected : validate")
    print(f"Got      : {result_validate.get('intent')}")
    print(f"Reasoning: {result_validate.get('orchestrator_reasoning')}")
    print(f"Tokens   : {result_validate.get('total_tokens')} (Cost: ${result_validate.get('estimated_cost', 0):.6f})\n")

    # ---------------------------------------------------------
    # Test Case 3: EXPLAIN Intent
    # ---------------------------------------------------------
    print("--- Test Case 3: EXPLAIN Intent ---")
    state_explain: ComplianceState = {
        "query": "Explain why recursion is banned in MISRA.",
        "code_snippet": "",
        "standard": "MISRA C:2023",
    }

    result_explain = await orchestrate(state_explain)
    print("Expected : explain")
    print(f"Got      : {result_explain.get('intent')}")
    print(f"Reasoning: {result_explain.get('orchestrator_reasoning')}")
    print(f"Tokens   : {result_explain.get('total_tokens')} (Cost: ${result_explain.get('estimated_cost', 0):.6f})\n")

    print("==================================================")
    print("✅ Live Test Complete")
    print("==================================================")


if __name__ == "__main__":
    # Ensure asyncio uses the correct event loop policy depending on the OS
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_live_tests())
