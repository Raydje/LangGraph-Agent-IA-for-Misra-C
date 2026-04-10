import asyncio
import os

from dotenv import load_dotenv

# Load environment variables (e.g., GEMINI_API_KEY)
load_dotenv()

from app.graph.nodes.critique import critique_node
from app.models.state import (
    ComplianceState,  # noqa: F401 — used for state type hints below
)


async def run_live_tests():
    print("==================================================")
    print("🕵️  Starting Live Test for Critique Node...")
    print("==================================================\n")

    # ---------------------------------------------------------
    # Test Case 1: PASSING CRITIQUE (Perfect Validation)
    # ---------------------------------------------------------
    print("--- Test Case 1: PASSING CRITIQUE ---")
    state_pass: ComplianceState = {
        "code_snippet": "int divide(int a, int b) { if (b == 0) return 0; return a / b; }",
        "retrieved_rules": [
            {"rule_id": "MISRA_RULE_7.1"},
            {"rule_id": "MISRA_DIR_4.15"},
            {"rule_id": "MISRA_RULE_1.3"},
            {"rule_id": "MISRA_RULE_8.16"},
            {"rule_id": "MISRA_DIR_2.1"},
        ],
        "validation_result": "Validation Complete.\nStandard: MISRA C:2023\nCompliant: True\nConfidence: 100%\nCited rules: Rule MISRA_RULE_1.3, Rule MISRA_DIR_4.15, Rule MISRA_DIR_2.1\nDetails: The provided code is compliant with the applicable rules regarding division by zero. The function `divide` explicitly checks for `b == 0` on line 1. If `b` is zero, it returns `0`, thereby preventing the integer division `a / b` from occurring. This avoids undefined behavior associated with integer division by zero, satisfying Rule MISRA_RULE_1.3 (Required). Rule MISRA_DIR_4.15 (Required) is not applicable as the code uses integer arithmetic, not floating-point expressions. The code is syntactically correct and would compile without errors, thus complying with Rule MISRA_DIR_2.1 (Required). Rules MISRA_RULE_8.16 (Advisory) and MISRA_RULE_7.1 (Required) are not relevant to the provided code snippet.",
        "cited_rules": ["Rule MISRA_RULE_1.3", "Rule MISRA_DIR_4.15", "Rule MISRA_DIR_2.1"],
        "is_compliant": True,
        "iteration_count": 0,
    }

    result_pass = await critique_node(state_pass)
    print("Expected Approved : True")
    print(f"Got Approved      : {result_pass.get('critique_approved')}")
    print(f"Feedback          : {result_pass.get('critique_feedback')}")
    print(
        f"Tokens            : {result_pass.get('total_tokens')} (Cost: ${result_pass.get('estimated_cost', 0):.6f})\n"
    )

    # ---------------------------------------------------------
    # Test Case 2: FAILING CRITIQUE (Rule Hallucination)
    # ---------------------------------------------------------
    print("--- Test Case 2: FAILING CRITIQUE (Hallucination) ---")
    state_hallucination: ComplianceState = {
        "code_snippet": "int *p = (int *)0x1234;",
        "retrieved_rules": [{"rule_id": "Rule 11.4"}],  # Only 11.4 was retrieved
        "validation_result": "The code violates Rule 11.6 (Required) because it casts an integer to a pointer.",
        "cited_rules": ["Rule 11.6"],  # Agent hallucinates 11.6
        "is_compliant": False,
        "iteration_count": 0,
    }

    result_fail_hallucination = await critique_node(state_hallucination)
    print("Expected Approved : False")
    print(f"Got Approved      : {result_fail_hallucination.get('critique_approved')}")
    print(f"Feedback          : {result_fail_hallucination.get('critique_feedback')}")
    print(
        f"Tokens            : {result_fail_hallucination.get('total_tokens')} (Cost: ${result_fail_hallucination.get('estimated_cost', 0):.6f})\n"
    )

    # ---------------------------------------------------------
    # Test Case 3: FAILING CRITIQUE (Logical Inconsistency)
    # ---------------------------------------------------------
    print("--- Test Case 3: FAILING CRITIQUE (Logic Error) ---")
    state_logic_error: ComplianceState = {
        "code_snippet": "void func() { return; }",
        "retrieved_rules": [{"rule_id": "Rule 2.1"}],
        "validation_result": "The code violates Rule 2.1 (Required) because it contains unreachable code.",
        "cited_rules": ["Rule 2.1"],
        "is_compliant": True,  # <--- Logic Error: text says violated, boolean says compliant
        "iteration_count": 0,
    }

    result_fail_logic = await critique_node(state_logic_error)
    print("Expected Approved : False")
    print(f"Got Approved      : {result_fail_logic.get('critique_approved')}")
    print(f"Feedback          : {result_fail_logic.get('critique_feedback')}")
    print(
        f"Tokens            : {result_fail_logic.get('total_tokens')} (Cost: ${result_fail_logic.get('estimated_cost', 0):.6f})\n"
    )

    # ---------------------------------------------------------
    # Test Case 4: FAILING CRITIQUE (Missing Rule Category format)
    # ---------------------------------------------------------
    print("--- Test Case 4: FAILING CRITIQUE (Missing Category) ---")
    state_format_error: ComplianceState = {
        "code_snippet": "int main() { return 0; }",
        "retrieved_rules": [{"rule_id": "Rule 8.4"}],
        "validation_result": "The function main does not have a visible prototype, violating Rule 8.4.",
        "cited_rules": ["Rule 8.4"],
        "is_compliant": False,  # Text matches boolean, but format "Rule 8.4 (Category)" is missing
        "iteration_count": 0,
    }

    result_fail_format = await critique_node(state_format_error)
    print("Expected Approved : False")
    print(f"Got Approved      : {result_fail_format.get('critique_approved')}")
    print(f"Feedback          : {result_fail_format.get('critique_feedback')}")
    print(
        f"Tokens            : {result_fail_format.get('total_tokens')} (Cost: ${result_fail_format.get('estimated_cost', 0):.6f})\n"
    )

    print("==================================================")
    print("✅ Critique Live Test Complete")
    print("==================================================")


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_live_tests())
