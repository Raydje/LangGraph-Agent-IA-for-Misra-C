import asyncio
import os
from dotenv import load_dotenv

# Load environment variables (e.g., GEMINI_API_KEY)
load_dotenv()

from app.graph.nodes.validation import validation_node
from app.models.state import ComplianceState  # noqa: F401 — used for state type hints below

async def run_live_tests():
    print("==================================================")
    print("🔬 Starting Live Test for Validation Node...")
    print("==================================================\n")

    # ---------------------------------------------------------
    # Test Case 1: COMPLIANT CODE
    # ---------------------------------------------------------
    print("--- Test Case 1: Compliant Code ---")
    state_compliant: ComplianceState = {
        "query": "Is this function compliant?",
        "code_snippet": "uint16_t add(uint16_t a, uint16_t b) {\n    return a + b;\n}",
        "retrieved_rules": [
            {
                "rule_id": "Rule MISRA_10.4",
                "category": "Required",
                "title": "Both operands of an operator in which the usual arithmetic conversions are performed shall have the same essential type category",
                "full_text": "To avoid implicit casting issues, operands should match."
            }
        ],
        "critique_feedback": "",
        "iteration_count": 0
    }
    
    result_compliant = await validation_node(state_compliant)
    print(f"Is Compliant     : {result_compliant.get('is_compliant')} (Expected: True)")
    print(f"Confidence       : {result_compliant.get('confidence_score')}")
    print(f"Cited Rules      : {result_compliant.get('cited_rules')}")
    print(f"Validation Result:\n{result_compliant.get('validation_result')}")
    print(f"Tokens           : {result_compliant.get('total_tokens')} (Cost: ${result_compliant.get('estimated_cost', 0):.6f})\n")

    # ---------------------------------------------------------
    # Test Case 2: NON-COMPLIANT CODE (Rule Violation)
    # ---------------------------------------------------------
    print("--- Test Case 2: Non-Compliant Code ---")
    state_violating: ComplianceState = {
        "query": "Check this code for prototype visibility violations.",
        "code_snippet": "void do_something(void) {\n    /* doing something */\n}",
        "retrieved_rules": [
            {
                "rule_id": "Rule MISRA_8.4",
                "category": "Required",
                "title": "A compatible declaration shall be visible when an object or function with external linkage is defined",
                "full_text": "If a function is defined, there must be a prototype in an included header or earlier in the file to ensure type safety across translation units."
            }
        ],
        "critique_feedback": "",
        "iteration_count": 0
    }
    
    result_violating = await validation_node(state_violating)
    print(f"Is Compliant     : {result_violating.get('is_compliant')} (Expected: False)")
    print(f"Confidence       : {result_violating.get('confidence_score')}")
    print(f"Cited Rules      : {result_violating.get('cited_rules')}")
    print(f"Validation Result:\n{result_violating.get('validation_result')}")
    print(f"Tokens           : {result_violating.get('total_tokens')} (Cost: ${result_violating.get('estimated_cost', 0):.6f})\n")

    # ---------------------------------------------------------
    # Test Case 3: CRITIQUE FEEDBACK INCORPORATION
    # ---------------------------------------------------------
    print("--- Test Case 3: Re-evaluation after Critique ---")
    state_critique: ComplianceState = {
        "query": "Does this cast violate MISRA?",
        "code_snippet": "int *p = (int *)0x1234;",
        "retrieved_rules": [
            {
                "rule_id": "Rule MISRA_11.4",
                "category": "Advisory",
                "title": "A conversion should not be performed between a pointer to object and an integer type",
                "full_text": "Casting integers to pointers leads to undefined behavior and hardware-dependent bugs."
            }
        ],
        "critique_feedback": "CRITIQUE REJECTION: You forgot to include the rule category in your explanation. You MUST write 'Rule MISRA_11.4 (Advisory): ...'. Fix this.",
        "iteration_count": 1
    }
    
    result_critique = await validation_node(state_critique)
    print(f"Is Compliant     : {result_critique.get('is_compliant')} (Expected: False)")
    print(f"Confidence       : {result_critique.get('confidence_score')}")
    print(f"Cited Rules      : {result_critique.get('cited_rules')}")
    print(f"Validation Result:\n{result_critique.get('validation_result')}")
    print(f"Tokens           : {result_critique.get('total_tokens')} (Cost: ${result_critique.get('estimated_cost', 0):.6f})\n")

    print("==================================================")
    print("✅ Validation Live Test Complete")
    print("==================================================")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(run_live_tests())