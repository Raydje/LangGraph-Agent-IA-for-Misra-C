"""
Integration tests — Validation node.

Verifies that the validation LLM produces correct compliance verdicts for
three canonical scenarios taken directly from the non-regression scripts.

Node contract
-------------
Input  : ComplianceState {query, code_snippet, retrieved_rules,
                          critique_feedback, iteration_count}
Output : dict containing {is_compliant, validation_result, confidence_score,
                          cited_rules, iteration_count, *_tokens, estimated_cost}

Key assertions per test
-----------------------
- is_compliant must match the ground-truth boolean for the given input.
- validation_result must be a non-empty string explanation.
- confidence_score must be in [0.0, 1.0].
- cited_rules must be a non-empty list of strings.
- iteration_count must be incremented by 1 relative to the input.
"""

import pytest

from app.graph.nodes.validation import validation_node
from app.models.state import ComplianceState

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Shared rule fixtures (mirror the live_test_validation.py rule dicts)
# ---------------------------------------------------------------------------

_RULE_10_4 = {
    "rule_id": "Rule MISRA_10.4",
    "category": "Required",
    "title": (
        "Both operands of an operator in which the usual arithmetic conversions "
        "are performed shall have the same essential type category"
    ),
    "full_text": "To avoid implicit casting issues, operands should match.",
}

_RULE_8_4 = {
    "rule_id": "Rule MISRA_8.4",
    "category": "Required",
    "title": (
        "A compatible declaration shall be visible when an object or function "
        "with external linkage is defined"
    ),
    "full_text": (
        "If a function is defined, there must be a prototype in an included header "
        "or earlier in the file to ensure type safety across translation units."
    ),
}

_RULE_11_4 = {
    "rule_id": "Rule MISRA_11.4",
    "category": "Advisory",
    "title": "A conversion should not be performed between a pointer to object and an integer type",
    "full_text": "Casting integers to pointers leads to undefined behavior and hardware-dependent bugs.",
}


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


async def test_validation_compliant_code():
    """
    A simple uint16_t addition function checked against Rule MISRA_10.4 (type
    matching) must be assessed as compliant.

    Rationale: both operands are uint16_t — same essential type category.
    """
    state: ComplianceState = {
        "query": "Is this function compliant?",
        "code_snippet": "uint16_t add(uint16_t a, uint16_t b) {\n    return a + b;\n}",
        "retrieved_rules": [_RULE_10_4],
        "critique_feedback": "",
        "iteration_count": 0,
    }

    result = await validation_node(state)

    assert result["is_compliant"] is True, (
        f"Expected is_compliant=True for compliant uint16_t code. "
        f"Got False.\nValidation result: {result.get('validation_result')}"
    )
    assert isinstance(result["validation_result"], str) and result["validation_result"]
    assert 0.0 <= result["confidence_score"] <= 1.0
    assert isinstance(result["cited_rules"], list) and len(result["cited_rules"]) >= 1
    # iteration_count must be incremented
    assert result["iteration_count"] == 1
    assert result.get("total_tokens", 0) >= 0
    assert result.get("estimated_cost", 0.0) >= 0.0


async def test_validation_non_compliant_code():
    """
    A function without a visible prototype violates Rule MISRA_8.4 (Required).
    The node must return is_compliant=False.
    """
    state: ComplianceState = {
        "query": "Check this code for prototype visibility violations.",
        "code_snippet": "void do_something(void) {\n    /* doing something */\n}",
        "retrieved_rules": [_RULE_8_4],
        "critique_feedback": "",
        "iteration_count": 0,
    }

    result = await validation_node(state)

    assert result["is_compliant"] is False, (
        f"Expected is_compliant=False for code missing a visible prototype. "
        f"Got True.\nValidation result: {result.get('validation_result')}"
    )
    assert isinstance(result["validation_result"], str) and result["validation_result"]
    assert 0.0 <= result["confidence_score"] <= 1.0
    assert isinstance(result["cited_rules"], list) and len(result["cited_rules"]) >= 1
    assert result["iteration_count"] == 1
    assert result.get("total_tokens", 0) >= 0


async def test_validation_incorporates_critique_feedback():
    """
    On a second iteration (iteration_count=1) with critique_feedback set,
    the validation node must incorporate the feedback and produce a revised
    verdict.  The pointer cast 'int *p = (int *)0x1234;' violates Rule
    MISRA_11.4 (Advisory) — is_compliant must remain False.

    This also verifies that the critique re-evaluation pathway is functional:
    the presence of critique_feedback in the prompt causes the LLM to address
    the format requirement raised by the critique.
    """
    state: ComplianceState = {
        "query": "Does this cast violate MISRA?",
        "code_snippet": "int *p = (int *)0x1234;",
        "retrieved_rules": [_RULE_11_4],
        "critique_feedback": (
            "CRITIQUE REJECTION: You forgot to include the rule category in your "
            "explanation. You MUST write 'Rule MISRA_11.4 (Advisory): ...'. Fix this."
        ),
        "iteration_count": 1,
    }

    result = await validation_node(state)

    assert result["is_compliant"] is False, (
        f"Expected is_compliant=False for integer-to-pointer cast. "
        f"Got True.\nValidation result: {result.get('validation_result')}"
    )
    # The revised output must include the category format demanded by the critique
    validation_text: str = result["validation_result"]
    assert "Advisory" in validation_text, (
        "Expected the word 'Advisory' in the revised validation result after "
        f"critique feedback, but it was absent.\nResult: {validation_text}"
    )
    assert result["iteration_count"] == 2  # incremented from input value of 1
    assert result.get("total_tokens", 0) >= 0
