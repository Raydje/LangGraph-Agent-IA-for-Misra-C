"""
Integration tests — Critique node.

Verifies that the critique LLM correctly approves well-formed validation
outputs and rejects outputs that contain hallucinations, logical
inconsistencies, or formatting errors.

Node contract
-------------
Input  : ComplianceState {code_snippet, retrieved_rules, validation_result,
                          cited_rules, is_compliant, iteration_count}
Output : dict containing {critique_approved, critique_feedback,
                          critique_history, *_tokens, estimated_cost}

Test matrix (mirrors live_test_critique.py scenarios)
------------------------------------------------------
1. Perfect validation         → critique_approved=True
2. Rule hallucination         → critique_approved=False
3. Logical inconsistency      → critique_approved=False
4. Missing rule category fmt  → critique_approved=False
"""

import pytest

from app.graph.nodes.critique import critique_node
from app.models.state import ComplianceState

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_base_shape(result: dict) -> None:
    """Assert that the critique node always returns the minimum required keys."""
    assert isinstance(result.get("critique_approved"), bool)
    assert isinstance(result.get("critique_feedback"), str) and result["critique_feedback"]
    assert isinstance(result.get("critique_history"), list)
    assert result.get("total_tokens", 0) >= 0
    assert result.get("estimated_cost", 0.0) >= 0.0


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


async def test_critique_approves_perfect_validation():
    """
    A well-formed validation that correctly references only retrieved rules,
    is logically consistent with is_compliant=True, and includes rule
    categories must be approved by the critique node.
    """
    state: ComplianceState = {
        "code_snippet": "int divide(int a, int b) { if (b == 0) return 0; return a / b; }",
        "retrieved_rules": [
            {"rule_id": "MISRA_RULE_7.1"},
            {"rule_id": "MISRA_DIR_4.15"},
            {"rule_id": "MISRA_RULE_1.3"},
            {"rule_id": "MISRA_RULE_8.16"},
            {"rule_id": "MISRA_DIR_2.1"},
        ],
        "validation_result": (
            "Validation Complete.\n"
            "Standard: MISRA C:2023\n"
            "Compliant: True\n"
            "Confidence: 100%\n"
            "Cited rules: Rule MISRA_RULE_1.3, Rule MISRA_DIR_4.15, Rule MISRA_DIR_2.1\n"
            "Details: The provided code is compliant with the applicable rules regarding "
            "division by zero. The function `divide` explicitly checks for `b == 0` on "
            "line 1. If `b` is zero, it returns `0`, thereby preventing the integer "
            "division `a / b` from occurring. This avoids undefined behavior associated "
            "with integer division by zero, satisfying Rule MISRA_RULE_1.3 (Required). "
            "Rule MISRA_DIR_4.15 (Required) is not applicable as the code uses integer "
            "arithmetic, not floating-point expressions. The code is syntactically "
            "correct and would compile without errors, thus complying with Rule "
            "MISRA_DIR_2.1 (Required). Rules MISRA_RULE_8.16 (Advisory) and "
            "MISRA_RULE_7.1 (Required) are not relevant to the provided code snippet."
        ),
        "cited_rules": ["Rule MISRA_RULE_1.3", "Rule MISRA_DIR_4.15", "Rule MISRA_DIR_2.1"],
        "is_compliant": True,
        "iteration_count": 0,
    }

    result = await critique_node(state)

    _assert_base_shape(result)
    assert result["critique_approved"] is True, (
        f"Expected critique_approved=True for a perfect validation. "
        f"Got False.\nFeedback: {result['critique_feedback']}"
    )


async def test_critique_rejects_hallucinated_rule():
    """
    If the validation cites Rule 11.6 but only Rule 11.4 was retrieved
    (i.e., the agent hallucinated a rule ID), the critique must reject it.

    This guards against the most dangerous failure mode: an agent confidently
    citing non-existent or irrelevant MISRA rules.
    """
    state: ComplianceState = {
        "code_snippet": "int *p = (int *)0x1234;",
        "retrieved_rules": [{"rule_id": "Rule 11.4"}],  # Only 11.4 was retrieved
        "validation_result": (
            "The code violates Rule 11.6 (Required) because it casts an integer to a pointer."
        ),
        "cited_rules": ["Rule 11.6"],  # Agent cites 11.6 — hallucination
        "is_compliant": False,
        "iteration_count": 0,
    }

    result = await critique_node(state)

    _assert_base_shape(result)
    assert result["critique_approved"] is False, (
        f"Expected critique_approved=False for a hallucinated rule ID. "
        f"Got True.\nFeedback: {result['critique_feedback']}"
    )


async def test_critique_rejects_logical_inconsistency():
    """
    If the validation text says the code violates a rule but is_compliant=True,
    the critique must detect the logical contradiction and reject it.
    """
    state: ComplianceState = {
        "code_snippet": "void func() { return; }",
        "retrieved_rules": [{"rule_id": "Rule 2.1"}],
        "validation_result": (
            "The code violates Rule 2.1 (Required) because it contains unreachable code."
        ),
        "cited_rules": ["Rule 2.1"],
        "is_compliant": True,  # Contradiction: text says violated, boolean says compliant
        "iteration_count": 0,
    }

    result = await critique_node(state)

    _assert_base_shape(result)
    assert result["critique_approved"] is False, (
        f"Expected critique_approved=False for a logical inconsistency "
        f"(is_compliant=True while text says 'violates'). "
        f"Got True.\nFeedback: {result['critique_feedback']}"
    )


async def test_critique_rejects_missing_rule_category_format():
    """
    If a cited rule is written as 'Rule 8.4' without the category qualifier
    '(Required)' or '(Advisory)', Criterion 4 of the critique (Standard
    Accuracy) requires rejection.

    The format 'Rule ID (Category): ...' is mandatory per the system prompt.
    """
    state: ComplianceState = {
        "code_snippet": "int main() { return 0; }",
        "retrieved_rules": [{"rule_id": "Rule 8.4"}],
        "validation_result": (
            "The function main does not have a visible prototype, violating Rule 8.4."
            # Missing the '(Category)' qualifier — must be 'Rule 8.4 (Required): ...'
        ),
        "cited_rules": ["Rule 8.4"],
        "is_compliant": False,  # is_compliant matches text — only format is wrong
        "iteration_count": 0,
    }

    result = await critique_node(state)

    _assert_base_shape(result)
    assert result["critique_approved"] is False, (
        f"Expected critique_approved=False for a validation missing the rule "
        f"category format '(Category)'. "
        f"Got True.\nFeedback: {result['critique_feedback']}"
    )
