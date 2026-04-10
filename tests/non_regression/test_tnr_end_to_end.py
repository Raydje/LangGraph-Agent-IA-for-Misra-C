"""
Non-regression end-to-end tests — Golden dataset validation.

Each test case sends a real HTTP request to the live docker-compose container
and asserts that the API's is_compliant verdict matches the expected value from
the golden dataset.

Running
-------
    pytest tests/non_regression/ -v

    # Override the container URL (defaults to http://localhost:8000):
    TNR_BASE_URL=http://staging.internal:8000 pytest tests/non_regression/ -v

Dataset
-------
data/golden_dataset.json — 10 entries (5 compliant, 5 non-compliant).
The is_compliant field is stripped from the payload sent to the API and used
only as the expected value for assertion.

Test IDs
--------
IDs are derived from the dataset itself so CI logs are immediately readable:
    non_compliant-does_this_memory_allocation
    compliant-is_this_function_misra

Failure output
--------------
On a verdict mismatch the assertion message includes:
    - expected vs actual is_compliant
    - confidence_score returned by the model
    - first 400 chars of final_response
    - cited_rules list
"""

import json
from pathlib import Path

import httpx
import pytest

# ---------------------------------------------------------------------------
# Golden dataset loading
# ---------------------------------------------------------------------------

_DATASET_PATH = Path(__file__).parent.parent.parent / "data" / "golden_dataset.json"


def _load_golden_cases() -> tuple[list[dict], list[str]]:
    """
    Parse the golden dataset and return (cases, ids).

    Each case dict contains:
        payload   — dict sent verbatim to POST /api/v1/query
        expected  — bool, the ground-truth is_compliant value
    """
    raw: list[dict] = json.loads(_DATASET_PATH.read_text(encoding="utf-8"))

    cases: list[dict] = []
    ids: list[str] = []

    for entry in raw:
        expected = entry["is_compliant"]

        cases.append(
            {
                "payload": {
                    "query": entry["query"],
                    "code_snippet": entry["code_snippet"],
                    "standard": entry["standard"],
                },
                "expected": expected,
            }
        )

        # Build a readable ID:  compliant-first_four_words_of_query
        prefix = "compliant" if expected else "non_compliant"
        words = entry["query"].lower().split()[:4]
        slug = "_".join(w.strip(".,?!") for w in words)
        ids.append(f"{prefix}-{slug}")

    return cases, ids


_golden_cases, _golden_ids = _load_golden_cases()


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------


@pytest.mark.non_regression
@pytest.mark.parametrize("case", _golden_cases, ids=_golden_ids)
def test_compliance_golden_case(case: dict, tnr_client: httpx.Client) -> None:
    """
    Send one golden dataset entry to the live API and assert the verdict.

    The test fails if:
      - The HTTP response is not 200.
      - is_compliant is None (model could not decide).
      - is_compliant does not match the expected ground-truth value.
    """
    payload: dict = case["payload"]
    expected: bool = case["expected"]

    response = tnr_client.post("/api/v1/query", json=payload)

    assert response.status_code == 200, (
        f"Expected HTTP 200, got {response.status_code}.\n"
        f"Query   : {payload['query'][:120]}\n"
        f"Response: {response.text[:400]}"
    )

    data: dict = response.json()

    is_compliant: bool | None = data.get("is_compliant")

    assert is_compliant is not None, (
        "API returned is_compliant=None — the model could not reach a verdict.\n"
        f"Query         : {payload['query'][:120]}\n"
        f"final_response: {data.get('final_response', '')[:400]}"
    )

    assert is_compliant == expected, (
        f"Verdict mismatch.\n"
        f"  Expected      : is_compliant={expected}\n"
        f"  Got           : is_compliant={is_compliant}\n"
        f"  Confidence    : {data.get('confidence_score')}\n"
        f"  Cited rules   : {data.get('cited_rules', [])}\n"
        f"  final_response: {data.get('final_response', '')[:400]}"
    )
