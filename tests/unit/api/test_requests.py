# tests/unit/api/test_requests.py
"""
Unit tests for app/api/v1/requests.py — ComplianceQueryRequest validation.

The model_validator calls get_settings() to read max_input_length.
The session-scoped override_settings fixture (conftest.py) already patches
this to a Settings with max_input_length=3000, so no extra patching needed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.v1.requests import ComplianceQueryRequest

# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_valid_minimal_request():
    req = ComplianceQueryRequest(query="Is this compliant?")
    assert req.query == "Is this compliant?"
    assert req.standard == "MISRA C:2023"
    assert req.code_snippet is None
    assert req.thread_id is None


def test_valid_full_request():
    req = ComplianceQueryRequest(
        query="Validate this",
        code_snippet="int x = 0;",
        standard="MISRA C:2023",
        thread_id="abc-123",
    )
    assert req.code_snippet == "int x = 0;"
    assert req.thread_id == "abc-123"


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_empty_query_raises_validation_error():
    with pytest.raises(ValidationError):
        ComplianceQueryRequest(query="")


def test_query_exceeds_max_length_raises_validation_error():
    with pytest.raises(ValidationError, match="query must be at most"):
        ComplianceQueryRequest(query="x" * 3001)


def test_code_snippet_exceeds_max_length_raises_validation_error():
    with pytest.raises(ValidationError, match="code_snippet must be at most"):
        ComplianceQueryRequest(query="valid query", code_snippet="y" * 3001)


def test_standard_exceeds_max_length_raises_validation_error():
    with pytest.raises(ValidationError, match="standard must be at most"):
        ComplianceQueryRequest(query="valid query", standard="s" * 3001)


def test_thread_id_exceeds_100_chars_raises_validation_error():
    with pytest.raises(ValidationError):
        ComplianceQueryRequest(query="valid query", thread_id="t" * 101)


def test_empty_standard_raises_validation_error():
    with pytest.raises(ValidationError):
        ComplianceQueryRequest(query="valid query", standard="")
