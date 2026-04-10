# tests/unit/services/test_llm_service.py
"""
Unit tests for app/services/llm_service.py.

Patches ChatGoogleGenerativeAI so no real API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from app.services.llm_service import get_llm, get_structured_llm

# ---------------------------------------------------------------------------
# get_llm
# ---------------------------------------------------------------------------


@patch("app.services.llm_service.get_settings")
@patch("app.services.llm_service.ChatGoogleGenerativeAI")
def test_get_llm_instantiates_with_settings_values(mock_llm_cls, mock_get_settings):
    mock_llm_cls.return_value = MagicMock()
    mock_get_settings.return_value.gemini_api_key = "test-fake-gemini-key"
    mock_get_settings.return_value.gemini_model = "gemini-2.5-flash"

    get_llm(temperature=0.5, timeout=30)

    mock_llm_cls.assert_called_once()
    call_kwargs = mock_llm_cls.call_args[1]
    assert call_kwargs["temperature"] == 0.5
    assert call_kwargs["request_timeout"] == 30
    # model and api_key come from patched settings
    assert call_kwargs["google_api_key"] == "test-fake-gemini-key"
    assert call_kwargs["model"] == "gemini-2.5-flash"


@patch("app.services.llm_service.ChatGoogleGenerativeAI")
def test_get_llm_returns_llm_instance(mock_llm_cls):
    fake_llm = MagicMock()
    mock_llm_cls.return_value = fake_llm

    result = get_llm()

    assert result is fake_llm


# ---------------------------------------------------------------------------
# get_structured_llm
# ---------------------------------------------------------------------------


class _DummySchema(BaseModel):
    value: str


@patch("app.services.llm_service.ChatGoogleGenerativeAI")
def test_get_structured_llm_calls_with_structured_output(mock_llm_cls):
    fake_llm = MagicMock()
    fake_chain = MagicMock()
    fake_llm.with_structured_output.return_value = fake_chain
    mock_llm_cls.return_value = fake_llm

    result = get_structured_llm(_DummySchema)

    fake_llm.with_structured_output.assert_called_once_with(_DummySchema, include_raw=True)
    assert result is fake_chain


@patch("app.services.llm_service.ChatGoogleGenerativeAI")
def test_get_structured_llm_raw_bool_false_passes_through(mock_llm_cls):
    fake_llm = MagicMock()
    mock_llm_cls.return_value = fake_llm

    get_structured_llm(_DummySchema, raw_bool=False)

    call_kwargs = fake_llm.with_structured_output.call_args[1]
    assert call_kwargs["include_raw"] is False
