# tests/unit/test_config.py
"""
Unit tests for app/config.py — Settings validation and computed properties.

Settings uses pydantic-settings which reads env vars as fallback.
Required-field tests temporarily remove the relevant env var so validation
fires correctly, then restore it.
"""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from app.config import Settings

_REQUIRED = {
    "gemini_api_key": "g-key",
    "pinecone_api_key": "p-key",
    "mongodb_uri": "mongodb://localhost/test",
    "jwt_secret_key": "jwt-secret",
}


# ---------------------------------------------------------------------------
# Required fields — env vars must be absent for validation to fire
# ---------------------------------------------------------------------------


def test_settings_instantiates_with_all_required_fields():
    s = Settings(**_REQUIRED)
    assert s.gemini_api_key == "g-key"
    assert s.pinecone_api_key == "p-key"


def test_settings_missing_gemini_api_key_raises():
    kwargs = {k: v for k, v in _REQUIRED.items() if k != "gemini_api_key"}
    env_key = "GEMINI_API_KEY"
    old = os.environ.pop(env_key, None)
    try:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **kwargs)
    finally:
        if old is not None:
            os.environ[env_key] = old


def test_settings_missing_jwt_secret_key_raises():
    kwargs = {k: v for k, v in _REQUIRED.items() if k != "jwt_secret_key"}
    env_key = "JWT_SECRET_KEY"
    old = os.environ.pop(env_key, None)
    try:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **kwargs)
    finally:
        if old is not None:
            os.environ[env_key] = old


def test_settings_missing_pinecone_api_key_raises():
    kwargs = {k: v for k, v in _REQUIRED.items() if k != "pinecone_api_key"}
    env_key = "PINECONE_API_KEY"
    old = os.environ.pop(env_key, None)
    try:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **kwargs)
    finally:
        if old is not None:
            os.environ[env_key] = old


def test_settings_missing_mongodb_uri_raises():
    kwargs = {k: v for k, v in _REQUIRED.items() if k != "mongodb_uri"}
    env_key = "MONGODB_URI"
    old = os.environ.pop(env_key, None)
    try:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **kwargs)
    finally:
        if old is not None:
            os.environ[env_key] = old


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_settings_default_gemini_model():
    s = Settings(**_REQUIRED)
    assert s.gemini_model == "gemini-2.5-flash"


def test_settings_default_max_input_length():
    s = Settings(**_REQUIRED)
    assert s.max_input_length == 3000


def test_settings_default_cors_origins_contain_localhost():
    s = Settings(**_REQUIRED)
    assert any("localhost" in o for o in s.cors_allowed_origins)


# ---------------------------------------------------------------------------
# redis_uri property
# ---------------------------------------------------------------------------


def test_redis_uri_no_auth_returns_plain_url():
    s = Settings(**_REQUIRED, redis_host="myredis", redis_port=6380, redis_password="")
    assert s.redis_uri == "redis://myredis:6380"


def test_redis_uri_with_password_includes_credentials():
    s = Settings(**_REQUIRED, redis_host="myredis", redis_port=6379, redis_user="admin", redis_password="secret")
    assert "admin:secret@myredis" in s.redis_uri


# ---------------------------------------------------------------------------
# set_model_pricing validator
# ---------------------------------------------------------------------------


def test_known_model_populates_pricing():
    s = Settings(**_REQUIRED, gemini_model="gemini-2.5-flash")
    assert s.llm_input_cost_per_1m == 0.30
    assert s.llm_output_cost_per_1m == 2.50


def test_unknown_model_falls_back_to_flash_pricing():
    s = Settings(**_REQUIRED, gemini_model="totally-unknown-model-xyz")
    # Falls back to gemini-2.5-flash pricing
    assert s.llm_input_cost_per_1m == 0.30
    assert s.llm_output_cost_per_1m == 2.50
