# tests/unit/auth/test_auth_service.py
"""
Unit tests for app/auth/service.py.

All cryptographic operations are exercised with real bcrypt/jose calls —
no mocking needed because there are no network calls involved.
"""
from __future__ import annotations

import pytest
from jose import jwt, JWTError

from app.auth.service import (
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    hash_password,
    parse_api_key,
    verify_api_key_secret,
    verify_password,
)


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

def test_hash_password_returns_bcrypt_string():
    hashed = hash_password("correct-horse-battery")
    # bcrypt hashes start with $2b$ (or $2a$)
    assert hashed.startswith("$2")


def test_verify_password_correct_plaintext_returns_true():
    plaintext = "correct-horse-battery"
    hashed = hash_password(plaintext)
    assert verify_password(plaintext, hashed) is True


def test_verify_password_wrong_plaintext_returns_false():
    hashed = hash_password("correct-horse-battery")
    assert verify_password("wrong-password", hashed) is False


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

def test_generate_api_key_returns_three_tuple():
    full_key, key_id, key_hash = generate_api_key()
    assert isinstance(full_key, str)
    assert isinstance(key_id, str)
    assert isinstance(key_hash, str)


def test_generate_api_key_full_key_has_ak_prefix():
    full_key, _, _ = generate_api_key()
    assert full_key.startswith("ak_")


def test_generate_api_key_key_id_is_8_hex_chars():
    _, key_id, _ = generate_api_key()
    assert len(key_id) == 8
    int(key_id, 16)  # raises ValueError if not valid hex


def test_verify_api_key_secret_correct_secret_returns_true():
    full_key, key_id, key_hash = generate_api_key()
    _, secret = parse_api_key(full_key)
    assert verify_api_key_secret(secret, key_hash) is True


def test_verify_api_key_secret_wrong_secret_returns_false():
    _, _, key_hash = generate_api_key()
    assert verify_api_key_secret("not-the-real-secret", key_hash) is False


def test_parse_api_key_happy_path():
    full_key, expected_key_id, _ = generate_api_key()
    key_id, secret = parse_api_key(full_key)
    assert key_id == expected_key_id
    assert len(secret) > 0


def test_parse_api_key_wrong_prefix_raises_value_error():
    with pytest.raises(ValueError, match="Malformed API key"):
        parse_api_key("sk_abc123_somesecret")


def test_parse_api_key_too_few_segments_raises_value_error():
    with pytest.raises(ValueError, match="Malformed API key"):
        parse_api_key("ak_onlytwoparts")


# ---------------------------------------------------------------------------
# JWT — access token
# ---------------------------------------------------------------------------

def test_create_access_token_returns_decodable_jwt():
    token, expires_in = create_access_token("user-1", "a@b.com", ["query:read"])
    assert isinstance(token, str)
    assert expires_in > 0


def test_create_access_token_has_correct_claims():
    token, _ = create_access_token("user-42", "test@example.com", ["query:read", "admin:all"])
    payload = decode_token(token)
    assert payload["sub"] == "user-42"
    assert payload["email"] == "test@example.com"
    assert payload["scopes"] == ["query:read", "admin:all"]
    assert payload["type"] == "access"


# ---------------------------------------------------------------------------
# JWT — refresh token
# ---------------------------------------------------------------------------

def test_create_refresh_token_returns_decodable_jwt():
    token = create_refresh_token("user-1")
    payload = decode_token(token)
    assert payload["sub"] == "user-1"
    assert payload["type"] == "refresh"
    assert "jti" in payload  # unique token ID must be present


def test_create_refresh_token_jti_is_unique():
    t1 = create_refresh_token("user-1")
    t2 = create_refresh_token("user-1")
    p1 = decode_token(t1)
    p2 = decode_token(t2)
    assert p1["jti"] != p2["jti"]


# ---------------------------------------------------------------------------
# JWT — decode_token error paths
# ---------------------------------------------------------------------------

def test_decode_token_raises_on_tampered_signature():
    token, _ = create_access_token("u", "u@u.com", [])
    # Flip one character in the signature segment
    parts = token.split(".")
    tampered = parts[0] + "." + parts[1] + "." + parts[2][:-1] + ("A" if parts[2][-1] != "A" else "B")
    with pytest.raises(JWTError):
        decode_token(tampered)
