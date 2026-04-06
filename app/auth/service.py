"""
Auth service — all cryptographic primitives in one place.

Rules:
- Passwords: bcrypt directly (passlib 1.7.4 is incompatible with bcrypt>=4)
- API keys: bcrypt hash of the secret portion; key_id enables O(1) DB lookup
- JWTs: HS256 signed with JWT_SECRET_KEY from settings
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import jwt

from app.config import get_settings


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

def hash_password(plaintext: str) -> str:
    return bcrypt.hashpw(plaintext.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plaintext: str, hashed: str) -> bool:
    return bcrypt.checkpw(plaintext.encode("utf-8"), hashed.encode("utf-8"))


# ---------------------------------------------------------------------------
# API keys
#
# Format: ak_<key_id>_<secret>
#   key_id  — 8 hex chars — stored in plaintext, used for fast DB lookup
#   secret  — 43 url-safe base64 chars (32 bytes) — bcrypt hashed in DB
# ---------------------------------------------------------------------------

def generate_api_key() -> tuple[str, str, str]:
    """Return (full_key, key_id, key_hash).

    full_key  — shown to the caller exactly once; never stored.
    key_id    — stored plaintext; used to retrieve the matching hash from DB.
    key_hash  — bcrypt hash of secret; stored in DB.
    """
    key_id = secrets.token_hex(4)          # 8 hex chars
    secret = secrets.token_urlsafe(32)     # 43 url-safe base64 chars
    full_key = f"ak_{key_id}_{secret}"
    key_hash = bcrypt.hashpw(secret.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    return full_key, key_id, key_hash


def verify_api_key_secret(secret: str, key_hash: str) -> bool:
    return bcrypt.checkpw(secret.encode("utf-8"), key_hash.encode("utf-8"))


def parse_api_key(full_key: str) -> tuple[str, str]:
    """Split 'ak_<key_id>_<secret>' → (key_id, secret).

    Raises ValueError on malformed input.
    """
    parts = full_key.split("_", 2)
    if len(parts) != 3 or parts[0] != "ak":
        raise ValueError("Malformed API key")
    return parts[1], parts[2]


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------

def create_access_token(user_id: str, email: str, scopes: list[str]) -> tuple[str, int]:
    """Return (jwt_string, expires_in_seconds)."""
    settings = get_settings()
    delta = timedelta(minutes=settings.jwt_access_token_expire_minutes)
    expire = datetime.now(timezone.utc) + delta
    payload = {
        "sub": user_id,
        "email": email,
        "scopes": scopes,
        "exp": expire,
        "type": "access",
    }
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm="HS256")
    return token, int(delta.total_seconds())


def create_refresh_token(user_id: str) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_token_expire_days)
    payload = {
        "sub": user_id,
        "exp": expire,
        "jti": secrets.token_hex(16),  # unique token ID — enables per-token revocation
        "type": "refresh",
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm="HS256")


def decode_token(token: str) -> dict:
    """Decode and verify signature + expiry. Raises jose.JWTError on any failure."""
    settings = get_settings()
    return jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
