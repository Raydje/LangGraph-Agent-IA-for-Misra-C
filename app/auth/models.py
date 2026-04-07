from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field

# ---------------------------------------------------------------------------
# Inbound request bodies
# ---------------------------------------------------------------------------


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    # Present only when caller wants admin scopes; validated against settings.admin_registration_token
    admin_token: str | None = None


class APIKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    scopes: list[str] = ["query:read"]
    expires_at: datetime | None = None


class RefreshRequest(BaseModel):
    refresh_token: str


# ---------------------------------------------------------------------------
# Outbound response bodies
# ---------------------------------------------------------------------------


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until access token expires


class APIKeyResponse(BaseModel):
    """Returned once at creation time — full_key is never retrievable again."""

    key_id: str
    name: str
    full_key: str
    scopes: list[str]
    expires_at: datetime | None


class APIKeyInfo(BaseModel):
    """Safe listing representation — no secret material."""

    key_id: str
    name: str
    scopes: list[str]
    expires_at: datetime | None
    last_used_at: datetime | None
    is_active: bool
    created_at: datetime


# ---------------------------------------------------------------------------
# Internal / dependency injection
# ---------------------------------------------------------------------------


class Principal(BaseModel):
    """Resolved identity propagated through authenticated request handlers."""

    user_id: str
    email: str
    scopes: list[str]
    auth_method: Literal["jwt", "api_key"]
    key_id: str | None = None  # populated only when auth_method == "api_key"

    def has_scope(self, required: str) -> bool:
        return required in self.scopes or "admin:all" in self.scopes
