"""
Auth router — mounted at /api/v1/auth

Public endpoints (no token required):
    POST /auth/register           — create account (default: query:read scope)
    POST /auth/token              — OAuth2 password flow → access + refresh tokens
    POST /auth/refresh            — rotate refresh token → new access + refresh tokens

Authenticated endpoints:
    POST   /auth/api-keys         — create an API key (requires any valid token)
    GET    /auth/api-keys         — list caller's active API keys
    DELETE /auth/api-keys/{key_id} — revoke an API key

Admin self-registration:
    Include {"admin_token": "<ADMIN_REGISTRATION_TOKEN>"} in /register body
    to receive scopes ["query:read", "admin:seed", "admin:replay", "admin:all"].
    Set ADMIN_REGISTRATION_TOKEN= in .env (empty = feature disabled).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Security, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError

from app.auth.dependencies import get_current_principal
from app.auth.models import (
    APIKeyCreate,
    APIKeyInfo,
    APIKeyResponse,
    Principal,
    RefreshRequest,
    TokenResponse,
    UserCreate,
)
from app.auth.service import (
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    hash_password,
    verify_password,
)
from app.config import get_settings

auth_router = APIRouter(prefix="/auth", tags=["Auth"])

_ADMIN_SCOPES = ["query:read", "admin:seed", "admin:replay", "admin:all"]
_DEFAULT_SCOPES = ["query:read"]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@auth_router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(body: UserCreate, request: Request):
    """Create a new user account.

    Pass ``admin_token`` matching ``ADMIN_REGISTRATION_TOKEN`` in .env to
    receive full admin scopes. Leave it out for a standard ``query:read`` account.
    """
    settings = get_settings()
    db = request.app.state.mongodb.db

    if await db["users"].find_one({"email": body.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    # Determine scopes based on whether a valid admin token was supplied
    if body.admin_token:
        if not settings.admin_registration_token:
            raise HTTPException(status_code=403, detail="Admin registration is disabled")
        if body.admin_token != settings.admin_registration_token:
            raise HTTPException(status_code=403, detail="Invalid admin token")
        scopes = _ADMIN_SCOPES
    else:
        scopes = _DEFAULT_SCOPES

    user_id = str(uuid.uuid4())
    await db["users"].insert_one(
        {
            "_id": user_id,
            "email": body.email,
            "hashed_password": hash_password(body.password),
            "scopes": scopes,
            "is_active": True,
            "refresh_tokens": [],
            "created_at": datetime.now(UTC),
            # Usage tracking (updated atomically by UsageService after each query)
            "total_cost": 0.0,
            "total_requests": 0,
        }
    )

    return {"user_id": user_id, "email": body.email, "scopes": scopes}


# ---------------------------------------------------------------------------
# Token issuance (OAuth2 password flow)
# ---------------------------------------------------------------------------


@auth_router.post("/token", response_model=TokenResponse)
async def login(request: Request, form: OAuth2PasswordRequestForm = Depends()):
    """Exchange email + password for an access token and refresh token.

    Swagger UI treats this endpoint as the OAuth2 token URL — clicking
    "Authorize" will POST here.  Use your email as ``username``.
    """
    db = request.app.state.mongodb.db
    user = await db["users"].find_one({"email": form.username})

    if not user or not verify_password(form.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account disabled")

    access_token, expires_in = create_access_token(user["_id"], user["email"], user["scopes"])
    refresh_token = create_refresh_token(user["_id"])

    await db["users"].update_one(
        {"_id": user["_id"]},
        {
            "$push": {
                "refresh_tokens": {
                    "token": refresh_token,
                    "issued_at": datetime.now(UTC),
                }
            }
        },
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_in,
    )


# ---------------------------------------------------------------------------
# Token refresh (refresh token rotation)
# ---------------------------------------------------------------------------


@auth_router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest, request: Request):
    """Rotate a refresh token.

    The old refresh token is revoked; a new access + refresh pair is returned.
    This implements refresh token rotation — a compromised token can only be
    used once before it's detected.
    """
    try:
        payload = decode_token(body.refresh_token)
        if payload.get("type") != "refresh":
            raise ValueError("Not a refresh token")
    except (JWTError, ValueError) as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token") from e

    db = request.app.state.mongodb.db
    user = await db["users"].find_one(
        {
            "_id": payload["sub"],
            "refresh_tokens.token": body.refresh_token,
        }
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token not found or already revoked",
        )

    new_refresh = create_refresh_token(user["_id"])
    # Atomic revoke-old + store-new
    await db["users"].update_one(
        {"_id": user["_id"]},
        {
            "$pull": {"refresh_tokens": {"token": body.refresh_token}},
            "$push": {
                "refresh_tokens": {
                    "token": new_refresh,
                    "issued_at": datetime.now(UTC),
                }
            },
        },
    )

    access_token, expires_in = create_access_token(user["_id"], user["email"], user["scopes"])
    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh,
        expires_in=expires_in,
    )


# ---------------------------------------------------------------------------
# API key management
# ---------------------------------------------------------------------------


@auth_router.post("/api-keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    body: APIKeyCreate,
    request: Request,
    principal: Principal = Security(get_current_principal, scopes=["query:read"]),
):
    """Generate a new API key.

    The ``full_key`` in the response is shown **once** — store it securely.
    Callers cannot request scopes beyond those held by their own account.
    """
    # Prevent privilege escalation: strip scopes the caller doesn't hold
    allowed_scopes = (
        body.scopes if "admin:all" in principal.scopes else [s for s in body.scopes if s in principal.scopes]
    )
    if not allowed_scopes:
        raise HTTPException(status_code=400, detail="No valid scopes — you cannot grant scopes you don't hold")

    full_key, key_id, key_hash = generate_api_key()
    db = request.app.state.mongodb.db

    await db["api_keys"].insert_one(
        {
            "key_id": key_id,
            "name": body.name,
            "key_hash": key_hash,
            "user_id": principal.user_id,
            "scopes": allowed_scopes,
            "expires_at": body.expires_at,
            "is_active": True,
            "last_used_at": None,
            "created_at": datetime.now(UTC),
        }
    )

    return APIKeyResponse(
        key_id=key_id,
        name=body.name,
        full_key=full_key,
        scopes=allowed_scopes,
        expires_at=body.expires_at,
    )


@auth_router.get("/api-keys", response_model=list[APIKeyInfo])
async def list_api_keys(
    request: Request,
    principal: Principal = Security(get_current_principal, scopes=["query:read"]),
):
    """List all active API keys owned by the caller."""
    db = request.app.state.mongodb.db
    keys: list[APIKeyInfo] = []
    async for doc in db["api_keys"].find({"user_id": principal.user_id, "is_active": True}):
        keys.append(
            APIKeyInfo(
                key_id=doc["key_id"],
                name=doc["name"],
                scopes=doc.get("scopes", []),
                expires_at=doc.get("expires_at"),
                last_used_at=doc.get("last_used_at"),
                is_active=doc["is_active"],
                created_at=doc["created_at"],
            )
        )
    return keys


@auth_router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: str,
    request: Request,
    principal: Principal = Security(get_current_principal, scopes=["query:read"]),
):
    """Soft-delete an API key (sets is_active=False).  Callers can only revoke their own keys."""
    db = request.app.state.mongodb.db
    result = await db["api_keys"].update_one(
        {"key_id": key_id, "user_id": principal.user_id},
        {"$set": {"is_active": False}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="API key not found")
