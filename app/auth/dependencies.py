"""
FastAPI Security dependencies.

Usage in route handlers:
    from fastapi import Security
    from app.auth.dependencies import get_current_principal

    @router.post("/query")
    async def my_route(
        principal: Principal = Security(get_current_principal, scopes=["query:read"]),
    ): ...

Scope catalogue:
    query:read    — invoke /query, /history
    admin:seed    — invoke /seed
    admin:replay  — invoke /replay
    admin:all     — unrestricted (satisfies any scope check)
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from jose import JWTError

from app.auth.models import Principal
from app.auth.service import decode_token, parse_api_key, verify_api_key_secret

# Swagger UI will render a "Authorize" button pointing at our token endpoint.
_oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/token",
    auto_error=False,  # We raise the error ourselves for consistent JSON shape
)


async def get_current_principal(
    request: Request,
    security_scopes: SecurityScopes,
    token: str | None = Depends(_oauth2_scheme),
) -> Principal:
    """Dual-token resolver: accepts both JWTs and API keys under the same
    ``Authorization: Bearer <token>`` header.

    Detection:
    - Starts with ``ak_``  → API key path (DB lookup + bcrypt verify)
    - Anything else        → JWT path (stateless signature verify)
    """
    if not token:
        raise _build_401(security_scopes, "Not authenticated")

    if token.startswith("ak_"):
        principal = await _resolve_api_key(request, token)
    else:
        principal = _resolve_jwt(token)

    # Scope enforcement: admin:all is a wildcard
    if security_scopes.scopes and "admin:all" not in principal.scopes:
        for required in security_scopes.scopes:
            if required not in principal.scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions — requires scope: '{required}'",
                )

    return principal


# ---------------------------------------------------------------------------
# Internal resolvers
# ---------------------------------------------------------------------------


def _resolve_jwt(token: str) -> Principal:
    try:
        payload = decode_token(token)
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not an access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return Principal(
        user_id=payload["sub"],
        email=payload["email"],
        scopes=payload.get("scopes", []),
        auth_method="jwt",
    )


async def _resolve_api_key(request: Request, full_key: str) -> Principal:
    try:
        key_id, secret = parse_api_key(full_key)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed API key",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    db = request.app.state.mongodb.db
    key_doc = await db["api_keys"].find_one({"key_id": key_id, "is_active": True})

    if key_doc is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key not found or revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Expiry check
    expires_at = key_doc.get("expires_at")
    if expires_at and expires_at.replace(tzinfo=UTC) < datetime.now(UTC):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Constant-time bcrypt verification
    if not verify_api_key_secret(secret, key_doc["key_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Fire-and-forget last_used_at update (non-blocking)
    await db["api_keys"].update_one(
        {"key_id": key_id},
        {"$set": {"last_used_at": datetime.now(UTC)}},
    )

    user_doc = await db["users"].find_one({"_id": key_doc["user_id"]})

    return Principal(
        user_id=key_doc["user_id"],
        email=user_doc["email"] if user_doc else "unknown",
        scopes=key_doc.get("scopes", []),
        auth_method="api_key",
        key_id=key_id,
    )


def _build_401(security_scopes: SecurityScopes, detail: str) -> HTTPException:
    www_auth = f'Bearer scope="{security_scopes.scope_str}"' if security_scopes.scopes else "Bearer"
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": www_auth},
    )
