"""FastAPI dependencies for user authentication, token extraction, and RBAC authorization."""

from typing import Any, Dict, Optional
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.app.core.security import decode_access_token
from backend.app.services.user_service import user_service

# Standard HTTP Bearer authentication scheme for OpenAPI / Swagger UI integration
oauth2_bearer = HTTPBearer(
    auto_error=False,
    description="Enter your JWT access token obtained from /api/v1/auth/login",
)


def extract_bearer_token(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_bearer),
    request: Request = None,
) -> Optional[str]:
    """
    Extract token string from standard HTTPBearer scheme or fallback Authorization header.
    """
    if auth and auth.credentials:
        return auth.credentials

    # Robust fallback: inspect raw request header in case client passed token directly without 'Bearer '
    if request is not None:
        raw_auth = request.headers.get("authorization")
        if raw_auth:
            parts = raw_auth.strip().split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1]
            elif len(parts) == 1:
                return parts[0]

    return None


def get_current_user(token: Optional[str] = Depends(extract_bearer_token)) -> Dict[str, Any]:
    """
    Validate access token and return authenticated user.
    Raises HTTP 401 if token is missing, invalid, or expired.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = user_service.get_user_by_id(payload["sub"])
    if not user or not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found or deactivated.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def get_optional_current_user(token: Optional[str] = Depends(extract_bearer_token)) -> Optional[Dict[str, Any]]:
    """Return authenticated user if valid token is provided, or None if omitted/invalid."""
    if not token:
        return None
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        return None
    user = user_service.get_user_by_id(payload["sub"])
    if not user or not user.get("is_active", True):
        return None
    return user


def require_developer(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Require authenticated user to hold the 'developer' role."""
    if current_user.get("role") != "developer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Developer or administrator privileges are required to access this resource.",
        )
    return current_user
