"""Authentication and user management router."""

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.dependencies import get_current_user
from backend.app.core.security import create_access_token
from backend.app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from backend.app.services.user_service import user_service

router = APIRouter(prefix="/auth", tags=["Authentication & User Management"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register New Household User",
    description="Create a new household user account. Role is strictly assigned as 'household'.",
)
def register(request: RegisterRequest) -> UserResponse:
    """Register a new household user."""
    try:
        user = user_service.create_user(
            username=request.username,
            password=request.password,
            role="household",
        )
        return UserResponse(**user)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="User Login",
    description="Authenticate with username and password to obtain an HS256 signed bearer access token.",
)
def login(request: LoginRequest) -> TokenResponse:
    """Authenticate user credentials and issue access token."""
    user = user_service.authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token({
        "sub": user["id"],
        "username": user["username"],
        "role": user["role"],
    })

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse(**user),
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get Current User Profile",
    description="Retrieve the authenticated caller's profile and assigned role.",
)
def get_me(current_user: Dict[str, Any] = Depends(get_current_user)) -> UserResponse:
    """Return currently authenticated user profile."""
    return UserResponse(**current_user)


@router.post(
    "/logout",
    summary="Logout User",
    description="Informational logout endpoint; clients should discard stored bearer tokens.",
)
def logout() -> Dict[str, str]:
    """Acknowledge user logout."""
    return {"message": "Successfully logged out."}
