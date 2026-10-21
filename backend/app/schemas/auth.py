"""Pydantic schemas for user registration, authentication, and authorization."""

from typing import Optional
from pydantic import BaseModel, Field


class UserResponse(BaseModel):
    """Public representation of a user identity (excluding password hash)."""

    id: str = Field(..., description="Unique user identifier (e.g. 'usr_...')")
    username: str = Field(..., description="Unique normalized username")
    role: str = Field(..., description="User role: 'household' or 'developer'")
    created_at: str = Field(..., description="UTC creation timestamp")
    is_active: bool = Field(default=True, description="Account active status")


class RegisterRequest(BaseModel):
    """Payload for registering a new household user account."""

    model_config = {"extra": "forbid"}

    username: str = Field(..., min_length=3, max_length=32, description="Alphanumeric username (3 to 32 characters)")
    password: str = Field(..., min_length=6, description="Password (at least 6 characters)")


class LoginRequest(BaseModel):
    """Payload for user authentication."""

    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class TokenResponse(BaseModel):
    """Authentication token response containing signed access token and user metadata."""

    access_token: str = Field(..., description="HS256 signed bearer access token")
    token_type: str = Field(default="bearer", description="Token type")
    user: UserResponse = Field(..., description="Authenticated user profile")
