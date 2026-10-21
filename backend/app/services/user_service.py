"""
User service for managing authentication, user creation, credential verification, and RBAC.
"""

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.app.core.config import settings
from backend.app.core.security import hash_password, verify_password
from backend.app.db.database import DatabaseManager, db_manager


class UserService:
    """Service handling user account lifecycle, authentication, and role assignment."""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self._db = db or db_manager

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Retrieve user record by username."""
        clean_username = username.strip().lower()
        conn = self._db.get_connection()
        try:
            cursor = conn.execute(
                "SELECT id, username, hashed_password, role, created_at, is_active FROM users WHERE username = ?;",
                (clean_username,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user record by unique user ID."""
        conn = self._db.get_connection()
        try:
            cursor = conn.execute(
                "SELECT id, username, hashed_password, role, created_at, is_active FROM users WHERE id = ?;",
                (user_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def create_user(
        self,
        username: str,
        password: str,
        role: str = "household",
    ) -> Dict[str, Any]:
        """
        Create a new user account with Argon2id hashed password.
        Role must be 'household' or 'developer'.
        """
        clean_username = username.strip().lower()

        if not re.match(r"^[a-zA-Z0-9_-]{3,32}$", clean_username):
            raise ValueError(
                "Username must be between 3 and 32 characters and contain only letters, numbers, underscores, or hyphens."
            )

        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters long.")

        if role not in ("household", "developer"):
            raise ValueError(f"Invalid role '{role}'. Allowed roles are 'household' or 'developer'.")

        existing = self.get_user_by_username(clean_username)
        if existing is not None:
            raise ValueError(f"Username '{clean_username}' is already registered.")

        user_id = f"usr_{uuid.uuid4().hex[:12]}"
        hashed_pw = hash_password(password)
        created_at = datetime.now(timezone.utc).isoformat()

        conn = self._db.get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO users (id, username, hashed_password, role, created_at, is_active)
                    VALUES (?, ?, ?, ?, ?, 1);
                    """,
                    (user_id, clean_username, hashed_pw, role, created_at),
                )
        finally:
            conn.close()

        return {
            "id": user_id,
            "username": clean_username,
            "role": role,
            "created_at": created_at,
            "is_active": True,
        }

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user by username and password. Returns user dict without hash if valid."""
        clean_username = username.strip().lower()
        user = self.get_user_by_username(clean_username)
        if user is None:
            return None

        if not user.get("is_active", True):
            return None

        if not verify_password(password, user["hashed_password"]):
            return None

        # Return safe dictionary without password hash
        return {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "created_at": user["created_at"],
            "is_active": bool(user["is_active"]),
        }

    def bootstrap_initial_developer(self) -> Optional[Dict[str, Any]]:
        """Bootstrap an initial developer user from environment variables if configured and missing."""
        dev_username = settings.INITIAL_DEV_USERNAME.strip().lower()
        dev_password = settings.INITIAL_DEV_PASSWORD.strip()

        if not dev_username or not dev_password:
            return None

        existing = self.get_user_by_username(dev_username)
        if existing is not None:
            return existing

        return self.create_user(
            username=dev_username,
            password=dev_password,
            role="developer",
        )


user_service = UserService()
