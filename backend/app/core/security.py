"""
Cryptographic security, password hashing, and token handling.

Features:
- Password hashing via Argon2id (OWASP recommended standard).
- Standard HS256 JWT-compatible bearer tokens via HMAC-SHA256.
- Constant-time signature comparison to eliminate timing attacks.
- Expiration enforcement and token integrity verification.
"""

import base64
import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import argon2
from backend.app.core.config import settings

# Initialize Argon2 password hasher
_hasher = argon2.PasswordHasher(
    time_cost=2,
    memory_cost=65536,  # 64 MB
    parallelism=1,
    hash_len=32,
    type=argon2.Type.ID,
)


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password using Argon2id."""
    if not plain_password:
        raise ValueError("Password cannot be empty.")
    return _hasher.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against an Argon2 hash in constant time."""
    if not plain_password or not hashed_password:
        return False
    try:
        return _hasher.verify(hashed_password, plain_password)
    except (argon2.exceptions.VerifyMismatchError, argon2.exceptions.InvalidHashError):
        return False


def _b64url_encode(data: bytes) -> str:
    """Encode bytes to base64url string without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(encoded: str) -> bytes:
    """Decode base64url string with padding restoration."""
    rem = len(encoded) % 4
    if rem > 0:
        encoded += "=" * (4 - rem)
    return base64.urlsafe_b64decode(encoded.encode("ascii"))


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Generate an HS256 signed JWT-compatible access token.
    Token format: <header_b64>.<payload_b64>.<signature_b64>
    """
    to_encode = data.copy()
    now_utc = datetime.now(timezone.utc)
    
    if expires_delta:
        expire_dt = now_utc + expires_delta
    else:
        expire_dt = now_utc + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode["iat"] = int(now_utc.timestamp())
    to_encode["exp"] = int(expire_dt.timestamp())

    header = {"alg": "HS256", "typ": "JWT"}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload_bytes = json.dumps(to_encode, separators=(",", ":")).encode("utf-8")

    header_b64 = _b64url_encode(header_bytes)
    payload_b64 = _b64url_encode(payload_bytes)

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    secret_key = settings.JWT_SECRET_KEY.encode("utf-8")
    signature = hmac.new(secret_key, signing_input, hashlib.sha256).digest()
    signature_b64 = _b64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode an HS256 signed token.
    Returns the token payload if valid and unexpired; returns None otherwise.
    """
    try:
        parts = token.strip().split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, signature_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        secret_key = settings.JWT_SECRET_KEY.encode("utf-8")

        expected_sig = hmac.new(secret_key, signing_input, hashlib.sha256).digest()
        provided_sig = _b64url_decode(signature_b64)

        if not hmac.compare_digest(expected_sig, provided_sig):
            return None

        payload_bytes = _b64url_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))

        # Verify expiration
        exp = payload.get("exp")
        if exp is None or int(exp) < int(time.time()):
            return None

        return payload
    except Exception:
        return None
