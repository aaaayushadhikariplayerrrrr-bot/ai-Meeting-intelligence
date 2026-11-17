"""Security and cryptographic utility module for password hashing and JWT token handling.

Provides:
- Argon2id password hashing and verification using OWASP/RFC 9106 recommended parameters.
- Cryptographically signed JWT access token creation and decoding using PyJWT.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import argon2
import argon2.exceptions
import jwt
import jwt.exceptions

from app.core.config import settings

# Configure Argon2id PasswordHasher with RFC 9106 / OWASP recommended profile:
# - Type: Argon2id (resistant to side-channel and GPU attacks)
# - Time cost: 2 iterations
# - Memory cost: 65536 KiB (64 MiB)
# - Parallelism: 1 thread
# - Hash length: 32 bytes
# - Salt length: 16 bytes
_PASSWORD_HASHER = argon2.PasswordHasher(
    time_cost=2,
    memory_cost=65536,
    parallelism=1,
    hash_len=32,
    salt_len=16,
    type=argon2.Type.ID,
)

ACCESS_TOKEN_TYPE = "access"


def hash_password(password: str) -> str:
    """Hash a plaintext password using Argon2id with random salt.

    Args:
        password: Raw plaintext password to hash.

    Returns:
        Encoded Argon2id hash string containing parameters, salt, and digest.

    Raises:
        ValueError: If password is empty or whitespace-only.
    """
    if not password or not password.strip():
        raise ValueError("Password cannot be empty or whitespace-only.")

    return _PASSWORD_HASHER.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against an Argon2 hash.

    Safely handles malformed hashes, wrong passwords, and type mismatches without
    raising exceptions or leaking account existence.

    Args:
        plain_password: The plaintext password candidate to check.
        hashed_password: The stored Argon2 hash.

    Returns:
        True if the password matches the hash, False otherwise.
    """
    if not plain_password or not hashed_password:
        return False

    if not isinstance(plain_password, str) or not isinstance(hashed_password, str):
        return False

    try:
        return _PASSWORD_HASHER.verify(hashed_password, plain_password)
    except (
        argon2.exceptions.VerificationError,
        argon2.exceptions.InvalidHashError,
        ValueError,
        TypeError,
    ):
        return False


def create_access_token(
    subject: str,
    token_version: int,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a signed, strictly minimal JWT access token.

    Contains ONLY the five permitted claims:
    - 'sub': Subject (User UUID string)
    - 'iat': Issued-at Unix timestamp
    - 'exp': Expiration Unix timestamp
    - 'type': 'access'
    - 'token_version': Integer user token generation counter (>= 1) for revocation

    Tenant IDs, roles, permissions, passwords, and profile metadata are strictly
    prohibited from the token payload and must be resolved from the database during
    authenticated request processing.

    Args:
        subject: User ID or unique identifier to be stored in the 'sub' claim.
        token_version: Mandatory user's current token version integer (>= 1).
        expires_delta: Optional custom duration before expiry. Defaults to
                       settings.ACCESS_TOKEN_EXPIRE_MINUTES.

    Returns:
        Encoded and signed JWT string.

    Raises:
        ValueError: If subject is empty, token_version is not a positive integer,
                    or expires_delta is non-positive.
        TypeError: If token_version is omitted.
    """
    if not subject or not str(subject).strip():
        raise ValueError("Token subject ('sub') must be a non-empty string.")

    if not isinstance(token_version, int) or isinstance(token_version, bool) or token_version < 1:
        raise ValueError("Token version ('token_version') must be an integer >= 1.")

    now = datetime.now(timezone.utc)
    if expires_delta is not None:
        if expires_delta.total_seconds() <= 0:
            raise ValueError("Token expires_delta must be positive.")
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    # Strictly include only the permitted claims
    payload: Dict[str, Any] = {
        "sub": str(subject).strip(),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "type": ACCESS_TOKEN_TYPE,
        "token_version": token_version,
    }

    token = jwt.encode(
        payload,
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    return token


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a JWT access token.

    Validates signature, expiration, required claims ('sub', 'iat', 'exp', 'type', 'token_version'),
    and ensures the token type is 'access'. Restricts algorithms strictly to
    settings.JWT_ALGORITHM.

    Args:
        token: The encoded JWT token string.

    Returns:
        Dictionary of decoded claims on success, or None if the token is invalid,
        expired, malformed, or has an incorrect type.
    """
    if not token or not isinstance(token, str) or not token.strip():
        return None

    try:
        payload = jwt.decode(
            token.strip(),
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={
                "verify_exp": True,
                "verify_signature": True,
                "require": ["sub", "iat", "exp", "type", "token_version"],
            },
        )
    except (jwt.PyJWTError, ValueError, TypeError):
        return None

    # Validate token type
    if payload.get("type") != ACCESS_TOKEN_TYPE:
        return None

    # Validate that subject is a non-empty string
    sub = payload.get("sub")
    if not sub or not isinstance(sub, str) or not sub.strip():
        return None

    # Validate that token_version is a valid integer >= 1
    tv = payload.get("token_version")
    if not isinstance(tv, int) or isinstance(tv, bool) or tv < 1:
        return None

    return payload
