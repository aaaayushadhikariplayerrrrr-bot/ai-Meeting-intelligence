import time
from datetime import timedelta
import pytest
import jwt

from app.core.config import settings
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    ACCESS_TOKEN_TYPE,
)


# --- 1. Password Hashing Tests ---

def test_password_correct_verifies():
    """Verify that a correct password matches its Argon2id hash."""
    plain = "SuperSecurePassword123!"
    hashed = hash_password(plain)
    assert hashed.startswith("$argon2id$")
    assert verify_password(plain, hashed) is True


def test_password_incorrect_returns_false():
    """Verify that an incorrect password returns False."""
    plain = "SuperSecurePassword123!"
    hashed = hash_password(plain)
    assert verify_password("WrongPassword123!", hashed) is False


def test_password_random_salt_produces_unique_hashes():
    """Verify that hashing the same password twice yields distinct hashes due to random salting."""
    plain = "IdenticalPasswordToHash"
    h1 = hash_password(plain)
    h2 = hash_password(plain)
    assert h1 != h2
    assert verify_password(plain, h1) is True
    assert verify_password(plain, h2) is True


def test_password_empty_rejected():
    """Verify that empty or whitespace-only passwords raise ValueError."""
    with pytest.raises(ValueError, match="Password cannot be empty"):
        hash_password("")

    with pytest.raises(ValueError, match="Password cannot be empty"):
        hash_password("   ")


def test_password_malformed_hash_handled_safely():
    """Verify that malformed or corrupted hashes do not crash and return False."""
    assert verify_password("password123", "not-a-valid-argon2-hash") is False
    assert verify_password("password123", "$argon2id$v=19$m=65536,t=2,p=1$corrupted") is False
    assert verify_password("password123", "") is False
    assert verify_password("", "some-hash") is False
    assert verify_password(None, "some-hash") is False
    assert verify_password("password123", None) is False


# --- 2. JWT Access Token Tests ---

def test_jwt_valid_access_token_decodes():
    """Verify that a valid access token encodes and decodes properly."""
    sub = "user-uuid-12345"
    token = create_access_token(subject=sub, token_version=1)
    claims = decode_access_token(token)
    assert claims is not None
    assert claims["sub"] == sub
    assert claims["type"] == ACCESS_TOKEN_TYPE
    assert claims["token_version"] == 1


def test_jwt_strictly_minimal_claims():
    """Verify that tokens contain ONLY the permitted claims (sub, iat, exp, type, token_version)
    and contain NO organization, role, permission, or profile information.
    """
    sub = "user-uuid-67890"
    token = create_access_token(
        subject=sub,
        token_version=2,
        expires_delta=timedelta(minutes=30),
    )
    claims = decode_access_token(token)
    assert claims is not None

    # Strictly only the permitted claims
    assert set(claims.keys()) == {"sub", "iat", "exp", "type", "token_version"}
    assert claims["sub"] == sub
    assert claims["type"] == ACCESS_TOKEN_TYPE
    assert claims["token_version"] == 2
    assert isinstance(claims["iat"], int)
    assert isinstance(claims["exp"], int)
    assert isinstance(claims["token_version"], int)
    assert claims["exp"] > claims["iat"]

    # Explicitly verify NO organization, role, permission, email, or credentials exist in claims
    forbidden_keys = {
        "org_id",
        "organization_id",
        "role",
        "roles",
        "permissions",
        "email",
        "full_name",
        "password",
        "hashed_password",
        "is_super_admin",
    }
    for key in forbidden_keys:
        assert key not in claims


def test_create_access_token_rejects_extra_claims_argument():
    """Verify that create_access_token strictly prohibits arbitrary extra claims."""
    with pytest.raises(TypeError):
        create_access_token(
            subject="user-123",
            token_version=1,
            extra_claims={"org_id": "org-99", "role": "ORG_ADMIN"},  # type: ignore
        )


def test_create_access_token_requires_token_version():
    """Verify that omitting the mandatory token_version parameter raises TypeError."""
    with pytest.raises(TypeError):
        create_access_token(subject="user-123")  # type: ignore


def test_jwt_expired_token_rejected():
    """Verify that an expired JWT access token is rejected and returns None."""
    now = int(time.time())
    expired_payload = {
        "sub": "user-uuid-expired",
        "iat": now - 3600,
        "exp": now - 60,  # Expired 60 seconds ago
        "type": ACCESS_TOKEN_TYPE,
        "token_version": 1,
    }
    expired_token = jwt.encode(
        expired_payload,
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    assert decode_access_token(expired_token) is None


def test_jwt_invalid_signature_rejected():
    """Verify that a token signed with a different key returns None."""
    alt_secret = "different-secret-key-for-forgery-test-at-least-32-chars"
    now = int(time.time())
    payload = {
        "sub": "user-uuid-tampered",
        "iat": now,
        "exp": now + 3600,
        "type": ACCESS_TOKEN_TYPE,
        "token_version": 1,
    }
    forged_token = jwt.encode(payload, alt_secret, algorithm="HS256")
    assert decode_access_token(forged_token) is None


def test_jwt_malformed_token_rejected():
    """Verify that malformed or truncated tokens return None."""
    assert decode_access_token("malformed.jwt.token") is None
    assert decode_access_token("completely-invalid-string") is None
    assert decode_access_token("") is None
    assert decode_access_token(None) is None


def test_jwt_incorrect_type_rejected():
    """Verify that tokens with a type other than 'access' are rejected."""
    now = int(time.time())
    refresh_payload = {
        "sub": "user-uuid-refresh",
        "iat": now,
        "exp": now + 3600,
        "type": "refresh",  # Not an access token
        "token_version": 1,
    }
    refresh_token = jwt.encode(
        refresh_payload,
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    assert decode_access_token(refresh_token) is None


def test_jwt_missing_or_empty_subject_rejected():
    """Verify that tokens missing a valid subject are rejected on creation and decoding."""
    # Creation rejects empty subject
    with pytest.raises(ValueError, match="subject"):
        create_access_token(subject="", token_version=1)

    with pytest.raises(ValueError, match="subject"):
        create_access_token(subject="   ", token_version=1)

    # Decoding rejects tokens with missing or empty sub claim
    now = int(time.time())
    token_empty_sub = jwt.encode(
        {"sub": "", "iat": now, "exp": now + 3600, "type": ACCESS_TOKEN_TYPE, "token_version": 1},
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    assert decode_access_token(token_empty_sub) is None

    token_no_sub = jwt.encode(
        {"iat": now, "exp": now + 3600, "type": ACCESS_TOKEN_TYPE, "token_version": 1},
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    assert decode_access_token(token_no_sub) is None


def test_jwt_verification_uses_only_configured_algorithm():
    """Verify that tokens signed with unauthorized algorithms (e.g. 'none' or different HMAC) are rejected."""
    now = int(time.time())
    payload = {
        "sub": "user-uuid-algo-test",
        "iat": now,
        "exp": now + 3600,
        "type": ACCESS_TOKEN_TYPE,
        "token_version": 1,
    }

    # Algorithm 'none'
    token_none = jwt.encode(payload, key="", algorithm="none")
    assert decode_access_token(token_none) is None

    # Algorithm HS384 when configured as HS256
    token_hs384 = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS384")
    assert decode_access_token(token_hs384) is None


def test_jwt_token_version_validation():
    """Verify that create_access_token and decode_access_token enforce strict token_version constraints."""
    sub = "user-token-ver-test"

    # Valid token versions >= 1 work
    for valid_ver in [1, 2, 42, 9999]:
        token = create_access_token(subject=sub, token_version=valid_ver)
        claims = decode_access_token(token)
        assert claims is not None
        assert claims["token_version"] == valid_ver

    # create_access_token rejects invalid versions
    for invalid_ver in [0, -1, -100, 1.5, "1", None, True, False]:
        with pytest.raises(ValueError, match="token_version"):
            create_access_token(subject=sub, token_version=invalid_ver)  # type: ignore

    # decode_access_token rejects tokens missing token_version claim
    now = int(time.time())
    token_no_tv = jwt.encode(
        {"sub": sub, "iat": now, "exp": now + 3600, "type": ACCESS_TOKEN_TYPE},
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    assert decode_access_token(token_no_tv) is None

    # decode_access_token rejects tokens with non-integer or boolean or <= 0 token_version
    for bad_tv in [0, -1, -50, 1.5, "1", True, False, [], {}]:
        token_bad_tv = jwt.encode(
            {"sub": sub, "iat": now, "exp": now + 3600, "type": ACCESS_TOKEN_TYPE, "token_version": bad_tv},
            settings.SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )
        assert decode_access_token(token_bad_tv) is None
