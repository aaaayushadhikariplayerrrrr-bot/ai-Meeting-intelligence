import pytest
from pydantic import ValidationError
from app.core.config import Settings


def test_argon2_and_jwt_installed_and_importable():
    """Verify that argon2-cffi and pyjwt are installed in the backend environment."""
    import argon2
    import jwt

    assert argon2 is not None
    assert jwt is not None
    # PyJWT version check
    assert hasattr(jwt, "encode") and hasattr(jwt, "decode")


def test_security_settings_valid_test_environment():
    """Verify that settings load with explicit test secret in testing environment."""
    s = Settings(
        ENVIRONMENT="testing",
        SECRET_KEY="test-explicit-secret-key-configured-for-ci-32chars"
    )
    assert s.SECRET_KEY == "test-explicit-secret-key-configured-for-ci-32chars"
    assert s.JWT_ALGORITHM == "HS256"
    assert s.ACCESS_TOKEN_EXPIRE_MINUTES == 60


def test_security_settings_test_environment_fallback():
    """Verify that testing environment provides safe designated test secret if empty."""
    s = Settings(ENVIRONMENT="testing", SECRET_KEY="")
    assert len(s.SECRET_KEY) >= 32
    assert s.JWT_ALGORITHM == "HS256"


def test_security_settings_rejects_missing_secret_in_development_or_production():
    """Verify that development/production cannot start without a SECRET_KEY."""
    with pytest.raises(ValidationError, match="SECRET_KEY is required"):
        Settings(ENVIRONMENT="production", SECRET_KEY="")

    with pytest.raises(ValidationError, match="SECRET_KEY is required"):
        Settings(ENVIRONMENT="development", SECRET_KEY="")


def test_security_settings_rejects_insecure_placeholder_secrets():
    """Verify that common insecure placeholders are strictly rejected."""
    placeholders = [
        "changeme",
        "secret",
        "secretkey",
        "default",
        "password",
        "generate-with-openssl-rand-hex-32-min-32-chars",
        "your-secret-key-here-must-be-32-chars-or-more",
    ]
    for ph in placeholders:
        with pytest.raises(ValidationError, match="known insecure placeholder"):
            Settings(ENVIRONMENT="production", SECRET_KEY=ph)


def test_security_settings_rejects_short_secret():
    """Verify that secrets shorter than 32 characters (256 bits) are rejected."""
    with pytest.raises(ValidationError, match="at least 32 characters"):
        Settings(ENVIRONMENT="development", SECRET_KEY="short-secret-under-32-chars")


def test_security_settings_rejects_invalid_jwt_algorithm():
    """Verify that insecure or unsupported JWT algorithms are rejected."""
    with pytest.raises(ValidationError, match="Unsupported JWT_ALGORITHM"):
        Settings(
            ENVIRONMENT="development",
            SECRET_KEY="a" * 32,
            JWT_ALGORITHM="none"
        )


def test_security_settings_rejects_invalid_token_expiry():
    """Verify that non-positive token expiration minutes are rejected."""
    with pytest.raises(ValidationError, match="ACCESS_TOKEN_EXPIRE_MINUTES must be greater than 0"):
        Settings(
            ENVIRONMENT="development",
            SECRET_KEY="a" * 32,
            ACCESS_TOKEN_EXPIRE_MINUTES=0
        )
