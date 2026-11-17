"""Authentication and Identity Pydantic Schemas.

Provides validation and serialization models for registration, login,
token issuance, user profile introspection, and organization memberships.
Adheres strictly to Pydantic v2 conventions and security best practices.
"""

import re
from typing import List
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Standard RFC-compliant email pattern for robust validation without external package dependencies
EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+$"
)


def validate_and_normalize_email(v: str) -> str:
    """Normalize and validate an email string."""
    if not isinstance(v, str):
        raise ValueError("Email must be a string.")
    v = v.strip().lower()
    if not v or len(v) > 255:
        raise ValueError("Email must be between 1 and 255 characters.")
    if not EMAIL_REGEX.match(v):
        raise ValueError("Invalid email address format.")
    return v


class RegisterRequest(BaseModel):
    """Schema for self-service user registration with initial organization."""
    email: str = Field(..., max_length=255, description="User's unique email address")
    full_name: str = Field(..., min_length=1, max_length=255, description="Full name of the user")
    password: str = Field(..., min_length=8, max_length=128, description="User password (8-128 characters)")
    organization_name: str = Field(..., min_length=1, max_length=255, description="Organization or workspace name")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return validate_and_normalize_email(v)

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Full name cannot be empty or whitespace only.")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Password cannot be empty or whitespace only.")
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        if len(v) > 128:
            raise ValueError("Password cannot exceed 128 characters.")
        return v

    @field_validator("organization_name")
    @classmethod
    def validate_organization_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Organization name cannot be empty or whitespace only.")
        return v


class LoginRequest(BaseModel):
    """Schema for credential-based user authentication."""
    email: str = Field(..., max_length=255, description="User's registered email address")
    password: str = Field(..., min_length=1, max_length=128, description="User password")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return validate_and_normalize_email(v)


class TokenResponse(BaseModel):
    """Schema for JWT access token responses."""
    access_token: str = Field(..., description="Signed JWT access token")
    token_type: str = Field("bearer", description="Token type, always 'bearer'")


class UserResponse(BaseModel):
    """Schema for public user profile data. Never exposes password hashes."""
    id: str = Field(..., description="Unique user identifier (UUID)")
    email: str = Field(..., description="User's registered email address")
    full_name: str = Field(..., description="User's full name")
    is_super_admin: bool = Field(..., description="Global system administrator flag")
    is_active: bool = Field(..., description="Account active status")

    model_config = ConfigDict(from_attributes=True)


class OrgMembershipResponse(BaseModel):
    """Schema representing an active organization membership for a user."""
    organization_id: str = Field(..., description="Organization identifier (UUID)")
    organization_name: str = Field(..., description="Organization display name")
    organization_slug: str = Field(..., description="Organization URL-safe slug")
    role: str = Field(..., description="Role within organization (ORG_ADMIN, MEMBER)")

    model_config = ConfigDict(from_attributes=True)


class AuthMeResponse(BaseModel):
    """Schema for /me endpoint returning current user profile and tenant memberships."""
    user: UserResponse = Field(..., description="Current authenticated user profile")
    organizations: List[OrgMembershipResponse] = Field(
        default_factory=list,
        description="Active organization memberships belonging to the user"
    )

    model_config = ConfigDict(from_attributes=True)
