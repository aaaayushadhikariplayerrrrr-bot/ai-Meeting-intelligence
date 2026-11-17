"""Authentication and Identity API Endpoints.

Provides:
- POST /register: Atomic user registration, organization creation, and ORG_ADMIN assignment.
- POST /login: Secure credential verification with timing attack mitigation and minimal JWT issuance.
- GET /me: Current authenticated user profile and active tenant memberships.
- GET /organizations: List of active organizations available to the authenticated user.
"""

import re
import secrets
import unicodedata
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from app.database.session import get_db
from app.models.identity import User, Organization, OrganizationMembership, OrgRole
from app.core.security import hash_password, verify_password, create_access_token
from app.core.rate_limit import login_rate_limiter, register_rate_limiter
from app.api.deps import get_current_active_user
from app.schemas.auth import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    UserResponse,
    OrgMembershipResponse,
    AuthMeResponse,
)

router = APIRouter(tags=["Authentication"])

# Precomputed Argon2id hash generated once at module load to mitigate account-enumeration
# timing differences for non-existent users and accounts with NULL password hashes.
_DUMMY_TIMING_HASH: str = hash_password("timing-mitigation-dummy-secret-hash-salt")


def slugify(text: str) -> str:
    """Normalize string into a URL-friendly alphanumeric slug."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[-\s]+", "-", text).strip("-")
    return text[:60] or "org"


def generate_unique_org_slug(db: Session, org_name: str) -> str:
    """Generate a unique organization slug with collision avoidance."""
    base_slug = slugify(org_name)
    stmt = select(Organization.id).where(Organization.slug == base_slug)
    if db.execute(stmt).first() is None:
        return base_slug

    for _ in range(5):
        suffix = secrets.token_hex(2)
        candidate = f"{base_slug[:50]}-{suffix}"
        stmt = select(Organization.id).where(Organization.slug == candidate)
        if db.execute(stmt).first() is None:
            return candidate

    return f"{base_slug[:45]}-{secrets.token_hex(4)}"


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(register_rate_limiter)],
    summary="Register new user, create organization, and assign ORG_ADMIN",
)
def register(
    payload: RegisterRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """Register a new user, create their tenant organization, and issue access token."""
    normalized_email = payload.email.strip().lower()

    # Pre-check for duplicate email
    existing_user = db.execute(
        select(User.id).where(User.email == normalized_email)
    ).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    slug = generate_unique_org_slug(db, payload.organization_name)
    hashed_pw = hash_password(payload.password)

    try:
        new_user = User(
            email=normalized_email,
            full_name=payload.full_name.strip(),
            hashed_password=hashed_pw,
            is_super_admin=False,
            is_active=True,
        )
        new_org = Organization(
            name=payload.organization_name.strip(),
            slug=slug,
            is_active=True,
        )
        membership = OrganizationMembership(
            organization=new_org,
            user=new_user,
            role=OrgRole.ORG_ADMIN.value,
            is_active=True,
        )

        db.add(new_user)
        db.add(new_org)
        db.add(membership)
        db.commit()
        db.refresh(new_user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Registration conflict: email or organization identifier already in use.",
        )
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed. Please try again.",
        )

    access_token = create_access_token(subject=new_user.id, token_version=new_user.token_version)
    return TokenResponse(access_token=access_token, token_type="bearer")


@router.post(
    "/login",
    response_model=TokenResponse,
    dependencies=[Depends(login_rate_limiter)],
    summary="Authenticate user and issue minimal JWT access token",
)
def login(
    payload: LoginRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """Authenticate user with email and password."""
    unauthorized_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

    normalized_email = payload.email.strip().lower()
    user = db.execute(
        select(User).where(User.email == normalized_email)
    ).scalar_one_or_none()

    if user is None:
        # Run dummy verification to equalize execution time against timing attacks
        verify_password(payload.password, _DUMMY_TIMING_HASH)
        raise unauthorized_exc

    if user.hashed_password is None:
        # User exists but has no usable password hash set
        verify_password(payload.password, _DUMMY_TIMING_HASH)
        raise unauthorized_exc

    if not verify_password(payload.password, user.hashed_password):
        raise unauthorized_exc

    if not user.is_active:
        # Inactive user returns identical 401 to prevent account enumeration
        raise unauthorized_exc

    access_token = create_access_token(subject=user.id, token_version=user.token_version)
    return TokenResponse(access_token=access_token, token_type="bearer")


@router.get(
    "/me",
    response_model=AuthMeResponse,
    summary="Get current user profile and active organization memberships",
)
def get_me(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> AuthMeResponse:
    """Introspect authenticated user profile and memberships."""
    stmt = (
        select(OrganizationMembership, Organization)
        .join(Organization, OrganizationMembership.organization_id == Organization.id)
        .where(
            OrganizationMembership.user_id == current_user.id,
            OrganizationMembership.is_active == True,
            Organization.is_active == True,
        )
    )
    rows = db.execute(stmt).all()
    org_memberships = [
        OrgMembershipResponse(
            organization_id=org.id,
            organization_name=org.name,
            organization_slug=org.slug,
            role=membership.role,
        )
        for membership, org in rows
    ]

    return AuthMeResponse(
        user=UserResponse.model_validate(current_user),
        organizations=org_memberships,
    )


@router.get(
    "/organizations",
    response_model=List[OrgMembershipResponse],
    summary="List active organization memberships for current authenticated user",
)
def get_my_organizations(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> List[OrgMembershipResponse]:
    """Return all active organizations the authenticated user belongs to."""
    stmt = (
        select(OrganizationMembership, Organization)
        .join(Organization, OrganizationMembership.organization_id == Organization.id)
        .where(
            OrganizationMembership.user_id == current_user.id,
            OrganizationMembership.is_active == True,
            Organization.is_active == True,
        )
    )
    rows = db.execute(stmt).all()
    return [
        OrgMembershipResponse(
            organization_id=org.id,
            organization_name=org.name,
            organization_slug=org.slug,
            role=membership.role,
        )
        for membership, org in rows
    ]


@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="Revoke all active access tokens for current user",
)
def logout(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Revoke all issued access tokens for the authenticated user by atomically incrementing token_version."""
    db.execute(
        update(User)
        .where(User.id == current_user.id)
        .values(token_version=User.token_version + 1)
    )
    db.commit()
    db.refresh(current_user)
    return {"message": "Successfully logged out"}
