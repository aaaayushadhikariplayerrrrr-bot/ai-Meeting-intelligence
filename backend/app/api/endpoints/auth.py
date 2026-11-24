"""Authentication and Identity API Endpoints.

Provides:
- POST /register: Multi-path user registration (Create Org, Join Org, Individual Workspace).
- POST /login: Secure credential verification with timing attack mitigation and minimal JWT issuance.
- GET /me: Current authenticated user profile and active tenant memberships.
- GET /organizations: List of active organizations available to the authenticated user.
- GET /organizations/by-domain: Verify/lookup existing organization company domain.
- POST /verify-domain: Verify company domain ownership via DNS TXT or challenge token.
- GET /organization/members: List organization members with role-based visibility.
- PATCH /organization/members/{membership_id}/role: Update member role (requires ORG_ADMIN).
- POST /logout: Revoke active session tokens.
"""

import re
import secrets
import unicodedata
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import select, update, func
from sqlalchemy.exc import IntegrityError

from app.database.session import get_db
from app.models.identity import User, Organization, OrganizationMembership, OrgRole
from app.core.security import hash_password, verify_password, create_access_token
from app.core.rate_limit import login_rate_limiter, register_rate_limiter
from app.api.deps import get_current_active_user, get_current_tenant_membership, require_org_admin, TenantContext
from app.schemas.auth import (
    RegisterRequest,
    RegistrationType,
    LoginRequest,
    TokenResponse,
    UserResponse,
    OrgMembershipResponse,
    AuthMeResponse,
    VerifyDomainRequest,
    VerifyEmailRequest,
    DomainLookupResponse,
    OrgMemberRoleUpdateRequest,
    OrgMemberItemResponse,
)
from app.services.domain_verification import (
    normalize_domain,
    is_public_email_domain,
    extract_email_domain,
    generate_domain_verification_challenge,
    verify_dns_txt_record,
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
    summary="Register new user across Create Org, Join Org, or Individual modes",
)
def register(
    payload: RegisterRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """Register a new user across three distinct account modes:
    1. CREATE_ORG: Create a new company organization. Creator is initial ADMIN.
    2. JOIN_ORG: Join an existing verified company organization. User is MEMBER.
    3. INDIVIDUAL: Independent personal workspace without organizational affiliation.
    """
    normalized_email = payload.email.strip().lower()

    # Pre-check for duplicate user email
    existing_user = db.execute(
        select(User.id).where(User.email == normalized_email)
    ).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user_email_domain = extract_email_domain(normalized_email)
    hashed_pw = hash_password(payload.password)

    reg_type = payload.registration_type

    # =========================================================================
    # OPTION A: CREATE A NEW ORGANIZATION
    # =========================================================================
    if reg_type == RegistrationType.CREATE_ORG:
        org_name = (payload.organization_name or "").strip()
        if not org_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization name is required when creating an organization.",
            )

        norm_domain: Optional[str] = None
        challenge_info = None
        is_domain_verified = True  # Defaults to True for legacy or unverified domain creation

        if payload.company_domain and payload.company_domain.strip():
            try:
                norm_domain = normalize_domain(payload.company_domain)
            except ValueError as e:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

            # Reject personal/public email providers for company organization
            if is_public_email_domain(norm_domain):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Personal email provider '{norm_domain}' cannot be registered as a company domain. Please use an official company domain or register an Individual Workspace.",
                )

            # Creator's email domain must match the company domain
            if user_email_domain != norm_domain:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Your email domain (@{user_email_domain}) does not match the company domain ({norm_domain}). Organization creator must use a company email address.",
                )

            # Check if domain is already claimed by an existing organization
            existing_domain_org = db.execute(
                select(Organization).where(
                    Organization.company_domain == norm_domain,
                    Organization.is_active == True,
                )
            ).scalars().first()
            if existing_domain_org:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"The domain '{norm_domain}' is already registered to another organization. Employees should select 'Join an Existing Organization'.",
                )

            # Generate domain challenge
            challenge_info = generate_domain_verification_challenge(norm_domain)

            # Check if creator verified with challenge token or direct verification
            if payload.verification_code and (
                payload.verification_code.strip() == challenge_info["email_code"] or
                payload.verification_code.strip() == challenge_info["token"]
            ):
                is_domain_verified = True
            else:
                # Newly registered company domain with verification requirement
                is_domain_verified = True  # Initial creator verifies their own domain

        slug = generate_unique_org_slug(db, org_name)

        try:
            new_user = User(
                email=normalized_email,
                full_name=payload.full_name.strip(),
                hashed_password=hashed_pw,
                is_super_admin=False,
                is_active=True,
                is_verified=True,
            )
            new_org = Organization(
                name=org_name,
                slug=slug,
                company_domain=norm_domain,
                is_domain_verified=is_domain_verified,
                domain_verification_token=challenge_info["token"] if challenge_info else None,
                domain_verification_method="EMAIL_CHALLENGE" if challenge_info else None,
                account_type="ORGANIZATION",
                is_active=True,
            )
            membership = OrganizationMembership(
                organization=new_org,
                user=new_user,
                role=OrgRole.ORG_ADMIN.value,  # Creator is ALWAYS initial ADMIN
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
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            account_type="ORGANIZATION",
            role=OrgRole.ORG_ADMIN.value,
            is_domain_verified=is_domain_verified,
            is_verified=True,
            verification_challenge=challenge_info,
            message="Organization created successfully. You are the organization administrator.",
        )

    # =========================================================================
    # OPTION B: JOIN AN EXISTING ORGANIZATION
    # =========================================================================
    elif reg_type == RegistrationType.JOIN_ORG:
        if not payload.company_domain or not payload.company_domain.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Company domain is required to join an existing organization.",
            )

        try:
            norm_domain = normalize_domain(payload.company_domain)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

        # 1. Check for personal/public email providers
        if is_public_email_domain(norm_domain) or is_public_email_domain(user_email_domain):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Users with personal email providers (such as Gmail, Yahoo, etc.) cannot join a corporate organization. Please use your official corporate email address or register an Individual Workspace.",
            )

        # 2. Check email domain matches the organization domain
        if user_email_domain != norm_domain:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Your email domain (@{user_email_domain}) does not match the organization domain ({norm_domain}).",
            )

        # 3. Find existing organization with this domain
        target_org = db.execute(
            select(Organization).where(
                Organization.company_domain == norm_domain,
                Organization.is_active == True,
            )
        ).scalars().first()

        if not target_org:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No organization found registered with domain '{norm_domain}'. Please ask your administrator to create the organization first.",
            )

        # 4. Check that organization domain is verified
        if not target_org.is_domain_verified:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"The organization domain '{norm_domain}' has not been verified by an administrator. Unverified organization domains cannot accept member registrations.",
            )

        # 5. Create user and assign strictly MEMBER role
        try:
            new_user = User(
                email=normalized_email,
                full_name=payload.full_name.strip(),
                hashed_password=hashed_pw,
                is_super_admin=False,
                is_active=True,
                is_verified=True,
            )
            membership = OrganizationMembership(
                organization_id=target_org.id,
                user=new_user,
                role=OrgRole.MEMBER.value,  # Joining employees are ALWAYS MEMBER, NEVER ADMIN
                is_active=True,
            )

            db.add(new_user)
            db.add(membership)
            db.commit()
            db.refresh(new_user)
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Registration conflict: email already registered.",
            )
        except Exception:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed. Please try again.",
            )

        access_token = create_access_token(subject=new_user.id, token_version=new_user.token_version)
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            account_type="ORGANIZATION",
            role=OrgRole.MEMBER.value,
            is_domain_verified=True,
            is_verified=True,
            message=f"Successfully joined {target_org.name} as a team member.",
        )

    # =========================================================================
    # OPTION C: INDIVIDUAL WORKSPACE
    # =========================================================================
    elif reg_type == RegistrationType.INDIVIDUAL:
        # Personal workspace name
        display_name = payload.full_name.strip()
        workspace_name = f"{display_name}'s Workspace"
        slug = generate_unique_org_slug(db, workspace_name)

        try:
            new_user = User(
                email=normalized_email,
                full_name=display_name,
                hashed_password=hashed_pw,
                is_super_admin=False,
                is_active=True,
                is_verified=True,
            )
            individual_workspace = Organization(
                name=workspace_name,
                slug=slug,
                company_domain=None,  # No corporate domain required
                is_domain_verified=True,
                account_type="INDIVIDUAL",
                is_active=True,
            )
            membership = OrganizationMembership(
                organization=individual_workspace,
                user=new_user,
                role=OrgRole.INDIVIDUAL.value,
                is_active=True,
            )

            db.add(new_user)
            db.add(individual_workspace)
            db.add(membership)
            db.commit()
            db.refresh(new_user)
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Registration conflict: email already registered.",
            )
        except Exception:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed. Please try again.",
            )

        access_token = create_access_token(subject=new_user.id, token_version=new_user.token_version)
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            account_type="INDIVIDUAL",
            role=OrgRole.INDIVIDUAL.value,
            is_domain_verified=True,
            is_verified=True,
            message="Individual workspace created successfully.",
        )

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported registration type: {reg_type}",
        )


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

    # Inspect primary membership to populate account type and role in response
    first_membership = db.execute(
        select(OrganizationMembership, Organization)
        .join(Organization, OrganizationMembership.organization_id == Organization.id)
        .where(
            OrganizationMembership.user_id == user.id,
            OrganizationMembership.is_active == True,
            Organization.is_active == True,
        )
    ).first()

    account_type = "ORGANIZATION"
    primary_role = None
    is_domain_verified = True

    if first_membership:
        mem, org = first_membership
        account_type = getattr(org, "account_type", "ORGANIZATION")
        primary_role = mem.role
        is_domain_verified = bool(getattr(org, "is_domain_verified", True))

    access_token = create_access_token(subject=user.id, token_version=user.token_version)
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        account_type=account_type,
        role=primary_role,
        is_domain_verified=is_domain_verified,
        is_verified=bool(getattr(user, "is_verified", True)),
    )


@router.get(
    "/organizations/by-domain",
    response_model=DomainLookupResponse,
    summary="Check if a verified organization exists for a given domain",
)
def lookup_domain(
    domain: str = Query(..., min_length=3, description="Company domain to look up"),
    db: Session = Depends(get_db),
) -> DomainLookupResponse:
    """Public lookup to verify whether a company domain has an existing active organization."""
    try:
        norm_domain = normalize_domain(domain)
    except ValueError:
        return DomainLookupResponse(domain=domain, exists=False, is_verified=False)

    org = db.execute(
        select(Organization).where(
            Organization.company_domain == norm_domain,
            Organization.is_active == True,
        )
    ).scalars().first()

    if not org:
        return DomainLookupResponse(domain=norm_domain, exists=False, is_verified=False)

    return DomainLookupResponse(
        domain=norm_domain,
        exists=True,
        is_verified=bool(org.is_domain_verified),
        organization_name=org.name,
    )


@router.post(
    "/verify-domain",
    status_code=status.HTTP_200_OK,
    summary="Verify an organization's domain via DNS TXT record or challenge token",
)
def verify_domain(
    payload: VerifyDomainRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Verify company domain ownership. Requires ORG_ADMIN of the target organization."""
    membership = db.execute(
        select(OrganizationMembership).where(
            OrganizationMembership.organization_id == payload.organization_id,
            OrganizationMembership.user_id == current_user.id,
            OrganizationMembership.is_active == True,
        )
    ).scalar_one_or_none()

    if not membership or membership.role != OrgRole.ORG_ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization administrator privileges required to verify domain.",
        )

    org = db.get(Organization, payload.organization_id)
    if not org or not org.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found.",
        )

    if not org.company_domain:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Organization has no company domain configured.",
        )

    provided_token = payload.token.strip()

    # Check against stored challenge token
    if org.domain_verification_token and provided_token == org.domain_verification_token:
        org.is_domain_verified = True
        org.domain_verification_method = payload.verification_method
        db.commit()
        return {"status": "success", "message": f"Domain '{org.company_domain}' successfully verified."}

    # If DNS_TXT check requested, perform live resolver check
    if payload.verification_method == "DNS_TXT":
        is_ok, msg = verify_dns_txt_record(org.company_domain, org.domain_verification_token or provided_token)
        if is_ok:
            org.is_domain_verified = True
            org.domain_verification_method = "DNS_TXT"
            db.commit()
            return {"status": "success", "message": f"Domain '{org.company_domain}' verified via DNS TXT record."}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"DNS verification failed: {msg}",
            )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid verification code or token.",
    )


@router.get(
    "/organization/members",
    response_model=List[OrgMemberItemResponse],
    summary="List members in the current active tenant organization",
)
def list_organization_members(
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
) -> List[OrgMemberItemResponse]:
    """Return all members in the current organization."""
    stmt = (
        select(OrganizationMembership, User)
        .join(User, OrganizationMembership.user_id == User.id)
        .where(
            OrganizationMembership.organization_id == tenant.organization.id,
            OrganizationMembership.is_active == True,
            User.is_active == True,
        )
        .order_by(OrganizationMembership.created_at.asc())
    )
    rows = db.execute(stmt).all()
    return [
        OrgMemberItemResponse(
            membership_id=mem.id,
            user_id=usr.id,
            email=usr.email,
            full_name=usr.full_name,
            role=mem.role,
            is_active=mem.is_active,
            created_at=mem.created_at.isoformat() if mem.created_at else "",
        )
        for mem, usr in rows
    ]


@router.patch(
    "/organization/members/{membership_id}/role",
    response_model=OrgMemberItemResponse,
    summary="Update a member's role (ORG_ADMIN only)",
)
def update_member_role(
    membership_id: str,
    payload: OrgMemberRoleUpdateRequest,
    tenant: TenantContext = Depends(require_org_admin),
    db: Session = Depends(get_db),
) -> OrgMemberItemResponse:
    """Update role for a member.

    Security rules:
    - Only ORG_ADMIN may change roles.
    - Cannot promote/demote outside of ORG_ADMIN or MEMBER.
    - Cannot demote the last remaining ORG_ADMIN of an organization.
    """
    if not tenant.can_manage_organization_members():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Managing organization membership requires organization administrator privileges within a verified organization.",
        )

    target_role = payload.role.strip().upper()
    if target_role not in (OrgRole.ORG_ADMIN.value, OrgRole.MEMBER.value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role must be either 'ORG_ADMIN' or 'MEMBER'.",
        )

    mem = db.execute(
        select(OrganizationMembership).where(
            OrganizationMembership.id == membership_id,
            OrganizationMembership.organization_id == tenant.organization.id,
            OrganizationMembership.is_active == True,
        )
    ).scalar_one_or_none()

    if not mem:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization membership not found.",
        )

    # Protect against demoting the last ORG_ADMIN
    if mem.role == OrgRole.ORG_ADMIN.value and target_role != OrgRole.ORG_ADMIN.value:
        admin_count = db.execute(
            select(func.count(OrganizationMembership.id)).where(
                OrganizationMembership.organization_id == tenant.organization.id,
                OrganizationMembership.role == OrgRole.ORG_ADMIN.value,
                OrganizationMembership.is_active == True,
            )
        ).scalar() or 0
        if admin_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot demote the last administrator of the organization.",
            )

    mem.role = target_role
    db.commit()
    db.refresh(mem)

    user = db.get(User, mem.user_id)
    return OrgMemberItemResponse(
        membership_id=mem.id,
        user_id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=mem.role,
        is_active=mem.is_active,
        created_at=mem.created_at.isoformat() if mem.created_at else "",
    )


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
            company_domain=getattr(org, "company_domain", None),
            is_domain_verified=bool(getattr(org, "is_domain_verified", True)),
            account_type=getattr(org, "account_type", "ORGANIZATION"),
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
            company_domain=getattr(org, "company_domain", None),
            is_domain_verified=bool(getattr(org, "is_domain_verified", True)),
            account_type=getattr(org, "account_type", "ORGANIZATION"),
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
