"""API Dependencies for Multilingual Meeting Intelligence System.

Provides:
- get_current_active_user: Validates JWT access token and resolves active User.
- TenantContext: Dataclass holding authenticated User, Organization, and OrganizationMembership.
- get_current_tenant_membership: Strictly verifies active user membership in the selected organization.
- require_org_admin: Enforces ORG_ADMIN role within the active tenant organization.
- require_super_admin: Enforces global Super Administrator privilege.
"""

from dataclasses import dataclass
from typing import Optional, List
from fastapi import Header, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.database.session import get_db
from app.models.identity import Organization, User, OrganizationMembership, OrgRole
from app.core.security import decode_access_token

security_scheme = HTTPBearer(auto_error=False)


@dataclass
class TenantContext:
    """Encapsulates the verified tenant authorization context for a request."""
    user: User
    organization: Organization
    membership: OrganizationMembership

    @property
    def role(self) -> str:
        return self.membership.role


def get_current_active_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Validate JWT access token and return the active User entity.

    Enforces:
    - HTTP Bearer authentication scheme.
    - Cryptographic signature, expiration, and claim verification via decode_access_token().
    - Loading user from database using JWT 'sub' claim.
    - Rejection of missing, deleted, inactive, or invalid users.
    - Returns HTTP 401 Unauthorized with 'WWW-Authenticate: Bearer' header on failure.
    - Zero trust of client-supplied or token-embedded roles/organizations.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.get(User, user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Server-side token revocation check: compare token claims against database user record
    token_ver = payload.get("token_version")
    if token_ver != user.token_version:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked or superseded",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def get_current_tenant_membership(
    x_organization_id: Optional[str] = Header(None, alias="X-Organization-Id"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> TenantContext:
    """Validate and resolve tenant organization membership for the authenticated user.

    Strict rules:
    - Requires valid Bearer JWT authentication (current_user).
    - X-Organization-Id acts only as an organization selector, NEVER as proof of authorization.
    - Verified against active database membership for current_user.
    - Both OrganizationMembership.is_active and Organization.is_active must be True.
    - If X-Organization-Id is omitted:
        - Auto-selects if user has exactly ONE active membership in an active organization.
        - Raises HTTP 400 if user has MULTIPLE active memberships (explicit selector required).
        - Raises HTTP 403 if user has ZERO active memberships.
    - If X-Organization-Id is provided:
        - Raises HTTP 403 if the user lacks active membership in that organization.
    - User.is_super_admin does NOT bypass tenant membership.
    - Never auto-provisions users or organizations.
    - Never trusts arbitrary client-supplied IDs.
    """
    stmt = (
        select(OrganizationMembership)
        .join(Organization, OrganizationMembership.organization_id == Organization.id)
        .where(
            OrganizationMembership.user_id == current_user.id,
            OrganizationMembership.is_active == True,
            Organization.is_active == True,
        )
    )
    active_memberships = db.execute(stmt).scalars().all()

    if not active_memberships:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have an active membership in any active organization.",
        )

    selected_membership: Optional[OrganizationMembership] = None

    if x_organization_id:
        target_org_id = x_organization_id.strip()
        for m in active_memberships:
            if m.organization_id == target_org_id:
                selected_membership = m
                break
        if not selected_membership:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Active membership in the specified organization is required.",
            )
    else:
        if len(active_memberships) == 1:
            selected_membership = active_memberships[0]
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="X-Organization-Id header is required when user belongs to multiple organizations.",
            )

    return TenantContext(
        user=current_user,
        organization=selected_membership.organization,
        membership=selected_membership,
    )


def require_org_admin(
    tenant: TenantContext = Depends(get_current_tenant_membership),
) -> TenantContext:
    """Ensure the user possesses ORG_ADMIN role in the active tenant organization.

    Strict rule:
    User.is_super_admin does NOT bypass tenant membership or the organization role check.
    A global Super Admin must possess explicit ORG_ADMIN membership in the selected organization.
    """
    if tenant.membership.role != OrgRole.ORG_ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization administrator privileges required.",
        )
    return tenant


def require_super_admin(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """Ensure the authenticated user possesses global Super Admin privilege.

    Used strictly for platform-wide operations (e.g. system hardware diagnostics).
    """
    if not current_user.is_super_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Global Super Administrator privileges required.",
        )
    return current_user
