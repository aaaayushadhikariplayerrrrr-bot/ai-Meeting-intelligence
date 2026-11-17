"""Tests for Phase 6 Step 4B: Authentication Schemas, Endpoints, and Dependencies.

Covers:
- Self-service registration with organization creation
- Credential authentication (login) with timing mitigation
- Email normalization and validation
- Slug generation and collision handling
- Atomic rollback on failed registration
- Password verification failures (wrong password, unknown email, NULL password hash)
- Inactive user rejection
- Token validation (invalid, expired, malformed, wrong type, missing)
- Authenticated user profile (/me) and organization introspection (/organizations)
- Filtering of inactive organizations and memberships
- Response sanitization (no passwords or sensitive hashes exposed)
- Bootstrap CLI tool behavior and --force protection
"""

from datetime import timedelta
import jwt
import pytest
from sqlalchemy import select

from app.core.config import settings
from app.core.security import decode_access_token, verify_password, hash_password
from app.core.rate_limit import register_rate_limiter
from app.models.identity import User, Organization, OrganizationMembership, OrgRole
from scripts.bootstrap_dev_password import bootstrap_dev_password


def test_register_success(client, db_session):
    """Verify successful user registration, organization creation, and minimal JWT response."""
    payload = {
        "email": "alice@example.com",
        "full_name": "Alice Smith",
        "password": "SecurePassword123!",
        "organization_name": "Acme Innovations",
    }
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

    # Verify minimal JWT token claims
    token = data["access_token"]
    claims = decode_access_token(token)
    assert claims is not None
    assert set(claims.keys()) == {"sub", "iat", "exp", "type", "token_version"}
    assert claims["type"] == "access"
    assert claims["token_version"] == 1
    assert "organization_id" not in claims
    assert "role" not in claims

    # Verify database state
    user = db_session.execute(
        select(User).where(User.email == "alice@example.com")
    ).scalar_one_or_none()
    assert user is not None
    assert user.id == claims["sub"]
    assert user.full_name == "Alice Smith"
    assert user.is_super_admin is False
    assert user.is_active is True
    assert user.hashed_password.startswith("$argon2id$")
    assert verify_password("SecurePassword123!", user.hashed_password) is True

    # Verify organization and ORG_ADMIN membership
    org = db_session.execute(
        select(Organization).where(Organization.name == "Acme Innovations")
    ).scalar_one_or_none()
    assert org is not None
    assert org.slug == "acme-innovations"

    membership = db_session.execute(
        select(OrganizationMembership).where(
            OrganizationMembership.user_id == user.id,
            OrganizationMembership.organization_id == org.id,
        )
    ).scalar_one_or_none()
    assert membership is not None
    assert membership.role == OrgRole.ORG_ADMIN.value
    assert membership.is_active is True


def test_register_duplicate_email(client):
    """Verify duplicate email registration returns HTTP 409 Conflict."""
    payload = {
        "email": "duplicate@example.com",
        "full_name": "User One",
        "password": "Password12345",
        "organization_name": "First Org",
    }
    resp1 = client.post("/api/v1/auth/register", json=payload)
    assert resp1.status_code == 201

    payload2 = {
        "email": "duplicate@example.com",
        "full_name": "User Two",
        "password": "Password67890",
        "organization_name": "Second Org",
    }
    resp2 = client.post("/api/v1/auth/register", json=payload2)
    assert resp2.status_code == 409
    assert "already exists" in resp2.json()["detail"].lower()


def test_register_email_normalization(client, db_session):
    """Verify email is normalized to lowercase and whitespace trimmed."""
    payload = {
        "email": "   SpacesAndCaps@EXAMPLE.COM   ",
        "full_name": "Normalized User",
        "password": "Password12345",
        "organization_name": "Normalize Org",
    }
    resp = client.post("/api/v1/auth/register", json=payload)
    assert resp.status_code == 201

    user = db_session.execute(
        select(User).where(User.email == "spacesandcaps@example.com")
    ).scalar_one_or_none()
    assert user is not None
    assert user.email == "spacesandcaps@example.com"

    # Login works with normalized variations
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "SPACESANDCAPS@EXAMPLE.COM", "password": "Password12345"},
    )
    assert login_resp.status_code == 200


def test_register_slug_collision_resolution(client, db_session):
    """Verify organization slug collision is safely resolved with a unique suffix."""
    payload1 = {
        "email": "user1@org.com",
        "full_name": "User One",
        "password": "Password12345",
        "organization_name": "Tech Corp",
    }
    resp1 = client.post("/api/v1/auth/register", json=payload1)
    assert resp1.status_code == 201

    payload2 = {
        "email": "user2@org.com",
        "full_name": "User Two",
        "password": "Password12345",
        "organization_name": "Tech Corp",
    }
    resp2 = client.post("/api/v1/auth/register", json=payload2)
    assert resp2.status_code == 201

    orgs = db_session.execute(
        select(Organization).where(Organization.name == "Tech Corp").order_by(Organization.created_at)
    ).scalars().all()
    assert len(orgs) == 2
    assert orgs[0].slug == "tech-corp"
    assert orgs[1].slug.startswith("tech-corp-")
    assert orgs[0].slug != orgs[1].slug


def test_register_validation_errors(client):
    """Verify Pydantic schema validation rejects invalid inputs."""
    register_rate_limiter.reset()
    # Invalid email
    r1 = client.post("/api/v1/auth/register", json={
        "email": "not-an-email",
        "full_name": "Test",
        "password": "Password123",
        "organization_name": "Org",
    })
    assert r1.status_code == 422

    register_rate_limiter.reset()
    # Password too short (<8)
    r2 = client.post("/api/v1/auth/register", json={
        "email": "valid@email.com",
        "full_name": "Test",
        "password": "short",
        "organization_name": "Org",
    })
    assert r2.status_code == 422

    register_rate_limiter.reset()
    # Whitespace-only password
    r3 = client.post("/api/v1/auth/register", json={
        "email": "valid@email.com",
        "full_name": "Test",
        "password": "          ",
        "organization_name": "Org",
    })
    assert r3.status_code == 422

    register_rate_limiter.reset()
    # Whitespace-only full name
    r4 = client.post("/api/v1/auth/register", json={
        "email": "valid@email.com",
        "full_name": "   ",
        "password": "ValidPassword123",
        "organization_name": "Org",
    })
    assert r4.status_code == 422

    register_rate_limiter.reset()
    # Whitespace-only organization name
    r5 = client.post("/api/v1/auth/register", json={
        "email": "valid@email.com",
        "full_name": "Test User",
        "password": "ValidPassword123",
        "organization_name": "   ",
    })
    assert r5.status_code == 422


def test_register_atomic_rollback(client, db_session, monkeypatch):
    """Verify atomic transaction rollback leaves no orphaned user or organization if error occurs."""
    from sqlalchemy.orm import Session

    original_commit = Session.commit

    def failing_commit(self):
        # Force a database failure on commit
        raise RuntimeError("Simulated transaction failure during registration")

    monkeypatch.setattr(Session, "commit", failing_commit)

    payload = {
        "email": "atomic@example.com",
        "full_name": "Atomic User",
        "password": "AtomicPassword123",
        "organization_name": "Atomic Org",
    }
    resp = client.post("/api/v1/auth/register", json=payload)
    assert resp.status_code == 500

    # Restore commit and check that neither user nor org was saved
    monkeypatch.undo()
    user = db_session.execute(
        select(User).where(User.email == "atomic@example.com")
    ).scalar_one_or_none()
    assert user is None

    org = db_session.execute(
        select(Organization).where(Organization.name == "Atomic Org")
    ).scalar_one_or_none()
    assert org is None


def test_login_success(client):
    """Verify successful login returns valid JWT token."""
    client.post("/api/v1/auth/register", json={
        "email": "bob@example.com",
        "full_name": "Bob Jones",
        "password": "BobSecretPassword123",
        "organization_name": "Bob Org",
    })

    login_resp = client.post("/api/v1/auth/login", json={
        "email": "bob@example.com",
        "password": "BobSecretPassword123",
    })
    assert login_resp.status_code == 200
    data = login_resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password(client):
    """Verify wrong password returns generic 401 with WWW-Authenticate header."""
    client.post("/api/v1/auth/register", json={
        "email": "carol@example.com",
        "full_name": "Carol Danvers",
        "password": "CorrectPassword123",
        "organization_name": "Carol Org",
    })

    resp = client.post("/api/v1/auth/login", json={
        "email": "carol@example.com",
        "password": "WrongPassword999",
    })
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid email or password"
    assert resp.headers.get("www-authenticate") == "Bearer"


def test_login_unknown_email(client):
    """Verify non-existent email returns identical 401 with WWW-Authenticate header."""
    resp = client.post("/api/v1/auth/login", json={
        "email": "nonexistent@example.com",
        "password": "SomePassword123",
    })
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid email or password"
    assert resp.headers.get("www-authenticate") == "Bearer"


def test_login_null_password_hash(client, db_session):
    """Verify account with NULL password hash cannot login and returns identical 401."""
    null_pw_user = User(
        email="nullhash@example.com",
        full_name="Null Hash User",
        hashed_password=None,
        is_active=True,
    )
    db_session.add(null_pw_user)
    db_session.commit()

    resp = client.post("/api/v1/auth/login", json={
        "email": "nullhash@example.com",
        "password": "AttemptPassword123",
    })
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid email or password"


def test_login_inactive_user(client, db_session):
    """Verify deactivated user cannot login and returns identical 401."""
    inactive_user = User(
        email="inactive@example.com",
        full_name="Inactive User",
        hashed_password=hash_password("UserPassword123"),
        is_active=False,
    )
    db_session.add(inactive_user)
    db_session.commit()

    resp = client.post("/api/v1/auth/login", json={
        "email": "inactive@example.com",
        "password": "UserPassword123",
    })
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid email or password"


def test_auth_me_authenticated(client):
    """Verify /me returns current user and active memberships."""
    reg_resp = client.post("/api/v1/auth/register", json={
        "email": "me_user@example.com",
        "full_name": "Me User",
        "password": "Password12345",
        "organization_name": "Me Workspace",
    })
    token = reg_resp.json()["access_token"]

    resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["user"]["email"] == "me_user@example.com"
    assert data["user"]["full_name"] == "Me User"
    assert data["user"]["is_super_admin"] is False
    assert len(data["organizations"]) == 1
    assert data["organizations"][0]["organization_name"] == "Me Workspace"
    assert data["organizations"][0]["role"] == "ORG_ADMIN"


def test_auth_me_unauthenticated_and_missing_token(client):
    """Verify calling /me without authorization returns 401."""
    resp = client.get("/api/v1/auth/me")
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"


def test_auth_me_invalid_and_malformed_token(client):
    """Verify malformed token returns 401."""
    resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer not-a-valid-jwt-token"},
    )
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"


def test_auth_me_expired_token(client, db_session):
    """Verify expired token is rejected with 401."""
    user = User(
        email="expired@example.com",
        full_name="Expired User",
        hashed_password=hash_password("Pass12345"),
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()

    # Craft an expired token
    now_ts = 1000000000
    payload = {
        "sub": user.id,
        "iat": now_ts - 3600,
        "exp": now_ts - 100,  # Expired in past
        "type": "access",
    }
    expired_token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert resp.status_code == 401


def test_auth_me_wrong_token_type(client, db_session):
    """Verify token with wrong type (not 'access') is rejected."""
    user = User(
        email="refreshtoken@example.com",
        full_name="Refresh User",
        hashed_password=hash_password("Pass12345"),
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()

    import time
    now = int(time.time())
    payload = {
        "sub": user.id,
        "iat": now,
        "exp": now + 3600,
        "type": "refresh",  # Not "access"
    }
    wrong_type_token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {wrong_type_token}"},
    )
    assert resp.status_code == 401


def test_auth_me_inactive_user_token(client, db_session):
    """Verify token for deactivated user is rejected on authenticated routes."""
    reg_resp = client.post("/api/v1/auth/register", json={
        "email": "will_deactivate@example.com",
        "full_name": "Will Deactivate",
        "password": "Password12345",
        "organization_name": "Deactivate Org",
    })
    token = reg_resp.json()["access_token"]

    # Deactivate the user in DB
    user = db_session.execute(
        select(User).where(User.email == "will_deactivate@example.com")
    ).scalar_one()
    user.is_active = False
    db_session.commit()

    resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401


def test_organizations_endpoint_and_inactive_filtering(client, db_session):
    """Verify /organizations returns only active memberships in active organizations."""
    reg_resp = client.post("/api/v1/auth/register", json={
        "email": "multi_org@example.com",
        "full_name": "Multi Org User",
        "password": "Password12345",
        "organization_name": "Active Org 1",
    })
    token = reg_resp.json()["access_token"]

    user = db_session.execute(
        select(User).where(User.email == "multi_org@example.com")
    ).scalar_one()

    # Create Org 2 (active org, active membership)
    org2 = Organization(name="Active Org 2", slug="active-org-2", is_active=True)
    m2 = OrganizationMembership(organization=org2, user=user, role="MEMBER", is_active=True)

    # Create Org 3 (inactive org, active membership)
    org3 = Organization(name="Inactive Org 3", slug="inactive-org-3", is_active=False)
    m3 = OrganizationMembership(organization=org3, user=user, role="MEMBER", is_active=True)

    # Create Org 4 (active org, inactive membership)
    org4 = Organization(name="Active Org 4", slug="active-org-4", is_active=True)
    m4 = OrganizationMembership(organization=org4, user=user, role="MEMBER", is_active=False)

    db_session.add_all([org2, m2, org3, m3, org4, m4])
    db_session.commit()

    resp = client.get(
        "/api/v1/auth/organizations",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    org_names = [o["organization_name"] for o in data]

    assert "Active Org 1" in org_names
    assert "Active Org 2" in org_names
    assert "Inactive Org 3" not in org_names
    assert "Active Org 4" not in org_names
    assert len(data) == 2


def test_no_sensitive_info_in_responses(client):
    """Verify sensitive fields like password hashes are absent from all auth responses."""
    reg_resp = client.post("/api/v1/auth/register", json={
        "email": "sensitive_check@example.com",
        "full_name": "Sensitive Check",
        "password": "Password12345",
        "organization_name": "Sensitive Org",
    })
    token = reg_resp.json()["access_token"]
    assert "password" not in reg_resp.text.lower() or "password12345" not in reg_resp.text.lower()
    assert "hashed_password" not in reg_resp.text
    assert "argon2" not in reg_resp.text

    me_resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert "hashed_password" not in me_resp.text
    assert "password" not in me_resp.text


def test_bootstrap_cli_refuse_overwrite_without_force(db_session):
    """Verify bootstrap script sets password once and refuses overwrite without --force."""
    # Seed dev user with NULL password
    dev_user = User(
        email="dev@local.internal",
        full_name="Local Developer",
        hashed_password=None,
        is_super_admin=True,
        is_active=True,
    )
    db_session.add(dev_user)
    db_session.commit()

    # Initial set should succeed
    res1 = bootstrap_dev_password(
        force=False,
        password="FirstDevPassword123",
        db=db_session,
    )
    assert res1 == 0

    db_session.refresh(dev_user)
    assert dev_user.hashed_password is not None
    assert verify_password("FirstDevPassword123", dev_user.hashed_password) is True

    # Overwrite without force must be refused (returns 1)
    res2 = bootstrap_dev_password(
        force=False,
        password="SecondDevPassword123",
        db=db_session,
    )
    assert res2 == 1

    db_session.refresh(dev_user)
    # Password must remain the first one
    assert verify_password("FirstDevPassword123", dev_user.hashed_password) is True

    # Overwrite WITH force should succeed (returns 0)
    res3 = bootstrap_dev_password(
        force=True,
        password="SecondDevPassword123",
        db=db_session,
    )
    assert res3 == 0

    db_session.refresh(dev_user)
    assert verify_password("SecondDevPassword123", dev_user.hashed_password) is True
