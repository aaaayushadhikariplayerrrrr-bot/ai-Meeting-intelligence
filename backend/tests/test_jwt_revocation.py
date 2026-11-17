"""Tests for server-side JWT revocation using per-user token_version.

Covers:
- Valid token matching user's current token_version succeeds.
- Outdated token_version is rejected with HTTP 401.
- Missing or malformed token_version claims are rejected with HTTP 401.
- POST /api/v1/auth/logout atomically increments token_version in the database.
- Tokens issued prior to logout are immediately rejected on subsequent requests.
- New tokens issued after logout authenticate successfully with the incremented version.
- Multi-session invalidation: logging out from one session invalidates all active sessions.
- Inactive users are rejected even if token_version matches.
- Registration sets token_version=1 and returns a token with token_version=1.
"""

import time
import jwt
import pytest
from app.core.config import settings
from app.core.security import ACCESS_TOKEN_TYPE, create_access_token, decode_access_token
from app.models.identity import User


def test_valid_token_with_matching_version_succeeds(client, org_admin_user, test_org_a):
    """Verify that a token with matching token_version authenticates successfully."""
    token = create_access_token(subject=org_admin_user.id, token_version=org_admin_user.token_version)
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.get("/api/v1/auth/me", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user"]["id"] == org_admin_user.id


def test_token_with_outdated_version_rejected_with_401(client, db_session, org_admin_user):
    """Verify that a token with a superseded token_version is rejected with 401."""
    # Issue token with current version
    old_token = create_access_token(subject=org_admin_user.id, token_version=org_admin_user.token_version)

    # Increment token_version in database (simulating revocation or logout)
    org_admin_user.token_version += 1
    db_session.add(org_admin_user)
    db_session.commit()
    db_session.refresh(org_admin_user)

    headers = {"Authorization": f"Bearer {old_token}"}
    resp = client.get("/api/v1/auth/me", headers=headers)
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"
    assert "revoked or superseded" in resp.json()["detail"]


def test_token_with_future_token_version_rejected_with_401(client, org_admin_user):
    """Verify that a correctly signed token with a future/tampered token_version is rejected with 401."""
    future_version = org_admin_user.token_version + 10
    future_token = create_access_token(subject=org_admin_user.id, token_version=future_version)
    headers = {"Authorization": f"Bearer {future_token}"}

    resp = client.get("/api/v1/auth/me", headers=headers)
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"
    assert "revoked or superseded" in resp.json()["detail"]


def test_token_missing_token_version_claim_rejected_with_401(client, org_admin_user):
    """Verify that tokens missing the required token_version claim are rejected with 401."""
    now = int(time.time())
    payload_no_tv = {
        "sub": org_admin_user.id,
        "iat": now,
        "exp": now + 3600,
        "type": ACCESS_TOKEN_TYPE,
    }
    raw_token = jwt.encode(payload_no_tv, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    headers = {"Authorization": f"Bearer {raw_token}"}

    resp = client.get("/api/v1/auth/me", headers=headers)
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"


@pytest.mark.parametrize("bad_tv", [0, -1, 1.5, "1", True, False, None])
def test_token_with_invalid_token_version_type_rejected_with_401(client, org_admin_user, bad_tv):
    """Verify that tokens with boolean, float, string, zero, or negative token_version are rejected with 401."""
    now = int(time.time())
    payload = {
        "sub": org_admin_user.id,
        "iat": now,
        "exp": now + 3600,
        "type": ACCESS_TOKEN_TYPE,
        "token_version": bad_tv,
    }
    raw_token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    headers = {"Authorization": f"Bearer {raw_token}"}

    resp = client.get("/api/v1/auth/me", headers=headers)
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"


def test_logout_endpoint_increments_token_version(client, db_session, org_admin_user):
    """Verify POST /api/v1/auth/logout increments token_version in the database."""
    initial_version = org_admin_user.token_version
    token = create_access_token(subject=org_admin_user.id, token_version=initial_version)
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post("/api/v1/auth/logout", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"message": "Successfully logged out"}

    db_session.refresh(org_admin_user)
    assert org_admin_user.token_version == initial_version + 1


def test_token_issued_before_logout_rejected_after_logout(client, org_admin_user):
    """Verify that a token used to call /logout cannot be used for any subsequent requests."""
    token = create_access_token(subject=org_admin_user.id, token_version=org_admin_user.token_version)
    headers = {"Authorization": f"Bearer {token}"}

    # Token works initially
    resp1 = client.get("/api/v1/auth/me", headers=headers)
    assert resp1.status_code == 200

    # User logs out
    resp_logout = client.post("/api/v1/auth/logout", headers=headers)
    assert resp_logout.status_code == 200

    # Token is now immediately revoked
    resp2 = client.get("/api/v1/auth/me", headers=headers)
    assert resp2.status_code == 401
    assert "revoked or superseded" in resp2.json()["detail"]


def test_new_token_issued_after_logout_succeeds(client, org_admin_user):
    """Verify that logging in after logout generates a fresh token with updated token_version."""
    initial_version = org_admin_user.token_version
    old_token = create_access_token(subject=org_admin_user.id, token_version=initial_version)
    headers_old = {"Authorization": f"Bearer {old_token}"}

    # Logout
    client.post("/api/v1/auth/logout", headers=headers_old)

    # Login again with credentials
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": org_admin_user.email, "password": "AdminPass123!"},
    )
    assert login_resp.status_code == 200
    new_token = login_resp.json()["access_token"]
    assert new_token != old_token

    # Directly decode the new token and assert token_version equals incremented database value
    claims = decode_access_token(new_token)
    assert claims is not None
    assert claims["token_version"] == initial_version + 1

    # New token works
    headers_new = {"Authorization": f"Bearer {new_token}"}
    me_resp = client.get("/api/v1/auth/me", headers=headers_new)
    assert me_resp.status_code == 200
    assert me_resp.json()["user"]["id"] == org_admin_user.id


def test_simultaneous_sessions_all_revoked_on_logout(client, org_admin_user):
    """Verify logging out invalidates all concurrently active tokens for that user."""
    # Simulate two active sessions (e.g. desktop and mobile)
    session1_token = create_access_token(subject=org_admin_user.id, token_version=org_admin_user.token_version)
    session2_token = create_access_token(subject=org_admin_user.id, token_version=org_admin_user.token_version)

    # Both work
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {session1_token}"}).status_code == 200
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {session2_token}"}).status_code == 200

    # Session 1 logs out
    resp_logout = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {session1_token}"})
    assert resp_logout.status_code == 200

    # Both Session 1 and Session 2 are now revoked
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {session1_token}"}).status_code == 401
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {session2_token}"}).status_code == 401


def test_logout_unauthenticated_rejected(client):
    """Verify calling /logout without credentials returns 401."""
    resp = client.post("/api/v1/auth/logout")
    assert resp.status_code == 401


def test_logout_with_revoked_token_rejected(client, org_admin_user):
    """Verify calling /logout with an already revoked token returns 401."""
    token = create_access_token(subject=org_admin_user.id, token_version=org_admin_user.token_version)
    headers = {"Authorization": f"Bearer {token}"}

    # First logout succeeds
    assert client.post("/api/v1/auth/logout", headers=headers).status_code == 200

    # Second logout with same token fails
    assert client.post("/api/v1/auth/logout", headers=headers).status_code == 401


def test_inactive_user_with_valid_token_version_rejected(client, db_session, org_admin_user):
    """Verify an inactive user is rejected even if the token has the current token_version."""
    token = create_access_token(subject=org_admin_user.id, token_version=org_admin_user.token_version)
    headers = {"Authorization": f"Bearer {token}"}

    org_admin_user.is_active = False
    db_session.add(org_admin_user)
    db_session.commit()

    resp = client.get("/api/v1/auth/me", headers=headers)
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"


def test_registration_sets_initial_token_version_1(client, db_session):
    """Verify new user registration assigns token_version=1 and encodes token_version=1 in JWT."""
    unique_suffix = int(time.time() * 1000)
    reg_payload = {
        "email": f"newuser_{unique_suffix}@example.com",
        "full_name": "New Version Test User",
        "password": "Password123!",
        "organization_name": f"New Org {unique_suffix}",
    }

    resp = client.post("/api/v1/auth/register", json=reg_payload)
    assert resp.status_code == 201
    token = resp.json()["access_token"]

    # Verify user record in database
    created_user = db_session.query(User).filter(User.email == reg_payload["email"]).first()
    assert created_user is not None
    assert created_user.token_version == 1

    # Verify token claims
    claims = jwt.decode(
        token,
        settings.SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
        options={"verify_signature": True},
    )
    assert claims["token_version"] == 1
    assert claims["sub"] == created_user.id

    # Verify token works to authenticate
    me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
