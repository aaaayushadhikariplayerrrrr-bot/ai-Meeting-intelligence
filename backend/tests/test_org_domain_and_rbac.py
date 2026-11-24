"""Tests for Organization-Based Registration, Verified Company-Domain Membership, and RBAC."""

import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.identity import User, Organization, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, InputSourceType, MeetingStatus
from app.models.transcription import Transcript, TranscriptType, TranscriptSegment
from app.core.security import hash_password, create_access_token
from app.core.rate_limit import register_rate_limiter, login_rate_limiter


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset rate limiters before each test run."""
    register_rate_limiter.reset()
    login_rate_limiter.reset()


def test_1_create_org_first_user_becomes_admin(client: TestClient, db_session: Session):
    """Scenario 1: First user creates an organization with company domain and becomes ADMIN."""
    unique_domain = f"corp-{uuid.uuid4().hex[:6]}.com"
    email = f"founder@{unique_domain}"

    resp = client.post(
        "/api/v1/auth/register",
        json={
            "registration_type": "CREATE_ORG",
            "email": email,
            "full_name": "Org Founder",
            "password": "Password123!",
            "organization_name": "Corporation Alpha",
            "company_domain": unique_domain,
        },
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert "access_token" in data
    assert data["role"] == OrgRole.ORG_ADMIN.value
    assert data["account_type"] == "ORGANIZATION"

    # Verify database state
    user = db_session.execute(select(User).where(User.email == email)).scalar_one()
    org = db_session.execute(select(Organization).where(Organization.company_domain == unique_domain)).scalar_one()
    mem = db_session.execute(
        select(OrganizationMembership).where(
            OrganizationMembership.user_id == user.id,
            OrganizationMembership.organization_id == org.id,
        )
    ).scalar_one()

    assert mem.role == OrgRole.ORG_ADMIN.value
    assert org.is_domain_verified is True


def test_2_and_3_second_user_joins_as_member_not_admin(client: TestClient, db_session: Session):
    """Scenarios 2 & 3: Second user joins existing verified organization as MEMBER, not ADMIN."""
    domain = f"tech-{uuid.uuid4().hex[:6]}.com"
    founder_email = f"alice@{domain}"
    emp_email = f"bob@{domain}"

    # 1. Founder creates org
    r1 = client.post(
        "/api/v1/auth/register",
        json={
            "registration_type": "CREATE_ORG",
            "email": founder_email,
            "full_name": "Alice Admin",
            "password": "Password123!",
            "organization_name": "Tech Corp",
            "company_domain": domain,
        },
    )
    assert r1.status_code == 201

    # 2. Employee Bob joins
    r2 = client.post(
        "/api/v1/auth/register",
        json={
            "registration_type": "JOIN_ORG",
            "email": emp_email,
            "full_name": "Bob Employee",
            "password": "Password123!",
            "company_domain": domain,
        },
    )
    assert r2.status_code == 201, r2.text
    data2 = r2.json()
    assert data2["role"] == OrgRole.MEMBER.value
    assert data2["role"] != OrgRole.ORG_ADMIN.value

    # Verify membership in DB
    bob = db_session.execute(select(User).where(User.email == emp_email)).scalar_one()
    org = db_session.execute(select(Organization).where(Organization.company_domain == domain)).scalar_one()
    bob_mem = db_session.execute(
        select(OrganizationMembership).where(
            OrganizationMembership.user_id == bob.id,
            OrganizationMembership.organization_id == org.id,
        )
    ).scalar_one()

    assert bob_mem.role == OrgRole.MEMBER.value
    assert bob_mem.role != OrgRole.ORG_ADMIN.value


def test_4_mismatched_email_domain_rejected(client: TestClient, db_session: Session):
    """Scenario 4: User with a different email domain cannot join the organization."""
    domain = f"secure-{uuid.uuid4().hex[:6]}.com"

    # Create org
    client.post(
        "/api/v1/auth/register",
        json={
            "registration_type": "CREATE_ORG",
            "email": f"admin@{domain}",
            "full_name": "Admin",
            "password": "Password123!",
            "organization_name": "Secure Co",
            "company_domain": domain,
        },
    )

    # Attempt to join with a different email domain
    r = client.post(
        "/api/v1/auth/register",
        json={
            "registration_type": "JOIN_ORG",
            "email": "hacker@differentdomain.com",
            "full_name": "Intruder",
            "password": "Password123!",
            "company_domain": domain,
        },
    )
    assert r.status_code == 400
    assert "does not match" in r.json()["detail"]


def test_5_cannot_join_nonexistent_organization(client: TestClient):
    """Scenario 5: Attempting to join a domain that has no registered organization returns 404."""
    r = client.post(
        "/api/v1/auth/register",
        json={
            "registration_type": "JOIN_ORG",
            "email": "nobody@nonexistent-startup-12345.com",
            "full_name": "Nobody",
            "password": "Password123!",
            "company_domain": "nonexistent-startup-12345.com",
        },
    )
    assert r.status_code == 404
    assert "No organization found" in r.json()["detail"]


def test_6_unverified_organization_domain_cannot_accept_members(client: TestClient, db_session: Session):
    """Scenario 6: An unverified organization domain cannot be used for employee membership."""
    domain = f"pending-{uuid.uuid4().hex[:6]}.com"

    # Manually insert an unverified organization
    unverified_org = Organization(
        name="Pending Corp",
        slug=f"pending-{uuid.uuid4().hex[:4]}",
        company_domain=domain,
        is_domain_verified=False,
        is_active=True,
    )
    db_session.add(unverified_org)
    db_session.commit()

    # Employee attempts to join unverified org
    r = client.post(
        "/api/v1/auth/register",
        json={
            "registration_type": "JOIN_ORG",
            "email": f"worker@{domain}",
            "full_name": "Worker",
            "password": "Password123!",
            "company_domain": domain,
        },
    )
    assert r.status_code == 400
    assert "not been verified" in r.json()["detail"]


def test_7_cannot_claim_already_registered_domain(client: TestClient):
    """Scenario 7: An employee/user cannot claim a domain already registered to another organization."""
    domain = f"claimed-{uuid.uuid4().hex[:6]}.com"

    # First registration claims domain
    r1 = client.post(
        "/api/v1/auth/register",
        json={
            "registration_type": "CREATE_ORG",
            "email": f"founder@{domain}",
            "full_name": "Founder",
            "password": "Password123!",
            "organization_name": "Claimed Org",
            "company_domain": domain,
        },
    )
    assert r1.status_code == 201

    # Second user attempts to CREATE an organization with the same domain
    r2 = client.post(
        "/api/v1/auth/register",
        json={
            "registration_type": "CREATE_ORG",
            "email": f"second@{domain}",
            "full_name": "Impostor",
            "password": "Password123!",
            "organization_name": "Rival Org",
            "company_domain": domain,
        },
    )
    assert r2.status_code == 409
    assert "already registered" in r2.json()["detail"]


def test_8_individual_user_registers_without_organization(client: TestClient, db_session: Session):
    """Scenario 8: An individual user can register without an organization (personal workspace)."""
    email = f"solo_freelancer_{uuid.uuid4().hex[:6]}@gmail.com"

    r = client.post(
        "/api/v1/auth/register",
        json={
            "registration_type": "INDIVIDUAL",
            "email": email,
            "full_name": "Freelancer Jane",
            "password": "Password123!",
        },
    )
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["account_type"] == "INDIVIDUAL"
    assert data["role"] == OrgRole.INDIVIDUAL.value

    # Verify DB
    usr = db_session.execute(select(User).where(User.email == email)).scalar_one()
    mem = db_session.execute(select(OrganizationMembership).where(OrganizationMembership.user_id == usr.id)).scalar_one()
    org = db_session.execute(select(Organization).where(Organization.id == mem.organization_id)).scalar_one()

    assert org.account_type == "INDIVIDUAL"
    assert org.company_domain is None


def test_9_existing_accounts_can_still_login_normally(client: TestClient, db_session: Session):
    """Scenario 9: Existing accounts can log in with email and password without re-entering domain."""
    email = f"legacy_user_{uuid.uuid4().hex[:6]}@example.com"
    pw = "LegacyPass123!"

    # Create legacy user and org
    org = Organization(name="Legacy Org", slug=f"legacy-{uuid.uuid4().hex[:4]}", is_active=True)
    db_session.add(org)
    db_session.flush()

    usr = User(email=email, full_name="Legacy User", hashed_password=hash_password(pw), is_active=True)
    db_session.add(usr)
    db_session.flush()

    mem = OrganizationMembership(organization_id=org.id, user_id=usr.id, role=OrgRole.ORG_ADMIN.value, is_active=True)
    db_session.add(mem)
    db_session.commit()

    # Log in normally using only email and password
    r = client.post(
        "/api/v1/auth/login",
        json={
            "email": email,
            "password": pw,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "access_token" in data
    assert data["role"] == OrgRole.ORG_ADMIN.value


def test_10_members_cannot_access_unauthorized_data(client: TestClient, db_session: Session):
    """Scenario 10: Members cannot access another organization's meetings or data."""
    # Org A
    org_a = Organization(name="Org A", slug=f"orga-{uuid.uuid4().hex[:4]}", is_active=True)
    usr_a = User(email=f"usera_{uuid.uuid4().hex[:4]}@a.com", full_name="User A", hashed_password=hash_password("Pass123!"), is_active=True)
    db_session.add_all([org_a, usr_a])
    db_session.flush()
    db_session.add(OrganizationMembership(organization_id=org_a.id, user_id=usr_a.id, role=OrgRole.MEMBER.value, is_active=True))

    meeting_a = Meeting(
        organization_id=org_a.id,
        title="Top Secret Meeting Org A",
        input_source_type=InputSourceType.AUDIO.value,
        status=MeetingStatus.COMPLETED.value,
    )
    db_session.add(meeting_a)

    # Org B
    org_b = Organization(name="Org B", slug=f"orgb-{uuid.uuid4().hex[:4]}", is_active=True)
    usr_b = User(email=f"userb_{uuid.uuid4().hex[:4]}@b.com", full_name="User B", hashed_password=hash_password("Pass123!"), is_active=True)
    db_session.add_all([org_b, usr_b])
    db_session.flush()
    db_session.add(OrganizationMembership(organization_id=org_b.id, user_id=usr_b.id, role=OrgRole.MEMBER.value, is_active=True))
    db_session.commit()

    token_b = create_access_token(subject=usr_b.id, token_version=usr_b.token_version)

    # User B attempts to access Meeting from Org A
    r = client.get(
        f"/api/v1/media/meetings/{meeting_a.id}",
        headers={"Authorization": f"Bearer {token_b}", "X-Organization-Id": org_b.id},
    )
    assert r.status_code == 404


def test_11_members_cannot_promote_themselves_to_admin(client: TestClient, db_session: Session):
    """Scenario 11: Regular members cannot elevate their own role to ORG_ADMIN."""
    org = Organization(name="Company X", slug=f"compx-{uuid.uuid4().hex[:4]}", is_active=True)
    member = User(email=f"regular_{uuid.uuid4().hex[:4]}@x.com", full_name="Regular Employee", hashed_password=hash_password("Pass123!"), is_active=True)
    db_session.add_all([org, member])
    db_session.flush()

    mem = OrganizationMembership(organization_id=org.id, user_id=member.id, role=OrgRole.MEMBER.value, is_active=True)
    db_session.add(mem)
    db_session.commit()

    token_member = create_access_token(subject=member.id, token_version=member.token_version)

    # Member tries to call role update endpoint to promote self to ORG_ADMIN
    r = client.patch(
        f"/api/v1/auth/organization/members/{mem.id}/role",
        headers={"Authorization": f"Bearer {token_member}", "X-Organization-Id": org.id},
        json={"role": "ORG_ADMIN"},
    )
    assert r.status_code == 403
    assert "administrator privileges required" in r.json()["detail"].lower()


def test_12_personal_email_provider_cannot_join_corporate_domain(client: TestClient):
    """Scenario 12: Users with personal email providers (Gmail, etc.) cannot join a corporate domain."""
    r = client.post(
        "/api/v1/auth/register",
        json={
            "registration_type": "JOIN_ORG",
            "email": "employee@gmail.com",
            "full_name": "Gmail User",
            "password": "Password123!",
            "company_domain": "apple.com",
        },
    )
    assert r.status_code == 400
    assert "personal email providers" in r.json()["detail"].lower()


def test_13_individual_owner_can_view_own_original_and_derived_transcripts(client: TestClient, db_session: Session):
    """Individual workspace owner can view their own ORIGINAL transcripts and derive new ones."""
    org = Organization(name="Ritik Workspace", slug=f"ind-{uuid.uuid4().hex[:6]}", account_type="INDIVIDUAL", is_active=True)
    user = User(email=f"owner_{uuid.uuid4().hex[:4]}@gmail.com", full_name="Individual Owner", hashed_password=hash_password("Pass123!"), is_active=True)
    db_session.add_all([org, user])
    db_session.flush()

    mem = OrganizationMembership(organization_id=org.id, user_id=user.id, role=OrgRole.INDIVIDUAL.value, is_active=True)
    db_session.add(mem)

    meeting = Meeting(
        organization_id=org.id,
        created_by_user_id=user.id,
        title="Individual Standup",
        input_source_type=InputSourceType.AUDIO.value,
        status=MeetingStatus.COMPLETED.value,
    )
    db_session.add(meeting)
    db_session.flush()

    t_orig = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
        language_detected="en",
        is_primary=True,
    )
    db_session.add(t_orig)
    db_session.flush()

    seg = TranscriptSegment(
        transcript_id=t_orig.id,
        sequence_order=0,
        start_time=0.0,
        end_time=5.0,
        text="Hello world test for individual owner",
        language_code="en",
    )
    db_session.add(seg)
    db_session.commit()

    token = create_access_token(subject=user.id, token_version=user.token_version)
    headers = {"Authorization": f"Bearer {token}", "X-Organization-Id": org.id}

    # 1. Individual owner can view ORIGINAL transcript
    resp = client.get(f"/api/v1/transcription/{meeting.id}", headers=headers)
    assert resp.status_code == 200, resp.text
    transcripts = resp.json()
    assert len(transcripts) == 1
    assert transcripts[0]["transcript_type"] == "ORIGINAL"
    assert len(transcripts[0]["segments"]) == 1

    # 2. Individual owner can derive unredacted CLEANED transcript
    resp_clean = client.post(
        f"/api/v1/transcription/{meeting.id}/process",
        headers=headers,
        json={"clean": True, "redact": False},
    )
    assert resp_clean.status_code == 201, resp_clean.text
    assert resp_clean.json()["transcript_type"] == "CLEANED"

    # 3. Individual owner can derive privacy-safe CLEANED_REDACTED transcript
    resp_both = client.post(
        f"/api/v1/transcription/{meeting.id}/process",
        headers=headers,
        json={"clean": True, "redact": True},
    )
    assert resp_both.status_code == 201, resp_both.text
    assert resp_both.json()["transcript_type"] == "CLEANED_REDACTED"


def test_14_individual_owner_blocked_from_cross_tenant_transcripts(client: TestClient, db_session: Session):
    """Individual workspace owner cannot access another organization's transcripts."""
    # Org A (Individual)
    org_a = Organization(name="Ind A", slug=f"inda-{uuid.uuid4().hex[:6]}", account_type="INDIVIDUAL", is_active=True)
    user_a = User(email=f"ind_a_{uuid.uuid4().hex[:4]}@gmail.com", full_name="User A", hashed_password=hash_password("Pass123!"), is_active=True)
    db_session.add_all([org_a, user_a])
    db_session.flush()
    db_session.add(OrganizationMembership(organization_id=org_a.id, user_id=user_a.id, role=OrgRole.INDIVIDUAL.value, is_active=True))

    meeting_a = Meeting(
        organization_id=org_a.id,
        created_by_user_id=user_a.id,
        title="Ind A Meeting",
        input_source_type=InputSourceType.AUDIO.value,
        status=MeetingStatus.COMPLETED.value,
    )
    db_session.add(meeting_a)

    # Org B (Individual)
    org_b = Organization(name="Ind B", slug=f"indb-{uuid.uuid4().hex[:6]}", account_type="INDIVIDUAL", is_active=True)
    user_b = User(email=f"ind_b_{uuid.uuid4().hex[:4]}@gmail.com", full_name="User B", hashed_password=hash_password("Pass123!"), is_active=True)
    db_session.add_all([org_b, user_b])
    db_session.flush()
    db_session.add(OrganizationMembership(organization_id=org_b.id, user_id=user_b.id, role=OrgRole.INDIVIDUAL.value, is_active=True))
    db_session.commit()

    token_b = create_access_token(subject=user_b.id, token_version=user_b.token_version)
    headers_b = {"Authorization": f"Bearer {token_b}", "X-Organization-Id": org_b.id}

    # User B cannot access Meeting A's transcripts
    resp = client.get(f"/api/v1/transcription/{meeting_a.id}", headers=headers_b)
    assert resp.status_code == 404


def test_15_org_member_cannot_view_or_generate_unredacted_transcripts(client: TestClient, db_session: Session):
    """MEMBER can only view/generate privacy-safe transcripts; cannot access ORIGINAL or CLEANED."""
    org = Organization(name="Corporate Beta", slug=f"beta-{uuid.uuid4().hex[:6]}", company_domain="beta.com", is_active=True)
    admin = User(email=f"admin_{uuid.uuid4().hex[:4]}@beta.com", full_name="Beta Admin", hashed_password=hash_password("Pass123!"), is_active=True)
    member = User(email=f"member_{uuid.uuid4().hex[:4]}@beta.com", full_name="Beta Member", hashed_password=hash_password("Pass123!"), is_active=True)
    db_session.add_all([org, admin, member])
    db_session.flush()

    db_session.add(OrganizationMembership(organization_id=org.id, user_id=admin.id, role=OrgRole.ORG_ADMIN.value, is_active=True))
    db_session.add(OrganizationMembership(organization_id=org.id, user_id=member.id, role=OrgRole.MEMBER.value, is_active=True))

    meeting = Meeting(
        organization_id=org.id,
        created_by_user_id=admin.id,
        title="Corporate Sync",
        input_source_type=InputSourceType.AUDIO.value,
        status=MeetingStatus.COMPLETED.value,
    )
    db_session.add(meeting)
    db_session.flush()

    # Create ORIGINAL transcript
    t_orig = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
        language_detected="en",
        is_primary=True,
    )
    # Create REDACTED transcript
    t_red = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.REDACTED.value,
        version=1,
        language_detected="en",
        is_primary=False,
    )
    db_session.add_all([t_orig, t_red])
    db_session.flush()

    seg1 = TranscriptSegment(transcript_id=t_orig.id, sequence_order=0, start_time=0.0, end_time=2.0, text="Secret unredacted text", language_code="en")
    seg2 = TranscriptSegment(transcript_id=t_red.id, sequence_order=0, start_time=0.0, end_time=2.0, text="Secret [REDACTED] text", language_code="en")
    db_session.add_all([seg1, seg2])
    db_session.commit()

    token_member = create_access_token(subject=member.id, token_version=member.token_version)
    headers_member = {"Authorization": f"Bearer {token_member}", "X-Organization-Id": org.id}

    # 1. MEMBER querying /transcription/{meeting_id} only sees REDACTED, NOT ORIGINAL
    r = client.get(f"/api/v1/transcription/{meeting.id}", headers=headers_member)
    assert r.status_code == 200
    returned = r.json()
    assert len(returned) == 1
    assert returned[0]["transcript_type"] == "REDACTED"

    # 2. MEMBER explicitly requesting ORIGINAL gets 403 Forbidden
    r_orig = client.get(f"/api/v1/transcription/{meeting.id}?transcript_type=ORIGINAL", headers=headers_member)
    assert r_orig.status_code == 403

    # 3. MEMBER requesting unredacted CLEANED derivation gets 403 Forbidden
    r_clean = client.post(
        f"/api/v1/transcription/{meeting.id}/process",
        headers=headers_member,
        json={"clean": True, "redact": False},
    )
    assert r_clean.status_code == 403

    # 4. MEMBER requesting privacy-safe CLEANED_REDACTED derivation is allowed (201)
    r_cr = client.post(
        f"/api/v1/transcription/{meeting.id}/process",
        headers=headers_member,
        json={"clean": True, "redact": True},
    )
    assert r_cr.status_code == 201
    assert r_cr.json()["transcript_type"] == "CLEANED_REDACTED"


def test_16_individual_owner_cannot_manage_org_membership(client: TestClient, db_session: Session):
    """Individual workspace owners cannot manage organization membership."""
    org = Organization(name="Solo Workspace", slug=f"solo-{uuid.uuid4().hex[:6]}", account_type="INDIVIDUAL", is_active=True)
    user = User(email=f"solo_{uuid.uuid4().hex[:4]}@gmail.com", full_name="Solo Dev", hashed_password=hash_password("Pass123!"), is_active=True)
    db_session.add_all([org, user])
    db_session.flush()

    mem = OrganizationMembership(organization_id=org.id, user_id=user.id, role=OrgRole.INDIVIDUAL.value, is_active=True)
    db_session.add(mem)
    db_session.commit()

    token = create_access_token(subject=user.id, token_version=user.token_version)
    headers = {"Authorization": f"Bearer {token}", "X-Organization-Id": org.id}

    # Attempting to change membership role returns 403 Forbidden
    resp = client.patch(
        f"/api/v1/auth/organization/members/{mem.id}/role",
        headers=headers,
        json={"role": "ORG_ADMIN"},
    )
    assert resp.status_code == 403
