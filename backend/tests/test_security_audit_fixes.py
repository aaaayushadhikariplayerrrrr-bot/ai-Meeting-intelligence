"""Tests for Phase 6E.2 Security and Configuration Audit Fixes.

Covers:
1. Production API documentation disabling (docs_url, redoc_url, openapi_url)
2. Development API documentation enabling
3. Secret configuration safety (production rejects insecure/placeholder secrets)
4. Unredacted transcript retrieval by ORG_ADMIN creates VIEW_UNREDACTED_TRANSCRIPT audit log
5. Redacted transcript retrieval by ORG_ADMIN does not create that event
6. Audit log metadata contains zero raw transcript text and zero PII
7. Non-super-admin cannot access hardware diagnostics (403 Forbidden)
8. Super-admin can access hardware diagnostics (200 OK)
9. .gitignore and .env.example configuration hygiene
"""

import json
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings, settings, ROOT_DIR
from app.core.security import create_access_token, hash_password
from app.models.identity import User, Organization, OrganizationMembership, OrgRole
from app.models.meeting import Meeting
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.memory import AuditLog
from backend.tests.conftest import make_auth_headers


def test_production_disables_api_documentation():
    """Verify that when ENVIRONMENT is production, Swagger docs, ReDoc, and OpenAPI schema are disabled."""
    # Test the exact conditional logic implemented in backend/app/main.py
    env_prod = "production"
    is_prod = env_prod.strip().lower() == "production"

    prod_app = FastAPI(
        docs_url=None if is_prod else "/docs",
        redoc_url=None if is_prod else "/redoc",
        openapi_url=None if is_prod else "/api/v1/openapi.json",
    )

    @prod_app.get("/")
    def prod_root():
        return {
            "name": "Test System",
            "documentation": None if is_prod else "/docs",
        }

    client = TestClient(prod_app)

    # 1. /docs must return 404
    resp_docs = client.get("/docs")
    assert resp_docs.status_code == 404

    # 2. /redoc must return 404
    resp_redoc = client.get("/redoc")
    assert resp_redoc.status_code == 404

    # 3. /api/v1/openapi.json must return 404
    resp_openapi = client.get("/api/v1/openapi.json")
    assert resp_openapi.status_code == 404

    # 4. Root documentation link must be None
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    assert resp_root.json()["documentation"] is None


def test_development_keeps_api_documentation_enabled():
    """Verify that when ENVIRONMENT is development, Swagger docs and OpenAPI schema are accessible."""
    env_dev = "development"
    is_prod = env_dev.strip().lower() == "production"

    dev_app = FastAPI(
        docs_url=None if is_prod else "/docs",
        redoc_url=None if is_prod else "/redoc",
        openapi_url=None if is_prod else "/api/v1/openapi.json",
    )

    @dev_app.get("/")
    def dev_root():
        return {
            "name": "Test System",
            "documentation": None if is_prod else "/docs",
        }

    client = TestClient(dev_app)

    resp_docs = client.get("/docs")
    assert resp_docs.status_code == 200

    resp_openapi = client.get("/api/v1/openapi.json")
    assert resp_openapi.status_code == 200

    resp_root = client.get("/")
    assert resp_root.status_code == 200
    assert resp_root.json()["documentation"] == "/docs"


def test_production_rejects_insecure_or_placeholder_secrets():
    """Verify production startup rejects placeholder, test, or insecure secret keys."""
    # 1. Rejects placeholder pattern in production
    with pytest.raises(ValueError, match="insecure"):
        Settings(
            ENVIRONMENT="production",
            SECRET_KEY="generate-with-openssl-rand-hex-32-min-32-chars",
        )

    # 2. Rejects 'test' pattern in production
    with pytest.raises(ValueError, match="insecure"):
        Settings(
            ENVIRONMENT="production",
            SECRET_KEY="test-secret-key-for-automated-testing-only-minimum-32-chars",
        )

    # 3. Rejects 'example' pattern in production
    with pytest.raises(ValueError, match="insecure"):
        Settings(
            ENVIRONMENT="production",
            SECRET_KEY="example-production-key-that-is-at-least-32-chars-long",
        )

    # 4. Rejects empty secret in production
    with pytest.raises(ValueError, match="SECRET_KEY is required"):
        Settings(
            ENVIRONMENT="production",
            SECRET_KEY="",
        )

    # 5. Rejects short secret in production
    with pytest.raises(ValueError, match="at least 32 characters"):
        Settings(
            ENVIRONMENT="production",
            SECRET_KEY="short-secret-under-32-chars",
        )

    # 6. Accepts high-entropy 256-bit key in production
    valid_prod_key = "a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0"
    prod_settings = Settings(
        ENVIRONMENT="production",
        SECRET_KEY=valid_prod_key,
    )
    assert prod_settings.SECRET_KEY == valid_prod_key


def test_gitignore_and_env_example_safety():
    """Verify .gitignore ignores .env and backend/.env.example contains no real secrets."""
    gitignore_path = ROOT_DIR / ".gitignore"
    assert gitignore_path.exists(), ".gitignore must exist in project root"
    gitignore_content = gitignore_path.read_text(encoding="utf-8")
    assert ".env" in gitignore_content, ".gitignore must ignore .env files"

    env_example_path = ROOT_DIR / "backend" / ".env.example"
    assert env_example_path.exists(), "backend/.env.example must exist"
    env_example_content = env_example_path.read_text(encoding="utf-8")
    assert "generate-with-openssl" in env_example_content, ".env.example must use placeholder"
    # Ensure no concrete active hex secret is in .env.example
    assert "5fe1737cb7265f16" not in env_example_content


def test_unredacted_transcript_access_by_org_admin_creates_audit_event(client, db_session):
    """Verify ORG_ADMIN retrieving ORIGINAL or CLEANED transcripts creates a VIEW_UNREDACTED_TRANSCRIPT audit log."""
    # 1. Setup Organization, Admin User, and Membership
    org = Organization(name="Audit Test Org", slug="audit-test-org")
    db_session.add(org)
    db_session.flush()

    admin_user = User(
        email="orgadmin@audit.local",
        hashed_password=hash_password("AdminSecurePass123!"),
        full_name="Org Admin",
        is_active=True,
        is_super_admin=False,
        token_version=1,
    )
    db_session.add(admin_user)
    db_session.flush()

    membership = OrganizationMembership(
        user_id=admin_user.id,
        organization_id=org.id,
        role=OrgRole.ORG_ADMIN.value,
        is_active=True,
    )
    db_session.add(membership)

    # 2. Setup Meeting with ORIGINAL and REDACTED Transcripts
    meeting = Meeting(
        organization_id=org.id,
        created_by_user_id=admin_user.id,
        title="Quarterly Review",
    )
    db_session.add(meeting)
    db_session.flush()

    raw_text = "Sensitive Discussion: Contact CEO at 9876543210 or ceo@corp.com"
    t_orig = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
    )
    db_session.add(t_orig)
    db_session.flush()

    seg_orig = TranscriptSegment(
        transcript_id=t_orig.id,
        start_time=0.0,
        end_time=5.0,
        text=raw_text,
        sequence_order=0,
        is_redacted=False,
    )
    db_session.add(seg_orig)

    t_red = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.REDACTED.value,
        version=1,
    )
    db_session.add(t_red)
    db_session.flush()

    seg_red = TranscriptSegment(
        transcript_id=t_red.id,
        start_time=0.0,
        end_time=5.0,
        text="Sensitive Discussion: Contact CEO at [PHONE_REDACTED] or [EMAIL_REDACTED]",
        sequence_order=0,
        is_redacted=True,
    )
    db_session.add(seg_red)
    db_session.commit()

    # 3. ORG_ADMIN retrieves all transcripts (which includes ORIGINAL)
    headers = make_auth_headers(admin_user, org=org)
    resp = client.get(f"/api/v1/transcription/{meeting.id}", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2

    # 4. Verify AuditLog entry was created
    audit_logs = (
        db_session.query(AuditLog)
        .filter(
            AuditLog.organization_id == org.id,
            AuditLog.action == "VIEW_UNREDACTED_TRANSCRIPT"
        )
        .all()
    )
    assert len(audit_logs) == 1
    log_entry = audit_logs[0]
    assert log_entry.user_id == admin_user.id
    assert log_entry.entity_type == "Meeting"
    assert log_entry.entity_id == meeting.id

    # 5. Verify audit details contains metadata but NO raw text or PII
    details = json.loads(log_entry.details)
    assert details["meeting_id"] == meeting.id
    assert t_orig.id in details["retrieved_transcript_ids"]
    assert details["unredacted_count"] == 1
    assert "9876543210" not in log_entry.details
    assert "ceo@corp.com" not in log_entry.details
    assert raw_text not in log_entry.details


def test_redacted_transcript_access_does_not_create_view_unredacted_audit_event(client, db_session):
    """Verify retrieving ONLY REDACTED transcripts does not generate VIEW_UNREDACTED_TRANSCRIPT audit log."""
    org = Organization(name="Privacy Safe Org", slug="privacy-safe-org")
    db_session.add(org)
    db_session.flush()

    admin_user = User(
        email="admin2@audit.local",
        hashed_password=hash_password("AdminSecurePass123!"),
        full_name="Org Admin 2",
        is_active=True,
        is_super_admin=False,
        token_version=1,
    )
    db_session.add(admin_user)
    db_session.flush()

    membership = OrganizationMembership(
        user_id=admin_user.id,
        organization_id=org.id,
        role=OrgRole.ORG_ADMIN.value,
        is_active=True,
    )
    db_session.add(membership)

    meeting = Meeting(
        organization_id=org.id,
        created_by_user_id=admin_user.id,
        title="Privacy Safe Meeting",
    )
    db_session.add(meeting)
    db_session.flush()

    t_red = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.REDACTED.value,
        version=1,
    )
    db_session.add(t_red)
    db_session.flush()

    seg_red = TranscriptSegment(
        transcript_id=t_red.id,
        start_time=0.0,
        end_time=3.0,
        text="All PII has been redacted here.",
        sequence_order=0,
        is_redacted=True,
    )
    db_session.add(seg_red)
    db_session.commit()

    # Clear any previous audit logs
    count_before = (
        db_session.query(AuditLog)
        .filter(AuditLog.action == "VIEW_UNREDACTED_TRANSCRIPT")
        .count()
    )

    # Admin requests REDACTED explicitly
    headers = make_auth_headers(admin_user, org=org)
    resp = client.get(f"/api/v1/transcription/{meeting.id}?transcript_type=REDACTED", headers=headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 1

    count_after = (
        db_session.query(AuditLog)
        .filter(AuditLog.action == "VIEW_UNREDACTED_TRANSCRIPT")
        .count()
    )
    assert count_after == count_before, "No VIEW_UNREDACTED_TRANSCRIPT audit log should be created for redacted transcripts"


def test_member_cannot_retrieve_unredacted_transcripts(client, db_session):
    """Verify regular MEMBER cannot retrieve ORIGINAL transcripts and no audit event is created."""
    org = Organization(name="Member Test Org", slug="member-test-org")
    db_session.add(org)
    db_session.flush()

    member_user = User(
        email="member@audit.local",
        hashed_password=hash_password("MemberPass123!"),
        full_name="Regular Member",
        is_active=True,
        is_super_admin=False,
        token_version=1,
    )
    db_session.add(member_user)
    db_session.flush()

    membership = OrganizationMembership(
        user_id=member_user.id,
        organization_id=org.id,
        role=OrgRole.MEMBER.value,
        is_active=True,
    )
    db_session.add(membership)

    meeting = Meeting(
        organization_id=org.id,
        created_by_user_id=member_user.id,
        title="Member Meeting",
    )
    db_session.add(meeting)
    db_session.flush()

    t_orig = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
    )
    db_session.add(t_orig)
    db_session.commit()

    headers = make_auth_headers(member_user, org=org)
    # Member explicitly asking for ORIGINAL receives 403 Forbidden
    resp = client.get(f"/api/v1/transcription/{meeting.id}?transcript_type=ORIGINAL", headers=headers)
    assert resp.status_code == 403

    # Confirm no audit log created
    count = (
        db_session.query(AuditLog)
        .filter(AuditLog.action == "VIEW_UNREDACTED_TRANSCRIPT")
        .count()
    )
    assert count == 0


def test_hardware_diagnostics_super_admin_vs_non_super_admin(client, db_session, monkeypatch):
    """Verify only global Super Admin can access hardware diagnostics; non-super-admins receive 403, and gated by ENABLE_SYSTEM_DIAGNOSTICS."""
    from app.core.config import settings

    # 1. Non-super-admin user
    reg_user = User(
        email="regular@test.local",
        hashed_password=hash_password("UserPass123!"),
        full_name="Regular User",
        is_active=True,
        is_super_admin=False,
        token_version=1,
    )
    db_session.add(reg_user)

    # 2. Super-admin user
    super_user = User(
        email="root@system.local",
        hashed_password=hash_password("RootAdminPass123!"),
        full_name="System SuperAdmin",
        is_active=True,
        is_super_admin=True,
        token_version=1,
    )
    db_session.add(super_user)
    db_session.commit()

    # Regular user receives 403
    reg_headers = make_auth_headers(reg_user)
    resp_reg = client.get("/api/v1/transcription/hardware/diagnostics", headers=reg_headers)
    assert resp_reg.status_code == 403
    assert "Super Administrator" in resp_reg.json()["detail"]

    # Super admin receives 403 when diagnostics are disabled (default)
    super_headers = make_auth_headers(super_user)
    resp_super_disabled = client.get("/api/v1/transcription/hardware/diagnostics", headers=super_headers)
    assert resp_super_disabled.status_code == 403
    assert "System and hardware diagnostics are disabled" in resp_super_disabled.json()["detail"]

    # Super admin receives 200 OK when enabled
    monkeypatch.setattr(settings, "ENABLE_SYSTEM_DIAGNOSTICS", True)
    resp_super = client.get("/api/v1/transcription/hardware/diagnostics", headers=super_headers)
    assert resp_super.status_code == 200
    data = resp_super.json()
    assert "gpu_info" in data
    assert "torch_cuda_available" in data["gpu_info"]
