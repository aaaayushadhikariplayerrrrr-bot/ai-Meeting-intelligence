"""Comprehensive Test Suite for Phase 6 Step 5B:
Authentication Enforcement, Tenant Isolation, and Role-Based Authorization.

Covers:
- Authentication Enforcement (Missing token, invalid token, expired token, inactive user)
- Tenant Isolation (Valid access, spoofed org header, cross-tenant isolation, inactive membership, inactive org, multi-org selector)
- Role Authorization (MEMBER vs ORG_ADMIN for ORIGINAL/CLEANED transcripts and raw media deletion)
- Global Super Admin (No membership, MEMBER membership, ORG_ADMIN membership)
- Anti-enumeration (Foreign and non-existent IDs return 404 consistently)
- Hardware diagnostics platform protection
"""

from datetime import datetime, timezone
from pathlib import Path
import time
import jwt
import pytest
from app.core.config import settings
from app.core.security import ACCESS_TOKEN_TYPE, create_access_token, hash_password
from app.models.meeting import Meeting, MediaAsset, ProcessingJob, JobType, JobStatus
from app.models.memory import AuditLog
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.identity import Organization, OrganizationMembership, OrgRole
from app.services.media_storage import resolve_safe_storage_path
from backend.tests.conftest import make_auth_headers


def _seed_meeting_with_all_transcripts(db_session, org_id, user_id):
    """Helper to create a meeting with media asset and all 4 transcript versions."""
    meeting = Meeting(
        organization_id=org_id,
        created_by_user_id=user_id,
        title="Strategy Session",
    )
    db_session.add(meeting)
    db_session.flush()

    asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="meeting.wav",
        storage_path=f"storage/meetings/{meeting.id}/meeting.wav",
        media_type="AUDIO",
        mime_type="audio/wav",
        is_raw_deleted=False,
    )
    db_session.add(asset)

    t_orig = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
    )
    t_clean = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED.value,
        version=1,
    )
    t_red = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.REDACTED.value,
        version=1,
    )
    t_cr = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=1,
    )
    job = ProcessingJob(
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        status=JobStatus.COMPLETED.value,
        progress=100.0,
    )
    db_session.add_all([t_orig, t_clean, t_red, t_cr, job])
    db_session.commit()
    db_session.refresh(meeting)
    return meeting, asset, job


def _seed_meeting_with_source_transcript_and_pii(db_session, org_id, user_id):
    """Helper to seed a meeting with an ORIGINAL transcript containing sensitive PII."""
    meeting = Meeting(
        organization_id=org_id,
        created_by_user_id=user_id,
        title="Confidential Product Strategy",
    )
    db_session.add(meeting)
    db_session.flush()

    asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="recording.wav",
        storage_path=f"storage/meetings/{meeting.id}/recording.wav",
        media_type="AUDIO",
        mime_type="audio/wav",
        is_raw_deleted=False,
    )
    db_session.add(asset)

    t_orig = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
    )
    db_session.add(t_orig)
    db_session.flush()

    seg1 = TranscriptSegment(
        transcript_id=t_orig.id,
        sequence_order=0,
        start_time=0.0,
        end_time=4.0,
        text="um, uh, please contact Rahul at rahul.private@example.com for confidential access.",
        language_code="en",
        confidence=0.98,
    )
    seg2 = TranscriptSegment(
        transcript_id=t_orig.id,
        sequence_order=1,
        start_time=4.0,
        end_time=8.0,
        text="His direct phone is 9876543210 and the secret pass is topsecret99.",
        language_code="en",
        confidence=0.95,
    )
    db_session.add_all([seg1, seg2])
    db_session.commit()
    db_session.refresh(meeting)
    return meeting, t_orig


# --- 1. Authentication Enforcement ---

def test_expired_jwt_rejected_on_tenant_endpoint(client, test_org_a, org_admin_user):
    """Verify calling a tenant endpoint with an actually expired JWT returns HTTP 401 with WWW-Authenticate header."""
    now = int(time.time())
    expired_payload = {
        "sub": org_admin_user.id,
        "iat": now - 3600,
        "exp": now - 60,  # Token expired 60 seconds ago
        "type": ACCESS_TOKEN_TYPE,
        "token_version": 1,
    }
    expired_token = jwt.encode(
        expired_payload,
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    headers = {
        "Authorization": f"Bearer {expired_token}",
        "X-Organization-Id": test_org_a.id,
    }

    resp = client.get("/api/v1/media/meetings", headers=headers)
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"
    assert "credentials" in resp.json()["detail"].lower()


PROTECTED_ENDPOINTS = [
    (
        "POST",
        "/api/v1/media/upload",
        {"files": {"file": ("audio.wav", b"RIFF....WAVEfmt ...", "audio/wav")}},
        "media upload",
    ),
    (
        "POST",
        "/api/v1/media/youtube",
        {"json": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "confirmed_authorization": True}},
        "youtube ingestion",
    ),
    (
        "GET",
        "/api/v1/media/meetings",
        {},
        "meeting history",
    ),
    (
        "GET",
        "/api/v1/media/meetings/test-meeting-uuid",
        {},
        "meeting details",
    ),
    (
        "GET",
        "/api/v1/media/jobs/test-job-uuid",
        {},
        "media job status",
    ),
    (
        "DELETE",
        "/api/v1/media/assets/test-asset-uuid",
        {},
        "media deletion",
    ),
    (
        "POST",
        "/api/v1/transcription/start/test-meeting-uuid",
        {"json": {"transcription_mode": "standard"}},
        "transcription start",
    ),
    (
        "GET",
        "/api/v1/transcription/test-meeting-uuid/default",
        {},
        "transcript default retrieval",
    ),
    (
        "GET",
        "/api/v1/transcription/test-meeting-uuid",
        {},
        "transcript retrieval",
    ),
    (
        "POST",
        "/api/v1/transcription/test-meeting-uuid/clean",
        {"json": {}},
        "transcript cleaning",
    ),
    (
        "POST",
        "/api/v1/transcription/test-meeting-uuid/redact",
        {"json": {}},
        "transcript redaction",
    ),
    (
        "POST",
        "/api/v1/transcription/test-meeting-uuid/process",
        {"json": {"clean": True, "redact": True}},
        "transcript processing",
    ),
    (
        "GET",
        "/api/v1/transcription/jobs/test-job-uuid",
        {},
        "transcription job status",
    ),
    (
        "GET",
        "/api/v1/transcription/hardware/diagnostics",
        {},
        "hardware diagnostics",
    ),
]


@pytest.mark.parametrize("method,endpoint,kwargs,description", PROTECTED_ENDPOINTS)
def test_missing_authentication_rejected_across_all_14_protected_endpoints(
    client, test_org_a, method, endpoint, kwargs, description
):
    """Verify all 14 protected endpoints reject unauthenticated requests with HTTP 401 and WWW-Authenticate."""
    call_kwargs = dict(kwargs)
    # Supply tenant organization header to prove rejection occurs on missing Bearer auth
    call_kwargs["headers"] = {"X-Organization-Id": test_org_a.id}

    resp = client.request(method, endpoint, **call_kwargs)
    assert resp.status_code == 401, f"{description} ({method} {endpoint}) returned {resp.status_code}, expected 401"
    assert resp.headers.get("www-authenticate") == "Bearer"
    assert "not authenticated" in resp.json().get("detail", "").lower()


def test_missing_token_rejected_on_protected_endpoints(client, test_org_a):
    """Verify calling protected endpoints without an auth token returns 401 with WWW-Authenticate header."""
    # Media endpoints
    r1 = client.get("/api/v1/media/meetings", headers={"X-Organization-Id": test_org_a.id})
    assert r1.status_code == 401
    assert r1.headers.get("www-authenticate") == "Bearer"

    # Transcription endpoints
    r2 = client.get("/api/v1/transcription/some-meeting-id/default", headers={"X-Organization-Id": test_org_a.id})
    assert r2.status_code == 401
    assert r2.headers.get("www-authenticate") == "Bearer"

    # Hardware diagnostics
    r3 = client.get("/api/v1/transcription/hardware/diagnostics")
    assert r3.status_code == 401
    assert r3.headers.get("www-authenticate") == "Bearer"


def test_invalid_token_rejected(client, test_org_a):
    """Verify calling with malformed token returns 401."""
    headers = {
        "Authorization": "Bearer not-a-valid-token-string",
        "X-Organization-Id": test_org_a.id,
    }
    r = client.get("/api/v1/media/meetings", headers=headers)
    assert r.status_code == 401
    assert r.headers.get("www-authenticate") == "Bearer"


def test_inactive_user_token_rejected(client, db_session, org_admin_user, test_org_a):
    """Verify calling with token of a deactivated user returns 401."""
    headers = make_auth_headers(org_admin_user, test_org_a)

    # Deactivate the user
    org_admin_user.is_active = False
    db_session.commit()

    r = client.get("/api/v1/media/meetings", headers=headers)
    assert r.status_code == 401
    assert "inactive" in r.json()["detail"].lower()


# --- 2. Tenant Isolation & Header Selection ---

def test_valid_access_to_own_organization(client, db_session, org_admin_user, test_org_a, org_admin_headers):
    """Verify user can access their own organization's meetings."""
    meeting, _, _ = _seed_meeting_with_all_transcripts(db_session, test_org_a.id, org_admin_user.id)

    resp = client.get("/api/v1/media/meetings", headers=org_admin_headers)
    assert resp.status_code == 200
    meetings = resp.json()
    assert len(meetings) == 1
    assert meetings[0]["id"] == meeting.id


def test_spoofed_organization_header_rejected(client, db_session, org_admin_user, org_admin_headers):
    """Verify specifying an organization where user has no membership returns 403."""
    headers = dict(org_admin_headers)
    headers["X-Organization-Id"] = "e9d9bac0-0000-0000-0000-000000000000"  # Arbitrary org UUID

    resp = client.get("/api/v1/media/meetings", headers=headers)
    assert resp.status_code == 403
    assert "active membership" in resp.json()["detail"].lower()


def test_cross_tenant_meeting_access_blocked(client, db_session, org_admin_user, test_org_a, test_org_b, org_admin_headers):
    """Verify member of Org A cannot access meeting belonging to Org B (returns 404)."""
    # Seed meeting in Org B
    meeting_b, _, _ = _seed_meeting_with_all_transcripts(db_session, test_org_b.id, org_admin_user.id)

    # Org A admin requests Org B meeting with Org A headers -> 404
    resp = client.get(f"/api/v1/media/meetings/{meeting_b.id}", headers=org_admin_headers)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Meeting not found."


def test_cross_tenant_job_access_blocked(client, db_session, org_admin_user, test_org_a, test_org_b, org_admin_headers):
    """Verify member of Org A cannot access job belonging to Org B (returns 404)."""
    _, _, job_b = _seed_meeting_with_all_transcripts(db_session, test_org_b.id, org_admin_user.id)

    resp = client.get(f"/api/v1/media/jobs/{job_b.id}", headers=org_admin_headers)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Job not found."


def test_inactive_membership_rejected(client, db_session, org_admin_user, test_org_a, org_admin_headers):
    """Verify deactivated membership in a valid organization returns 403."""
    m = db_session.query(OrganizationMembership).filter(
        OrganizationMembership.user_id == org_admin_user.id,
        OrganizationMembership.organization_id == test_org_a.id,
    ).first()
    m.is_active = False
    db_session.commit()

    resp = client.get("/api/v1/media/meetings", headers=org_admin_headers)
    assert resp.status_code == 403
    assert "active membership" in resp.json()["detail"].lower()


def test_inactive_organization_rejected(client, db_session, test_org_a, org_admin_headers):
    """Verify user accessing deactivated organization returns 403."""
    test_org_a.is_active = False
    db_session.commit()

    resp = client.get("/api/v1/media/meetings", headers=org_admin_headers)
    assert resp.status_code == 403
    assert "active membership" in resp.json()["detail"].lower()


def test_multiple_memberships_require_organization_selector(client, multi_org_user, test_org_a, test_org_b):
    """Verify user with multiple memberships must supply X-Organization-Id header."""
    headers_without_org = make_auth_headers(multi_org_user)  # No X-Organization-Id

    resp = client.get("/api/v1/media/meetings", headers=headers_without_org)
    assert resp.status_code == 400
    assert "x-organization-id header is required" in resp.json()["detail"].lower()

    # Supplying explicit header succeeds
    headers_with_org = make_auth_headers(multi_org_user, test_org_a)
    resp2 = client.get("/api/v1/media/meetings", headers=headers_with_org)
    assert resp2.status_code == 200


def test_single_membership_auto_selected_when_header_omitted(client, org_admin_user, test_org_a):
    """Verify user with exactly one membership has organization resolved automatically when header is omitted."""
    headers_without_org = make_auth_headers(org_admin_user)  # No X-Organization-Id

    resp = client.get("/api/v1/media/meetings", headers=headers_without_org)
    assert resp.status_code == 200


# --- 3. Role Authorization (MEMBER vs ORG_ADMIN) ---

def test_member_cannot_retrieve_original_transcript(client, db_session, regular_member_user, test_org_a, regular_member_headers):
    """Verify MEMBER role requesting ORIGINAL transcript returns HTTP 403 Forbidden."""
    meeting, _, _ = _seed_meeting_with_all_transcripts(db_session, test_org_a.id, regular_member_user.id)

    resp = client.get(
        f"/api/v1/transcription/{meeting.id}?transcript_type=ORIGINAL",
        headers=regular_member_headers,
    )
    assert resp.status_code == 403
    assert "administrator privileges" in resp.json()["detail"].lower()


def test_member_cannot_retrieve_cleaned_transcript(client, db_session, regular_member_user, test_org_a, regular_member_headers):
    """Verify MEMBER role requesting CLEANED transcript returns HTTP 403 Forbidden."""
    meeting, _, _ = _seed_meeting_with_all_transcripts(db_session, test_org_a.id, regular_member_user.id)

    resp = client.get(
        f"/api/v1/transcription/{meeting.id}?transcript_type=CLEANED",
        headers=regular_member_headers,
    )
    assert resp.status_code == 403
    assert "administrator privileges" in resp.json()["detail"].lower()


def test_member_can_retrieve_redacted_and_cleaned_redacted(client, db_session, regular_member_user, test_org_a, regular_member_headers):
    """Verify MEMBER role can retrieve privacy-safe REDACTED and CLEANED_REDACTED transcripts."""
    meeting, _, _ = _seed_meeting_with_all_transcripts(db_session, test_org_a.id, regular_member_user.id)

    # REDACTED
    r_red = client.get(
        f"/api/v1/transcription/{meeting.id}?transcript_type=REDACTED",
        headers=regular_member_headers,
    )
    assert r_red.status_code == 200
    assert len(r_red.json()) == 1
    assert r_red.json()[0]["transcript_type"] == "REDACTED"

    # CLEANED_REDACTED
    r_cr = client.get(
        f"/api/v1/transcription/{meeting.id}?transcript_type=CLEANED_REDACTED",
        headers=regular_member_headers,
    )
    assert r_cr.status_code == 200
    assert len(r_cr.json()) == 1
    assert r_cr.json()[0]["transcript_type"] == "CLEANED_REDACTED"


def test_member_omitting_type_receives_only_privacy_safe_transcripts(client, db_session, regular_member_user, test_org_a, regular_member_headers):
    """Verify MEMBER omitting transcript_type receives only redacted versions, never original or cleaned."""
    meeting, _, _ = _seed_meeting_with_all_transcripts(db_session, test_org_a.id, regular_member_user.id)

    resp = client.get(f"/api/v1/transcription/{meeting.id}", headers=regular_member_headers)
    assert resp.status_code == 200
    items = resp.json()
    types = [i["transcript_type"] for i in items]
    assert "ORIGINAL" not in types
    assert "CLEANED" not in types
    assert "REDACTED" in types
    assert "CLEANED_REDACTED" in types


def test_member_can_retrieve_default_transcript(client, db_session, regular_member_user, test_org_a, regular_member_headers):
    """Verify MEMBER can retrieve the default privacy-safe transcript."""
    meeting, _, _ = _seed_meeting_with_all_transcripts(db_session, test_org_a.id, regular_member_user.id)

    resp = client.get(f"/api/v1/transcription/{meeting.id}/default", headers=regular_member_headers)
    assert resp.status_code == 200
    assert resp.json()["transcript_type"] == "CLEANED_REDACTED"


def test_member_cannot_delete_raw_media(client, db_session, regular_member_user, test_org_a, regular_member_headers):
    """Verify MEMBER cannot delete raw media assets (returns 403)."""
    meeting, asset, _ = _seed_meeting_with_all_transcripts(db_session, test_org_a.id, regular_member_user.id)

    resp = client.delete(f"/api/v1/media/assets/{asset.id}", headers=regular_member_headers)
    assert resp.status_code == 403
    assert "administrator privileges required" in resp.json()["detail"].lower()


def test_org_admin_can_retrieve_all_transcript_versions(client, db_session, org_admin_user, test_org_a, org_admin_headers):
    """Verify ORG_ADMIN can retrieve ORIGINAL, CLEANED, and all versions."""
    meeting, _, _ = _seed_meeting_with_all_transcripts(db_session, test_org_a.id, org_admin_user.id)

    # ORIGINAL
    r_orig = client.get(f"/api/v1/transcription/{meeting.id}?transcript_type=ORIGINAL", headers=org_admin_headers)
    assert r_orig.status_code == 200
    assert len(r_orig.json()) == 1
    assert r_orig.json()[0]["transcript_type"] == "ORIGINAL"

    # CLEANED
    r_clean = client.get(f"/api/v1/transcription/{meeting.id}?transcript_type=CLEANED", headers=org_admin_headers)
    assert r_clean.status_code == 200
    assert len(r_clean.json()) == 1
    assert r_clean.json()[0]["transcript_type"] == "CLEANED"

    # ALL
    r_all = client.get(f"/api/v1/transcription/{meeting.id}", headers=org_admin_headers)
    assert r_all.status_code == 200
    assert len(r_all.json()) == 4


def test_org_admin_can_delete_raw_media(client, db_session, org_admin_user, test_org_a, org_admin_headers, monkeypatch):
    """Verify ORG_ADMIN can delete raw media assets."""
    meeting, asset, _ = _seed_meeting_with_all_transcripts(db_session, test_org_a.id, org_admin_user.id)

    # Monkeypatch physical file deletion
    monkeypatch.setattr("app.services.lifecycle.Path.exists", lambda self: True)
    monkeypatch.setattr("app.services.lifecycle.Path.unlink", lambda self: None)

    resp = client.delete(f"/api/v1/media/assets/{asset.id}", headers=org_admin_headers)
    assert resp.status_code == 200
    assert resp.json()["is_raw_deleted"] is True


def test_member_calling_clean_endpoint_receives_403(
    client, db_session, regular_member_user, test_org_a, regular_member_headers
):
    """Verify MEMBER calling POST /transcription/{meeting_id}/clean is rejected with 403,

    creating no database transcript, no audit log, and leaking no sensitive text.
    """
    meeting, _ = _seed_meeting_with_source_transcript_and_pii(db_session, test_org_a.id, regular_member_user.id)

    initial_cleaned_count = (
        db_session.query(Transcript)
        .filter(Transcript.meeting_id == meeting.id, Transcript.transcript_type == TranscriptType.CLEANED.value)
        .count()
    )
    assert initial_cleaned_count == 0

    resp = client.post(
        f"/api/v1/transcription/{meeting.id}/clean",
        json={},
        headers=regular_member_headers,
    )
    assert resp.status_code == 403
    assert "administrator privileges" in resp.json()["detail"].lower()

    # Verify no sensitive PII was leaked in the response
    raw_response_text = resp.text
    assert "rahul.private@example.com" not in raw_response_text
    assert "9876543210" not in raw_response_text
    assert "topsecret99" not in raw_response_text

    # Verify database state was not modified
    db_session.expire_all()
    cleaned_count_after = (
        db_session.query(Transcript)
        .filter(Transcript.meeting_id == meeting.id, Transcript.transcript_type == TranscriptType.CLEANED.value)
        .count()
    )
    assert cleaned_count_after == 0

    # Verify no audit log entry was created
    audit_entry = (
        db_session.query(AuditLog)
        .filter(AuditLog.organization_id == test_org_a.id, AuditLog.action == "TRANSCRIPT_CLEANED")
        .first()
    )
    assert audit_entry is None


def test_member_calling_process_with_redact_false_receives_403(
    client, db_session, regular_member_user, test_org_a, regular_member_headers
):
    """Verify MEMBER calling POST /transcription/{meeting_id}/process with redact=False is rejected with 403."""
    meeting, _ = _seed_meeting_with_source_transcript_and_pii(db_session, test_org_a.id, regular_member_user.id)

    resp = client.post(
        f"/api/v1/transcription/{meeting.id}/process",
        json={"clean": True, "redact": False},
        headers=regular_member_headers,
    )
    assert resp.status_code == 403
    assert "administrator privileges" in resp.json()["detail"].lower()

    # Verify no sensitive PII was leaked
    raw_response_text = resp.text
    assert "rahul.private@example.com" not in raw_response_text
    assert "9876543210" not in raw_response_text

    # Verify database side effect prevention
    db_session.expire_all()
    cleaned_count = (
        db_session.query(Transcript)
        .filter(Transcript.meeting_id == meeting.id, Transcript.transcript_type == TranscriptType.CLEANED.value)
        .count()
    )
    assert cleaned_count == 0


def test_member_can_perform_permitted_redacted_processing(
    client, db_session, regular_member_user, test_org_a, regular_member_headers
):
    """Verify MEMBER can still perform privacy-safe processing (CLEANED_REDACTED and REDACTED)."""
    meeting, _ = _seed_meeting_with_source_transcript_and_pii(db_session, test_org_a.id, regular_member_user.id)

    # 1. MEMBER calling /redact
    r_redact = client.post(
        f"/api/v1/transcription/{meeting.id}/redact",
        json={},
        headers=regular_member_headers,
    )
    assert r_redact.status_code == 201
    data_redact = r_redact.json()
    assert data_redact["transcript_type"] == "REDACTED"
    assert "rahul.private@example.com" not in str(data_redact["segments"])
    assert "9876543210" not in str(data_redact["segments"])
    assert "[EMAIL_REDACTED]" in str(data_redact["segments"])
    assert "[PHONE_REDACTED]" in str(data_redact["segments"])

    # 2. MEMBER calling /process with clean=True, redact=True (CLEANED_REDACTED)
    r_process = client.post(
        f"/api/v1/transcription/{meeting.id}/process",
        json={"clean": True, "redact": True},
        headers=regular_member_headers,
    )
    assert r_process.status_code == 201
    data_process = r_process.json()
    assert data_process["transcript_type"] == "CLEANED_REDACTED"
    assert "rahul.private@example.com" not in str(data_process["segments"])
    assert "[EMAIL_REDACTED]" in str(data_process["segments"])

    # Verify database records both privacy-safe transcripts
    db_session.expire_all()
    transcripts = (
        db_session.query(Transcript)
        .filter(Transcript.meeting_id == meeting.id)
        .all()
    )
    types = {t.transcript_type for t in transcripts}
    assert "ORIGINAL" in types
    assert "REDACTED" in types
    assert "CLEANED_REDACTED" in types
    assert "CLEANED" not in types  # No unredacted CLEANED was created


def test_org_admin_retains_cleaning_and_unredacted_processing(
    client, db_session, org_admin_user, test_org_a, org_admin_headers
):
    """Verify ORG_ADMIN retains full ability to generate CLEANED and unredacted transcripts."""
    meeting, _ = _seed_meeting_with_source_transcript_and_pii(db_session, test_org_a.id, org_admin_user.id)

    # 1. ORG_ADMIN calling /clean
    r_clean = client.post(
        f"/api/v1/transcription/{meeting.id}/clean",
        json={},
        headers=org_admin_headers,
    )
    assert r_clean.status_code == 201
    data_clean = r_clean.json()
    assert data_clean["transcript_type"] == "CLEANED"
    # Unredacted content is preserved for admin
    assert "rahul.private@example.com" in str(data_clean["segments"])

    # 2. ORG_ADMIN calling /process with redact=False
    r_process_unredacted = client.post(
        f"/api/v1/transcription/{meeting.id}/process",
        json={"clean": True, "redact": False},
        headers=org_admin_headers,
    )
    assert r_process_unredacted.status_code == 201
    assert r_process_unredacted.json()["transcript_type"] == "CLEANED"

    # 3. ORG_ADMIN calling /process with redact=True
    r_process_redacted = client.post(
        f"/api/v1/transcription/{meeting.id}/process",
        json={"clean": True, "redact": True},
        headers=org_admin_headers,
    )
    assert r_process_redacted.status_code == 201
    assert r_process_redacted.json()["transcript_type"] == "CLEANED_REDACTED"


def test_cross_tenant_clean_and_process_endpoints_rejected(
    client, db_session, org_admin_user, regular_member_user, test_org_a, test_org_b, org_admin_headers, regular_member_headers
):
    """Verify cross-tenant attempts against /clean and /process return 404 and create no data."""
    # Seed meeting in Organization B
    meeting_b, _ = _seed_meeting_with_source_transcript_and_pii(db_session, test_org_b.id, org_admin_user.id)

    initial_transcripts_org_b = (
        db_session.query(Transcript)
        .filter(Transcript.meeting_id == meeting_b.id)
        .count()
    )

    # 1. Org A admin calls /clean on Org B meeting -> 404
    r1 = client.post(
        f"/api/v1/transcription/{meeting_b.id}/clean",
        json={},
        headers=org_admin_headers,
    )
    assert r1.status_code == 404
    assert r1.json()["detail"] == f"Meeting {meeting_b.id} not found."

    # 2. Org A admin calls /process on Org B meeting -> 404
    r2 = client.post(
        f"/api/v1/transcription/{meeting_b.id}/process",
        json={"clean": True, "redact": False},
        headers=org_admin_headers,
    )
    assert r2.status_code == 404
    assert r2.json()["detail"] == f"Meeting {meeting_b.id} not found."

    # 3. Org A member calls /process on Org B meeting -> 404
    r3 = client.post(
        f"/api/v1/transcription/{meeting_b.id}/process",
        json={"clean": True, "redact": True},
        headers=regular_member_headers,
    )
    assert r3.status_code == 404
    assert r3.json()["detail"] == f"Meeting {meeting_b.id} not found."

    # Verify no transcript was added to Organization B
    db_session.expire_all()
    transcripts_org_b_after = (
        db_session.query(Transcript)
        .filter(Transcript.meeting_id == meeting_b.id)
        .count()
    )
    assert transcripts_org_b_after == initial_transcripts_org_b


# --- 4. Global Super Admin Policy ---

def test_super_admin_without_membership_cannot_access_tenant_meetings(client, db_session, super_admin_no_membership, test_org_a):
    """Verify Super Admin without membership in Org A cannot access Org A data (returns 403)."""
    meeting, _, _ = _seed_meeting_with_all_transcripts(db_session, test_org_a.id, super_admin_no_membership.id)

    headers = make_auth_headers(super_admin_no_membership, test_org_a)
    resp = client.get("/api/v1/media/meetings", headers=headers)
    assert resp.status_code == 403
    assert "active membership" in resp.json()["detail"].lower()


def test_super_admin_with_member_role_cannot_delete_raw_media(client, db_session, super_admin_member, test_org_a):
    """Verify Super Admin with MEMBER role in Org A cannot perform ORG_ADMIN actions (returns 403)."""
    meeting, asset, _ = _seed_meeting_with_all_transcripts(db_session, test_org_a.id, super_admin_member.id)

    headers = make_auth_headers(super_admin_member, test_org_a)
    resp = client.delete(f"/api/v1/media/assets/{asset.id}", headers=headers)
    assert resp.status_code == 403
    assert "administrator privileges required" in resp.json()["detail"].lower()


def test_super_admin_with_explicit_org_admin_can_perform_admin_actions(client, db_session, super_admin_org_admin, test_org_a, monkeypatch):
    """Verify Super Admin with explicit ORG_ADMIN role can perform tenant admin actions."""
    meeting, asset, _ = _seed_meeting_with_all_transcripts(db_session, test_org_a.id, super_admin_org_admin.id)

    monkeypatch.setattr("app.services.lifecycle.Path.exists", lambda self: True)
    monkeypatch.setattr("app.services.lifecycle.Path.unlink", lambda self: None)

    headers = make_auth_headers(super_admin_org_admin, test_org_a)
    resp = client.delete(f"/api/v1/media/assets/{asset.id}", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["is_raw_deleted"] is True


# --- 5. Anti-Enumeration (404 Consistency) ---

def test_anti_enumeration_unknown_and_foreign_ids_return_404(client, db_session, org_admin_user, test_org_a, test_org_b, org_admin_headers):
    """Verify both non-existent IDs and foreign tenant IDs return identical 404."""
    # Seed entity in Org B
    meeting_b, asset_b, job_b = _seed_meeting_with_all_transcripts(db_session, test_org_b.id, org_admin_user.id)

    # 1. Foreign meeting ID vs Non-existent meeting ID
    r_foreign_meeting = client.get(f"/api/v1/media/meetings/{meeting_b.id}", headers=org_admin_headers)
    r_unknown_meeting = client.get("/api/v1/media/meetings/unknown-meeting-uuid", headers=org_admin_headers)
    assert r_foreign_meeting.status_code == 404
    assert r_unknown_meeting.status_code == 404
    assert r_foreign_meeting.json() == r_unknown_meeting.json()

    # 2. Foreign job ID vs Non-existent job ID
    r_foreign_job = client.get(f"/api/v1/media/jobs/{job_b.id}", headers=org_admin_headers)
    r_unknown_job = client.get("/api/v1/media/jobs/unknown-job-uuid", headers=org_admin_headers)
    assert r_foreign_job.status_code == 404
    assert r_unknown_job.status_code == 404
    assert r_foreign_job.json() == r_unknown_job.json()

    # 3. Foreign media asset ID vs Non-existent media asset ID
    r_foreign_asset = client.delete(f"/api/v1/media/assets/{asset_b.id}", headers=org_admin_headers)
    r_unknown_asset = client.delete("/api/v1/media/assets/unknown-asset-uuid", headers=org_admin_headers)
    assert r_foreign_asset.status_code == 404
    assert r_unknown_asset.status_code == 404
    assert r_foreign_asset.json() == r_unknown_asset.json()


def test_cross_tenant_media_asset_deletion_integrity(
    client, db_session, org_admin_user, test_org_a, test_org_b, org_admin_headers, monkeypatch
):
    """Verify attempting to delete an asset belonging to Organization B using an authenticated

    Organization A administrator:
    1. Returns HTTP 404 (anti-enumeration).
    2. Asset remains present in the database.
    3. Asset's deletion status remains unchanged (is_raw_deleted is False).
    4. Asset's physical file has not been deleted and Path.unlink is never called.
    5. No audit log entry is recorded for the cross-tenant delete attempt.
    """
    # 1. Seed meeting and asset in Organization B
    meeting_b, asset_b, _ = _seed_meeting_with_all_transcripts(db_session, test_org_b.id, org_admin_user.id)

    # 2. Stage a real physical file at the resolved storage path
    resolved_path = resolve_safe_storage_path(asset_b.storage_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    test_content = b"CRITICAL_ORGANIZATION_B_AUDIO_DATA_DO_NOT_DELETE"
    resolved_path.write_bytes(test_content)

    unlink_invocations = []
    orig_unlink = Path.unlink

    def spy_unlink(self, *args, **kwargs):
        unlink_invocations.append(str(self))
        return orig_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", spy_unlink)

    try:
        # 3. Authenticated Org A administrator attempts to delete Org B's asset
        resp = client.delete(f"/api/v1/media/assets/{asset_b.id}", headers=org_admin_headers)

        # 4. Assert HTTP 404 anti-enumeration response
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Media asset not found."

        # 5. Assert physical file was NOT deleted and unlink was never invoked
        assert len(unlink_invocations) == 0
        assert resolved_path.exists() is True
        assert resolved_path.read_bytes() == test_content

        # 6. Query the database to verify asset remains present and is_raw_deleted is False
        db_session.expire_all()
        persisted_asset = db_session.get(MediaAsset, asset_b.id)
        assert persisted_asset is not None
        assert persisted_asset.is_raw_deleted is False
        assert persisted_asset.meeting_id == meeting_b.id

        # 7. Verify no audit log record was created
        audit_record = db_session.query(AuditLog).filter(AuditLog.entity_id == asset_b.id).first()
        assert audit_record is None
    finally:
        # Cleanup temporary test file
        if resolved_path.exists():
            orig_unlink(resolved_path)


# --- 6. Hardware Diagnostics Platform Authorization ---

def test_hardware_diagnostics_requires_super_admin(client, regular_member_user, super_admin_no_membership, test_org_a):
    """Verify hardware diagnostics is strictly restricted to global Super Admin."""
    # 1. Anonymous returns 401
    r_anon = client.get("/api/v1/transcription/hardware/diagnostics")
    assert r_anon.status_code == 401

    # 2. Regular user (even ORG_ADMIN) returns 403
    regular_headers = make_auth_headers(regular_member_user, test_org_a)
    r_member = client.get("/api/v1/transcription/hardware/diagnostics", headers=regular_headers)
    assert r_member.status_code == 403
    assert "super administrator" in r_member.json()["detail"].lower()

    # 3. Super admin succeeds
    super_headers = make_auth_headers(super_admin_no_membership)
    r_super = client.get("/api/v1/transcription/hardware/diagnostics", headers=super_headers)
    assert r_super.status_code == 200
    data = r_super.json()
    assert "gpu_info" in data
    assert "ctranslate2_cuda_support" in data
