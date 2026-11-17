"""Integration tests for Phase 7.5: Structured Meeting Intelligence API Endpoints.

Covers:
1. POST /meetings/{meeting_id}/intelligence/extract (trigger extraction job).
2. GET /meetings/{meeting_id}/intelligence (retrieve intelligence).
3. PATCH /action-items/{item_id}/status (update status with audit log).
4. Authentication rejection (401).
5. Cross-tenant isolation rejection (404).
6. Invalid and malformed request payload handling.
7. Privacy-safe transcript requirement enforcement.
8. Duplicate active extraction job prevention.
9. Consistent empty response before extraction.
10. Complete structured intelligence retrieval after extraction.
11. Audit logging on action item status changes.
"""

import json
from uuid import uuid4
import pytest
from sqlalchemy.orm import Session

from app.models.identity import Organization, User, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, ProcessingJob, JobType, JobStatus, MeetingStatus
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.intelligence import (
    MeetingSummary,
    Topic,
    Decision,
    ActionItem,
    Risk,
    UnresolvedQuestion,
    ActionItemStatus,
)
from app.models.memory import AuditLog
from app.core.security import hash_password
from app.services.job_worker import run_single_job_cycle
from tests.conftest import make_auth_headers


@pytest.fixture(autouse=True)
def bind_worker_sessions(db_session, monkeypatch):
    """Ensure worker handlers use the test database session."""
    monkeypatch.setattr("app.database.session.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.services.job_worker.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.services.intelligence.job_handler.SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_session, "close", lambda: None)


def _setup_org_user(db: Session, name_prefix: str, role: str = OrgRole.MEMBER.value):
    """Create an organization, user, and membership."""
    suffix = uuid4().hex[:6]
    org = Organization(name=f"{name_prefix} Org {suffix}", slug=f"{name_prefix.lower()}-{suffix}", is_active=True)
    user = User(
        email=f"{name_prefix.lower()}_{suffix}@example.com",
        full_name=f"{name_prefix} User",
        hashed_password=hash_password("SecurePass123!"),
        is_active=True,
    )
    db.add_all([org, user])
    db.flush()

    membership = OrganizationMembership(
        organization_id=org.id,
        user_id=user.id,
        role=role,
        is_active=True,
    )
    db.add(membership)
    db.commit()
    return org, user


def _setup_meeting_with_transcript(db: Session, org: Organization, transcript_type: str = TranscriptType.CLEANED_REDACTED.value):
    """Create a meeting with a transcript and sample segments."""
    meeting = Meeting(
        organization_id=org.id,
        title="Intelligence Strategy Session",
        status=MeetingStatus.COMPLETED.value,
    )
    db.add(meeting)
    db.flush()

    transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=transcript_type,
        version=1,
    )
    db.add(transcript)
    db.flush()

    segs = [
        TranscriptSegment(
            transcript_id=transcript.id,
            sequence_order=0,
            start_time=0.0,
            end_time=12.0,
            text="We decided to launch the intelligence endpoints today.",
            language_code="en",
        ),
        TranscriptSegment(
            transcript_id=transcript.id,
            sequence_order=1,
            start_time=12.5,
            end_time=25.0,
            text="I will verify the tenant isolation tests by EOD.",
            language_code="en",
        ),
        TranscriptSegment(
            transcript_id=transcript.id,
            sequence_order=2,
            start_time=25.5,
            end_time=35.0,
            text="Risk is potential cross-tenant leak in unredacted endpoints.",
            language_code="en",
        ),
        TranscriptSegment(
            transcript_id=transcript.id,
            sequence_order=3,
            start_time=35.5,
            end_time=45.0,
            text="How will we monitor response latency?",
            language_code="en",
        ),
    ]
    db.add_all(segs)
    db.commit()
    return meeting, transcript


# --- 1. Authentication Tests ---

def test_unauthenticated_requests_rejected(client):
    """Verify all intelligence endpoints reject requests lacking valid bearer credentials."""
    dummy_id = str(uuid4())

    # Extract endpoint
    res = client.post(f"/api/v1/meetings/{dummy_id}/intelligence/extract", json={})
    assert res.status_code == 401
    assert "WWW-Authenticate" in res.headers

    # Get intelligence endpoint
    res = client.get(f"/api/v1/meetings/{dummy_id}/intelligence")
    assert res.status_code == 401

    # Action item status update
    res = client.patch(f"/api/v1/action-items/{dummy_id}/status", json={"status": "COMPLETED"})
    assert res.status_code == 401


# --- 2. Trigger Extraction Endpoint Tests ---

def test_extract_intelligence_job_success(client, db_session):
    """Verify authenticated user can trigger extraction and receive a 202 queued job response."""
    org, user = _setup_org_user(db_session, "Alpha", role=OrgRole.MEMBER.value)
    meeting, transcript = _setup_meeting_with_transcript(db_session, org)
    headers = make_auth_headers(user, org=org)

    response = client.post(
        f"/api/v1/meetings/{meeting.id}/intelligence/extract",
        json={
            "output_language": "en",
            "provider_override": "deterministic",
        },
        headers=headers,
    )
    assert response.status_code == 202
    data = response.json()
    assert data["job_type"] == "EXTRACTION"
    assert data["status"] == "PENDING"
    assert data["meeting_id"] == meeting.id
    assert "job_id" in data

    # Verify job persisted in DB with validated payload
    job = db_session.get(ProcessingJob, data["job_id"])
    assert job is not None
    assert job.status == JobStatus.PENDING.value
    payload = json.loads(job.payload)
    assert payload["meeting_id"] == meeting.id
    assert payload["organization_id"] == org.id
    assert payload["provider_override"] == "deterministic"


def test_extract_intelligence_cross_tenant_rejected(client, db_session):
    """Verify user in Org B cannot trigger extraction for meeting in Org A (returns 404)."""
    org_a, user_a = _setup_org_user(db_session, "OrgA")
    org_b, user_b = _setup_org_user(db_session, "OrgB")
    meeting_a, _ = _setup_meeting_with_transcript(db_session, org_a)

    headers_b = make_auth_headers(user_b, org=org_b)
    response = client.post(
        f"/api/v1/meetings/{meeting_a.id}/intelligence/extract",
        json={},
        headers=headers_b,
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_extract_intelligence_no_privacy_safe_transcript_rejected(client, db_session):
    """Verify extraction is rejected if meeting has only unredacted ORIGINAL transcript."""
    org, user = _setup_org_user(db_session, "PrivacyOrg")
    meeting, _ = _setup_meeting_with_transcript(db_session, org, transcript_type=TranscriptType.ORIGINAL.value)
    headers = make_auth_headers(user, org=org)

    response = client.post(
        f"/api/v1/meetings/{meeting.id}/intelligence/extract",
        json={},
        headers=headers,
    )
    assert response.status_code == 400
    assert "privacy-safe" in response.json()["detail"].lower()


def test_extract_intelligence_explicit_unredacted_transcript_rejected(client, db_session):
    """Verify explicitly specifying an unredacted transcript ID is rejected with 400."""
    org, user = _setup_org_user(db_session, "PrivacyOrg2")
    meeting, safe_t = _setup_meeting_with_transcript(db_session, org, transcript_type=TranscriptType.CLEANED_REDACTED.value)

    # Add an unredacted ORIGINAL transcript
    orig_t = Transcript(meeting_id=meeting.id, transcript_type=TranscriptType.ORIGINAL.value, version=2)
    db_session.add(orig_t)
    db_session.commit()

    headers = make_auth_headers(user, org=org)
    response = client.post(
        f"/api/v1/meetings/{meeting.id}/intelligence/extract",
        json={"source_transcript_id": orig_t.id},
        headers=headers,
    )
    assert response.status_code == 400
    assert "strictly requires privacy-safe transcripts" in response.json()["detail"]


def test_extract_intelligence_invalid_provider_override(client, db_session):
    """Verify invalid provider override returns 400."""
    org, user = _setup_org_user(db_session, "ProviderOrg")
    meeting, _ = _setup_meeting_with_transcript(db_session, org)
    headers = make_auth_headers(user, org=org)

    response = client.post(
        f"/api/v1/meetings/{meeting.id}/intelligence/extract",
        json={"provider_override": "unsupported_llm"},
        headers=headers,
    )
    assert response.status_code == 400
    assert "Invalid provider_override" in response.json()["detail"]


def test_extract_intelligence_duplicate_active_job_prevention(client, db_session):
    """Verify duplicate extraction requests return the existing active job instead of creating a new one."""
    org, user = _setup_org_user(db_session, "DupOrg")
    meeting, _ = _setup_meeting_with_transcript(db_session, org)
    headers = make_auth_headers(user, org=org)

    # First request
    res1 = client.post(f"/api/v1/meetings/{meeting.id}/intelligence/extract", json={}, headers=headers)
    assert res1.status_code == 202
    job1_id = res1.json()["job_id"]

    # Second request while first is still PENDING
    res2 = client.post(f"/api/v1/meetings/{meeting.id}/intelligence/extract", json={}, headers=headers)
    assert res2.status_code == 202
    data2 = res2.json()
    assert data2["job_id"] == job1_id
    assert "already active" in data2["message"]

    # Exactly 1 job exists in DB
    jobs_count = db_session.query(ProcessingJob).filter(
        ProcessingJob.meeting_id == meeting.id,
        ProcessingJob.job_type == JobType.EXTRACTION.value,
    ).count()
    assert jobs_count == 1


# --- 3. Retrieve Meeting Intelligence Endpoint Tests ---

def test_get_meeting_intelligence_empty_state(client, db_session):
    """Verify GET before extraction returns consistent empty response with has_intelligence=False."""
    org, user = _setup_org_user(db_session, "EmptyOrg")
    meeting, _ = _setup_meeting_with_transcript(db_session, org)
    headers = make_auth_headers(user, org=org)

    response = client.get(f"/api/v1/meetings/{meeting.id}/intelligence", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["meeting_id"] == meeting.id
    assert data["has_intelligence"] is False
    assert data["summary"] is None
    assert data["topics"] == []
    assert data["decisions"] == []
    assert data["action_items"] == []
    assert data["risks"] == []
    assert data["unresolved_questions"] == []


def test_get_meeting_intelligence_cross_tenant_rejected(client, db_session):
    """Verify GET intelligence for meeting in another tenant returns 404."""
    org_a, user_a = _setup_org_user(db_session, "OrgA")
    org_b, user_b = _setup_org_user(db_session, "OrgB")
    meeting_a, _ = _setup_meeting_with_transcript(db_session, org_a)

    headers_b = make_auth_headers(user_b, org=org_b)
    response = client.get(f"/api/v1/meetings/{meeting_a.id}/intelligence", headers=headers_b)
    assert response.status_code == 404


def test_get_meeting_intelligence_complete_after_extraction(client, db_session):
    """Verify full intelligence payload after background extraction completes."""
    org, user = _setup_org_user(db_session, "ExtractedOrg")
    meeting, transcript = _setup_meeting_with_transcript(db_session, org)
    headers = make_auth_headers(user, org=org)

    # 1. Trigger extraction
    post_res = client.post(
        f"/api/v1/meetings/{meeting.id}/intelligence/extract",
        json={"provider_override": "deterministic"},
        headers=headers,
    )
    assert post_res.status_code == 202

    # 2. Execute the queued background job
    handled = run_single_job_cycle(db_session)
    assert handled is True

    # 3. Retrieve intelligence
    get_res = client.get(f"/api/v1/meetings/{meeting.id}/intelligence", headers=headers)
    assert get_res.status_code == 200
    intel = get_res.json()

    assert intel["has_intelligence"] is True
    assert intel["meeting_id"] == meeting.id

    # Verify Summary
    assert intel["summary"] is not None
    assert intel["summary"]["version"] == 1
    assert len(intel["summary"]["executive_summary"]) > 0

    # Verify Decisions with evidence grounding
    assert len(intel["decisions"]) >= 1
    d0 = intel["decisions"][0]
    assert d0["evidence_segment_id"] is not None
    assert d0["timestamp_seconds"] == 0.0

    # Verify Action Items
    assert len(intel["action_items"]) >= 1
    a0 = intel["action_items"][0]
    assert a0["status"] == "PENDING"

    # Verify Risks
    assert len(intel["risks"]) >= 1

    # Verify Unresolved Questions
    assert len(intel["unresolved_questions"]) >= 1

    # Ensure unredacted transcript text is not leaked
    assert "ORIGINAL" not in json.dumps(intel)


# --- 4. Update Action Item Status Tests ---

def test_update_action_item_status_success(client, db_session):
    """Verify action item status can be updated and generates an immutable audit log."""
    org, user = _setup_org_user(db_session, "ActionOrg")
    meeting, _ = _setup_meeting_with_transcript(db_session, org)
    headers = make_auth_headers(user, org=org)

    # Create an action item in the database
    item = ActionItem(
        meeting_id=meeting.id,
        description="Verify system audit log records",
        owner_name="Alice",
        status=ActionItemStatus.PENDING.value,
    )
    db_session.add(item)
    db_session.commit()

    # Update status to COMPLETED
    res = client.patch(
        f"/api/v1/action-items/{item.id}/status",
        json={"status": "COMPLETED"},
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == item.id
    assert data["status"] == "COMPLETED"
    assert data["description"] == "Verify system audit log records"
    assert data["owner_name"] == "Alice"

    # Check database state
    db_session.refresh(item)
    assert item.status == "COMPLETED"

    # Check AuditLog
    audit = db_session.query(AuditLog).filter(
        AuditLog.entity_id == item.id,
        AuditLog.action == "UPDATE_ACTION_ITEM_STATUS",
    ).first()
    assert audit is not None
    assert audit.organization_id == org.id
    assert audit.user_id == user.id
    details = json.loads(audit.details)
    assert details["old_status"] == "PENDING"
    assert details["new_status"] == "COMPLETED"
    assert details["meeting_id"] == meeting.id


def test_update_action_item_status_cross_tenant_rejected(client, db_session):
    """Verify user in Org B cannot update an action item belonging to Org A."""
    org_a, user_a = _setup_org_user(db_session, "OrgA")
    org_b, user_b = _setup_org_user(db_session, "OrgB")
    meeting_a, _ = _setup_meeting_with_transcript(db_session, org_a)

    item_a = ActionItem(
        meeting_id=meeting_a.id,
        description="Org A action item",
        status=ActionItemStatus.PENDING.value,
    )
    db_session.add(item_a)
    db_session.commit()

    headers_b = make_auth_headers(user_b, org=org_b)
    res = client.patch(
        f"/api/v1/action-items/{item_a.id}/status",
        json={"status": "COMPLETED"},
        headers=headers_b,
    )
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_update_action_item_invalid_status_rejected(client, db_session):
    """Verify submitting an invalid action item status returns 422."""
    org, user = _setup_org_user(db_session, "InvalidStatusOrg")
    meeting, _ = _setup_meeting_with_transcript(db_session, org)
    headers = make_auth_headers(user, org=org)

    item = ActionItem(
        meeting_id=meeting.id,
        description="Item to test invalid status",
        status=ActionItemStatus.PENDING.value,
    )
    db_session.add(item)
    db_session.commit()

    res = client.patch(
        f"/api/v1/action-items/{item.id}/status",
        json={"status": "INVALID_STATE"},
        headers=headers,
    )
    assert res.status_code == 422


def test_api_attribution_persistence_and_retrieval(client, db_session):
    """Verify provider attribution is returned via GET /meetings/{id}/intelligence."""
    org, user = _setup_org_user(db_session, "AttribApiOrg")
    meeting, _ = _setup_meeting_with_transcript(db_session, org)
    headers = make_auth_headers(user, org=org)

    # 1. Trigger extraction with deterministic provider
    post_res = client.post(
        f"/api/v1/meetings/{meeting.id}/intelligence/extract",
        json={"provider_override": "deterministic"},
        headers=headers,
    )
    assert post_res.status_code == 202

    # 2. Run extraction job cycle
    ran = run_single_job_cycle(db_session)
    assert ran is True

    # 3. Retrieve intelligence
    get_res = client.get(
        f"/api/v1/meetings/{meeting.id}/intelligence",
        headers=headers,
    )
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["has_intelligence"] is True
    assert data["provider_used"] == "deterministic"
    assert data["is_heuristic"] is True
    assert data["fallback_warning"] is None
    assert data["summary"] is not None
    assert data["summary"]["provider_used"] == "deterministic"
    assert data["summary"]["is_heuristic"] is True
    assert data["summary"]["fallback_warning"] is None


def test_api_legacy_records_with_missing_attribution(client, db_session):
    """Verify legacy meeting intelligence records with null attribution are handled safely as unknown/legacy."""
    org, user = _setup_org_user(db_session, "LegacyApiOrg")
    meeting, transcript = _setup_meeting_with_transcript(db_session, org)
    headers = make_auth_headers(user, org=org)

    # Manually insert legacy summary record without provider attribution
    legacy_summary = MeetingSummary(
        meeting_id=meeting.id,
        transcript_id=transcript.id,
        executive_summary="Legacy executive summary from previous system phase.",
        detailed_summary="Legacy discussion.",
        version=1,
        provider_used=None,
        is_heuristic=None,
        fallback_warning=None,
    )
    db_session.add(legacy_summary)
    db_session.commit()

    get_res = client.get(
        f"/api/v1/meetings/{meeting.id}/intelligence",
        headers=headers,
    )
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["has_intelligence"] is True
    assert data["provider_used"] is None
    assert data["is_heuristic"] is None
    assert data["fallback_warning"] is None
    assert data["summary"] is not None
    assert data["summary"]["provider_used"] is None
    assert data["summary"]["is_heuristic"] is None
    assert data["summary"]["fallback_warning"] is None
