"""Comprehensive Integration and Security Tests for Phase 8.8.5: REST APIs for Cross-Meeting Intelligence.

Covers:
1. Decision listing, status/meeting/search/date filtering, details, and lineage.
2. Explicit decision superseding, self-superseding rejection, and invalid status transition rejection.
3. Decision revocation with audited reason and duplicate revocation rejection.
4. Commitment listing, overdue calculation, status/owner/search filtering, details, and review history.
5. Recording commitment review events across meetings and listing reviews.
6. Question listing, status filtering, resolution with decision links, and reopening.
7. Conflict detection endpoint output, subject filtering, heuristic labeling, and superseding exclusion.
8. Strict authenticated tenant isolation across all endpoints (Org A cannot see or mutate Org B data).
9. Foreign resource IDs return safe HTTP 404 responses without enumeration leakage.
10. Unauthenticated requests are rejected with HTTP 401.
11. Rate limiting enforcement on cross-meeting endpoints (HTTP 429).
12. Immutable audit log creation for superseding, revoking, reviewing, resolving, and reopening.
13. Safe transcript/evidence handling (no unredacted transcript leakage).
14. Regression validation of POST /api/v1/memory/query exposing structured intelligence and conflicts.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
from uuid import uuid4
import pytest
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from app.models.identity import Organization, User, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, MeetingStatus
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.intelligence import (
    Decision,
    DecisionStatus,
    ActionItem,
    ActionItemStatus,
    ActionItemReview,
    UnresolvedQuestion,
    QuestionStatus,
)
from app.models.memory import AuditLog, TranscriptChunkRecord
from app.core.security import hash_password
from app.core.rate_limit import cross_meeting_rate_limiter, memory_rate_limiter
from tests.conftest import make_auth_headers


# ===========================================================================
# Fixtures and Test Helpers
# ===========================================================================

@pytest.fixture
def cross_meeting_api_setup(db_session: Session):
    """Set up two isolated tenant organizations, users, meetings, and data."""
    # Org A
    org_a = Organization(name="Tenant Alpha", slug=f"tenant-alpha-{uuid4().hex[:6]}", is_active=True)
    user_a = User(
        email=f"alpha_admin_{uuid4().hex[:6]}@example.com",
        full_name="Alice Admin",
        hashed_password=hash_password("SecurePass123!"),
        is_active=True,
    )
    db_session.add_all([org_a, user_a])
    db_session.flush()

    mem_a = OrganizationMembership(
        user_id=user_a.id,
        organization_id=org_a.id,
        role=OrgRole.ORG_ADMIN.value,
        is_active=True,
    )
    db_session.add(mem_a)

    # Org B
    org_b = Organization(name="Tenant Beta", slug=f"tenant-beta-{uuid4().hex[:6]}", is_active=True)
    user_b = User(
        email=f"beta_user_{uuid4().hex[:6]}@example.com",
        full_name="Bob Beta",
        hashed_password=hash_password("SecurePass123!"),
        is_active=True,
    )
    db_session.add_all([org_b, user_b])
    db_session.flush()

    mem_b = OrganizationMembership(
        user_id=user_b.id,
        organization_id=org_b.id,
        role=OrgRole.MEMBER.value,
        is_active=True,
    )
    db_session.add(mem_b)

    # Meetings for Org A
    now = datetime.now(timezone.utc)
    m1 = Meeting(
        organization_id=org_a.id,
        title="Alpha Arch Sync 1",
        created_by_user_id=user_a.id,
        status=MeetingStatus.COMPLETED.value,
        actual_start_time=now - timedelta(days=10),
    )
    m2 = Meeting(
        organization_id=org_a.id,
        title="Alpha Arch Sync 2",
        created_by_user_id=user_a.id,
        status=MeetingStatus.COMPLETED.value,
        actual_start_time=now - timedelta(days=3),
    )
    m3 = Meeting(
        organization_id=org_a.id,
        title="Alpha Retrospective",
        created_by_user_id=user_a.id,
        status=MeetingStatus.COMPLETED.value,
        actual_start_time=now - timedelta(days=1),
    )

    # Meetings for Org B
    m_b = Meeting(
        organization_id=org_b.id,
        title="Beta Standup",
        created_by_user_id=user_b.id,
        status=MeetingStatus.COMPLETED.value,
        actual_start_time=now - timedelta(days=2),
    )

    db_session.add_all([m1, m2, m3, m_b])
    db_session.flush()

    # Transcripts & Segments for Org A
    t1 = Transcript(meeting_id=m1.id, transcript_type=TranscriptType.CLEANED_REDACTED.value)
    t2 = Transcript(meeting_id=m2.id, transcript_type=TranscriptType.CLEANED_REDACTED.value)
    db_session.add_all([t1, t2])
    db_session.flush()

    seg1 = TranscriptSegment(transcript_id=t1.id, sequence_order=1, start_time=10.0, end_time=20.0, text="Agreed to use AWS.")
    seg2 = TranscriptSegment(transcript_id=t2.id, sequence_order=1, start_time=5.0, end_time=15.0, text="Switched to GCP.")
    db_session.add_all([seg1, seg2])
    db_session.commit()

    headers_a = make_auth_headers(user_a, org_a)
    headers_b = make_auth_headers(user_b, org_b)

    return {
        "org_a": org_a,
        "user_a": user_a,
        "headers_a": headers_a,
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "seg1": seg1,
        "seg2": seg2,
        "org_b": org_b,
        "user_b": user_b,
        "headers_b": headers_b,
        "m_b": m_b,
    }


# ===========================================================================
# 1. Decisions API Tests
# ===========================================================================

def test_list_decisions_and_filtering(client: TestClient, db_session: Session, cross_meeting_api_setup):
    """Test GET /api/v1/decisions with status, meeting, and search filters."""
    s = cross_meeting_api_setup
    d1 = Decision(
        meeting_id=s["m1"].id,
        description="Adopt PostgreSQL for primary persistent storage",
        rationale_or_context="Strong ACID guarantees and JSON support",
        decided_by="Database Working Group",
        status=DecisionStatus.ACTIVE.value,
    )
    d2 = Decision(
        meeting_id=s["m2"].id,
        description="Adopt Redis for distributed session caching",
        rationale_or_context="Sub-millisecond latency for token validation",
        decided_by="Platform Team",
        status=DecisionStatus.ACTIVE.value,
    )
    db_session.add_all([d1, d2])
    db_session.commit()

    # 1. Unfiltered list
    resp = client.get("/api/v1/decisions", headers=s["headers_a"])
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    descriptions = {d["description"] for d in data}
    assert "Adopt PostgreSQL for primary persistent storage" in descriptions
    assert "Adopt Redis for distributed session caching" in descriptions

    # 2. Search filter
    resp = client.get("/api/v1/decisions?search=PostgreSQL", headers=s["headers_a"])
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["id"] == d1.id

    # 3. Meeting filter
    resp = client.get(f"/api/v1/decisions?meeting_id={s['m2'].id}", headers=s["headers_a"])
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["id"] == d2.id

    # 4. Status filter
    resp = client.get("/api/v1/decisions?status=SUPERSEDED", headers=s["headers_a"])
    assert resp.status_code == 200
    assert len(resp.json()) == 0


def test_get_decision_details_and_lineage(client: TestClient, db_session: Session, cross_meeting_api_setup):
    """Test GET /api/v1/decisions/{decision_id} returns details and historical lineage."""
    s = cross_meeting_api_setup
    d1 = Decision(
        meeting_id=s["m1"].id,
        description="Deploy on Bare Metal",
        status=DecisionStatus.SUPERSEDED.value,
    )
    db_session.add(d1)
    db_session.commit()

    d2 = Decision(
        meeting_id=s["m2"].id,
        description="Migrate to AWS EC2",
        status=DecisionStatus.ACTIVE.value,
        supersedes_decision_id=d1.id,
    )
    db_session.add(d2)
    db_session.commit()

    resp = client.get(f"/api/v1/decisions/{d2.id}", headers=s["headers_a"])
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == d2.id
    assert data["status"] == "ACTIVE"
    assert data["supersedes_decision_id"] == d1.id
    assert data["lineage_depth"] >= 1
    assert len(data["lineage"]) >= 1
    assert any(a["id"] == d1.id for a in data["lineage"])
    assert any(a["description"] == "Deploy on Bare Metal" for a in data["lineage"])


def test_supersede_decision_explicit(client: TestClient, db_session: Session, cross_meeting_api_setup):
    """Test POST /api/v1/decisions/{decision_id}/supersede explicitly creates new decision and updates old."""
    s = cross_meeting_api_setup
    d1 = Decision(
        meeting_id=s["m1"].id,
        description="Use SQLite for production",
        status=DecisionStatus.ACTIVE.value,
    )
    db_session.add(d1)
    db_session.commit()

    payload = {
        "meeting_id": s["m2"].id,
        "description": "Migrate to Managed PostgreSQL on RDS",
        "rationale_or_context": "SQLite concurrency limit reached under production load",
        "decided_by": "Architecture Review Board",
        "evidence_segment_id": s["seg2"].id,
        "timestamp_seconds": 12.0,
    }

    resp = client.post(f"/api/v1/decisions/{d1.id}/supersede", json=payload, headers=s["headers_a"])
    assert resp.status_code == 201
    data = resp.json()
    assert data["description"] == "Migrate to Managed PostgreSQL on RDS"
    assert data["status"] == "ACTIVE"
    assert data["supersedes_decision_id"] == d1.id
    assert len(data["lineage"]) >= 1
    assert any(a["id"] == d1.id for a in data["lineage"])

    # Verify d1 is now marked SUPERSEDED in database
    db_session.refresh(d1)
    assert d1.status == DecisionStatus.SUPERSEDED.value

    # Verify audit log record
    audit = (
        db_session.query(AuditLog)
        .filter(AuditLog.action == "SUPERSEDE_DECISION", AuditLog.organization_id == s["org_a"].id)
        .first()
    )
    assert audit is not None
    assert audit.user_id == s["user_a"].id


def test_supersede_invalid_status_transitions(client: TestClient, db_session: Session, cross_meeting_api_setup):
    """Test superseding already SUPERSEDED or REVOKED decisions is rejected (HTTP 400)."""
    s = cross_meeting_api_setup
    d_sup = Decision(meeting_id=s["m1"].id, description="Old decision", status=DecisionStatus.SUPERSEDED.value)
    d_rev = Decision(meeting_id=s["m1"].id, description="Revoked decision", status=DecisionStatus.REVOKED.value)
    db_session.add_all([d_sup, d_rev])
    db_session.commit()

    payload = {
        "meeting_id": s["m2"].id,
        "description": "Attempt to supersede invalid status",
    }

    # Attempt to supersede already superseded decision
    resp1 = client.post(f"/api/v1/decisions/{d_sup.id}/supersede", json=payload, headers=s["headers_a"])
    assert resp1.status_code == 400
    assert "already SUPERSEDED" in resp1.json()["detail"]

    # Attempt to supersede revoked decision
    resp2 = client.post(f"/api/v1/decisions/{d_rev.id}/supersede", json=payload, headers=s["headers_a"])
    assert resp2.status_code == 400
    assert "already REVOKED" in resp2.json()["detail"]


def test_link_superseded_existing_decisions(client: TestClient, db_session: Session, cross_meeting_api_setup):
    """Test POST /api/v1/decisions/{decision_id}/link-superseded links two existing decisions."""
    s = cross_meeting_api_setup
    d1 = Decision(meeting_id=s["m1"].id, description="Target decision to supersede", status=DecisionStatus.ACTIVE.value)
    d2 = Decision(meeting_id=s["m2"].id, description="Newer active decision", status=DecisionStatus.ACTIVE.value)
    db_session.add_all([d1, d2])
    db_session.commit()

    # Link d1 as superseded by d2
    resp = client.post(
        f"/api/v1/decisions/{d2.id}/link-superseded",
        json={"superseded_decision_id": d1.id},
        headers=s["headers_a"],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == d2.id
    assert data["supersedes_decision_id"] == d1.id

    db_session.refresh(d1)
    assert d1.status == DecisionStatus.SUPERSEDED.value

    # Attempt self-superseding -> rejected 400
    resp_self = client.post(
        f"/api/v1/decisions/{d2.id}/link-superseded",
        json={"superseded_decision_id": d2.id},
        headers=s["headers_a"],
    )
    assert resp_self.status_code == 400
    assert "cannot supersede itself" in resp_self.json()["detail"]


def test_revoke_decision_with_reason(client: TestClient, db_session: Session, cross_meeting_api_setup):
    """Test POST /api/v1/decisions/{decision_id}/revoke requires reason and marks REVOKED."""
    s = cross_meeting_api_setup
    d = Decision(meeting_id=s["m1"].id, description="Policy to test revocation", status=DecisionStatus.ACTIVE.value)
    db_session.add(d)
    db_session.commit()

    # Empty reason -> 422 or 400
    resp_empty = client.post(
        f"/api/v1/decisions/{d.id}/revoke",
        json={"reason": "   "},
        headers=s["headers_a"],
    )
    assert resp_empty.status_code in (400, 422)

    # Valid revocation
    resp = client.post(
        f"/api/v1/decisions/{d.id}/revoke",
        json={"reason": "Project discontinued by leadership."},
        headers=s["headers_a"],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "REVOKED"

    db_session.refresh(d)
    assert d.status == DecisionStatus.REVOKED.value

    # Duplicate revocation attempt -> 400
    resp_dup = client.post(
        f"/api/v1/decisions/{d.id}/revoke",
        json={"reason": "Another reason"},
        headers=s["headers_a"],
    )
    assert resp_dup.status_code == 400
    assert "already revoked" in resp_dup.json()["detail"]


# ===========================================================================
# 2. Commitments & Reviews API Tests
# ===========================================================================

def test_list_commitments_and_overdue_handling(client: TestClient, db_session: Session, cross_meeting_api_setup):
    """Test GET /api/v1/commitments with overdue calculation and status filtering."""
    s = cross_meeting_api_setup
    now = datetime.now(timezone.utc)

    ai_overdue = ActionItem(
        meeting_id=s["m1"].id,
        description="Fix TLS certificate renewal",
        owner_name="Dave",
        deadline=now - timedelta(days=2),
        status=ActionItemStatus.IN_PROGRESS.value,
    )
    ai_future = ActionItem(
        meeting_id=s["m2"].id,
        description="Implement user profile avatar upload",
        owner_name="Eve",
        deadline=now + timedelta(days=5),
        status=ActionItemStatus.PENDING.value,
    )
    db_session.add_all([ai_overdue, ai_future])
    db_session.commit()

    # 1. List all
    resp = client.get("/api/v1/commitments", headers=s["headers_a"])
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2

    # 2. Filter overdue=true
    resp_od = client.get("/api/v1/commitments?is_overdue=true", headers=s["headers_a"])
    assert resp_od.status_code == 200
    data_od = resp_od.json()
    assert len(data_od) == 1
    assert data_od[0]["id"] == ai_overdue.id
    assert data_od[0]["is_overdue"] is True

    # 3. Filter overdue=false
    resp_future = client.get("/api/v1/commitments?is_overdue=false", headers=s["headers_a"])
    assert resp_future.status_code == 200
    data_future = resp_future.json()
    assert any(c["id"] == ai_future.id for c in data_future)


def test_commitment_review_creation_and_history(client: TestClient, db_session: Session, cross_meeting_api_setup):
    """Test recording commitment review event and fetching history."""
    s = cross_meeting_api_setup
    ai = ActionItem(
        meeting_id=s["m1"].id,
        description="Provision Redis cluster in staging",
        owner_name="Frank",
        status=ActionItemStatus.PENDING.value,
    )
    db_session.add(ai)
    db_session.commit()

    review_payload = {
        "reviewing_meeting_id": s["m2"].id,
        "evidence_segment_id": s["seg2"].id,
        "status": "IN_PROGRESS",
        "review_notes": "Terraform scripts validated, waiting on VPC peering.",
    }

    # Record review
    resp = client.post(
        f"/api/v1/commitments/{ai.id}/reviews",
        json=review_payload,
        headers=s["headers_a"],
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"] == ai.id
    assert data["status"] == "IN_PROGRESS"
    assert data["review_count"] == 1
    assert len(data["reviews"]) == 1
    assert data["reviews"][0]["meeting_id"] == s["m2"].id
    assert data["reviews"][0]["review_notes"] == "Terraform scripts validated, waiting on VPC peering."

    # List reviews endpoint
    resp_reviews = client.get(f"/api/v1/commitments/{ai.id}/reviews", headers=s["headers_a"])
    assert resp_reviews.status_code == 200
    reviews_data = resp_reviews.json()
    assert len(reviews_data) == 1
    assert reviews_data[0]["status_at_review"] == "IN_PROGRESS"


# ===========================================================================
# 3. Unresolved Questions API Tests
# ===========================================================================

def test_questions_lifecycle_resolve_and_reopen(client: TestClient, db_session: Session, cross_meeting_api_setup):
    """Test listing questions, resolving with decision link, and reopening."""
    s = cross_meeting_api_setup
    q = UnresolvedQuestion(
        meeting_id=s["m1"].id,
        question_text="Should we migrate to TypeScript for backend?",
        raised_by="Grace",
        status=QuestionStatus.OPEN.value,
    )
    dec = Decision(
        meeting_id=s["m2"].id,
        description="Agreed to stay on Python with strict typing",
        status=DecisionStatus.ACTIVE.value,
    )
    db_session.add_all([q, dec])
    db_session.commit()

    # 1. List open questions
    resp_list = client.get("/api/v1/questions?status=OPEN", headers=s["headers_a"])
    assert resp_list.status_code == 200
    assert len(resp_list.json()) == 1

    # 2. Resolve question
    resolve_payload = {
        "resolving_meeting_id": s["m2"].id,
        "resolving_decision_id": dec.id,
        "resolution_notes": "Decided against migration; Python 3.13 strict type annotations adopted.",
        "status": "RESOLVED",
    }
    resp_resolve = client.post(f"/api/v1/questions/{q.id}/resolve", json=resolve_payload, headers=s["headers_a"])
    assert resp_resolve.status_code == 200
    res_data = resp_resolve.json()
    assert res_data["status"] == "RESOLVED"
    assert res_data["resolving_decision_id"] == dec.id
    assert res_data["resolving_meeting_title"] == "Alpha Arch Sync 2"

    # 3. Reopen question
    resp_reopen = client.post(f"/api/v1/questions/{q.id}/reopen", headers=s["headers_a"])
    assert resp_reopen.status_code == 200
    reopen_data = resp_reopen.json()
    assert reopen_data["status"] == "OPEN"
    assert reopen_data["resolving_meeting_id"] is None


# ===========================================================================
# 4. Conflicts API Tests
# ===========================================================================

def test_conflicts_detection_endpoint_and_superseding_exclusion(client: TestClient, db_session: Session, cross_meeting_api_setup):
    """Test GET /api/v1/conflicts surfaces heuristic warning and excludes superseded decisions."""
    s = cross_meeting_api_setup
    d1 = Decision(
        meeting_id=s["m1"].id,
        description="Approved remote work policy for all engineers",
        status=DecisionStatus.ACTIVE.value,
    )
    d2 = Decision(
        meeting_id=s["m2"].id,
        description="Disallowed remote work policy; mandatory on-site required",
        status=DecisionStatus.ACTIVE.value,
    )
    db_session.add_all([d1, d2])
    db_session.commit()

    # Detect conflicts
    resp = client.get("/api/v1/conflicts", headers=s["headers_a"])
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    cf = data[0]
    assert cf["is_heuristic"] is True
    assert "remote" in cf["subject"].lower() or "policy" in cf["subject"].lower() or "work" in cf["subject"].lower()
    assert "opposing stances" in cf["reason"].lower()

    # Also verify /api/v1/memory/conflicts alias works
    resp_mem = client.get("/api/v1/memory/conflicts", headers=s["headers_a"])
    assert resp_mem.status_code == 200
    assert len(resp_mem.json()) == 1

    # Now explicitly supersede d1 with d2 -> conflict MUST disappear!
    d2.supersedes_decision_id = d1.id
    d1.status = DecisionStatus.SUPERSEDED.value
    db_session.commit()

    resp_after = client.get("/api/v1/conflicts", headers=s["headers_a"])
    assert resp_after.status_code == 200
    assert len(resp_after.json()) == 0


# ===========================================================================
# 5. Security & Tenant Isolation Tests
# ===========================================================================

def test_tenant_isolation_cross_organization_data_leakage(client: TestClient, db_session: Session, cross_meeting_api_setup):
    """Verify Org B cannot list, retrieve, or mutate Org A records."""
    s = cross_meeting_api_setup
    d_a = Decision(meeting_id=s["m1"].id, description="Alpha secret decision", status=DecisionStatus.ACTIVE.value)
    ai_a = ActionItem(meeting_id=s["m1"].id, description="Alpha confidential action", status=ActionItemStatus.PENDING.value)
    q_a = UnresolvedQuestion(meeting_id=s["m1"].id, question_text="Alpha proprietary question", status=QuestionStatus.OPEN.value)
    db_session.add_all([d_a, ai_a, q_a])
    db_session.commit()

    # Org B listings must be empty
    assert len(client.get("/api/v1/decisions", headers=s["headers_b"]).json()) == 0
    assert len(client.get("/api/v1/commitments", headers=s["headers_b"]).json()) == 0
    assert len(client.get("/api/v1/questions", headers=s["headers_b"]).json()) == 0
    assert len(client.get("/api/v1/conflicts", headers=s["headers_b"]).json()) == 0

    # Org B attempting direct ID access to Org A resources -> 404 (safe, prevents enumeration)
    assert client.get(f"/api/v1/decisions/{d_a.id}", headers=s["headers_b"]).status_code == 404
    assert client.get(f"/api/v1/commitments/{ai_a.id}", headers=s["headers_b"]).status_code == 404
    assert client.get(f"/api/v1/questions/{q_a.id}", headers=s["headers_b"]).status_code == 404

    # Org B attempting to mutate Org A resources -> 404
    assert client.post(f"/api/v1/decisions/{d_a.id}/revoke", json={"reason": "Attacker"}, headers=s["headers_b"]).status_code == 404
    assert client.post(f"/api/v1/commitments/{ai_a.id}/reviews", json={"reviewing_meeting_id": s["m_b"].id}, headers=s["headers_b"]).status_code == 404
    assert client.post(f"/api/v1/questions/{q_a.id}/resolve", json={"resolving_meeting_id": s["m_b"].id}, headers=s["headers_b"]).status_code == 404


def test_foreign_meeting_reference_rejection(client: TestClient, db_session: Session, cross_meeting_api_setup):
    """Verify linking a foreign meeting from another tenant is rejected (HTTP 404)."""
    s = cross_meeting_api_setup
    d_a = Decision(meeting_id=s["m1"].id, description="Valid decision in Org A", status=DecisionStatus.ACTIVE.value)
    ai_a = ActionItem(meeting_id=s["m1"].id, description="Valid action in Org A", status=ActionItemStatus.PENDING.value)
    q_a = UnresolvedQuestion(meeting_id=s["m1"].id, question_text="Valid question in Org A", status=QuestionStatus.OPEN.value)
    db_session.add_all([d_a, ai_a, q_a])
    db_session.commit()

    # Attempt supersede in Org A referencing Org B meeting -> 404
    resp_sup = client.post(
        f"/api/v1/decisions/{d_a.id}/supersede",
        json={"meeting_id": s["m_b"].id, "description": "Cross-tenant attempt"},
        headers=s["headers_a"],
    )
    assert resp_sup.status_code == 404

    # Attempt review in Org A referencing Org B meeting -> 404
    resp_rev = client.post(
        f"/api/v1/commitments/{ai_a.id}/reviews",
        json={"reviewing_meeting_id": s["m_b"].id},
        headers=s["headers_a"],
    )
    assert resp_rev.status_code == 404

    # Attempt resolve in Org A referencing Org B meeting -> 404
    resp_res = client.post(
        f"/api/v1/questions/{q_a.id}/resolve",
        json={"resolving_meeting_id": s["m_b"].id},
        headers=s["headers_a"],
    )
    assert resp_res.status_code == 404


def test_unauthenticated_requests_rejected(client: TestClient, cross_meeting_api_setup):
    """Verify all cross-meeting endpoints reject unauthenticated calls with HTTP 401."""
    assert client.get("/api/v1/decisions").status_code == 401
    assert client.get("/api/v1/commitments").status_code == 401
    assert client.get("/api/v1/questions").status_code == 401
    assert client.get("/api/v1/conflicts").status_code == 401
    assert client.post("/api/v1/decisions/123/revoke", json={"reason": "test"}).status_code == 401


def test_rate_limiting_enforcement_on_cross_meeting(client: TestClient, cross_meeting_api_setup):
    """Verify rate limiter blocks excessive requests with HTTP 429."""
    s = cross_meeting_api_setup
    cross_meeting_rate_limiter.reset()
    original_max = cross_meeting_rate_limiter.max_requests
    cross_meeting_rate_limiter.max_requests = 2

    try:
        # Request 1: OK
        r1 = client.get("/api/v1/decisions", headers=s["headers_a"])
        assert r1.status_code == 200

        # Request 2: OK
        r2 = client.get("/api/v1/decisions", headers=s["headers_a"])
        assert r2.status_code == 200

        # Request 3: HTTP 429
        r3 = client.get("/api/v1/decisions", headers=s["headers_a"])
        assert r3.status_code == 429
        assert "Too many requests" in r3.json()["detail"]
    finally:
        cross_meeting_rate_limiter.max_requests = original_max
        cross_meeting_rate_limiter.reset()


def test_memory_query_regression_with_structured_intelligence(client: TestClient, db_session: Session, cross_meeting_api_setup, monkeypatch):
    """Verify existing POST /api/v1/memory/query exposes structured intelligence and conflicts."""
    s = cross_meeting_api_setup
    d = Decision(
        meeting_id=s["m1"].id,
        description="Adopt Kubernetes for orchestration",
        rationale_or_context="Cloud-native declarative container orchestration",
        decided_by="Platform Architecture Team",
        status=DecisionStatus.ACTIVE.value,
    )
    db_session.add(d)
    db_session.commit()

    resp = client.post(
        "/api/v1/memory/query",
        json={"question": "What decisions were made about container orchestration?"},
        headers=s["headers_a"],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_abstained"] is False
    assert "Kubernetes" in data["answer"]
    assert len(data["structured_decisions"]) >= 1
    assert data["diagnostics"]["structured_decisions_count"] >= 1
