"""Tests for Phase 8.8.3: Cross-Meeting Intelligence Services.

Verifies:
1. DecisionLifecycleService:
   - Cross-meeting listing with status, meeting, and date filtering
   - Lineage reconstruction and ancestry chain
   - Explicit decision superseding and audit logging
   - Existing decision linking and revocation
   - Cross-tenant rejection
2. CommitmentTrackingService:
   - Cross-meeting listing and filtering (status, owner, meeting)
   - Dynamic overdue calculation logic
   - Append-only review history accumulation across multiple meetings
   - Origin immutability vs latest review cache
   - Cross-tenant rejection
3. UnresolvedQuestionService:
   - Cross-meeting listing and filtering (OPEN, RESOLVED, DEFERRED)
   - Explicit question resolution with resolving meeting/decision and notes
   - Question reopening lifecycle
   - Cross-tenant rejection
4. Transaction integrity and audit logging.
"""

from datetime import datetime, timezone, timedelta
import pytest
from sqlalchemy.orm import Session

from app.models.identity import Organization, User
from app.models.meeting import Meeting, InputSourceType, MeetingStatus
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
from app.models.memory import AuditLog
from app.schemas.intelligence import DecisionCreate
from app.services.intelligence.cross_meeting import (
    DecisionLifecycleService,
    CommitmentTrackingService,
    UnresolvedQuestionService,
)
from app.services.intelligence.lineage import (
    LineageValidationError,
    CrossTenantLineageError,
    CircularLineageError,
    SelfSupersedingError,
)


def _make_meeting(db: Session, org: Organization, title: str = "Test Meeting", days_ago: int = 0) -> Meeting:
    """Helper to create a meeting for a specific organization with configurable date."""
    meeting_date = datetime.now(timezone.utc) - timedelta(days=days_ago)
    m = Meeting(
        organization_id=org.id,
        title=title,
        input_source_type=InputSourceType.AUDIO.value,
        status=MeetingStatus.COMPLETED.value,
        created_at=meeting_date,
        updated_at=meeting_date,
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


def _make_transcript_segment(db: Session, meeting: Meeting, text: str = "Evidence text") -> TranscriptSegment:
    """Helper to create a privacy-safe transcript and segment."""
    transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        language_detected="en",
    )
    db.add(transcript)
    db.flush()

    segment = TranscriptSegment(
        transcript_id=transcript.id,
        sequence_order=1,
        start_time=10.0,
        end_time=15.0,
        text=text,
        language_code="en",
    )
    db.add(segment)
    db.commit()
    db.refresh(segment)
    return segment


# =========================================================================
# 1. Decision Lifecycle Service Tests
# =========================================================================

def test_list_decisions_cross_meeting_with_filters(db_session, test_org_a, test_org_b):
    """Verify listing decisions across meetings for an organization with filtering."""
    m1 = _make_meeting(db_session, test_org_a, "Q1 Planning", days_ago=10)
    m2 = _make_meeting(db_session, test_org_a, "Q2 Review", days_ago=5)
    m_other = _make_meeting(db_session, test_org_b, "Org B Meeting", days_ago=2)

    # Decisions for Org A
    dec1 = Decision(meeting_id=m1.id, description="Adopt Kubernetes for orchestration", status=DecisionStatus.ACTIVE.value)
    dec2 = Decision(meeting_id=m1.id, description="Use PostgreSQL for DB", status=DecisionStatus.SUPERSEDED.value)
    dec3 = Decision(meeting_id=m2.id, description="Revoke legacy LDAP auth", status=DecisionStatus.REVOKED.value)

    # Decision for Org B (should never appear in Org A queries)
    dec_b = Decision(meeting_id=m_other.id, description="Org B Internal Policy", status=DecisionStatus.ACTIVE.value)

    db_session.add_all([dec1, dec2, dec3, dec_b])
    db_session.commit()

    # 1. List all for Org A
    all_a = DecisionLifecycleService.list_decisions(db_session, organization_id=test_org_a.id)
    assert len(all_a) == 3
    assert all(d.meeting_title in ["Q1 Planning", "Q2 Review"] for d in all_a)

    # 2. Filter by status ACTIVE
    active_a = DecisionLifecycleService.list_decisions(db_session, organization_id=test_org_a.id, status=DecisionStatus.ACTIVE)
    assert len(active_a) == 1
    assert active_a[0].id == dec1.id

    # 3. Filter by meeting_id
    m2_decisions = DecisionLifecycleService.list_decisions(db_session, organization_id=test_org_a.id, meeting_id=m2.id)
    assert len(m2_decisions) == 1
    assert m2_decisions[0].id == dec3.id

    # 4. Search filter
    search_res = DecisionLifecycleService.list_decisions(db_session, organization_id=test_org_a.id, search="Kubernetes")
    assert len(search_res) == 1
    assert search_res[0].id == dec1.id

    # 5. Org B isolation
    all_b = DecisionLifecycleService.list_decisions(db_session, organization_id=test_org_b.id)
    assert len(all_b) == 1
    assert all_b[0].id == dec_b.id


def test_get_decision_with_full_lineage(db_session, test_org_a):
    """Verify decision ancestry reconstruction: Root A -> Superseded by B -> Superseded by C."""
    m1 = _make_meeting(db_session, test_org_a, "Design v1")
    m2 = _make_meeting(db_session, test_org_a, "Design v2")
    m3 = _make_meeting(db_session, test_org_a, "Design v3")

    dec_a = Decision(meeting_id=m1.id, description="Use REST API v1")
    dec_b = Decision(meeting_id=m2.id, description="Migrate to GraphQL v2")
    dec_c = Decision(meeting_id=m3.id, description="Adopt gRPC for microservices v3")
    db_session.add_all([dec_a, dec_b, dec_c])
    db_session.commit()

    # Link lineage: A superseded by B, B superseded by C
    DecisionLifecycleService.supersede_existing_decision(
        db_session, organization_id=test_org_a.id, target_decision_id=dec_a.id, superseding_decision_id=dec_b.id
    )
    DecisionLifecycleService.supersede_existing_decision(
        db_session, organization_id=test_org_a.id, target_decision_id=dec_b.id, superseding_decision_id=dec_c.id
    )

    detail_c = DecisionLifecycleService.get_decision_with_lineage(
        db_session, organization_id=test_org_a.id, decision_id=dec_c.id
    )
    assert detail_c.id == dec_c.id
    assert detail_c.status == DecisionStatus.ACTIVE.value
    assert detail_c.lineage_depth == 3
    assert [d.id for d in detail_c.lineage] == [dec_a.id, dec_b.id, dec_c.id]

    # Verify B shows it was superseded by C
    detail_b = DecisionLifecycleService.get_decision_with_lineage(
        db_session, organization_id=test_org_a.id, decision_id=dec_b.id
    )
    assert detail_b.status == DecisionStatus.SUPERSEDED.value
    assert detail_b.superseded_by_decision_id == dec_c.id


def test_supersede_decision_explicit_and_audit_logging(db_session, test_org_a, org_admin_user):
    """Verify explicit superseding creates new decision, updates target, and logs audit trail."""
    m1 = _make_meeting(db_session, test_org_a, "Architecture Review 1")
    m2 = _make_meeting(db_session, test_org_a, "Architecture Review 2")

    dec_orig = Decision(meeting_id=m1.id, description="Deploy on AWS EC2", status=DecisionStatus.ACTIVE.value)
    db_session.add(dec_orig)
    db_session.commit()

    new_data = DecisionCreate(
        description="Deploy on AWS ECS Fargate",
        rationale_or_context="Reduce infrastructure maintenance overhead",
        decided_by="Infrastructure Team",
    )

    new_dec = DecisionLifecycleService.supersede_decision_explicit(
        db_session,
        organization_id=test_org_a.id,
        target_decision_id=dec_orig.id,
        new_decision_data=new_data,
        meeting_id=m2.id,
        user_id=org_admin_user.id,
        ip_address="192.168.1.100",
    )

    db_session.refresh(dec_orig)
    assert dec_orig.status == DecisionStatus.SUPERSEDED.value
    assert new_dec.status == DecisionStatus.ACTIVE.value
    assert new_dec.supersedes_decision_id == dec_orig.id

    # Verify AuditLog created
    audit = (
        db_session.query(AuditLog)
        .filter(AuditLog.action == "SUPERSEDE_DECISION", AuditLog.entity_id == new_dec.id)
        .first()
    )
    assert audit is not None
    assert audit.organization_id == test_org_a.id
    assert audit.user_id == org_admin_user.id
    assert audit.ip_address == "192.168.1.100"


def test_revoke_decision_with_reason(db_session, test_org_a, org_admin_user):
    """Verify revoking a decision sets status to REVOKED and logs audit record."""
    m = _make_meeting(db_session, test_org_a, "Emergency Sync")
    dec = Decision(meeting_id=m.id, description="Cancel project kickoff", status=DecisionStatus.ACTIVE.value)
    db_session.add(dec)
    db_session.commit()

    revoked = DecisionLifecycleService.revoke_decision(
        db_session,
        organization_id=test_org_a.id,
        decision_id=dec.id,
        reason="Budget restored by executive sponsor",
        user_id=org_admin_user.id,
    )

    assert revoked.status == DecisionStatus.REVOKED.value
    assert "Budget restored by executive sponsor" in revoked.rationale_or_context

    audit = (
        db_session.query(AuditLog)
        .filter(AuditLog.action == "REVOKE_DECISION", AuditLog.entity_id == dec.id)
        .first()
    )
    assert audit is not None
    assert audit.organization_id == test_org_a.id


def test_cross_organization_decision_access_rejected(db_session, test_org_a, test_org_b):
    """Verify tenant isolation rejects cross-tenant decision retrieval and superseding."""
    m_a = _make_meeting(db_session, test_org_a, "Org A Meeting")
    m_b = _make_meeting(db_session, test_org_b, "Org B Meeting")

    dec_b = Decision(meeting_id=m_b.id, description="Org B Confidential Strategy")
    db_session.add(dec_b)
    db_session.commit()

    # Org A cannot get Org B's decision
    with pytest.raises(CrossTenantLineageError):
        DecisionLifecycleService.get_decision_with_lineage(
            db_session, organization_id=test_org_a.id, decision_id=dec_b.id
        )

    # Org A cannot supersede Org B's decision
    new_data = DecisionCreate(description="Org A supersedes Org B")
    with pytest.raises(CrossTenantLineageError):
        DecisionLifecycleService.supersede_decision_explicit(
            db_session,
            organization_id=test_org_a.id,
            target_decision_id=dec_b.id,
            new_decision_data=new_data,
            meeting_id=m_a.id,
        )


# =========================================================================
# 2. Commitment Tracking Service Tests
# =========================================================================

def test_commitment_tracking_filters_and_overdue(db_session, test_org_a):
    """Verify commitment listing, owner filtering, and dynamic overdue calculation."""
    m = _make_meeting(db_session, test_org_a, "Sprint Backlog")
    now = datetime.now(timezone.utc)

    # 1. Overdue item: past deadline, PENDING
    ai_overdue = ActionItem(
        meeting_id=m.id,
        description="Fix critical security CVE",
        owner_name="Alice Smith",
        deadline=now - timedelta(days=2),
        status=ActionItemStatus.PENDING.value,
    )
    # 2. Not overdue: past deadline, but COMPLETED
    ai_completed = ActionItem(
        meeting_id=m.id,
        description="Implement user login",
        owner_name="Alice Smith",
        deadline=now - timedelta(days=5),
        status=ActionItemStatus.COMPLETED.value,
    )
    # 3. Future item: future deadline, PENDING
    ai_future = ActionItem(
        meeting_id=m.id,
        description="Write end-to-end integration tests",
        owner_name="Bob Jones",
        deadline=now + timedelta(days=7),
        status=ActionItemStatus.PENDING.value,
    )
    # 4. No deadline
    ai_no_deadline = ActionItem(
        meeting_id=m.id,
        description="Update readme documentation",
        owner_name="Charlie",
        deadline=None,
        status=ActionItemStatus.PENDING.value,
    )
    db_session.add_all([ai_overdue, ai_completed, ai_future, ai_no_deadline])
    db_session.commit()

    # Filter: is_overdue=True
    overdue_items = CommitmentTrackingService.list_commitments(
        db_session, organization_id=test_org_a.id, is_overdue=True
    )
    assert len(overdue_items) == 1
    assert overdue_items[0].id == ai_overdue.id
    assert overdue_items[0].is_overdue is True

    # Filter: owner_name="Alice"
    alice_items = CommitmentTrackingService.list_commitments(
        db_session, organization_id=test_org_a.id, owner_name="Alice"
    )
    assert len(alice_items) == 2

    # Filter: status=COMPLETED
    completed_items = CommitmentTrackingService.list_commitments(
        db_session, organization_id=test_org_a.id, status=ActionItemStatus.COMPLETED
    )
    assert len(completed_items) == 1
    assert completed_items[0].id == ai_completed.id
    assert completed_items[0].is_overdue is False


def test_commitment_review_and_history_accumulation(db_session, test_org_a, org_admin_user):
    """Verify reviewing an item accumulates append-only ActionItemReview history across meetings."""
    m1 = _make_meeting(db_session, test_org_a, "Sprint 1 Kickoff", days_ago=14)
    m2 = _make_meeting(db_session, test_org_a, "Sprint 2 Sync", days_ago=7)
    m3 = _make_meeting(db_session, test_org_a, "Sprint 3 Demo", days_ago=1)

    seg1 = _make_transcript_segment(db_session, m1, "Dave committed to setup Kafka by next week.")
    seg2 = _make_transcript_segment(db_session, m2, "Dave reported Kafka cluster is half-provisioned.")
    seg3 = _make_transcript_segment(db_session, m3, "Dave confirmed Kafka cluster is fully tested and live.")

    # Create original ActionItem in Meeting 1
    ai = ActionItem(
        meeting_id=m1.id,
        description="Setup Kafka cluster",
        owner_name="Dave",
        evidence_segment_id=seg1.id,
        timestamp_seconds=12.0,
        status=ActionItemStatus.PENDING.value,
    )
    db_session.add(ai)
    db_session.commit()
    db_session.refresh(ai)

    # Review 1: In Meeting 2 -> IN_PROGRESS
    CommitmentTrackingService.review_commitment(
        db_session,
        organization_id=test_org_a.id,
        action_item_id=ai.id,
        reviewing_meeting_id=m2.id,
        evidence_segment_id=seg2.id,
        status=ActionItemStatus.IN_PROGRESS,
        review_notes="Waiting on VPC peering approval",
        reviewer_user_id=org_admin_user.id,
    )

    # Review 2: In Meeting 3 -> COMPLETED
    CommitmentTrackingService.review_commitment(
        db_session,
        organization_id=test_org_a.id,
        action_item_id=ai.id,
        reviewing_meeting_id=m3.id,
        evidence_segment_id=seg3.id,
        status=ActionItemStatus.COMPLETED,
        review_notes="VPC peering active, topics created",
        reviewer_user_id=org_admin_user.id,
    )

    # Retrieve full commitment details with review history
    detail = CommitmentTrackingService.get_commitment_with_history(
        db_session, organization_id=test_org_a.id, action_item_id=ai.id
    )

    # Origin must remain strictly intact
    assert detail.meeting_id == m1.id
    assert detail.origin_meeting_title == "Sprint 1 Kickoff"
    assert detail.evidence_segment_id == seg1.id
    assert detail.timestamp_seconds == 12.0

    # Latest review pointer must match Meeting 3
    assert detail.status == ActionItemStatus.COMPLETED.value
    assert detail.last_reviewed_meeting_id == m3.id
    assert detail.last_reviewed_meeting_title == "Sprint 3 Demo"
    assert detail.last_evidence_segment_id == seg3.id

    # Complete review history verification
    assert detail.review_count == 2
    assert len(detail.reviews) == 2

    # First review in history
    rev1 = detail.reviews[0]
    assert rev1.meeting_id == m2.id
    assert rev1.meeting_title == "Sprint 2 Sync"
    assert rev1.status_at_review == ActionItemStatus.IN_PROGRESS.value
    assert rev1.review_notes == "Waiting on VPC peering approval"
    assert "half-provisioned" in rev1.evidence_quote

    # Second review in history
    rev2 = detail.reviews[1]
    assert rev2.meeting_id == m3.id
    assert rev2.meeting_title == "Sprint 3 Demo"
    assert rev2.status_at_review == ActionItemStatus.COMPLETED.value
    assert rev2.review_notes == "VPC peering active, topics created"
    assert "fully tested and live" in rev2.evidence_quote

    # Check AuditLog
    audits = (
        db_session.query(AuditLog)
        .filter(AuditLog.action == "REVIEW_COMMITMENT", AuditLog.entity_id == ai.id)
        .all()
    )
    assert len(audits) == 2


def test_cross_organization_commitment_rejected(db_session, test_org_a, test_org_b):
    """Verify cross-organization commitment access and review are rejected."""
    m_a = _make_meeting(db_session, test_org_a, "Org A Standup")
    m_b = _make_meeting(db_session, test_org_b, "Org B Standup")

    ai_a = ActionItem(meeting_id=m_a.id, description="Org A Task")
    db_session.add(ai_a)
    db_session.commit()

    # Org B cannot view Org A commitment
    with pytest.raises(CrossTenantLineageError):
        CommitmentTrackingService.get_commitment_with_history(
            db_session, organization_id=test_org_b.id, action_item_id=ai_a.id
        )

    # Org B cannot review Org A commitment
    with pytest.raises(CrossTenantLineageError):
        CommitmentTrackingService.review_commitment(
            db_session,
            organization_id=test_org_b.id,
            action_item_id=ai_a.id,
            reviewing_meeting_id=m_b.id,
        )


# =========================================================================
# 3. Unresolved Question Service Tests
# =========================================================================

def test_list_and_resolve_question(db_session, test_org_a, org_admin_user):
    """Verify question listing, explicit resolution with decision link, and reopening."""
    m1 = _make_meeting(db_session, test_org_a, "Sprint Planning")
    m2 = _make_meeting(db_session, test_org_a, "Sprint Retro")

    q = UnresolvedQuestion(
        meeting_id=m1.id,
        question_text="Should we migrate to Tailwind CSS?",
        raised_by="Frontend Dev",
        status=QuestionStatus.OPEN.value,
    )
    dec = Decision(
        meeting_id=m2.id,
        description="Approved Tailwind CSS migration for v2",
    )
    db_session.add_all([q, dec])
    db_session.commit()

    # 1. List open questions
    open_questions = UnresolvedQuestionService.list_questions(
        db_session, organization_id=test_org_a.id, status=QuestionStatus.OPEN
    )
    assert len(open_questions) == 1
    assert open_questions[0].id == q.id
    assert open_questions[0].status == "OPEN"

    # 2. Resolve question explicitly
    resolved_q = UnresolvedQuestionService.resolve_question(
        db_session,
        organization_id=test_org_a.id,
        question_id=q.id,
        resolving_meeting_id=m2.id,
        resolving_decision_id=dec.id,
        resolution_notes="Adopted after benchmarking build times.",
        user_id=org_admin_user.id,
        ip_address="10.0.0.1",
    )
    assert resolved_q.status == QuestionStatus.RESOLVED.value
    assert resolved_q.resolving_meeting_id == m2.id
    assert resolved_q.resolving_decision_id == dec.id
    assert resolved_q.resolution_notes == "Adopted after benchmarking build times."
    assert resolved_q.resolved_at is not None

    # Check question detail view
    detail = UnresolvedQuestionService.get_question(
        db_session, organization_id=test_org_a.id, question_id=q.id
    )
    assert detail.status == "RESOLVED"
    assert detail.origin_meeting_title == "Sprint Planning"
    assert detail.resolving_meeting_title == "Sprint Retro"
    assert detail.resolving_decision_description == "Approved Tailwind CSS migration for v2"

    # 3. Reopen question
    reopened = UnresolvedQuestionService.reopen_question(
        db_session, organization_id=test_org_a.id, question_id=q.id, user_id=org_admin_user.id
    )
    assert reopened.status == QuestionStatus.OPEN.value
    assert reopened.resolved_at is None
    assert reopened.resolving_meeting_id is None
    assert reopened.resolving_decision_id is None
    assert "[REOPENED]" in reopened.resolution_notes


def test_cross_organization_question_rejected(db_session, test_org_a, test_org_b):
    """Verify tenant isolation rejects cross-tenant question resolution and viewing."""
    m_a = _make_meeting(db_session, test_org_a, "Org A Meeting")
    m_b = _make_meeting(db_session, test_org_b, "Org B Meeting")

    q_a = UnresolvedQuestion(meeting_id=m_a.id, question_text="Org A Question")
    db_session.add(q_a)
    db_session.commit()

    # Org B cannot view Org A question
    with pytest.raises(CrossTenantLineageError):
        UnresolvedQuestionService.get_question(
            db_session, organization_id=test_org_b.id, question_id=q_a.id
        )

    # Org B cannot resolve Org A question
    with pytest.raises(CrossTenantLineageError):
        UnresolvedQuestionService.resolve_question(
            db_session,
            organization_id=test_org_b.id,
            question_id=q_a.id,
            resolving_meeting_id=m_b.id,
        )


# =========================================================================
# 4. Transaction Integrity and Rollback Tests
# =========================================================================

def test_transaction_rollback_on_error(db_session, test_org_a):
    """Verify database rolls back cleanly without partial commits when an error occurs."""
    m1 = _make_meeting(db_session, test_org_a, "Meeting 1")
    dec_a = Decision(meeting_id=m1.id, description="Original policy")
    db_session.add(dec_a)
    db_session.commit()

    # Try invalid self-superseding with invalid meeting
    with pytest.raises(LineageValidationError):
        DecisionLifecycleService.supersede_existing_decision(
            db_session,
            organization_id=test_org_a.id,
            target_decision_id=dec_a.id,
            superseding_decision_id="non-existent-id",
        )

    # Ensure dec_a status remains unchanged
    db_session.rollback()
    db_session.refresh(dec_a)
    assert dec_a.status == DecisionStatus.ACTIVE.value
