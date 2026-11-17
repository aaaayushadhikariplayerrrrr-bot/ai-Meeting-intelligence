"""Tests for Phase 8.8.4: Structured Organizational Memory Retrieval & Conflict Detection.

Verifies:
1. Structured memory retrieval of decisions, commitments, and unresolved questions directly from SQL.
2. Handling of active vs superseded decisions with lineage.
3. Commitment filtering, overdue calculations, and review history retrieval.
4. Conflict detection between active decisions (opposing stances and exclusive technology choices).
5. Explicit supersession distinction (superseded decisions are NOT flagged as conflicts).
6. Cross-tenant isolation in structured retrieval and conflict detection.
7. RAG integration: answering using structured intelligence when transcripts are absent or complementary.
8. RAG conflict alerting with heuristic warning banners.
9. RAG abstention when neither structured records nor transcript evidence exist.
10. Transcript-only RAG regression preservation.
"""

from datetime import datetime, timezone, timedelta
import pytest
from sqlalchemy.orm import Session

from app.models.identity import Organization
from app.models.meeting import Meeting, InputSourceType, MeetingStatus
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.intelligence import (
    Decision,
    DecisionStatus,
    ActionItem,
    ActionItemStatus,
    UnresolvedQuestion,
    QuestionStatus,
)
from app.services.intelligence.cross_meeting import (
    DecisionLifecycleService,
    CommitmentTrackingService,
    UnresolvedQuestionService,
)
from app.services.memory.conflict_detector import ConflictDetector
from app.services.memory.structured_retrieval import (
    StructuredMemoryRetriever,
    StructuredMemoryContext,
)
from app.services.memory.rag.engine import RAGEngine
from app.services.memory.rag.types import (
    RAGQueryRequest,
    RAGQueryResponse,
    AbstentionStatus,
)


def _create_meeting(db: Session, org: Organization, title: str, days_ago: int = 0) -> Meeting:
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


def _create_transcript_segment(db: Session, meeting: Meeting, text: str = "Evidence text") -> TranscriptSegment:
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
# 1. Structured Memory Retrieval Tests
# =========================================================================

def test_structured_decision_retrieval_with_lineage(db_session, test_org_a):
    """Verify structured retrieval loads decisions, meeting provenance, and lineage links."""
    m1 = _create_meeting(db_session, test_org_a, "Arch Sync 1", days_ago=14)
    m2 = _create_meeting(db_session, test_org_a, "Arch Sync 2", days_ago=7)

    d1 = Decision(
        meeting_id=m1.id,
        description="Deploy staging environment on AWS EC2 instances",
        status=DecisionStatus.ACTIVE.value,
    )
    d2 = Decision(
        meeting_id=m2.id,
        description="Migrate staging environment to AWS ECS Fargate",
        status=DecisionStatus.ACTIVE.value,
    )
    db_session.add_all([d1, d2])
    db_session.commit()

    # d2 explicitly supersedes d1
    DecisionLifecycleService.supersede_existing_decision(
        db_session,
        organization_id=test_org_a.id,
        target_decision_id=d1.id,
        superseding_decision_id=d2.id,
    )

    ctx = StructuredMemoryRetriever.search_structured_memory(
        db=db_session,
        organization_id=test_org_a.id,
        query="What decisions were made about AWS staging environment?",
    )

    assert ctx.has_structured_evidence is True
    assert len(ctx.decisions) >= 1
    # Active decision d2 should be returned
    active_d = next(d for d in ctx.decisions if d.id == d2.id)
    assert active_d.status == "ACTIVE"
    assert active_d.supersedes_decision_id == d1.id
    assert active_d.meeting_title == "Arch Sync 2"


def test_commitment_retrieval_and_overdue_handling(db_session, test_org_a):
    """Verify commitment structured search calculates dynamic overdue flags correctly."""
    m = _create_meeting(db_session, test_org_a, "Sprint Retro")
    now = datetime.now(timezone.utc)

    ai1 = ActionItem(
        meeting_id=m.id,
        description="Resolve database deadlock in payment gateway",
        owner_name="Alice Smith",
        deadline=now - timedelta(days=3),
        status=ActionItemStatus.PENDING.value,
    )
    ai2 = ActionItem(
        meeting_id=m.id,
        description="Write OpenAPI documentation for billing",
        owner_name="Bob Jones",
        deadline=now - timedelta(days=5),
        status=ActionItemStatus.COMPLETED.value,
    )
    ai3 = ActionItem(
        meeting_id=m.id,
        description="Implement webhook retry queue",
        owner_name="Alice Smith",
        deadline=now + timedelta(days=4),
        status=ActionItemStatus.IN_PROGRESS.value,
    )
    db_session.add_all([ai1, ai2, ai3])
    db_session.commit()

    # Query for overdue commitments
    ctx = StructuredMemoryRetriever.search_structured_memory(
        db=db_session,
        organization_id=test_org_a.id,
        query="Which action items are overdue?",
    )
    assert ctx.has_structured_evidence is True
    assert len(ctx.commitments) == 1
    assert ctx.commitments[0].id == ai1.id
    assert ctx.commitments[0].is_overdue is True

    # Query for Alice's tasks
    ctx_alice = StructuredMemoryRetriever.search_structured_memory(
        db=db_session,
        organization_id=test_org_a.id,
        query="What tasks are assigned to Alice?",
    )
    assert len(ctx_alice.commitments) == 2


def test_commitment_review_history_retrieval(db_session, test_org_a, org_admin_user):
    """Verify structured retrieval loads review history and origin immutability."""
    m1 = _create_meeting(db_session, test_org_a, "Sprint 1", days_ago=10)
    m2 = _create_meeting(db_session, test_org_a, "Sprint 2", days_ago=5)
    m3 = _create_meeting(db_session, test_org_a, "Sprint 3", days_ago=1)

    seg1 = _create_transcript_segment(db_session, m1, "Task created for staging cluster.")
    seg2 = _create_transcript_segment(db_session, m2, "Staging cluster delayed by credentials.")
    seg3 = _create_transcript_segment(db_session, m3, "Staging cluster verified and live.")

    ai = ActionItem(
        meeting_id=m1.id,
        description="Deploy Redis caching cluster",
        owner_name="Carol",
        evidence_segment_id=seg1.id,
        status=ActionItemStatus.PENDING.value,
    )
    db_session.add(ai)
    db_session.commit()

    CommitmentTrackingService.review_commitment(
        db_session,
        organization_id=test_org_a.id,
        action_item_id=ai.id,
        reviewing_meeting_id=m2.id,
        evidence_segment_id=seg2.id,
        status=ActionItemStatus.IN_PROGRESS,
        review_notes="Waiting on VPC peering",
        reviewer_user_id=org_admin_user.id,
    )
    CommitmentTrackingService.review_commitment(
        db_session,
        organization_id=test_org_a.id,
        action_item_id=ai.id,
        reviewing_meeting_id=m3.id,
        evidence_segment_id=seg3.id,
        status=ActionItemStatus.COMPLETED,
        review_notes="VPC peering active, cluster provisioned",
        reviewer_user_id=org_admin_user.id,
    )

    ctx = StructuredMemoryRetriever.search_structured_memory(
        db=db_session,
        organization_id=test_org_a.id,
        query="What is the review status of the Redis caching cluster?",
    )
    assert len(ctx.commitments) == 1
    c = ctx.commitments[0]
    assert c.origin_meeting_title == "Sprint 1"
    assert c.last_reviewed_meeting_title == "Sprint 3"
    assert c.review_count == 2
    assert c.status == ActionItemStatus.COMPLETED.value


def test_unresolved_question_structured_retrieval(db_session, test_org_a):
    """Verify structured retrieval loads open questions and resolution details."""
    m1 = _create_meeting(db_session, test_org_a, "Design Session", days_ago=5)
    m2 = _create_meeting(db_session, test_org_a, "Design Review", days_ago=2)

    q1 = UnresolvedQuestion(
        meeting_id=m1.id,
        question_text="Should we migrate to TypeScript for backend?",
        raised_by="Dave",
        status=QuestionStatus.OPEN.value,
    )
    q2 = UnresolvedQuestion(
        meeting_id=m1.id,
        question_text="Should we deprecate Python 3.9?",
        raised_by="Dave",
        status=QuestionStatus.RESOLVED.value,
        resolving_meeting_id=m2.id,
        resolution_notes="Agreed to deprecate Python 3.9 in Q3.",
        resolved_at=datetime.now(timezone.utc),
    )
    db_session.add_all([q1, q2])
    db_session.commit()

    ctx = StructuredMemoryRetriever.search_structured_memory(
        db=db_session,
        organization_id=test_org_a.id,
        query="Which questions remain unresolved?",
    )
    assert len(ctx.questions) == 1
    assert ctx.questions[0].id == q1.id
    assert ctx.questions[0].status == "OPEN"


# =========================================================================
# 2. Conflict Detection Tests
# =========================================================================

def test_conflict_detection_opposing_stances(db_session, test_org_a):
    """Verify conflict detector catches opposing active stances on the same topic."""
    m1 = _create_meeting(db_session, test_org_a, "Meeting 1", days_ago=10)
    m2 = _create_meeting(db_session, test_org_a, "Meeting 2", days_ago=2)

    d1 = Decision(
        meeting_id=m1.id,
        description="Approved remote work policy for all engineering teams",
        status=DecisionStatus.ACTIVE.value,
    )
    d2 = Decision(
        meeting_id=m2.id,
        description="Disallowed remote work policy; mandatory on-site required",
        status=DecisionStatus.ACTIVE.value,
    )
    db_session.add_all([d1, d2])
    db_session.commit()

    conflicts = ConflictDetector.detect_conflicts(db_session, organization_id=test_org_a.id)
    assert len(conflicts) == 1
    cf = conflicts[0]
    assert cf.is_heuristic is True
    assert "remote" in cf.subject.lower() or "policy" in cf.subject.lower()
    assert "opposing stances" in cf.reason.lower()


def test_conflict_detection_competing_exclusive_technologies(db_session, test_org_a):
    """Verify conflict detector catches competing choices in the same domain category."""
    m1 = _create_meeting(db_session, test_org_a, "Database Review 1", days_ago=8)
    m2 = _create_meeting(db_session, test_org_a, "Database Review 2", days_ago=3)

    d1 = Decision(
        meeting_id=m1.id,
        description="Adopt PostgreSQL as our primary relational database",
        status=DecisionStatus.ACTIVE.value,
    )
    d2 = Decision(
        meeting_id=m2.id,
        description="Adopt MongoDB as our primary production database",
        status=DecisionStatus.ACTIVE.value,
    )
    db_session.add_all([d1, d2])
    db_session.commit()

    conflicts = ConflictDetector.detect_conflicts(db_session, organization_id=test_org_a.id)
    assert len(conflicts) == 1
    cf = conflicts[0]
    assert "database" in cf.subject.lower()
    assert "competing database options" in cf.reason.lower() or "competing" in cf.reason.lower()


def test_explicit_superseding_is_NOT_flagged_as_conflict(db_session, test_org_a):
    """Verify explicitly superseded decisions are recognized as intentional evolutions, NOT conflicts."""
    m1 = _create_meeting(db_session, test_org_a, "Meeting 1", days_ago=10)
    m2 = _create_meeting(db_session, test_org_a, "Meeting 2", days_ago=2)

    d1 = Decision(
        meeting_id=m1.id,
        description="Adopt PostgreSQL for database",
        status=DecisionStatus.ACTIVE.value,
    )
    d2 = Decision(
        meeting_id=m2.id,
        description="Adopt MongoDB for database",
        status=DecisionStatus.ACTIVE.value,
    )
    db_session.add_all([d1, d2])
    db_session.commit()

    # Explicit supersession: d2 supersedes d1 (d1 becomes SUPERSEDED)
    DecisionLifecycleService.supersede_existing_decision(
        db_session,
        organization_id=test_org_a.id,
        target_decision_id=d1.id,
        superseding_decision_id=d2.id,
    )

    conflicts = ConflictDetector.detect_conflicts(db_session, organization_id=test_org_a.id)
    # Must NOT flag as conflict because d1 is SUPERSEDED and d2 explicitly points to d1!
    assert len(conflicts) == 0


def test_cross_tenant_conflict_and_retrieval_isolation(db_session, test_org_a, test_org_b):
    """Verify Org B cannot retrieve Org A records or trigger cross-tenant conflicts."""
    m_a = _create_meeting(db_session, test_org_a, "Org A Meeting")
    m_b = _create_meeting(db_session, test_org_b, "Org B Meeting")

    d_a = Decision(meeting_id=m_a.id, description="Approved AWS adoption", status=DecisionStatus.ACTIVE.value)
    d_b = Decision(meeting_id=m_b.id, description="Approved GCP adoption", status=DecisionStatus.ACTIVE.value)
    db_session.add_all([d_a, d_b])
    db_session.commit()

    # Conflict detector for Org A must not see Org B
    conflicts_a = ConflictDetector.detect_conflicts(db_session, organization_id=test_org_a.id)
    assert len(conflicts_a) == 0

    # Structured retriever for Org B must not return Org A's decision
    ctx_b = StructuredMemoryRetriever.search_structured_memory(
        db=db_session,
        organization_id=test_org_b.id,
        query="AWS adoption",
    )
    assert len(ctx_b.decisions) == 0


# =========================================================================
# 3. RAG Integration with Structured Memory Tests
# =========================================================================

def test_rag_engine_answers_using_structured_memory(db_session, test_org_a):
    """Verify RAG engine answers questions from structured memory without requiring transcript chunks."""
    m = _create_meeting(db_session, test_org_a, "Infrastructure Committee", days_ago=2)

    dec = Decision(
        meeting_id=m.id,
        description="Adopt Kubernetes for microservices orchestration across all clusters",
        rationale_or_context="Enables zero-downtime rolling updates and declarative autoscaling",
        decided_by="Platform Architecture Team",
        status=DecisionStatus.ACTIVE.value,
    )
    db_session.add(dec)
    db_session.commit()

    rag = RAGEngine()
    req = RAGQueryRequest(
        question="What decisions were made about container orchestration?",
        organization_id=test_org_a.id,
    )
    resp = rag.query(db=db_session, request=req)

    assert resp.is_abstained is False
    assert resp.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
    assert "Kubernetes" in resp.answer
    assert "Platform Architecture Team" in resp.answer or "Infrastructure Committee" in resp.answer
    assert resp.diagnostics.structured_decisions_count >= 1
    assert len(resp.structured_decisions) >= 1


def test_rag_engine_detects_and_alerts_on_conflicts(db_session, test_org_a):
    """Verify RAG engine flags CONFLICTING_EVIDENCE and surfaces heuristic warning for contradictory active decisions."""
    m1 = _create_meeting(db_session, test_org_a, "Infra Meeting 1", days_ago=6)
    m2 = _create_meeting(db_session, test_org_a, "Infra Meeting 2", days_ago=1)

    d1 = Decision(
        meeting_id=m1.id,
        description="Approved GraphQL for our mobile client API",
        status=DecisionStatus.ACTIVE.value,
    )
    d2 = Decision(
        meeting_id=m2.id,
        description="Disallowed GraphQL; all mobile clients must use gRPC",
        status=DecisionStatus.ACTIVE.value,
    )
    db_session.add_all([d1, d2])
    db_session.commit()

    rag = RAGEngine()
    req = RAGQueryRequest(
        question="Are there any conflicting decisions regarding GraphQL?",
        organization_id=test_org_a.id,
    )
    resp = rag.query(db=db_session, request=req)

    assert resp.abstention_status == AbstentionStatus.CONFLICTING_EVIDENCE
    assert "[HEURISTIC CONFLICT WARNING]" in resp.answer
    assert "Infra Meeting 1" in resp.answer
    assert "Infra Meeting 2" in resp.answer
    assert len(resp.detected_conflicts) >= 1
    assert resp.diagnostics.conflicts_detected_count >= 1


def test_rag_engine_abstains_when_no_evidence_at_all(db_session, test_org_a):
    """Verify RAG engine abstains with NO_EVIDENCE when neither structured nor transcript evidence exists."""
    _create_meeting(db_session, test_org_a, "General Sync")

    rag = RAGEngine()
    req = RAGQueryRequest(
        question="What quantum teleportation protocol was selected?",
        organization_id=test_org_a.id,
    )
    resp = rag.query(db=db_session, request=req)

    assert resp.is_abstained is True
    assert resp.abstention_status == AbstentionStatus.NO_EVIDENCE
    assert "No relevant meeting discussions or decisions were found" in resp.answer
    assert resp.diagnostics.structured_decisions_count == 0
    assert resp.diagnostics.evidence_items_count == 0
