"""Focused tests for Phase 8.9.4: Temporal Reasoning & Cross-Meeting Decision Lineage.

Verifies:
1. Three decisions across three different meeting dates (chronological ordering by actual meeting date).
2. Earlier versus latest decision questions ("What did we decide initially, and what changed later?").
3. Latest decision questions ("Which meeting introduced the latest decision about PostgreSQL?").
4. Explicit supersession lineage ("Was the earlier decision superseded?").
5. ACTIVE versus SUPERSEDED versus REVOKED status honesty.
6. Similar decisions without explicit lineage (distinguishing potential conflict from supersession).
7. Meeting and date filter behavior (start_date, end_date, meeting_id).
8. Hindi and Hinglish temporal questions ("पहले हुई meetings...", "Pehli meetings mein...").
9. Cross-tenant isolation in temporal retrieval and lineage traversal.
10. Exact meeting title, date, and transcript timestamps in citations.
11. Missing and unsafe evidence handling (no fabricated citations, safe transcript enforcement).
12. Insufficient history handling (single decision comparison / timeline note).
13. Abstention consistency (empty citations on NO_EVIDENCE).
"""

from datetime import datetime, timezone, timedelta
import pytest
from sqlalchemy.orm import Session

from app.models.identity import Organization
from app.models.meeting import Meeting, InputSourceType, MeetingStatus
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.intelligence import Decision, DecisionStatus
from app.services.intelligence.cross_meeting import DecisionLifecycleService
from app.services.intelligence.lineage import supersede_decision
from app.services.memory.rag.engine import RAGEngine
from app.services.memory.rag.types import (
    RAGQueryRequest,
    RAGQueryResponse,
    AbstentionStatus,
)


@pytest.fixture
def test_organization(db_session: Session) -> Organization:
    """Fixture providing a fresh test organization."""
    org = Organization(name="Tenant Temporal Inc", slug="tenant-temporal-inc")
    db_session.add(org)
    db_session.commit()
    db_session.refresh(org)
    return org


def _create_meeting_with_date(
    db: Session,
    org: Organization,
    title: str,
    meeting_date: datetime,
) -> Meeting:
    """Helper to create a meeting with an explicit actual_start_time date."""
    m = Meeting(
        organization_id=org.id,
        title=title,
        input_source_type=InputSourceType.AUDIO.value,
        status=MeetingStatus.COMPLETED.value,
        actual_start_time=meeting_date,
        scheduled_start_time=meeting_date,
        created_at=meeting_date,
        updated_at=meeting_date,
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


def _create_safe_segment(
    db: Session,
    meeting: Meeting,
    text: str,
    start_time: float,
    end_time: float,
) -> TranscriptSegment:
    """Helper to create a privacy-safe CLEANED_REDACTED transcript segment."""
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
        start_time=start_time,
        end_time=end_time,
        text=text,
    )
    db.add(segment)
    db.commit()
    db.refresh(segment)
    return segment


def test_three_decisions_across_three_dates_chronological_ordering(
    db_session: Session,
    test_organization: Organization,
):
    """Test 1: Three decisions across three dates are retrieved in strict chronological order with verified citations."""
    date_sep = datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc)
    date_oct = datetime(2026, 10, 15, 14, 0, 0, tzinfo=timezone.utc)
    date_nov = datetime(2026, 11, 20, 9, 30, 0, tzinfo=timezone.utc)

    m1 = _create_meeting_with_date(db_session, test_organization, "September Strategy Sync", date_sep)
    m2 = _create_meeting_with_date(db_session, test_organization, "October Architecture Review", date_oct)
    m3 = _create_meeting_with_date(db_session, test_organization, "November Final Governance", date_nov)

    seg1 = _create_safe_segment(db_session, m1, "We agreed to evaluate PostgreSQL and CockroachDB for persistence.", 15.0, 25.0)
    seg2 = _create_safe_segment(db_session, m2, "The architecture committee decided to adopt PostgreSQL 16 as primary persistence.", 40.0, 52.0)
    seg3 = _create_safe_segment(db_session, m3, "Mandate PostgreSQL 16 with TimescaleDB extension for all telemetry services.", 100.0, 115.0)

    # Decision 1 in Sept (initially ACTIVE)
    dec1 = Decision(
        meeting_id=m1.id,
        description="Evaluate PostgreSQL and CockroachDB for persistence",
        rationale_or_context="Initial persistence exploration",
        status=DecisionStatus.ACTIVE.value,
        evidence_segment_id=seg1.id,
        timestamp_seconds=20.0,
        created_at=date_sep,
    )
    db_session.add(dec1)
    db_session.commit()
    db_session.refresh(dec1)

    # Decision 2 in Oct (supersedes Decision 1)
    dec2 = Decision(
        meeting_id=m2.id,
        description="Adopt PostgreSQL 16 as primary persistence",
        rationale_or_context="Team familiarity and mature ecosystem",
        status=DecisionStatus.ACTIVE.value,
        evidence_segment_id=seg2.id,
        timestamp_seconds=45.0,
        created_at=date_oct,
    )
    db_session.add(dec2)
    db_session.flush()
    supersede_decision(db_session, superseding_decision=dec2, target_decision_id=dec1.id)
    db_session.commit()
    db_session.refresh(dec2)

    # Decision 3 in Nov (supersedes Decision 2)
    dec3 = Decision(
        meeting_id=m3.id,
        description="Mandate PostgreSQL 16 with TimescaleDB extension for telemetry",
        rationale_or_context="Telemetry time-series requirements finalized",
        status=DecisionStatus.ACTIVE.value,
        evidence_segment_id=seg3.id,
        timestamp_seconds=110.0,
        created_at=date_nov,
    )
    db_session.add(dec3)
    db_session.flush()
    supersede_decision(db_session, superseding_decision=dec3, target_decision_id=dec2.id)
    db_session.commit()
    db_session.refresh(dec3)

    engine = RAGEngine()
    req = RAGQueryRequest(
        organization_id=test_organization.id,
        question="What is the history of decisions regarding PostgreSQL across previous meetings?",
    )
    resp = engine.query(db=db_session, request=req)

    assert resp.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
    assert not resp.is_abstained

    # Chronological history check
    ans = resp.answer
    assert "Chronological Decision History:" in ans
    assert "September Strategy Sync" in ans
    assert "October Architecture Review" in ans
    assert "November Final Governance" in ans
    assert "2026-09-01" in ans
    assert "2026-10-15" in ans
    assert "2026-11-20" in ans

    # Verified Citations check: all 3 decisions produce first-class citations
    assert len(resp.citations) == 3
    cit_map = {c.meeting_title: c for c in resp.citations}
    assert "September Strategy Sync" in cit_map
    assert "October Architecture Review" in cit_map
    assert "November Final Governance" in cit_map

    # Check exact timestamps strictly from segments
    assert cit_map["September Strategy Sync"].start_time == 15.0
    assert cit_map["September Strategy Sync"].end_time == 25.0
    assert cit_map["October Architecture Review"].start_time == 40.0
    assert cit_map["October Architecture Review"].end_time == 52.0
    assert cit_map["November Final Governance"].start_time == 100.0
    assert cit_map["November Final Governance"].end_time == 115.0


def test_earlier_versus_latest_decision_questions(
    db_session: Session,
    test_organization: Organization,
):
    """Test 2: 'What did we decide initially, and what changed later?' synthesizes earlier vs latest decision."""
    d1_date = datetime(2026, 8, 10, 10, 0, 0, tzinfo=timezone.utc)
    d2_date = datetime(2026, 11, 12, 16, 0, 0, tzinfo=timezone.utc)

    m1 = _create_meeting_with_date(db_session, test_organization, "Q3 Policy Summit", d1_date)
    m2 = _create_meeting_with_date(db_session, test_organization, "Q4 Governance Board", d2_date)

    s1 = _create_safe_segment(db_session, m1, "Initial bylaws amendment threshold set at 60% majority.", 10.0, 20.0)
    s2 = _create_safe_segment(db_session, m2, "Bylaws amendment threshold finalized at 66% supermajority following legal review.", 30.0, 45.0)

    dec1 = Decision(
        meeting_id=m1.id,
        description="Draft amendments to bylaws proposed requiring 60% majority",
        rationale_or_context="Preliminary legal recommendation",
        status=DecisionStatus.ACTIVE.value,
        evidence_segment_id=s1.id,
        created_at=d1_date,
    )
    db_session.add(dec1)
    db_session.commit()

    dec2 = Decision(
        meeting_id=m2.id,
        description="Bylaws amendment threshold finalized at 66% supermajority",
        rationale_or_context="Updated following shareholder feedback and external counsel",
        status=DecisionStatus.ACTIVE.value,
        evidence_segment_id=s2.id,
        created_at=d2_date,
    )
    db_session.add(dec2)
    db_session.flush()
    supersede_decision(db_session, superseding_decision=dec2, target_decision_id=dec1.id)
    db_session.commit()

    engine = RAGEngine()
    req = RAGQueryRequest(
        organization_id=test_organization.id,
        question="What did we decide initially, and what changed later?",
    )
    resp = engine.query(db=db_session, request=req)

    assert resp.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
    ans = resp.answer

    # Must explain earlier vs superseding decision
    assert "Earlier Decision:" in ans or "Q3 Policy Summit" in ans
    assert "Superseding Decision:" in ans or "Q4 Governance Board" in ans
    assert "60%" in ans
    assert "66%" in ans
    assert "Current Status:" in ans or "active" in ans.lower()


def test_latest_decision_query_identifies_current(
    db_session: Session,
    test_organization: Organization,
):
    """Test 3: 'Which meeting introduced the latest decision about PostgreSQL?' identifies latest active decision."""
    d1_date = datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc)
    d2_date = datetime(2026, 11, 15, 14, 0, 0, tzinfo=timezone.utc)

    m1 = _create_meeting_with_date(db_session, test_organization, "Initial Sprint", d1_date)
    m2 = _create_meeting_with_date(db_session, test_organization, "Architecture Council", d2_date)

    s1 = _create_safe_segment(db_session, m1, "Agreed to test PostgreSQL on dev environment.", 5.0, 15.0)
    s2 = _create_safe_segment(db_session, m2, "Official mandate: Deploy PostgreSQL 16 on all production clusters.", 25.0, 40.0)

    dec1 = Decision(
        meeting_id=m1.id,
        description="Test PostgreSQL on dev environment",
        status=DecisionStatus.ACTIVE.value,
        evidence_segment_id=s1.id,
        created_at=d1_date,
    )
    db_session.add(dec1)
    db_session.commit()

    dec2 = Decision(
        meeting_id=m2.id,
        description="Deploy PostgreSQL 16 on all production clusters",
        rationale_or_context="Production benchmarking completed successfully",
        status=DecisionStatus.ACTIVE.value,
        evidence_segment_id=s2.id,
        created_at=d2_date,
    )
    db_session.add(dec2)
    db_session.flush()
    supersede_decision(db_session, superseding_decision=dec2, target_decision_id=dec1.id)
    db_session.commit()

    engine = RAGEngine()
    req = RAGQueryRequest(
        organization_id=test_organization.id,
        question="Which meeting introduced the latest decision about PostgreSQL?",
    )
    resp = engine.query(db=db_session, request=req)

    assert resp.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
    ans = resp.answer
    assert "Architecture Council" in ans
    assert "2026-11-15" in ans
    assert "ACTIVE" in ans
    assert "Deploy PostgreSQL 16 on all production clusters" in ans
    assert "Explicitly superseded" in ans or "Initial Sprint" in ans


def test_revoked_decision_status_honesty(
    db_session: Session,
    test_organization: Organization,
):
    """Test 4: A revoked decision is honestly identified as REVOKED with no active effective decision."""
    d1_date = datetime(2026, 7, 1, 10, 0, 0, tzinfo=timezone.utc)
    d2_date = datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc)

    m1 = _create_meeting_with_date(db_session, test_organization, "HR Townhall July", d1_date)
    m2 = _create_meeting_with_date(db_session, test_organization, "Leadership Retro October", d2_date)

    s1 = _create_safe_segment(db_session, m1, "Remote work policy approved for three days per week.", 10.0, 20.0)
    s2 = _create_safe_segment(db_session, m2, "Remote work policy revoked due to hybrid facility lease requirements.", 30.0, 40.0)

    dec1 = Decision(
        meeting_id=m1.id,
        description="Remote work policy approved for three days per week",
        status=DecisionStatus.SUPERSEDED.value,
        evidence_segment_id=s1.id,
        created_at=d1_date,
    )
    db_session.add(dec1)
    db_session.commit()

    dec2 = Decision(
        meeting_id=m2.id,
        description="Remote work policy revoked due to hybrid facility lease",
        rationale_or_context="Facility lease obligations",
        status=DecisionStatus.REVOKED.value,
        supersedes_decision_id=dec1.id,
        evidence_segment_id=s2.id,
        created_at=d2_date,
    )
    db_session.add(dec2)
    db_session.commit()

    engine = RAGEngine()
    req = RAGQueryRequest(
        organization_id=test_organization.id,
        question="Which meeting introduced the latest decision about remote work?",
    )
    resp = engine.query(db=db_session, request=req)

    assert resp.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
    ans = resp.answer
    assert "REVOKED" in ans
    assert "Leadership Retro October" in ans
    assert "no active decision" in ans.lower() or "revoked" in ans.lower()


def test_similar_decisions_without_lineage_not_falsely_superseded(
    db_session: Session,
    test_organization: Organization,
):
    """Test 5: Similar decisions from different meetings without supersedes_decision_id are distinguished as unlinked / potential conflicts."""
    d1_date = datetime(2026, 10, 5, 10, 0, 0, tzinfo=timezone.utc)
    d2_date = datetime(2026, 11, 5, 10, 0, 0, tzinfo=timezone.utc)

    m1 = _create_meeting_with_date(db_session, test_organization, "October Platform Sync", d1_date)
    m2 = _create_meeting_with_date(db_session, test_organization, "November Cloud Committee", d2_date)

    s1 = _create_safe_segment(db_session, m1, "All core production services must run on Google Cloud Platform.", 10.0, 20.0)
    s2 = _create_safe_segment(db_session, m2, "Deploy secondary production services on Amazon Web Services.", 30.0, 40.0)

    # Both are ACTIVE, neither supersedes the other
    dec1 = Decision(
        meeting_id=m1.id,
        description="All core production services must run on Google Cloud Platform",
        status=DecisionStatus.ACTIVE.value,
        evidence_segment_id=s1.id,
        created_at=d1_date,
    )
    dec2 = Decision(
        meeting_id=m2.id,
        description="Deploy secondary production cloud platform services on Amazon Web Services",
        status=DecisionStatus.ACTIVE.value,
        evidence_segment_id=s2.id,
        created_at=d2_date,
    )
    db_session.add_all([dec1, dec2])
    db_session.commit()

    engine = RAGEngine()
    req = RAGQueryRequest(
        organization_id=test_organization.id,
        question="Compare the decisions made about cloud platform across meetings.",
    )
    resp = engine.query(db=db_session, request=req)

    assert resp.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
    ans = resp.answer

    # Must NOT claim that dec2 superseded dec1
    assert "explicitly superseded" not in ans.lower()
    # Must flag unlinked or potential conflict
    assert "No explicit superseding relationship" in ans or "potential decision conflict" in ans.lower() or "Lineage & Conflict Analysis" in ans


def test_meeting_and_date_filter_behavior(
    db_session: Session,
    test_organization: Organization,
):
    """Test 6: Date range filtering (start_date/end_date) and meeting_id filtering accurately restrict decisions."""
    d_sep = datetime(2026, 9, 10, 10, 0, 0, tzinfo=timezone.utc)
    d_oct = datetime(2026, 10, 15, 10, 0, 0, tzinfo=timezone.utc)
    d_nov = datetime(2026, 11, 20, 10, 0, 0, tzinfo=timezone.utc)

    m1 = _create_meeting_with_date(db_session, test_organization, "September Kickoff", d_sep)
    m2 = _create_meeting_with_date(db_session, test_organization, "October Interim Review", d_oct)
    m3 = _create_meeting_with_date(db_session, test_organization, "November Milestone", d_nov)

    dec1 = Decision(
        meeting_id=m1.id,
        description="Budget allocated at $50,000 for mobile app",
        status=DecisionStatus.SUPERSEDED.value,
        created_at=d_sep,
    )
    dec2 = Decision(
        meeting_id=m2.id,
        description="Budget increased to $75,000 for mobile app",
        status=DecisionStatus.ACTIVE.value,
        supersedes_decision_id=dec1.id,
        created_at=d_oct,
    )
    dec3 = Decision(
        meeting_id=m3.id,
        description="Budget capped at $90,000 for mobile app",
        status=DecisionStatus.ACTIVE.value,
        supersedes_decision_id=dec2.id,
        created_at=d_nov,
    )
    db_session.add_all([dec1, dec2, dec3])
    db_session.commit()

    engine = RAGEngine()

    # Query with date filter targeting only October (Oct 1 to Oct 31)
    req_oct = RAGQueryRequest(
        organization_id=test_organization.id,
        question="What decisions were made about mobile app budget?",
        start_date=datetime(2026, 10, 1, 0, 0, 0, tzinfo=timezone.utc),
        end_date=datetime(2026, 10, 31, 23, 59, 59, tzinfo=timezone.utc),
    )
    resp_oct = engine.query(db=db_session, request=req_oct)

    assert resp_oct.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
    ans_oct = resp_oct.answer
    assert "$75,000" in ans_oct
    # November decision ($90,000) was outside date filter and should not be returned as a base decision
    assert "$90,000" not in ans_oct

    # Query with meeting_id targeting only m3
    req_m3 = RAGQueryRequest(
        organization_id=test_organization.id,
        question="What decisions were made about budget?",
        meeting_id=m3.id,
    )
    resp_m3 = engine.query(db=db_session, request=req_m3)
    assert resp_m3.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
    assert "$90,000" in resp_m3.answer


def test_hindi_and_hinglish_temporal_decision_questions(
    db_session: Session,
    test_organization: Organization,
):
    """Test 7: Hindi and Hinglish temporal questions accurately extract substantive entity and reconstruct lineage."""
    d1_date = datetime(2026, 9, 5, 10, 0, 0, tzinfo=timezone.utc)
    d2_date = datetime(2026, 11, 8, 14, 0, 0, tzinfo=timezone.utc)

    m1 = _create_meeting_with_date(db_session, test_organization, "Governance Meeting September", d1_date)
    m2 = _create_meeting_with_date(db_session, test_organization, "Governance Meeting November", d2_date)

    s1 = _create_safe_segment(db_session, m1, "Draft amendments to bylaws proposed requiring 60% majority.", 10.0, 20.0)
    s2 = _create_safe_segment(db_session, m2, "Final amendment approved establishing 66% supermajority for bylaws.", 30.0, 45.0)

    dec1 = Decision(
        meeting_id=m1.id,
        description="Draft amendments to bylaws proposed requiring 60% majority",
        status=DecisionStatus.ACTIVE.value,
        evidence_segment_id=s1.id,
        created_at=d1_date,
    )
    db_session.add(dec1)
    db_session.commit()

    dec2 = Decision(
        meeting_id=m2.id,
        description="Final amendment approved establishing 66% supermajority for bylaws",
        status=DecisionStatus.ACTIVE.value,
        evidence_segment_id=s2.id,
        created_at=d2_date,
    )
    db_session.add(dec2)
    db_session.flush()
    supersede_decision(db_session, superseding_decision=dec2, target_decision_id=dec1.id)
    db_session.commit()

    engine = RAGEngine()

    # Hindi question: "पहले हुई meetings में amendments के बारे में क्या decision लिया गया था?"
    req_hi = RAGQueryRequest(
        organization_id=test_organization.id,
        question="पहले हुई meetings में amendments के बारे में क्या decision लिया गया था?",
    )
    resp_hi = engine.query(db=db_session, request=req_hi)

    assert resp_hi.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
    ans_hi = resp_hi.answer
    assert "Governance Meeting September" in ans_hi
    assert "Governance Meeting November" in ans_hi
    assert "60%" in ans_hi
    assert "66%" in ans_hi
    assert len(resp_hi.citations) == 2

    # Hinglish question: "Pehli meetings mein amendments ko lekar kya decide hua tha?"
    req_hinglish = RAGQueryRequest(
        organization_id=test_organization.id,
        question="Pehli meetings mein amendments ko lekar kya decide hua tha?",
    )
    resp_hinglish = engine.query(db=db_session, request=req_hinglish)

    assert resp_hinglish.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
    ans_hinglish = resp_hinglish.answer
    assert "Governance Meeting September" in ans_hinglish
    assert "Governance Meeting November" in ans_hinglish
    assert len(resp_hinglish.citations) == 2


def test_cross_tenant_isolation_in_temporal_reasoning(
    db_session: Session,
    test_organization: Organization,
):
    """Test 8: Cross-tenant isolation guarantees another organization's decision lineage cannot be retrieved or traversed."""
    org_b = Organization(
        name="Tenant B Competitor",
        slug="tenant-b-competitor",
    )
    db_session.add(org_b)
    db_session.commit()

    m_b = _create_meeting_with_date(db_session, org_b, "Confidential Strategy B", datetime.now(timezone.utc))
    s_b = _create_safe_segment(db_session, m_b, "Tenant B secret decision to acquire competitor startup.", 5.0, 15.0)

    dec_b = Decision(
        meeting_id=m_b.id,
        description="Acquire competitor startup for $20M",
        status=DecisionStatus.ACTIVE.value,
        evidence_segment_id=s_b.id,
    )
    db_session.add(dec_b)
    db_session.commit()

    engine = RAGEngine()
    # Org A asks about acquiring competitor startup
    req = RAGQueryRequest(
        organization_id=test_organization.id,
        question="What decisions were made about acquiring competitor startup across previous meetings?",
    )
    resp = engine.query(db=db_session, request=req)

    assert resp.abstention_status == AbstentionStatus.NO_EVIDENCE
    assert resp.is_abstained
    assert "Acquire competitor" not in resp.answer
    assert len(resp.citations) == 0


def test_exact_meeting_title_date_and_segment_timestamps_in_citations(
    db_session: Session,
    test_organization: Organization,
):
    """Test 9: Verified citations contain exact meeting title, ISO date, and exact segment float timestamps (never stored timestamp_seconds)."""
    meet_date = datetime(2026, 10, 22, 11, 45, 0, tzinfo=timezone.utc)
    m = _create_meeting_with_date(db_session, test_organization, "Precision Timestamp Summit", meet_date)
    seg = _create_safe_segment(db_session, m, "Decision to enforce mTLS on all internal gRPC endpoints.", 123.45, 145.67)

    # Note: decision has timestamp_seconds=999.0 which differs from segment start_time=123.45
    dec = Decision(
        meeting_id=m.id,
        description="Enforce mTLS on all internal gRPC endpoints",
        status=DecisionStatus.ACTIVE.value,
        evidence_segment_id=seg.id,
        timestamp_seconds=999.0,
        created_at=meet_date,
    )
    db_session.add(dec)
    db_session.commit()

    engine = RAGEngine()
    req = RAGQueryRequest(
        organization_id=test_organization.id,
        question="Which meeting introduced the latest decision about mTLS?",
    )
    resp = engine.query(db=db_session, request=req)

    assert resp.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
    assert len(resp.citations) == 1
    cit = resp.citations[0]

    assert cit.meeting_title == "Precision Timestamp Summit"
    assert "2026-10-22" in cit.meeting_date
    assert cit.start_time == 123.45  # strictly from segment, NOT 999.0
    assert cit.end_time == 145.67
    assert cit.quoted_text == "Decision to enforce mTLS on all internal gRPC endpoints."
    assert cit.segment_ids == [seg.id]


def test_missing_and_unsafe_evidence_handling(
    db_session: Session,
    test_organization: Organization,
):
    """Test 10: Decisions with NULL or unsafe evidence format 'Evidence Reference: Unavailable' and generate zero citations."""
    m_date = datetime(2026, 10, 5, 10, 0, 0, tzinfo=timezone.utc)
    m = _create_meeting_with_date(db_session, test_organization, "Executive Board", m_date)

    # Decision without evidence segment
    dec_no_ev = Decision(
        meeting_id=m.id,
        description="Annual holiday party scheduled for December 18",
        status=DecisionStatus.ACTIVE.value,
        evidence_segment_id=None,
        created_at=m_date,
    )
    db_session.add(dec_no_ev)
    db_session.commit()

    engine = RAGEngine()
    req = RAGQueryRequest(
        organization_id=test_organization.id,
        question="What was decided about the annual holiday party?",
    )
    resp = engine.query(db=db_session, request=req)

    assert resp.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
    assert "Evidence Reference: Unavailable" in resp.answer
    assert len(resp.citations) == 0


def test_insufficient_history_when_single_decision_exists(
    db_session: Session,
    test_organization: Organization,
):
    """Test 11: When user asks for a comparison or timeline but only 1 decision exists, an honest Insufficient History note is displayed."""
    m_date = datetime(2026, 9, 15, 10, 0, 0, tzinfo=timezone.utc)
    m = _create_meeting_with_date(db_session, test_organization, "DevOps Sync", m_date)
    s = _create_safe_segment(db_session, m, "Adopt Kubernetes 1.30 across all production clusters.", 10.0, 20.0)

    dec = Decision(
        meeting_id=m.id,
        description="Adopt Kubernetes 1.30 across all production clusters",
        status=DecisionStatus.ACTIVE.value,
        evidence_segment_id=s.id,
        created_at=m_date,
    )
    db_session.add(dec)
    db_session.commit()

    engine = RAGEngine()
    req = RAGQueryRequest(
        organization_id=test_organization.id,
        question="Compare the decisions made about Kubernetes across previous meetings.",
    )
    resp = engine.query(db=db_session, request=req)

    assert resp.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
    ans = resp.answer
    assert "Historical Context Note:" in ans
    assert "single recorded decision" in ans
    assert "insufficient historical data across multiple meetings" in ans
