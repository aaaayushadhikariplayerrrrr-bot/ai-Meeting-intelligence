"""Tests for Phase 8.9.3: Structured Decision Evidence, Citation Verification & Timestamp Resolution.

Validates:
1. Decision with valid evidence segment produces a first-class verified Citation.
2. Segment timestamps (start_time, end_time) are derived strictly from database TranscriptSegment,
   never from decision.timestamp_seconds or LLM text.
3. Meeting title and meeting date are resolved strictly from database Meeting records.
4. Decision with NULL evidence_segment_id does NOT fabricate a citation and marks evidence as unavailable.
5. Decision referencing a non-existent segment ID does NOT fabricate a citation.
6. Cross-tenant isolation strictly rejects segments belonging to another organization.
7. Meeting ownership mismatch between decision and segment is rejected.
8. Safe transcript version enforcement (CLEANED_REDACTED or REDACTED required, mapping derived segments if on original).
9. ActionItem evidence verification produces first-class citations or marks unavailable.
10. UnresolvedQuestion evidence verification produces first-class citations or marks unavailable.
11. Combined retrieval with both transcript chunk and structured decision produces deduplicated sequential citations.
12. Strict abstention (NO_EVIDENCE, INSUFFICIENT_EVIDENCE) preserves empty citations.
"""

from datetime import datetime, timezone
import pytest
from sqlalchemy.orm import Session

from app.models.identity import Organization
from app.models.meeting import Meeting
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.intelligence import (
    Decision,
    DecisionStatus,
    ActionItem,
    ActionItemStatus,
    UnresolvedQuestion,
    QuestionStatus,
)
from app.schemas.memory import (
    AbstentionStatus,
    RAGQueryRequest,
    RAGQueryResponse,
    Citation,
)
from app.services.memory.rag.engine import RAGEngine
from app.services.memory.rag.verification import CitationVerifier, SAFE_TRANSCRIPT_TYPES


@pytest.fixture
def structured_citation_env(db_session: Session):
    """Sets up a multi-tenant environment with meetings, safe transcripts, segments, and structured records."""
    # Org A (Primary tenant)
    org_a = Organization(id="org-struct-a", name="Tenant A Alpha", slug="tenant-a-alpha")
    # Org B (Cross-tenant boundary test)
    org_b = Organization(id="org-struct-b", name="Tenant B Beta", slug="tenant-b-beta")
    db_session.add_all([org_a, org_b])
    db_session.commit()

    # Meetings for Org A
    m_a1 = Meeting(
        id="m-struct-a1",
        organization_id=org_a.id,
        title="Architecture Decision Review",
        actual_start_time=datetime(2026, 9, 20, 11, 30, tzinfo=timezone.utc),
    )
    m_a2 = Meeting(
        id="m-struct-a2",
        organization_id=org_a.id,
        title="Database Migration Sync",
        actual_start_time=datetime(2026, 9, 25, 14, 0, tzinfo=timezone.utc),
    )
    # Meeting for Org B
    m_b1 = Meeting(
        id="m-struct-b1",
        organization_id=org_b.id,
        title="Org B Secret Meeting",
        actual_start_time=datetime(2026, 9, 22, 10, 0, tzinfo=timezone.utc),
    )
    db_session.add_all([m_a1, m_a2, m_b1])
    db_session.commit()

    # Transcripts: Safe CLEANED_REDACTED for m_a1
    t_safe_a1 = Transcript(
        id="tr-safe-a1",
        meeting_id=m_a1.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=1,
    )
    # Unredacted ORIGINAL for m_a2 with derived CLEANED_REDACTED
    t_orig_a2 = Transcript(
        id="tr-orig-a2",
        meeting_id=m_a2.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
    )
    t_safe_a2 = Transcript(
        id="tr-safe-a2",
        meeting_id=m_a2.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=2,
    )
    # Org B Safe Transcript
    t_safe_b1 = Transcript(
        id="tr-safe-b1",
        meeting_id=m_b1.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=1,
    )
    db_session.add_all([t_safe_a1, t_orig_a2, t_safe_a2, t_safe_b1])
    db_session.commit()

    # Segments
    # seg_a1_1: Safe segment on m_a1
    seg_a1_1 = TranscriptSegment(
        id="seg-struct-a1-1",
        transcript_id=t_safe_a1.id,
        sequence_order=1,
        start_time=12.5,
        end_time=28.0,
        text="We decided to adopt Kafka for asynchronous event messaging across all backend services.",
        language_code="en",
    )
    # seg_a1_2: Safe segment on m_a1 for an action item
    seg_a1_2 = TranscriptSegment(
        id="seg-struct-a1-2",
        transcript_id=t_safe_a1.id,
        sequence_order=2,
        start_time=30.0,
        end_time=45.0,
        text="Alex will configure the production Kafka cluster by October 15th.",
        language_code="en",
    )
    # seg_a1_3: Safe segment on m_a1 for an unresolved question
    seg_a1_3 = TranscriptSegment(
        id="seg-struct-a1-3",
        transcript_id=t_safe_a1.id,
        sequence_order=3,
        start_time=50.0,
        end_time=65.0,
        text="Do we have budget approval for the managed Kafka hosting tier?",
        language_code="en",
    )
    # seg_orig_a2: Segment on ORIGINAL transcript
    seg_orig_a2 = TranscriptSegment(
        id="seg-orig-a2",
        transcript_id=t_orig_a2.id,
        sequence_order=1,
        start_time=5.0,
        end_time=20.0,
        text="Original unredacted confidential discussion about database failover.",
        language_code="en",
    )
    # seg_safe_a2: Derived safe segment referencing seg_orig_a2
    seg_safe_a2 = TranscriptSegment(
        id="seg-safe-a2",
        transcript_id=t_safe_a2.id,
        sequence_order=1,
        start_time=5.0,
        end_time=20.0,
        text="Redacted discussion regarding [CONFIDENTIAL] database failover.",
        language_code="en",
        source_segment_id=seg_orig_a2.id,
    )
    # seg_b1: Org B private segment
    seg_b1 = TranscriptSegment(
        id="seg-struct-b1",
        transcript_id=t_safe_b1.id,
        sequence_order=1,
        start_time=100.0,
        end_time=115.0,
        text="Org B confidential internal strategic plan.",
        language_code="en",
    )
    db_session.add_all([seg_a1_1, seg_a1_2, seg_a1_3, seg_orig_a2, seg_safe_a2, seg_b1])
    db_session.commit()

    return {
        "org_a": org_a,
        "org_b": org_b,
        "m_a1": m_a1,
        "m_a2": m_a2,
        "m_b1": m_b1,
        "t_safe_a1": t_safe_a1,
        "t_safe_a2": t_safe_a2,
        "seg_a1_1": seg_a1_1,
        "seg_a1_2": seg_a1_2,
        "seg_a1_3": seg_a1_3,
        "seg_orig_a2": seg_orig_a2,
        "seg_safe_a2": seg_safe_a2,
        "seg_b1": seg_b1,
    }


# ============================================================================
# 1. Decision with Valid Evidence Segment Produces First-Class Citation
# ============================================================================

def test_decision_with_valid_evidence_segment_produces_citation(structured_citation_env, db_session: Session):
    """Verify decision with valid evidence_segment_id produces verified Citation with database timestamps."""
    env = structured_citation_env
    org_a = env["org_a"]
    m_a1 = env["m_a1"]
    seg = env["seg_a1_1"]

    dec = Decision(
        meeting_id=m_a1.id,
        description="Adopt Apache Kafka for asynchronous messaging",
        rationale_or_context="Decouples microservices and guarantees at-least-once delivery",
        evidence_segment_id=seg.id,
        timestamp_seconds=999.0,  # Fabricated / different stored timestamp
        status=DecisionStatus.ACTIVE.value,
    )
    db_session.add(dec)
    db_session.commit()

    engine = RAGEngine()
    req = RAGQueryRequest(
        question="What decisions were made about Kafka messaging?",
        organization_id=org_a.id,
    )
    resp: RAGQueryResponse = engine.query(db=db_session, request=req)

    assert not resp.is_abstained
    assert resp.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
    assert "Adopt Apache Kafka" in resp.answer
    assert f"Segment [{seg.id}]" in resp.answer

    # Verified Citation exists
    assert len(resp.citations) == 1
    cit: Citation = resp.citations[0]
    assert cit.citation_id == "cit-1"
    assert cit.meeting_id == m_a1.id
    assert cit.meeting_title == "Architecture Decision Review"
    assert cit.meeting_date == m_a1.actual_start_time.isoformat()
    # Timestamps strictly from database TranscriptSegment, NEVER from decision.timestamp_seconds
    assert cit.start_time == 12.5
    assert cit.end_time == 28.0
    assert cit.start_time != 999.0
    assert cit.segment_ids == [seg.id]
    assert cit.quoted_text == seg.text
    assert cit.language_codes == ["en"]


# ============================================================================
# 2. Decision with NULL Evidence Segment Does NOT Fabricate Citation
# ============================================================================

def test_decision_with_null_evidence_segment_marks_unavailable(structured_citation_env, db_session: Session):
    """Verify decision without evidence_segment_id does not fabricate citation and displays unavailable."""
    env = structured_citation_env
    org_a = env["org_a"]
    m_a1 = env["m_a1"]

    dec = Decision(
        meeting_id=m_a1.id,
        description="Agreed to review infrastructure budget in Q4",
        evidence_segment_id=None,  # No evidence linked
        status=DecisionStatus.ACTIVE.value,
    )
    db_session.add(dec)
    db_session.commit()

    engine = RAGEngine()
    req = RAGQueryRequest(
        question="What decisions were made regarding infrastructure budget?",
        organization_id=org_a.id,
    )
    resp: RAGQueryResponse = engine.query(db=db_session, request=req)

    assert not resp.is_abstained
    assert "review infrastructure budget" in resp.answer
    assert "Evidence Reference: Unavailable" in resp.answer
    # No citations fabricated
    assert len(resp.citations) == 0


# ============================================================================
# 3. Decision with Non-Existent Segment ID Does NOT Fabricate Citation
# ============================================================================

def test_decision_with_non_existent_segment_id(structured_citation_env, db_session: Session):
    """Verify fabricated or non-existent segment IDs are rejected cleanly without crashing or citing."""
    env = structured_citation_env
    org_a = env["org_a"]
    m_a1 = env["m_a1"]

    verifier = CitationVerifier()
    cit = verifier.verify_structured_evidence(
        db=db_session,
        evidence_segment_id="00000000-0000-0000-0000-000000000000",
        meeting_id=m_a1.id,
        organization_id=org_a.id,
        entity_ref_id="dec-fake",
    )
    assert cit is None


# ============================================================================
# 4. Strict Cross-Tenant Isolation Rejection
# ============================================================================

def test_cross_tenant_evidence_segment_rejection(structured_citation_env, db_session: Session):
    """Verify segment belonging to Org B cannot be verified or cited for Org A."""
    env = structured_citation_env
    org_a = env["org_a"]
    m_a1 = env["m_a1"]
    seg_b1 = env["seg_b1"]  # Belongs to Org B

    # Decision in Org A maliciously or erroneously points to Org B's segment
    dec = Decision(
        meeting_id=m_a1.id,
        description="Decision claiming Org B evidence",
        evidence_segment_id=seg_b1.id,
        status=DecisionStatus.ACTIVE.value,
    )
    db_session.add(dec)
    db_session.commit()

    verifier = CitationVerifier()
    # Verification under Org A strictly rejects Org B's segment
    cit = verifier.verify_structured_evidence(
        db=db_session,
        evidence_segment_id=seg_b1.id,
        meeting_id=m_a1.id,
        organization_id=org_a.id,
        entity_ref_id=dec.id,
    )
    assert cit is None


# ============================================================================
# 5. Meeting Ownership Mismatch Rejection
# ============================================================================

def test_meeting_ownership_mismatch_rejection(structured_citation_env, db_session: Session):
    """Verify decision belonging to Meeting 1 referencing a segment in Meeting 2 is rejected."""
    env = structured_citation_env
    org_a = env["org_a"]
    m_a2 = env["m_a2"]
    seg_a1_1 = env["seg_a1_1"]  # Belongs to m_a1, not m_a2

    verifier = CitationVerifier()
    cit = verifier.verify_structured_evidence(
        db=db_session,
        evidence_segment_id=seg_a1_1.id,
        meeting_id=m_a2.id,  # Mismatched meeting ID
        organization_id=org_a.id,
        entity_ref_id="dec-test",
    )
    assert cit is None


# ============================================================================
# 6. Safe Transcript Version Enforcement & Derived Segment Mapping
# ============================================================================

def test_safe_transcript_version_mapping_from_original(structured_citation_env, db_session: Session):
    """Verify segment on ORIGINAL transcript maps to derived safe segment on CLEANED_REDACTED."""
    env = structured_citation_env
    org_a = env["org_a"]
    m_a2 = env["m_a2"]
    seg_orig = env["seg_orig_a2"]
    seg_safe = env["seg_safe_a2"]

    verifier = CitationVerifier()
    # Query referencing the original segment
    cit = verifier.verify_structured_evidence(
        db=db_session,
        evidence_segment_id=seg_orig.id,
        meeting_id=m_a2.id,
        organization_id=org_a.id,
        entity_ref_id="dec-orig",
    )
    assert cit is not None
    # Provenance mapped to the safe transcript and safe segment
    assert cit.segment_ids == [seg_safe.id]
    assert "[CONFIDENTIAL]" in cit.quoted_text
    assert "Original unredacted" not in cit.quoted_text


# ============================================================================
# 7. ActionItem Evidence Verification
# ============================================================================

def test_action_item_evidence_verification(structured_citation_env, db_session: Session):
    """Verify committed ActionItem with valid evidence_segment_id produces verified Citation."""
    env = structured_citation_env
    org_a = env["org_a"]
    m_a1 = env["m_a1"]
    seg_action = env["seg_a1_2"]

    ai = ActionItem(
        meeting_id=m_a1.id,
        description="Configure production Kafka cluster",
        owner_name="Alex",
        status=ActionItemStatus.PENDING.value,
        evidence_segment_id=seg_action.id,
    )
    db_session.add(ai)
    db_session.commit()

    engine = RAGEngine()
    req = RAGQueryRequest(
        question="What action items or commitments were assigned to Alex?",
        organization_id=org_a.id,
    )
    resp: RAGQueryResponse = engine.query(db=db_session, request=req)

    assert not resp.is_abstained
    assert "Configure production Kafka cluster" in resp.answer
    assert "Alex" in resp.answer

    # Citation produced for action item evidence
    assert len(resp.citations) >= 1
    cit = next((c for c in resp.citations if seg_action.id in c.segment_ids), None)
    assert cit is not None
    assert cit.start_time == 30.0
    assert cit.end_time == 45.0
    assert "October 15th" in cit.quoted_text


# ============================================================================
# 8. UnresolvedQuestion Evidence Verification
# ============================================================================

def test_unresolved_question_evidence_verification(structured_citation_env, db_session: Session):
    """Verify UnresolvedQuestion with valid evidence_segment_id produces verified Citation."""
    env = structured_citation_env
    org_a = env["org_a"]
    m_a1 = env["m_a1"]
    seg_q = env["seg_a1_3"]

    uq = UnresolvedQuestion(
        meeting_id=m_a1.id,
        question_text="Do we have budget approval for the managed Kafka hosting tier?",
        raised_by="DevOps Lead",
        status=QuestionStatus.OPEN.value,
        evidence_segment_id=seg_q.id,
    )
    db_session.add(uq)
    db_session.commit()

    engine = RAGEngine()
    req = RAGQueryRequest(
        question="What unresolved questions remain regarding Kafka?",
        organization_id=org_a.id,
    )
    resp: RAGQueryResponse = engine.query(db=db_session, request=req)

    assert not resp.is_abstained
    assert "budget approval for the managed Kafka" in resp.answer

    # Citation produced for question evidence
    assert len(resp.citations) >= 1
    cit = next((c for c in resp.citations if seg_q.id in c.segment_ids), None)
    assert cit is not None
    assert cit.start_time == 50.0
    assert cit.end_time == 65.0
    assert "managed Kafka hosting tier" in cit.quoted_text


# ============================================================================
# 9. Strict Abstention Preserves Empty Citations
# ============================================================================

def test_strict_abstention_preserves_empty_citations(structured_citation_env, db_session: Session):
    """Verify that when an answer abstains with NO_EVIDENCE, citations list is strictly empty."""
    env = structured_citation_env
    org_a = env["org_a"]

    engine = RAGEngine()
    req = RAGQueryRequest(
        question="What decisions were made about quantum superconducting circuits?",
        organization_id=org_a.id,
    )
    resp: RAGQueryResponse = engine.query(db=db_session, request=req)

    assert resp.is_abstained
    assert resp.abstention_status == AbstentionStatus.NO_EVIDENCE
    assert len(resp.citations) == 0


# ============================================================================
# 10. Combined Retrieval: Chunk and Decision Citations with Deduplication
# ============================================================================

def test_combined_retrieval_chunk_and_decision_citations(structured_citation_env, db_session: Session):
    """Verify combined verification assigns sequential citation IDs and deduplicates identical segments."""
    from app.models.memory import TranscriptChunkRecord
    from app.services.memory.rag.types import EvidenceItem
    from app.services.memory.structured_retrieval import StructuredMemoryContext
    from app.schemas.intelligence import CrossMeetingDecisionResponse

    env = structured_citation_env
    org_a = env["org_a"]
    m_a1 = env["m_a1"]
    seg_1 = env["seg_a1_1"]
    seg_2 = env["seg_a1_2"]

    # 1. Stored Chunk Record in DB for seg_1
    chunk_rec = TranscriptChunkRecord(
        id="chk-combo-1",
        chunk_index=0,
        meeting_id=m_a1.id,
        organization_id=org_a.id,
        transcript_id=env["t_safe_a1"].id,
        segment_ids=f'["{seg_1.id}"]',
        language_codes='["en"]',
        dominant_language="en",
        text=seg_1.text,
        start_time=seg_1.start_time,
        end_time=seg_1.end_time,
        token_count=15,
        dimension=384,
        embedding=b"\x00" * (384 * 4),
        model_name="test-model",
    )
    db_session.add(chunk_rec)
    db_session.commit()

    evidence_chunk = EvidenceItem(
        chunk_id=chunk_rec.id,
        organization_id=org_a.id,
        meeting_id=m_a1.id,
        meeting_title=m_a1.title,
        meeting_date=m_a1.actual_start_time.isoformat(),
        transcript_id=env["t_safe_a1"].id,
        segment_ids=[seg_1.id],
        start_time=seg_1.start_time,
        end_time=seg_1.end_time,
        text=seg_1.text,
        rrf_score=0.9,
    )

    # 2. Decision referencing seg_2 (different segment)
    dec_diff = CrossMeetingDecisionResponse(
        id="dec-diff-1",
        meeting_id=m_a1.id,
        description="Configure Kafka parameters",
        status="ACTIVE",
        evidence_segment_id=seg_2.id,
        meeting_title=m_a1.title,
        meeting_date=m_a1.actual_start_time,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    # 3. Decision referencing seg_1 (shared segment with chunk_rec)
    dec_shared = CrossMeetingDecisionResponse(
        id="dec-shared-2",
        meeting_id=m_a1.id,
        description="Adopt Kafka async",
        status="ACTIVE",
        evidence_segment_id=seg_1.id,
        meeting_title=m_a1.title,
        meeting_date=m_a1.actual_start_time,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    struct_ctx = StructuredMemoryContext(
        decisions=[dec_diff, dec_shared],
        has_structured_evidence=True,
    )

    verifier = CitationVerifier()
    verified = verifier.verify_citations(
        db=db_session,
        cited_chunk_ids=[chunk_rec.id],
        retrieved_evidence=[evidence_chunk],
        organization_id=org_a.id,
        structured_context=struct_ctx,
    )

    # Should have 2 citations: cit-1 for the chunk (seg_1), and cit-2 for the decision (seg_2).
    # dec_shared (seg_1) is deduplicated because seg_1 was already cited!
    assert len(verified) == 2
    assert verified[0].citation_id == "cit-1"
    assert verified[0].chunk_id == chunk_rec.id
    assert verified[0].segment_ids == [seg_1.id]

    assert verified[1].citation_id == "cit-2"
    assert verified[1].chunk_id == dec_diff.id
    assert verified[1].segment_ids == [seg_2.id]

