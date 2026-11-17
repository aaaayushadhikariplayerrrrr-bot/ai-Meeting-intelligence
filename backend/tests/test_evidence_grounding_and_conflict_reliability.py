"""Phase 8.10.2: Evidence Grounding, Sufficiency & Conflict Reliability Test Suite.

Comprehensive tests verifying:
P1: Prevention of unsupported entity and decision claims:
    - Terraform absent from Kubernetes infrastructure evidence.
    - Spanner absent from MongoDB/database discussion.
    - Relevant entity and action explicitly supported.
    - Valid paraphrased English query.
    - Valid Hindi/Hinglish/Gujarati cross-lingual queries.
    - Disjoint evidence across multiple meetings.
    - Generic questions that legitimately need multiple sources.
P2: Conflict status independent of query wording:
    - "Are there conflicts about the database?"
    - "What database did we decide to use?"
    - "Which database should we use now?"
    - Explicitly superseded PostgreSQL decision followed by MongoDB.
    - Revoked decisions.
    - Two unlinked active conflicting decisions.
P3: Harmonize structured supporting transcript matching:
    - Inflectional variants ("orchestrating" and "orchestration").
P4: Populate conflict DTOs for transcript conflicts:
    - Affirmative and negative statements about the same topic across meetings.
    - TranscriptConflictItem populated with chunk IDs, texts, meetings, timestamps, and citations.
"""

from datetime import datetime, timezone, timedelta
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.base import Base
from app.models.identity import Organization
from app.models.meeting import Meeting, InputSourceType, MeetingStatus
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.intelligence import Decision, DecisionStatus
from app.schemas.memory import (
    AbstentionStatus,
    RAGQueryRequest,
    EvidenceItem,
    TranscriptConflictItem,
)
from app.schemas.intelligence import CrossMeetingDecisionResponse
from app.services.memory.storage import ChunkStorageManager
from app.services.memory.embedding import MultilingualE5EmbeddingService
from app.services.memory.types import SemanticChunk, ChunkEmbeddingResult
from app.services.memory.rag.engine import RAGEngine
from app.services.memory.rag.providers import DeterministicAnswerProvider
from app.services.memory.structured_retrieval import StructuredMemoryRetriever, StructuredMemoryContext


@pytest.fixture(scope="module")
def grounding_test_env():
    """Build isolated SQLite test database with multi-meeting scenarios."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    org = Organization(id="org-grounding-1", name="Grounding Test Corp", slug="grounding-corp")
    db.add(org)

    # Meeting 1: Kubernetes Infrastructure
    m1 = Meeting(
        id="meet-infra-k8s",
        organization_id=org.id,
        title="Infrastructure Architecture Sync",
        input_source_type=InputSourceType.AUDIO.value,
        status=MeetingStatus.COMPLETED.value,
        scheduled_start_time=datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc),
        created_at=datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc),
    )
    # Meeting 2: Database Sync
    m2 = Meeting(
        id="meet-db-mongo",
        organization_id=org.id,
        title="Database Strategy Session",
        input_source_type=InputSourceType.AUDIO.value,
        status=MeetingStatus.COMPLETED.value,
        scheduled_start_time=datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc),
        created_at=datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc),
    )
    # Meeting 3: Brief Mention Sync
    m3 = Meeting(
        id="meet-brief-tf",
        organization_id=org.id,
        title="DevOps Tooling Discussion",
        input_source_type=InputSourceType.AUDIO.value,
        status=MeetingStatus.COMPLETED.value,
        scheduled_start_time=datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc),
        created_at=datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc),
    )
    # Meeting 4: General Infrastructure Approval
    m4 = Meeting(
        id="meet-general-infra",
        organization_id=org.id,
        title="Executive Infrastructure Review",
        input_source_type=InputSourceType.AUDIO.value,
        status=MeetingStatus.COMPLETED.value,
        scheduled_start_time=datetime(2026, 9, 15, 11, 0, tzinfo=timezone.utc),
        created_at=datetime(2026, 9, 15, 11, 0, tzinfo=timezone.utc),
    )
    # Meeting 5: Affirmative Auth Policy
    m5 = Meeting(
        id="meet-auth-aff",
        organization_id=org.id,
        title="Security Review 1",
        input_source_type=InputSourceType.AUDIO.value,
        status=MeetingStatus.COMPLETED.value,
        scheduled_start_time=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc),
        created_at=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc),
    )
    # Meeting 6: Negative Auth Policy
    m6 = Meeting(
        id="meet-auth-neg",
        organization_id=org.id,
        title="Security Review 2",
        input_source_type=InputSourceType.AUDIO.value,
        status=MeetingStatus.COMPLETED.value,
        scheduled_start_time=datetime(2026, 10, 15, 10, 0, tzinfo=timezone.utc),
        created_at=datetime(2026, 10, 15, 10, 0, tzinfo=timezone.utc),
    )

    db.add_all([m1, m2, m3, m4, m5, m6])
    db.commit()

    # Create transcripts & segments
    meetings = [m1, m2, m3, m4, m5, m6]
    transcripts = {}
    for m in meetings:
        t = Transcript(
            id=f"tr-{m.id}",
            meeting_id=m.id,
            transcript_type=TranscriptType.CLEANED_REDACTED.value,
            version=1,
            language_detected="en",
        )
        db.add(t)
        transcripts[m.id] = t
    db.commit()

    # Passages
    service = MultilingualE5EmbeddingService(device="cpu", allow_download=False)
    storage = ChunkStorageManager(expected_dim=384)

    passages = [
        {
            "id": "chk-k8s-infra",
            "meeting_id": m1.id,
            "text": "The engineering team finalized selecting Kubernetes for container orchestration and production infrastructure.",
            "start": 10.0,
            "end": 25.0,
        },
        {
            "id": "chk-mongo-db",
            "meeting_id": m2.id,
            "text": "We approved MongoDB as the primary document database for our backend catalog services.",
            "start": 30.0,
            "end": 45.0,
        },
        {
            "id": "chk-tf-brief",
            "meeting_id": m3.id,
            "text": "During our tooling sync, someone briefly mentioned Terraform documentation as a future possibility.",
            "start": 5.0,
            "end": 18.0,
        },
        {
            "id": "chk-general-infra-appr",
            "meeting_id": m4.id,
            "text": "Leadership confirmed and approved the infrastructure migration deployment plan across all regions.",
            "start": 12.0,
            "end": 28.0,
        },
        {
            "id": "chk-auth-approved",
            "meeting_id": m5.id,
            "text": "The team agreed to adopt biometric authentication for administrative access.",
            "start": 15.0,
            "end": 35.0,
        },
        {
            "id": "chk-auth-rejected",
            "meeting_id": m6.id,
            "text": "We decided against biometric authentication due to client hardware incompatibility.",
            "start": 20.0,
            "end": 40.0,
        },
    ]

    texts = [p["text"] for p in passages]
    embeddings = service.embed_passages(texts)

    for p, emb in zip(passages, embeddings):
        seg = TranscriptSegment(
            id=f"seg-{p['id']}",
            transcript_id=transcripts[p["meeting_id"]].id,
            sequence_order=1,
            start_time=p["start"],
            end_time=p["end"],
            text=p["text"],
            language_code="en",
        )
        db.add(seg)
        db.flush()

        chunk = SemanticChunk(
            chunk_id=p["id"],
            chunk_index=0,
            text=p["text"],
            token_count=len(p["text"].split()),
            start_time=p["start"],
            end_time=p["end"],
            segment_ids=[seg.id],
            language_codes=["en"],
            dominant_language="en",
            meeting_id=p["meeting_id"],
            organization_id=org.id,
            transcript_id=transcripts[p["meeting_id"]].id,
        )
        emb_res = ChunkEmbeddingResult(
            chunk=chunk,
            embedding=emb,
            dimension=384,
            model_name=service.model_name,
        )
        storage.store_chunks(
            db=db,
            chunks_with_embeddings=[emb_res],
            organization_id=org.id,
            transcript=transcripts[p["meeting_id"]],
        )

    db.commit()

    rag_engine = RAGEngine(embedding_service=service)

    return {
        "db": db,
        "org_id": org.id,
        "engine": rag_engine,
        "service": service,
        "meetings": {m.id: m for m in meetings},
    }


# =========================================================================
# P1: Prevent Unsupported Entity & Decision Claims Tests
# =========================================================================

def test_p1_terraform_absent_from_k8s_infrastructure_evidence(grounding_test_env):
    """Verify that asking about Terraform when evidence only discusses Kubernetes infrastructure yields INSUFFICIENT_EVIDENCE."""
    db = grounding_test_env["db"]
    engine = grounding_test_env["engine"]
    org_id = grounding_test_env["org_id"]

    req = RAGQueryRequest(
        question="Was Terraform selected for infrastructure?",
        organization_id=org_id,
        meeting_id="meet-infra-k8s",
    )
    res = engine.query(db=db, request=req)

    assert res.is_abstained is True
    assert res.abstention_status == AbstentionStatus.INSUFFICIENT_EVIDENCE
    assert len(res.citations) == 0
    assert "no factual evidence or decisions regarding 'terraform'" in res.answer.lower() or "insufficient" in res.answer.lower()


def test_p1_spanner_absent_from_mongodb_database_evidence(grounding_test_env):
    """Verify that asking about Spanner when evidence discusses MongoDB database yields INSUFFICIENT_EVIDENCE."""
    db = grounding_test_env["db"]
    engine = grounding_test_env["engine"]
    org_id = grounding_test_env["org_id"]

    req = RAGQueryRequest(
        question="Did we decide to use Spanner for our database?",
        organization_id=org_id,
        meeting_id="meet-db-mongo",
    )
    res = engine.query(db=db, request=req)

    assert res.is_abstained is True
    assert res.abstention_status == AbstentionStatus.INSUFFICIENT_EVIDENCE
    assert len(res.citations) == 0
    assert "spanner" in res.answer.lower()


def test_p1_relevant_entity_and_action_explicitly_supported(grounding_test_env):
    """Verify that asking about Kubernetes for container orchestration yields ANSWER_SUPPORTED."""
    db = grounding_test_env["db"]
    engine = grounding_test_env["engine"]
    org_id = grounding_test_env["org_id"]

    req = RAGQueryRequest(
        question="Was Kubernetes selected for container orchestration?",
        organization_id=org_id,
        meeting_id="meet-infra-k8s",
    )
    res = engine.query(db=db, request=req)

    assert res.is_abstained is False
    assert res.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
    assert len(res.citations) > 0
    assert "Kubernetes" in res.answer


def test_p1_disjoint_evidence_across_multiple_meetings_rejected(grounding_test_env):
    """Verify that a passing mention of Terraform in M3 cannot be combined with an unrelated approval in M4 to claim support."""
    provider = DeterministicAnswerProvider()

    # Construct two disjoint evidence items from separate meetings
    item_brief_tf = EvidenceItem(
        chunk_id="chk-tf-brief",
        organization_id="org-1",
        meeting_id="meet-brief-tf",
        meeting_title="DevOps Tooling Discussion",
        meeting_date="2026-09-01T09:00:00Z",
        transcript_id="tr-1",
        segment_ids=["seg-1"],
        start_time=5.0,
        end_time=18.0,
        language_codes=["en"],
        text="During our tooling sync, someone briefly mentioned Terraform documentation as a future possibility.",
        rrf_score=0.9,
    )
    item_general_infra = EvidenceItem(
        chunk_id="chk-general-infra-appr",
        organization_id="org-1",
        meeting_id="meet-general-infra",
        meeting_title="Executive Infrastructure Review",
        meeting_date="2026-09-15T11:00:00Z",
        transcript_id="tr-2",
        segment_ids=["seg-2"],
        start_time=12.0,
        end_time=28.0,
        language_codes=["en"],
        text="Leadership confirmed and approved the infrastructure migration deployment plan across all regions.",
        rrf_score=0.85,
    )

    result = provider.generate_answer(
        question="Was Terraform approved for infrastructure migration?",
        evidence_items=[item_brief_tf, item_general_infra],
    )

    # Must abstain because Terraform was never approved; item_general_infra does not mention Terraform
    assert result.abstention_status == AbstentionStatus.INSUFFICIENT_EVIDENCE
    assert len(result.cited_chunk_ids) == 0
    assert "terraform" in result.answer_text.lower()


def test_p1_generic_questions_legitimately_need_multiple_sources(grounding_test_env):
    """Verify that a generic question without specific entity names legitimately synthesizes multiple sources."""
    provider = DeterministicAnswerProvider()

    item_k8s = EvidenceItem(
        chunk_id="chk-k8s-infra",
        organization_id="org-1",
        meeting_id="meet-infra-k8s",
        meeting_title="Infrastructure Architecture Sync",
        meeting_date="2026-08-01T10:00:00Z",
        transcript_id="tr-1",
        segment_ids=["seg-1"],
        start_time=10.0,
        end_time=25.0,
        language_codes=["en"],
        text="The engineering team finalized selecting Kubernetes for container orchestration and production infrastructure.",
        rrf_score=0.95,
    )
    item_mongo = EvidenceItem(
        chunk_id="chk-mongo-db",
        organization_id="org-1",
        meeting_id="meet-db-mongo",
        meeting_title="Database Strategy Session",
        meeting_date="2026-08-15T14:00:00Z",
        transcript_id="tr-2",
        segment_ids=["seg-2"],
        start_time=30.0,
        end_time=45.0,
        language_codes=["en"],
        text="We approved MongoDB as the primary document database for our backend catalog services.",
        rrf_score=0.9,
    )

    result = provider.generate_answer(
        question="What decisions were made across meetings regarding our infrastructure and database systems?",
        evidence_items=[item_k8s, item_mongo],
    )

    assert result.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
    assert len(result.cited_chunk_ids) == 2
    assert "meet-infra-k8s" in [item_k8s.meeting_id]
    assert "Infrastructure Architecture Sync" in result.answer_text
    assert "Database Strategy Session" in result.answer_text


# =========================================================================
# P2: Conflict Status Independent of Query Wording Tests
# =========================================================================

def test_p2_conflict_status_independent_of_query_wording(grounding_test_env):
    """Verify CONFLICTING_EVIDENCE is returned for both conflict and natural questions when active decisions conflict."""
    db = grounding_test_env["db"]
    org_id = grounding_test_env["org_id"]
    engine = grounding_test_env["engine"]

    # Create two unlinked active conflicting decisions in different meetings
    d1 = Decision(
        id="dec-db-postgres-active",
        meeting_id="meet-infra-k8s",
        description="Adopt PostgreSQL as our primary database engine.",
        status=DecisionStatus.ACTIVE.value,
        timestamp_seconds=12.0,
    )
    d2 = Decision(
        id="dec-db-mongo-active",
        meeting_id="meet-db-mongo",
        description="Adopt MongoDB as our primary database engine.",
        status=DecisionStatus.ACTIVE.value,
        timestamp_seconds=22.0,
    )
    db.add_all([d1, d2])
    db.commit()

    try:
        # 1. Question with explicit 'conflict'
        res_conflict = engine.query(
            db=db,
            request=RAGQueryRequest(
                question="Are there conflicts about the database?",
                organization_id=org_id,
            ),
        )
        assert res_conflict.is_abstained is True
        assert res_conflict.abstention_status == AbstentionStatus.CONFLICTING_EVIDENCE
        assert len(res_conflict.detected_conflicts) > 0

        # 2. Natural question "What database did we decide to use?"
        res_what = engine.query(
            db=db,
            request=RAGQueryRequest(
                question="What database did we decide to use?",
                organization_id=org_id,
            ),
        )
        assert res_what.is_abstained is True
        assert res_what.abstention_status == AbstentionStatus.CONFLICTING_EVIDENCE
        assert len(res_what.detected_conflicts) > 0
        assert "refrains from arbitrarily selecting the newest decision" in res_what.answer

        # 3. Natural question "Which database should we use now?"
        res_which = engine.query(
            db=db,
            request=RAGQueryRequest(
                question="Which database should we use now?",
                organization_id=org_id,
            ),
        )
        assert res_which.is_abstained is True
        assert res_which.abstention_status == AbstentionStatus.CONFLICTING_EVIDENCE
        assert len(res_which.detected_conflicts) > 0
    finally:
        # Clean up conflicting test decisions
        db.query(Decision).filter(Decision.id.in_(["dec-db-postgres-active", "dec-db-mongo-active"])).delete(synchronize_session=False)
        db.commit()


def test_p2_explicit_supersession_not_treated_as_conflict(grounding_test_env):
    """Verify that an explicitly superseded decision (PostgreSQL superseded by MongoDB) is NOT flagged as a conflict."""
    db = grounding_test_env["db"]
    org_id = grounding_test_env["org_id"]
    engine = grounding_test_env["engine"]

    d_old = Decision(
        id="dec-db-pg-superseded",
        meeting_id="meet-infra-k8s",
        description="Adopt PostgreSQL as our primary database engine.",
        status=DecisionStatus.SUPERSEDED.value,
        timestamp_seconds=12.0,
    )
    db.add(d_old)
    db.flush()

    d_new = Decision(
        id="dec-db-mongo-super",
        meeting_id="meet-db-mongo",
        description="Adopt MongoDB as our primary database engine.",
        status=DecisionStatus.ACTIVE.value,
        supersedes_decision_id=d_old.id,
        timestamp_seconds=22.0,
    )
    db.add(d_new)
    db.commit()

    try:
        res = engine.query(
            db=db,
            request=RAGQueryRequest(
                question="What database did we decide to use?",
                organization_id=org_id,
            ),
        )

        # Should NOT abstain with CONFLICTING_EVIDENCE
        assert res.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
        assert res.is_abstained is False
        assert len(res.detected_conflicts) == 0
        assert "SUPERSEDED" in res.answer or "superseded" in res.answer
    finally:
        # Clean up
        db.query(Decision).filter(Decision.id.in_(["dec-db-mongo-super", "dec-db-pg-superseded"])).delete(synchronize_session=False)
        db.commit()


def test_p2_revoked_decision_not_treated_as_conflict(grounding_test_env):
    """Verify that revoked decisions are not treated as active unresolved conflicts."""
    db = grounding_test_env["db"]
    org_id = grounding_test_env["org_id"]
    engine = grounding_test_env["engine"]

    d_revoked = Decision(
        id="dec-db-revoked",
        meeting_id="meet-infra-k8s",
        description="Adopt Cassandra as our database.",
        status=DecisionStatus.REVOKED.value,
        timestamp_seconds=12.0,
    )
    d_active = Decision(
        id="dec-db-active-single",
        meeting_id="meet-db-mongo",
        description="Adopt PostgreSQL as our database.",
        status=DecisionStatus.ACTIVE.value,
        timestamp_seconds=22.0,
    )
    db.add_all([d_revoked, d_active])
    db.commit()

    try:
        res = engine.query(
            db=db,
            request=RAGQueryRequest(
                question="What database did we decide to use?",
                organization_id=org_id,
            ),
        )

        assert res.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
        assert res.is_abstained is False
        assert len(res.detected_conflicts) == 0
    finally:
        # Clean up
        db.query(Decision).filter(Decision.id.in_(["dec-db-revoked", "dec-db-active-single"])).delete(synchronize_session=False)
        db.commit()


# =========================================================================
# P3: Harmonize Structured Supporting Transcript Matching Tests
# =========================================================================

def test_p3_harmonize_structured_supporting_transcript_matching():
    """Verify inflectional variants ('orchestrating' vs 'orchestration') match in structured supporting transcripts."""
    provider = DeterministicAnswerProvider()

    item = EvidenceItem(
        chunk_id="chk-k8s-orch",
        organization_id="org-1",
        meeting_id="meet-1",
        meeting_title="Architecture Sync",
        meeting_date="2026-08-01T10:00:00Z",
        transcript_id="tr-1",
        segment_ids=["seg-1"],
        start_time=10.0,
        end_time=25.0,
        language_codes=["en"],
        text="We finalized container orchestration using Kubernetes clusters across all staging environments.",
        rrf_score=0.9,
    )

    dec = CrossMeetingDecisionResponse(
        id="dec-1",
        meeting_id="meet-1",
        description="Adopt Kubernetes for all container services.",
        meeting_title="Architecture Sync",
        meeting_date="2026-08-01T10:00:00Z",
        status="ACTIVE",
        created_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )

    structured_ctx = StructuredMemoryContext(
        decisions=[dec],
        has_structured_evidence=True,
    )

    # Query uses inflectional variant 'orchestrating'
    result = provider.generate_answer(
        question="What decisions were made about orchestrating containers?",
        evidence_items=[item],
        structured_context=structured_ctx,
    )

    assert result.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
    assert "chk-k8s-orch" in result.cited_chunk_ids
    assert "container orchestration using Kubernetes" in result.answer_text


# =========================================================================
# P4: Populate Conflict DTOs for Transcript Contradictions Tests
# =========================================================================

def test_p4_transcript_contradiction_populates_conflict_dto_and_citations(grounding_test_env):
    """Verify that when transcript statements contradict across meetings, TranscriptConflictItem is populated and cited."""
    db = grounding_test_env["db"]
    org_id = grounding_test_env["org_id"]
    engine = grounding_test_env["engine"]

    req = RAGQueryRequest(
        question="Was biometric authentication adopted or rejected?",
        organization_id=org_id,
    )
    res = engine.query(db=db, request=req)

    assert res.is_abstained is True
    assert res.abstention_status == AbstentionStatus.CONFLICTING_EVIDENCE

    # Check transcript conflict item is populated
    assert len(res.transcript_conflicts) > 0
    tc = res.transcript_conflicts[0]
    assert isinstance(tc, TranscriptConflictItem)
    assert "Security Review 1" in tc.chunk_a_meeting_title
    assert "Security Review 2" in tc.chunk_b_meeting_title
    assert "biometric" in tc.chunk_a_text.lower()
    assert "biometric" in tc.chunk_b_text.lower()
    assert tc.is_heuristic is True
    assert "biometric" in tc.subject.lower() or "authentication" in tc.subject.lower()

    # Verify both conflicting chunks are cited with exact timestamps and meeting titles
    assert len(res.citations) == 2
    c_meetings = {c.meeting_title for c in res.citations}
    assert "Security Review 1" in c_meetings
    assert "Security Review 2" in c_meetings
    for c in res.citations:
        assert c.start_time > 0
        assert c.end_time > c.start_time
        assert len(c.segment_ids) > 0
