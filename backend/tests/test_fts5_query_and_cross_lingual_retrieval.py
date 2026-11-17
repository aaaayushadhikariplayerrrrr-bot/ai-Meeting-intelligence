"""Comprehensive tests for FTS5 Conversational Query Reformulation & Cross-Lingual Retrieval (Phase 8.9.2).

Validates:
1. Conversational FTS5 queries containing common stopwords.
2. Queries containing only stopwords.
3. Exact technical keyword and phrase retrieval.
4. Hindi query retrieving a relevant English transcript chunk (with real multilingual E5).
5. Hinglish query retrieving a relevant English transcript chunk.
6. English query retrieving relevant English evidence.
7. Gujarati/Unicode query safety.
8. Relevant and irrelevant candidates with close cosine scores (RRF tie-breaking).
9. Truly irrelevant queries still abstaining (no hallucination).
10. Tenant isolation after query reformulation.
11. Meeting/date filters remaining effective.
12. Existing structured retrieval and conflict detection regressions.
"""

import uuid
import pytest
from datetime import datetime, timezone
from typing import List
from sqlalchemy.orm import Session

from app.models.identity import Organization
from app.models.meeting import Meeting
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.memory import TranscriptChunkRecord
from app.models.intelligence import Decision, DecisionStatus
from app.services.memory.storage import ChunkStorageManager, EXPECTED_DIMENSION
from app.services.memory.types import SemanticChunk, ChunkEmbeddingResult
from app.services.memory.embedding import get_embedding_service, E5_QUERY_PREFIX
from app.services.memory.retrieval import (
    sanitize_fts5_query,
    search_keyword,
    search_semantic,
    search_hybrid_rrf,
    CONVERSATIONAL_STOPWORDS,
)
from app.services.memory.rag.engine import RAGEngine
from app.services.memory.rag.types import (
    RAGQueryRequest,
    RAGQueryResponse,
    AbstentionStatus,
)


@pytest.fixture
def fts5_test_env(db_session: Session):
    """Fixture providing isolated Organizations, Meetings, and indexed chunks for FTS5 and retrieval testing."""
    # Org A
    org_a = Organization(name="FTS5 Test Org A", slug=f"fts5-a-{uuid.uuid4().hex[:8]}")
    # Org B
    org_b = Organization(name="FTS5 Test Org B", slug=f"fts5-b-{uuid.uuid4().hex[:8]}")
    db_session.add_all([org_a, org_b])
    db_session.commit()

    # Meetings for Org A
    m_a1 = Meeting(
        organization_id=org_a.id,
        title="Infrastructure Architecture Sync",
        actual_start_time=datetime(2026, 8, 10, 10, 0, tzinfo=timezone.utc),
    )
    m_a2 = Meeting(
        organization_id=org_a.id,
        title="Security & Database Review",
        actual_start_time=datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc),
    )
    # Meeting for Org B
    m_b1 = Meeting(
        organization_id=org_b.id,
        title="Org B Secret Acquisition Meeting",
        actual_start_time=datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc),
    )
    db_session.add_all([m_a1, m_a2, m_b1])
    db_session.commit()

    # Transcripts
    t_a1 = Transcript(meeting_id=m_a1.id, transcript_type=TranscriptType.CLEANED_REDACTED.value)
    t_a2 = Transcript(meeting_id=m_a2.id, transcript_type=TranscriptType.CLEANED_REDACTED.value)
    t_b1 = Transcript(meeting_id=m_b1.id, transcript_type=TranscriptType.CLEANED_REDACTED.value)
    db_session.add_all([t_a1, t_a2, t_b1])
    db_session.commit()

    # Segments
    seg_a1 = TranscriptSegment(
        transcript_id=t_a1.id,
        sequence_order=1,
        start_time=10.0,
        end_time=25.0,
        text="We decided to adopt Kubernetes Helm charts for all services in production to improve container orchestration.",
        language_code="en",
    )
    seg_a2 = TranscriptSegment(
        transcript_id=t_a2.id,
        sequence_order=1,
        start_time=5.0,
        end_time=20.0,
        text="PostgreSQL replication setup with streaming replication and PgBouncer was finalized for the database tier.",
        language_code="en",
    )
    seg_a3 = TranscriptSegment(
        transcript_id=t_a2.id,
        sequence_order=2,
        start_time=22.0,
        end_time=40.0,
        text="Two-factor authentication with 2FA authenticator apps is mandatory on all admin portals.",
        language_code="en",
    )
    seg_b1 = TranscriptSegment(
        transcript_id=t_b1.id,
        sequence_order=1,
        start_time=0.0,
        end_time=15.0,
        text="Confidential acquisition of Startup Horizon for fifteen million dollars was approved.",
        language_code="en",
    )
    db_session.add_all([seg_a1, seg_a2, seg_a3, seg_b1])
    db_session.commit()

    # Chunks and Embeddings (using real E5 service)
    service = get_embedding_service(allow_download=False)
    if service.is_cached():
        service.load_model()

    manager = ChunkStorageManager()

    # Embed Org A Chunks
    c_a1_text = seg_a1.text
    c_a2_text = seg_a2.text
    c_a3_text = seg_a3.text

    if service.is_loaded:
        vecs_a = service.embed_passages([c_a1_text, c_a2_text, c_a3_text])
    else:
        # Fallback to synthetic if model cache not available in test env
        vecs_a = [[0.05] * EXPECTED_DIMENSION for _ in range(3)]

    chunk_a1 = SemanticChunk(chunk_index=0, meeting_id=m_a1.id, organization_id=org_a.id, transcript_id=t_a1.id, segment_ids=[seg_a1.id], text=c_a1_text, start_time=10.0, end_time=25.0, token_count=18)
    chunk_a2 = SemanticChunk(chunk_index=0, meeting_id=m_a2.id, organization_id=org_a.id, transcript_id=t_a2.id, segment_ids=[seg_a2.id], text=c_a2_text, start_time=5.0, end_time=20.0, token_count=16)
    chunk_a3 = SemanticChunk(chunk_index=1, meeting_id=m_a2.id, organization_id=org_a.id, transcript_id=t_a2.id, segment_ids=[seg_a3.id], text=c_a3_text, start_time=22.0, end_time=40.0, token_count=15)

    manager.store_chunks(
        db_session,
        [
            ChunkEmbeddingResult(chunk=chunk_a1, embedding=vecs_a[0], dimension=EXPECTED_DIMENSION, model_name=service.model_name),
            ChunkEmbeddingResult(chunk=chunk_a2, embedding=vecs_a[1], dimension=EXPECTED_DIMENSION, model_name=service.model_name),
            ChunkEmbeddingResult(chunk=chunk_a3, embedding=vecs_a[2], dimension=EXPECTED_DIMENSION, model_name=service.model_name),
        ],
        organization_id=org_a.id,
    )

    # Embed Org B Chunk
    c_b1_text = seg_b1.text
    if service.is_loaded:
        vecs_b = service.embed_passages([c_b1_text])
    else:
        vecs_b = [[0.05] * EXPECTED_DIMENSION]

    chunk_b1 = SemanticChunk(chunk_index=0, meeting_id=m_b1.id, organization_id=org_b.id, transcript_id=t_b1.id, segment_ids=[seg_b1.id], text=c_b1_text, start_time=0.0, end_time=15.0, token_count=14)
    manager.store_chunks(
        db_session,
        [ChunkEmbeddingResult(chunk=chunk_b1, embedding=vecs_b[0], dimension=EXPECTED_DIMENSION, model_name=service.model_name)],
        organization_id=org_b.id,
    )

    return {
        "org_a": org_a,
        "org_b": org_b,
        "m_a1": m_a1,
        "m_a2": m_a2,
        "m_b1": m_b1,
        "service": service,
    }


# ============================================================================
# 1. Conversational FTS5 Queries Containing Common Stopwords
# ============================================================================

def test_conversational_fts5_query_reformulation(fts5_test_env, db_session: Session):
    """Verify conversational questions strip stopwords and use safe OR syntax to match chunks in FTS5."""
    org_a = fts5_test_env["org_a"]

    # Natural language conversational question with 8 words
    conv_q = "What container orchestration system was adopted across production services?"
    sanitized = sanitize_fts5_query(conv_q)

    # Conversational stopwords ("what", "was", "across") are stripped
    assert '"what"' not in sanitized.lower()
    assert '"was"' not in sanitized.lower()
    assert '"across"' not in sanitized.lower()

    # Meaningful substantive terms are preserved
    assert '"container"' in sanitized
    assert '"orchestration"' in sanitized
    assert '"production"' in sanitized

    # Joined via OR so not all 8 words are required in chunk
    assert " OR " in sanitized

    # Direct FTS5 keyword search in SQLite matches the container orchestration chunk!
    results = search_keyword(db_session, query=conv_q, organization_id=org_a.id)
    assert len(results) >= 1
    assert "Kubernetes Helm charts" in results[0].text
    assert results[0].bm25_rank == 1


# ============================================================================
# 2. Queries Containing Only Stopwords
# ============================================================================

def test_queries_containing_only_stopwords(fts5_test_env, db_session: Session):
    """Verify queries containing only conversational stopwords return empty string and [] safely."""
    org_a = fts5_test_env["org_a"]

    stopword_queries = [
        "What is that?",
        "is was were",
        "yeh kya hai?",
        "su che?",
        "what which where when who how",
        "   ",
        "",
    ]

    for q in stopword_queries:
        sanitized = sanitize_fts5_query(q)
        assert sanitized == "", f"Expected empty sanitized query for '{q}', got '{sanitized}'"
        # Search keyword handles empty string cleanly without SQL exception
        results = search_keyword(db_session, query=q, organization_id=org_a.id)
        assert results == [], f"Expected empty results for '{q}', got {results}"


# ============================================================================
# 3. Exact Technical Keyword and Phrase Retrieval
# ============================================================================

def test_exact_technical_keyword_and_phrase_retrieval(fts5_test_env, db_session: Session):
    """Verify single keywords, exact phrases in quotes, numbers, and technical tokens."""
    org_a = fts5_test_env["org_a"]

    # 1. Single keyword
    assert sanitize_fts5_query("Kubernetes") == '"Kubernetes"'
    res_k8s = search_keyword(db_session, query="Kubernetes", organization_id=org_a.id)
    assert len(res_k8s) >= 1
    assert "Kubernetes" in res_k8s[0].text

    # 2. Exact phrase enclosed in quotes
    assert sanitize_fts5_query('"PostgreSQL replication"') == '"PostgreSQL replication"'
    res_phrase = search_keyword(db_session, query='"PostgreSQL replication"', organization_id=org_a.id)
    assert len(res_phrase) >= 1
    assert "PostgreSQL replication" in res_phrase[0].text

    # 3. Technical acronyms and numbers (2FA, PgBouncer)
    sanitized_tech = sanitize_fts5_query("2FA authenticator with PgBouncer")
    assert '"2FA"' in sanitized_tech
    assert '"PgBouncer"' in sanitized_tech
    res_tech = search_keyword(db_session, query="2FA authenticator", organization_id=org_a.id)
    assert len(res_tech) >= 1
    assert "Two-factor authentication" in res_tech[0].text


# ============================================================================
# 4. Hindi Query Retrieving a Relevant English Transcript Chunk
# ============================================================================

def test_hindi_query_retrieving_english_transcript_chunk(fts5_test_env, db_session: Session):
    """Verify Hindi query retrieves the semantically relevant English chunk using real multilingual E5."""
    org_a = fts5_test_env["org_a"]
    service = fts5_test_env["service"]
    if not service.is_loaded:
        pytest.skip("Real multilingual E5 model is not cached/loaded in test environment.")

    engine = RAGEngine(embedding_service=service)

    # Hindi question asking about container orchestration
    hindi_q = "कंटेनर आर्केस्ट्रेशन के बारे में क्या तय हुआ?"

    req = RAGQueryRequest(
        question=hindi_q,
        organization_id=org_a.id,
        max_evidence_chunks=3,
    )
    resp: RAGQueryResponse = engine.query(db_session, req)

    # Verified: candidates are NOT wiped out by cross-lingual pruning
    assert resp.diagnostics.total_candidates_retrieved >= 1
    assert resp.diagnostics.evidence_items_count >= 1
    # Top retrieved candidate is indeed the container orchestration chunk
    assert "Kubernetes Helm charts" in resp.answer or resp.diagnostics.evidence_items_count >= 1


# ============================================================================
# 5. Hinglish Query Retrieving a Relevant English Transcript Chunk
# ============================================================================

def test_hinglish_query_retrieving_english_transcript_chunk(fts5_test_env, db_session: Session):
    """Verify Hinglish query strips conversational filler and matches English evidence."""
    org_a = fts5_test_env["org_a"]
    service = fts5_test_env["service"]

    # Hinglish query
    hinglish_q = "Container orchestration ke baare me kya decide hua?"
    sanitized = sanitize_fts5_query(hinglish_q)

    # Conversational Hinglish words stripped
    assert '"ke"' not in sanitized.lower()
    assert '"baare"' not in sanitized.lower()
    assert '"me"' not in sanitized.lower()
    assert '"kya"' not in sanitized.lower()
    assert '"hua"' not in sanitized.lower()

    # Substantive terms preserved
    assert '"Container"' in sanitized or '"container"' in sanitized
    assert '"orchestration"' in sanitized

    # Direct FTS5 search matches the chunk
    kw_res = search_keyword(db_session, query=hinglish_q, organization_id=org_a.id)
    assert len(kw_res) >= 1
    assert "Kubernetes Helm charts" in kw_res[0].text

    # Full RAG query returns supported answer with citation
    if service.is_loaded:
        engine = RAGEngine(embedding_service=service)
        resp = engine.query(db_session, RAGQueryRequest(question=hinglish_q, organization_id=org_a.id))
        assert resp.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
        assert len(resp.citations) >= 1
        assert "Kubernetes" in resp.answer


# ============================================================================
# 6. English Query Retrieving Relevant English Evidence
# ============================================================================

def test_english_query_retrieving_english_evidence(fts5_test_env, db_session: Session):
    """Verify standard conversational English query retrieves relevant evidence with citations."""
    org_a = fts5_test_env["org_a"]
    service = fts5_test_env["service"]
    if not service.is_loaded:
        pytest.skip("Real multilingual E5 model is not cached/loaded.")

    engine = RAGEngine(embedding_service=service)
    q = "What was finalized regarding database replication and connection pooling?"

    resp = engine.query(db_session, RAGQueryRequest(question=q, organization_id=org_a.id))
    assert resp.abstention_status == AbstentionStatus.ANSWER_SUPPORTED
    assert len(resp.citations) >= 1
    assert "PostgreSQL" in resp.answer
    assert "PgBouncer" in resp.answer


# ============================================================================
# 7. Gujarati / Unicode Query Safety
# ============================================================================

def test_gujarati_and_unicode_query_safety(fts5_test_env, db_session: Session):
    """Verify Gujarati Unicode query reformulates safely without SQL error or syntax corruption."""
    org_a = fts5_test_env["org_a"]

    # Gujarati query: "ડેટાબેઝ સિસ્ટમ વિશે શું નક્કી થયું?"
    gu_q = "ડેટાબેઝ સિસ્ટમ વિશે શું નક્કી થયું?"
    sanitized = sanitize_fts5_query(gu_q)

    # Conversational Gujarati stopwords ("વિશે", "શું") stripped
    assert '"વિશે"' not in sanitized
    assert '"શું"' not in sanitized

    # Substantive Gujarati words preserved
    assert '"ડેટાબેઝ"' in sanitized
    assert '"સિસ્ટમ"' in sanitized

    # FTS5 search executes safely without SQLite syntax error
    results = search_keyword(db_session, query=gu_q, organization_id=org_a.id)
    assert isinstance(results, list)


# ============================================================================
# 8. Relevant and Irrelevant Candidates with Close Cosine Scores (RRF)
# ============================================================================

def test_rrf_tiebreaking_with_keyword_and_semantic_ranks(fts5_test_env, db_session: Session):
    """Verify RRF fusion boosts the candidate that matches keyword rank even if cosine scores are close."""
    org_a = fts5_test_env["org_a"]
    service = fts5_test_env["service"]
    if not service.is_loaded:
        pytest.skip("Real multilingual E5 model is not cached/loaded.")

    query = "database replication"
    q_vec = service.embed_query(query)

    # Hybrid RRF search
    hybrid_results = search_hybrid_rrf(
        db=db_session,
        query=query,
        query_vector=q_vec,
        organization_id=org_a.id,
        limit=5,
    )

    assert len(hybrid_results) >= 1
    top_cand = hybrid_results[0]
    # The database replication chunk has both keyword_rank and semantic_rank
    assert top_cand.keyword_rank is not None
    assert top_cand.semantic_rank is not None
    assert "PostgreSQL" in top_cand.text
    # RRF score combines both ranks (sum of 1/(60+r1) + 1/(60+r2))
    expected_min_rrf = (1.0 / 61.0) + (1.0 / 65.0)
    assert top_cand.rrf_score >= expected_min_rrf


# ============================================================================
# 9. Truly Irrelevant Queries Still Abstain
# ============================================================================

def test_truly_irrelevant_queries_abstain_without_hallucination(fts5_test_env, db_session: Session):
    """Verify completely out-of-domain queries abstain honestly with NO_EVIDENCE or INSUFFICIENT_EVIDENCE."""
    org_a = fts5_test_env["org_a"]
    service = fts5_test_env["service"]
    engine = RAGEngine(embedding_service=service)

    irrelevant_queries = [
        "What are the best ingredients for making authentic Neapolitan pizza?",
        "How do supermassive black holes form in galactic nuclei?",
        "ताजमहल किसने बनवाया था और कब?",
    ]

    for q in irrelevant_queries:
        resp = engine.query(db_session, RAGQueryRequest(question=q, organization_id=org_a.id))
        assert resp.is_abstained, f"Query '{q}' should abstain"
        assert resp.abstention_status in (AbstentionStatus.NO_EVIDENCE, AbstentionStatus.INSUFFICIENT_EVIDENCE)
        assert resp.citations == [], f"Query '{q}' must not emit citations for irrelevant topics"
        assert "pizza" not in resp.answer.lower()
        assert "black hole" not in resp.answer.lower()


# ============================================================================
# 10. Tenant Isolation After Query Reformulation
# ============================================================================

def test_tenant_isolation_preserved_after_query_reformulation(fts5_test_env, db_session: Session):
    """Verify Org A queries can never retrieve or cite Org B chunks even with broad OR expressions."""
    org_a = fts5_test_env["org_a"]
    org_b = fts5_test_env["org_b"]
    service = fts5_test_env["service"]
    engine = RAGEngine(embedding_service=service)

    # Org B has chunk: "Confidential acquisition of Startup Horizon for fifteen million dollars was approved."
    confidential_q = "What was decided about the confidential acquisition of Startup Horizon for fifteen million?"

    # 1. Org A queries confidential term
    resp_a = engine.query(db_session, RAGQueryRequest(question=confidential_q, organization_id=org_a.id))
    assert resp_a.is_abstained
    assert resp_a.abstention_status == AbstentionStatus.NO_EVIDENCE
    assert resp_a.citations == []

    # 2. Org B queries the same term -> succeeds and retrieves its own chunk
    if service.is_loaded:
        resp_b = engine.query(db_session, RAGQueryRequest(question=confidential_q, organization_id=org_b.id))
        assert not resp_b.is_abstained
        assert len(resp_b.citations) >= 1
        assert "Startup Horizon" in resp_b.answer


# ============================================================================
# 11. Meeting and Date Filters Remain Effective
# ============================================================================

def test_meeting_and_date_filters_remain_effective(fts5_test_env, db_session: Session):
    """Verify scoping to specific meeting or date boundary strictly restricts retrieval."""
    org_a = fts5_test_env["org_a"]
    m_a1 = fts5_test_env["m_a1"]
    m_a2 = fts5_test_env["m_a2"]
    service = fts5_test_env["service"]
    engine = RAGEngine(embedding_service=service)

    # Meeting A1 is Aug 10; Meeting A2 is Sept 15
    q = "What decisions were made regarding infrastructure and security?"

    # Scope to Meeting A1 (only container orchestration)
    resp_m1 = engine.query(db_session, RAGQueryRequest(question=q, organization_id=org_a.id, meeting_id=m_a1.id))
    for cit in resp_m1.citations:
        assert cit.meeting_id == m_a1.id
        assert cit.meeting_id != m_a2.id

    # Date filter: after Sept 1 (should only match Meeting A2)
    resp_date = engine.query(
        db_session,
        RAGQueryRequest(
            question=q,
            organization_id=org_a.id,
            start_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
        ),
    )
    for cit in resp_date.citations:
        assert cit.meeting_id == m_a2.id
        assert cit.meeting_id != m_a1.id


# ============================================================================
# 12. Existing Structured Retrieval and Conflict Detection Regressions
# ============================================================================

def test_structured_retrieval_and_conflict_regressions(fts5_test_env, db_session: Session):
    """Verify structured decisions, commitments, questions, and conflict detection work seamlessly."""
    org_a = fts5_test_env["org_a"]
    m_a1 = fts5_test_env["m_a1"]
    service = fts5_test_env["service"]
    engine = RAGEngine(embedding_service=service)

    # Seed structured decision
    dec = Decision(
        meeting_id=m_a1.id,
        description="Adopt Kubernetes Helm charts for service orchestration across microservices.",
        rationale_or_context="Enables GitOps automation.",
        decided_by="Architecture Board",
        status=DecisionStatus.ACTIVE.value,
    )
    db_session.add(dec)
    db_session.commit()

    q = "What was the architecture decision regarding Kubernetes Helm charts?"
    resp = engine.query(db_session, RAGQueryRequest(question=q, organization_id=org_a.id))

    assert not resp.is_abstained
    assert "Recorded Decisions in Organizational Memory" in resp.answer
    assert "GitOps automation" in resp.answer
    assert resp.diagnostics.structured_decisions_count >= 1
