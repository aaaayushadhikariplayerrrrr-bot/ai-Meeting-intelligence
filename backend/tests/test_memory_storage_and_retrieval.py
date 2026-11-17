"""Tests for Phase 8.2: SQLite FTS5, Persistent Vector Storage, and Hybrid Retrieval."""

import json
import os
import tempfile
import uuid
from typing import List
import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.identity import Organization
from app.models.meeting import Meeting
from app.models.memory import TranscriptChunkRecord
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.services.memory.chunking import SemanticTranscriptChunker
from app.services.memory.exceptions import (
    VectorValidationError,
    FTS5UnavailableError,
    PrivacyEnforcementError,
    TenantSecurityError,
    RetrievalError,
)
from app.services.memory.retrieval import (
    sanitize_fts5_query,
    search_keyword,
    search_semantic,
    search_hybrid_rrf,
)
from app.services.memory.storage import (
    serialize_embedding,
    deserialize_embedding,
    check_fts5_available,
    ensure_fts5_ready,
    ChunkStorageManager,
    EXPECTED_DIMENSION,
)
from app.services.memory.types import SemanticChunk, ChunkEmbeddingResult


def make_synthetic_vector(seed: int, dim: int = EXPECTED_DIMENSION) -> List[float]:
    """Generate a deterministic unit-normalized 384-dimensional synthetic vector."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return (vec / norm).tolist()


@pytest.fixture
def memory_test_setup(db_session: Session):
    """Fixture providing isolated Organizations, Meetings, and Transcripts for retrieval testing."""
    # Org A
    org_a = Organization(name="Memory Org A", slug=f"mem-org-a-{uuid.uuid4().hex[:8]}")
    # Org B
    org_b = Organization(name="Memory Org B", slug=f"mem-org-b-{uuid.uuid4().hex[:8]}")
    db_session.add_all([org_a, org_b])
    db_session.commit()

    # Meetings for Org A
    meeting_a1 = Meeting(organization_id=org_a.id, title="Org A Architecture Meeting")
    meeting_a2 = Meeting(organization_id=org_a.id, title="Org A Financial Review")
    # Meeting for Org B
    meeting_b1 = Meeting(organization_id=org_b.id, title="Org B Confidential Strategy")
    db_session.add_all([meeting_a1, meeting_a2, meeting_b1])
    db_session.commit()

    # Transcripts
    # Org A - Redacted transcript (Privacy safe)
    transcript_a1 = Transcript(
        meeting_id=meeting_a1.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
    )
    # Org A - Second Redacted transcript
    transcript_a2 = Transcript(
        meeting_id=meeting_a2.id,
        transcript_type=TranscriptType.REDACTED.value,
    )
    # Org A - Unredacted original transcript (Privacy unsafe)
    transcript_a_unsafe = Transcript(
        meeting_id=meeting_a1.id,
        transcript_type=TranscriptType.ORIGINAL.value,
    )
    # Org B - Redacted transcript
    transcript_b1 = Transcript(
        meeting_id=meeting_b1.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
    )

    db_session.add_all([transcript_a1, transcript_a2, transcript_a_unsafe, transcript_b1])
    db_session.commit()

    return {
        "org_a": org_a,
        "org_b": org_b,
        "meeting_a1": meeting_a1,
        "meeting_a2": meeting_a2,
        "meeting_b1": meeting_b1,
        "transcript_a1": transcript_a1,
        "transcript_a2": transcript_a2,
        "transcript_a_unsafe": transcript_a_unsafe,
        "transcript_b1": transcript_b1,
    }


# ============================================================================
# 1. Vector Serialization & Validation Tests
# ============================================================================

def test_vector_blob_serialization_and_deserialization():
    """Verify float32 binary serialization and roundtrip deserialization."""
    vec = make_synthetic_vector(seed=42)
    blob = serialize_embedding(vec)
    assert isinstance(blob, bytes)
    assert len(blob) == 384 * 4  # 1536 bytes

    restored = deserialize_embedding(blob)
    assert isinstance(restored, np.ndarray)
    assert restored.dtype == np.float32
    assert restored.shape == (384,)
    np.testing.assert_allclose(restored, vec, atol=1e-6)


def test_vector_validation_dimension_mismatch():
    """Verify rejection of vectors not matching configured 384 dimensions."""
    # 128-dim
    vec_128 = make_synthetic_vector(seed=1, dim=128)
    with pytest.raises(VectorValidationError, match="dimension mismatch"):
        serialize_embedding(vec_128, expected_dim=384)

    # 512-dim
    vec_512 = make_synthetic_vector(seed=2, dim=512)
    with pytest.raises(VectorValidationError, match="dimension mismatch"):
        serialize_embedding(vec_512, expected_dim=384)


def test_vector_validation_nan_and_infinite_values():
    """Verify rejection of NaN and infinite values."""
    vec = make_synthetic_vector(seed=3)
    vec[10] = float("nan")
    with pytest.raises(VectorValidationError, match="non-finite"):
        serialize_embedding(vec)

    vec[10] = float("inf")
    with pytest.raises(VectorValidationError, match="non-finite"):
        serialize_embedding(vec)


def test_vector_validation_unnormalized():
    """Verify rejection of non-unit-normalized vectors."""
    vec = [1.0] * 384  # Norm is sqrt(384) ~ 19.59
    with pytest.raises(VectorValidationError, match="not unit normalized"):
        serialize_embedding(vec)


# ============================================================================
# 2. FTS5 Availability & Query Sanitization Tests
# ============================================================================

def test_fts5_availability_check(db_session: Session):
    """Verify SQLite FTS5 extension is available and functional."""
    assert check_fts5_available(db_session) is True
    # Should not raise
    ensure_fts5_ready(db_session)


def test_fts5_query_sanitizer():
    """Verify FTS5 query sanitizer handles phrases, multilingual text, and special chars."""
    # Empty inputs
    assert sanitize_fts5_query("") == ""
    assert sanitize_fts5_query("   ") == ""

    # Exact phrase enclosed in quotes
    assert sanitize_fts5_query('"neural network"') == '"neural network"'

    # Multilingual tokens (Devanagari & Gujarati)
    sanitized_hi = sanitize_fts5_query("मीटिंग की मुख्य चर्चा")
    assert '"मीटिंग"' in sanitized_hi
    assert '"चर्चा"' in sanitized_hi

    sanitized_gu = sanitize_fts5_query("મીટિંગ પ્રોજેક્ટ પ્લાન")
    assert '"મીટિંગ"' in sanitized_gu
    assert '"પ્લાન"' in sanitized_gu

    # Operator injection and syntax symbols
    sanitized_symbols = sanitize_fts5_query("cloud: architecture AND (security OR *risk)")
    # Must wrap words into quotes without rogue symbols
    assert '"cloud"' in sanitized_symbols
    assert '"architecture"' in sanitized_symbols
    assert '"security"' in sanitized_symbols
    assert "(" not in sanitized_symbols
    assert "*" not in sanitized_symbols


# ============================================================================
# 3. Persistent Storage & FTS Synchronization Tests
# ============================================================================

def test_chunk_storage_and_fts_synchronization(db_session: Session, memory_test_setup):
    """Verify chunks and vectors persist to DB and synchronize to FTS5 virtual table."""
    setup = memory_test_setup
    org_id = setup["org_a"].id
    meeting_id = setup["meeting_a1"].id
    transcript_id = setup["transcript_a1"].id

    manager = ChunkStorageManager()
    chunk = SemanticChunk(
        chunk_index=0,
        meeting_id=meeting_id,
        organization_id=org_id,
        transcript_id=transcript_id,
        segment_ids=["seg-1", "seg-2"],
        text="The new distributed consensus protocol improves latency by 40 percent.",
        start_time=0.0,
        end_time=15.5,
        language_codes=["en"],
        dominant_language="en",
        token_count=14,
        is_split_utterance=False,
        metadata={"department": "Engineering"},
    )
    vec = make_synthetic_vector(seed=10)
    item = ChunkEmbeddingResult(
        chunk=chunk,
        embedding=vec,
        dimension=384,
        model_name="intfloat/multilingual-e5-small",
    )

    records = manager.store_chunks(db_session, [item], organization_id=org_id)
    assert len(records) == 1
    rec = records[0]

    # Verify relational fields
    assert rec.organization_id == org_id
    assert rec.meeting_id == meeting_id
    assert rec.transcript_id == transcript_id
    assert rec.text == chunk.text
    assert rec.segment_ids_list == ["seg-1", "seg-2"]
    assert rec.extra_metadata_dict == {"department": "Engineering"}
    assert len(rec.embedding) == 384 * 4

    # Verify FTS5 virtual table row
    fts_row = db_session.execute(
        text("SELECT chunk_id, text FROM transcript_chunks_fts WHERE chunk_id = :cid"),
        {"cid": rec.id},
    ).fetchone()
    assert fts_row is not None
    assert fts_row[0] == rec.id
    assert "distributed consensus protocol" in fts_row[1]


def test_privacy_enforcement_rejects_unredacted_transcripts(db_session: Session, memory_test_setup):
    """Strict Privacy: Never index ORIGINAL or CLEANED transcripts."""
    setup = memory_test_setup
    org_id = setup["org_a"].id
    unsafe_transcript = setup["transcript_a_unsafe"]

    manager = ChunkStorageManager()
    chunk = SemanticChunk(
        chunk_index=0,
        meeting_id=setup["meeting_a1"].id,
        organization_id=org_id,
        transcript_id=unsafe_transcript.id,
        segment_ids=["seg-unsafe"],
        text="Raw audio containing unredacted phone number 555-1234.",
        start_time=0.0,
        end_time=5.0,
        language_codes=["en"],
        dominant_language="en",
        token_count=10,
    )
    item = ChunkEmbeddingResult(
        chunk=chunk,
        embedding=make_synthetic_vector(seed=11),
        dimension=384,
        model_name="intfloat/multilingual-e5-small",
    )

    # 1. Reject via database transcript type check
    with pytest.raises(PrivacyEnforcementError, match="unredacted transcript type"):
        manager.store_chunks(db_session, [item], organization_id=org_id)

    # 2. Reject via metadata hint check
    chunk_hint = chunk.model_copy()
    chunk_hint.metadata = {"transcript_type": "ORIGINAL"}
    item_hint = ChunkEmbeddingResult(
        chunk=chunk_hint,
        embedding=make_synthetic_vector(seed=12),
        dimension=384,
        model_name="intfloat/multilingual-e5-small",
    )
    with pytest.raises(PrivacyEnforcementError, match="unredacted transcript type"):
        manager.store_chunks(db_session, [item_hint], organization_id=org_id)


# ============================================================================
# 4. Keyword Search & Phrase Matching with BM25 Tests
# ============================================================================

def test_fts5_keyword_and_phrase_search_with_bm25(db_session: Session, memory_test_setup):
    """Verify keyword match, exact phrase match, and BM25 rank ordering."""
    setup = memory_test_setup
    org_id = setup["org_a"].id
    meeting_id = setup["meeting_a1"].id
    transcript_id = setup["transcript_a1"].id

    manager = ChunkStorageManager()
    # Chunk 1: High keyword frequency
    c1 = SemanticChunk(
        chunk_index=0,
        meeting_id=meeting_id,
        organization_id=org_id,
        transcript_id=transcript_id,
        segment_ids=["s1"],
        text="kubernetes deployment kubernetes cluster kubernetes autoscaling",
        start_time=0.0,
        end_time=10.0,
        token_count=6,
    )
    # Chunk 2: Low keyword frequency + phrase
    c2 = SemanticChunk(
        chunk_index=1,
        meeting_id=meeting_id,
        organization_id=org_id,
        transcript_id=transcript_id,
        segment_ids=["s2"],
        text="We configured kubernetes with automated failover and secure replication.",
        start_time=10.0,
        end_time=20.0,
        token_count=10,
    )
    # Chunk 3: Unrelated
    c3 = SemanticChunk(
        chunk_index=2,
        meeting_id=meeting_id,
        organization_id=org_id,
        transcript_id=transcript_id,
        segment_ids=["s3"],
        text="The quarterly revenue figures will be announced tomorrow.",
        start_time=20.0,
        end_time=30.0,
        token_count=8,
    )

    items = [
        ChunkEmbeddingResult(chunk=c1, embedding=make_synthetic_vector(seed=21), dimension=384, model_name="m"),
        ChunkEmbeddingResult(chunk=c2, embedding=make_synthetic_vector(seed=22), dimension=384, model_name="m"),
        ChunkEmbeddingResult(chunk=c3, embedding=make_synthetic_vector(seed=23), dimension=384, model_name="m"),
    ]
    recs = manager.store_chunks(db_session, items, organization_id=org_id)

    # Keyword search: "kubernetes"
    kw_results = search_keyword(db_session, query="kubernetes", organization_id=org_id)
    assert len(kw_results) == 2
    # c1 has higher term frequency, should be rank 1
    assert kw_results[0].chunk_id == recs[0].id
    assert kw_results[0].bm25_rank == 1
    assert kw_results[1].chunk_id == recs[1].id
    assert kw_results[1].bm25_rank == 2

    # Exact Phrase Search: "automated failover"
    phrase_results = search_keyword(db_session, query='"automated failover"', organization_id=org_id)
    assert len(phrase_results) == 1
    assert phrase_results[0].chunk_id == recs[1].id


# ============================================================================
# 5. Semantic Search & NumPy Cosine Similarity Tests
# ============================================================================

def test_semantic_similarity_ranking_with_numpy(db_session: Session, memory_test_setup):
    """Verify NumPy cosine similarity ranking against persistent BLOB vectors."""
    setup = memory_test_setup
    org_id = setup["org_a"].id
    meeting_id = setup["meeting_a1"].id
    transcript_id = setup["transcript_a1"].id

    vec1 = make_synthetic_vector(seed=31)
    vec2 = make_synthetic_vector(seed=32)
    vec3 = make_synthetic_vector(seed=33)

    manager = ChunkStorageManager()
    c1 = SemanticChunk(chunk_index=0, meeting_id=meeting_id, organization_id=org_id, transcript_id=transcript_id, segment_ids=["s1"], text="Topic 1", start_time=0.0, end_time=1.0, token_count=2)
    c2 = SemanticChunk(chunk_index=1, meeting_id=meeting_id, organization_id=org_id, transcript_id=transcript_id, segment_ids=["s2"], text="Topic 2", start_time=1.0, end_time=2.0, token_count=2)
    c3 = SemanticChunk(chunk_index=2, meeting_id=meeting_id, organization_id=org_id, transcript_id=transcript_id, segment_ids=["s3"], text="Topic 3", start_time=2.0, end_time=3.0, token_count=2)

    items = [
        ChunkEmbeddingResult(chunk=c1, embedding=vec1, dimension=384, model_name="m"),
        ChunkEmbeddingResult(chunk=c2, embedding=vec2, dimension=384, model_name="m"),
        ChunkEmbeddingResult(chunk=c3, embedding=vec3, dimension=384, model_name="m"),
    ]
    recs = manager.store_chunks(db_session, items, organization_id=org_id)

    # Query using vec2: should match c2 with cosine similarity near 1.0
    results = search_semantic(db_session, query_vector=vec2, organization_id=org_id)
    assert len(results) == 3
    assert results[0].chunk_id == recs[1].id
    assert results[0].semantic_rank == 1
    assert pytest.approx(results[0].cosine_similarity, abs=1e-4) == 1.0


# ============================================================================
# 6. Tenant-Isolation & Cross-Tenant Security Tests
# ============================================================================

def test_organization_level_isolation(db_session: Session, memory_test_setup):
    """Critical Security: Org A cannot retrieve or see Org B records in FTS, vector, or hybrid search."""
    setup = memory_test_setup
    org_a_id = setup["org_a"].id
    org_b_id = setup["org_b"].id

    manager = ChunkStorageManager()

    # Store identical keywords and vectors in both Org A and Org B
    shared_text = "Project Apollo launch date and budget allocation."
    vec_shared = make_synthetic_vector(seed=41)

    chunk_a = SemanticChunk(
        chunk_index=0,
        meeting_id=setup["meeting_a1"].id,
        organization_id=org_a_id,
        transcript_id=setup["transcript_a1"].id,
        segment_ids=["sa1"],
        text=shared_text,
        start_time=0.0,
        end_time=10.0,
        token_count=7,
    )
    chunk_b = SemanticChunk(
        chunk_index=0,
        meeting_id=setup["meeting_b1"].id,
        organization_id=org_b_id,
        transcript_id=setup["transcript_b1"].id,
        segment_ids=["sb1"],
        text=shared_text,
        start_time=0.0,
        end_time=10.0,
        token_count=7,
    )

    recs_a = manager.store_chunks(
        db_session,
        [ChunkEmbeddingResult(chunk=chunk_a, embedding=vec_shared, dimension=384, model_name="m")],
        organization_id=org_a_id,
    )
    recs_b = manager.store_chunks(
        db_session,
        [ChunkEmbeddingResult(chunk=chunk_b, embedding=vec_shared, dimension=384, model_name="m")],
        organization_id=org_b_id,
    )

    # 1. Keyword search in Org A
    kw_a = search_keyword(db_session, query="Apollo budget", organization_id=org_a_id)
    assert len(kw_a) == 1
    assert kw_a[0].chunk_id == recs_a[0].id
    assert kw_a[0].organization_id == org_a_id

    # 2. Semantic search in Org A
    sem_a = search_semantic(db_session, query_vector=vec_shared, organization_id=org_a_id)
    assert len(sem_a) == 1
    assert sem_a[0].chunk_id == recs_a[0].id
    assert sem_a[0].organization_id == org_a_id

    # 3. Hybrid search in Org A
    hyb_a = search_hybrid_rrf(db_session, query="Apollo budget", query_vector=vec_shared, organization_id=org_a_id)
    assert len(hyb_a) == 1
    assert hyb_a[0].chunk_id == recs_a[0].id
    assert hyb_a[0].organization_id == org_a_id

    # 4. Cross-tenant meeting ID attack: Org A credentials with Org B meeting ID
    cross_kw = search_keyword(db_session, query="Apollo", organization_id=org_a_id, meeting_id=setup["meeting_b1"].id)
    assert len(cross_kw) == 0

    cross_sem = search_semantic(db_session, query_vector=vec_shared, organization_id=org_a_id, meeting_id=setup["meeting_b1"].id)
    assert len(cross_sem) == 0


def test_meeting_level_filtering(db_session: Session, memory_test_setup):
    """Verify meeting_id filtering narrows search within an organization."""
    setup = memory_test_setup
    org_id = setup["org_a"].id
    m1_id = setup["meeting_a1"].id
    m2_id = setup["meeting_a2"].id

    manager = ChunkStorageManager()
    c1 = SemanticChunk(chunk_index=0, meeting_id=m1_id, organization_id=org_id, transcript_id=setup["transcript_a1"].id, segment_ids=["s1"], text="Database migration strategy for meeting 1", start_time=0.0, end_time=5.0, token_count=6)
    c2 = SemanticChunk(chunk_index=0, meeting_id=m2_id, organization_id=org_id, transcript_id=setup["transcript_a2"].id, segment_ids=["s2"], text="Database migration strategy for meeting 2", start_time=0.0, end_time=5.0, token_count=6)

    vec = make_synthetic_vector(seed=51)
    recs = manager.store_chunks(
        db_session,
        [
            ChunkEmbeddingResult(chunk=c1, embedding=vec, dimension=384, model_name="m"),
            ChunkEmbeddingResult(chunk=c2, embedding=vec, dimension=384, model_name="m"),
        ],
        organization_id=org_id,
    )

    # Without filter -> returns both
    all_res = search_keyword(db_session, query="migration", organization_id=org_id)
    assert len(all_res) == 2

    # Filtered by meeting 1
    m1_res = search_keyword(db_session, query="migration", organization_id=org_id, meeting_id=m1_id)
    assert len(m1_res) == 1
    assert m1_res[0].chunk_id == recs[0].id

    # Filtered by meeting 2
    m2_res = search_semantic(db_session, query_vector=vec, organization_id=org_id, meeting_id=m2_id)
    assert len(m2_res) == 1
    assert m2_res[0].chunk_id == recs[1].id


# ============================================================================
# 7. Hybrid Retrieval (RRF) & Deduplication Tests
# ============================================================================

def test_hybrid_rrf_ranking_and_deduplication(db_session: Session, memory_test_setup):
    """Verify Reciprocal Rank Fusion combines, deduplicates, and preserves individual ranks."""
    setup = memory_test_setup
    org_id = setup["org_a"].id
    meeting_id = setup["meeting_a1"].id
    transcript_id = setup["transcript_a1"].id

    manager = ChunkStorageManager()

    # Query vectors
    q_vec = make_synthetic_vector(seed=60)
    # v1 is very close to q_vec (high semantic match)
    v1 = make_synthetic_vector(seed=60)
    # v2 is far from q_vec (low semantic match)
    v2 = make_synthetic_vector(seed=99)
    # v3 is intermediate
    v3 = make_synthetic_vector(seed=61)

    # c1: Matches both keyword ("kubernetes") and semantic
    c1 = SemanticChunk(chunk_index=0, meeting_id=meeting_id, organization_id=org_id, transcript_id=transcript_id, segment_ids=["s1"], text="Kubernetes production pod scalability guidelines", start_time=0.0, end_time=10.0, token_count=5)
    # c2: Matches keyword ("kubernetes") only
    c2 = SemanticChunk(chunk_index=1, meeting_id=meeting_id, organization_id=org_id, transcript_id=transcript_id, segment_ids=["s2"], text="Legacy kubernetes cluster decommissioning note", start_time=10.0, end_time=20.0, token_count=5)
    # c3: Matches semantic (close vector) but different words
    c3 = SemanticChunk(chunk_index=2, meeting_id=meeting_id, organization_id=org_id, transcript_id=transcript_id, segment_ids=["s3"], text="Container orchestration engine horizontally scaling workloads", start_time=20.0, end_time=30.0, token_count=6)

    items = [
        ChunkEmbeddingResult(chunk=c1, embedding=v1, dimension=384, model_name="m"),
        ChunkEmbeddingResult(chunk=c2, embedding=v2, dimension=384, model_name="m"),
        ChunkEmbeddingResult(chunk=c3, embedding=v3, dimension=384, model_name="m"),
    ]
    recs = manager.store_chunks(db_session, items, organization_id=org_id)

    # Hybrid Search: query="kubernetes", query_vector=q_vec, k=60
    hybrid_results = search_hybrid_rrf(
        db=db_session,
        query="kubernetes",
        query_vector=q_vec,
        organization_id=org_id,
        k=60,
    )

    assert len(hybrid_results) >= 2
    # c1 should be rank 1 because it ranks in both keyword and semantic lists
    top = hybrid_results[0]
    assert top.chunk_id == recs[0].id
    assert top.keyword_rank is not None
    assert top.semantic_rank is not None
    assert top.keyword_score is not None
    assert top.semantic_score is not None
    # Validate RRF formula: sum(1 / (60 + rank))
    expected_rrf = (1.0 / (60 + top.keyword_rank)) + (1.0 / (60 + top.semantic_rank))
    assert pytest.approx(top.rrf_score, abs=1e-6) == expected_rrf

    # Verify deduplication: every chunk appears at most once
    chunk_ids = [r.chunk_id for r in hybrid_results]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_empty_index_behavior(db_session: Session, memory_test_setup):
    """Verify graceful handling when searching an organization with zero chunks."""
    setup = memory_test_setup
    empty_org_id = setup["org_b"].id
    vec = make_synthetic_vector(seed=70)

    assert search_keyword(db_session, query="anything", organization_id=empty_org_id) == []
    assert search_semantic(db_session, query_vector=vec, organization_id=empty_org_id) == []
    assert search_hybrid_rrf(db_session, query="anything", query_vector=vec, organization_id=empty_org_id) == []


# ============================================================================
# 8. Synchronization, Deletion, and Rollback Tests
# ============================================================================

def test_chunk_update_and_fts_synchronization(db_session: Session, memory_test_setup):
    """Verify chunk text and embedding update cleanly synchronize with FTS5."""
    setup = memory_test_setup
    org_id = setup["org_a"].id
    meeting_id = setup["meeting_a1"].id
    transcript_id = setup["transcript_a1"].id

    manager = ChunkStorageManager()
    chunk = SemanticChunk(chunk_index=0, meeting_id=meeting_id, organization_id=org_id, transcript_id=transcript_id, segment_ids=["s1"], text="Initial un-updated text version.", start_time=0.0, end_time=5.0, token_count=4)
    rec = manager.store_chunks(
        db_session,
        [ChunkEmbeddingResult(chunk=chunk, embedding=make_synthetic_vector(seed=81), dimension=384, model_name="m")],
        organization_id=org_id,
    )[0]

    # Verify initial search finds initial text
    assert len(search_keyword(db_session, query="un-updated", organization_id=org_id)) == 1

    # Update chunk text
    new_text = "Updated version mentioning quantum cryptography safeguards."
    manager.update_chunk(db_session, chunk_id=rec.id, organization_id=org_id, text_content=new_text)

    # Old term no longer matches
    assert len(search_keyword(db_session, query="un-updated", organization_id=org_id)) == 0
    # New term matches
    new_res = search_keyword(db_session, query="quantum cryptography", organization_id=org_id)
    assert len(new_res) == 1
    assert new_res[0].chunk_id == rec.id


def test_storage_deletion_by_transcript_and_meeting(db_session: Session, memory_test_setup):
    """Verify chunk and FTS deletions are strictly synchronized and tenant-isolated."""
    setup = memory_test_setup
    org_id = setup["org_a"].id
    m1_id = setup["meeting_a1"].id
    t1_id = setup["transcript_a1"].id
    m2_id = setup["meeting_a2"].id
    t2_id = setup["transcript_a2"].id

    manager = ChunkStorageManager()
    c1 = SemanticChunk(chunk_index=0, meeting_id=m1_id, organization_id=org_id, transcript_id=t1_id, segment_ids=["s1"], text="Transcript 1 specific information", start_time=0.0, end_time=5.0, token_count=4)
    c2 = SemanticChunk(chunk_index=0, meeting_id=m2_id, organization_id=org_id, transcript_id=t2_id, segment_ids=["s2"], text="Transcript 2 specific information", start_time=0.0, end_time=5.0, token_count=4)

    items = [
        ChunkEmbeddingResult(chunk=c1, embedding=make_synthetic_vector(seed=91), dimension=384, model_name="m"),
        ChunkEmbeddingResult(chunk=c2, embedding=make_synthetic_vector(seed=92), dimension=384, model_name="m"),
    ]
    recs = manager.store_chunks(db_session, items, organization_id=org_id)

    # 1. Delete by transcript 1
    del_count = manager.delete_chunks_by_transcript(db_session, transcript_id=t1_id, organization_id=org_id)
    assert del_count == 1

    # Verify chunk 1 is gone from DB and FTS
    assert manager.get_chunk_by_id(db_session, recs[0].id, org_id) is None
    fts_c1 = db_session.execute(text("SELECT count(*) FROM transcript_chunks_fts WHERE chunk_id = :cid"), {"cid": recs[0].id}).scalar()
    assert fts_c1 == 0

    # Verify chunk 2 remains
    assert manager.get_chunk_by_id(db_session, recs[1].id, org_id) is not None

    # 2. Delete by meeting 2
    del_m2 = manager.delete_chunks_by_meeting(db_session, meeting_id=m2_id, organization_id=org_id)
    assert del_m2 == 1
    assert manager.get_chunk_by_id(db_session, recs[1].id, org_id) is None


def test_transaction_rollback_preserves_consistency(db_session: Session, memory_test_setup):
    """Verify transaction rollback reverts both relational chunks and FTS5 entries atomically."""
    setup = memory_test_setup
    org_id = setup["org_a"].id
    meeting_id = setup["meeting_a1"].id
    transcript_id = setup["transcript_a1"].id

    manager = ChunkStorageManager()
    chunk = SemanticChunk(
        chunk_index=0,
        meeting_id=meeting_id,
        organization_id=org_id,
        transcript_id=transcript_id,
        segment_ids=["s1"],
        text="Atomic rollback test chunk text.",
        start_time=0.0,
        end_time=5.0,
        token_count=5,
    )
    item = ChunkEmbeddingResult(
        chunk=chunk,
        embedding=make_synthetic_vector(seed=101),
        dimension=384,
        model_name="m",
    )

    try:
        # Use nested transaction / savepoint
        with db_session.begin_nested():
            recs = manager.store_chunks(db_session, [item], organization_id=org_id)
            cid = recs[0].id
            # Intentionally raise an error to trigger rollback
            raise RuntimeError("Forced simulation error for rollback test")
    except RuntimeError:
        pass

    # Verify neither relational table nor FTS5 retained the chunk
    rel_count = db_session.query(TranscriptChunkRecord).filter(TranscriptChunkRecord.id == cid).count()
    assert rel_count == 0

    fts_count = db_session.execute(
        text("SELECT count(*) FROM transcript_chunks_fts WHERE chunk_id = :cid"),
        {"cid": cid},
    ).scalar()
    assert fts_count == 0


# ============================================================================
# 9. End-to-End Phase 8.1 Chunker Integration Test
# ============================================================================

def test_phase_8_1_chunker_integration(db_session: Session, memory_test_setup):
    """Verify SemanticTranscriptChunker chunks feed directly into ChunkStorageManager and hybrid retrieval."""
    setup = memory_test_setup
    org_id = setup["org_a"].id
    meeting_id = setup["meeting_a1"].id
    transcript_id = setup["transcript_a1"].id

    # Create TranscriptSegments
    seg1 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript_id,
        sequence_order=0,
        start_time=0.0,
        end_time=5.0,
        text="The client requested support for multilingual translation.",
        language_code="en",
    )
    seg2 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript_id,
        sequence_order=1,
        start_time=5.0,
        end_time=10.0,
        text="અમે નવી સુવિધા આગામી સપ્તાહમાં રિલીઝ કરીશું.",  # Gujarati
        language_code="gu",
    )
    db_session.add_all([seg1, seg2])
    db_session.commit()

    # Chunk with SemanticTranscriptChunker
    chunker = SemanticTranscriptChunker(target_tokens=100)
    chunks = chunker.chunk_segments(
        segments=[seg1, seg2],
        meeting_id=meeting_id,
        organization_id=org_id,
        transcript_id=transcript_id,
    )
    assert len(chunks) >= 1

    # Attach synthetic vectors
    items = []
    for idx, c in enumerate(chunks):
        vec = make_synthetic_vector(seed=200 + idx)
        items.append(ChunkEmbeddingResult(chunk=c, embedding=vec, dimension=384, model_name="intfloat/multilingual-e5-small"))

    # Persist via storage manager
    manager = ChunkStorageManager()
    persisted = manager.store_chunks(db_session, items, organization_id=org_id)
    assert len(persisted) == len(chunks)

    # Retrieve via hybrid search
    query_vec = make_synthetic_vector(seed=200)
    res = search_hybrid_rrf(
        db=db_session,
        query="translation રિલીઝ",
        query_vector=query_vec,
        organization_id=org_id,
    )
    assert len(res) >= 1
    assert res[0].organization_id == org_id
    assert res[0].rrf_score > 0.0


# ============================================================================
# 10. Alembic Migration Upgrade & Downgrade Test
# ============================================================================

def test_alembic_migration_upgrade_and_downgrade():
    """Verify Alembic migration upgrade, check, and downgrade cycle on a fresh database."""
    from alembic.config import Config
    from alembic import command

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db_path = f.name

    try:
        ini_path = "alembic.ini" if os.path.exists("alembic.ini") else "backend/alembic.ini"
        cfg = Config(ini_path)
        clean_url = "sqlite:///" + temp_db_path.replace("\\", "/")
        cfg.set_main_option("sqlalchemy.url", clean_url)

        # 1. Upgrade to head
        command.upgrade(cfg, "head")

        # 2. Check 0 schema drift
        command.check(cfg)

        # 3. Downgrade to d407b60ca77f
        command.downgrade(cfg, "d407b60ca77f")

        # 4. Re-upgrade to head
        command.upgrade(cfg, "head")
        command.check(cfg)
    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
